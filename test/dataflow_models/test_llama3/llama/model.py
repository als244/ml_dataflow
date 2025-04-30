# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
import numpy as np
import time
import pickle



SAVE_LAYER_ID = -1
SAVE_ITER = 0

SAVE_DIR = "/mnt/storage/research/ml_dataflow/correct_transformer_data"
TOKEN_ITERS_TO_PRINT = 1



@dataclass
class ModelArgs:
    dim: int = 8192
    n_layers: int = 80
    n_heads: int = 64
    n_kv_heads: Optional[int] = 8
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: Optional[bool] = None
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    xk_out_typed = xk_out.type_as(xk)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

        self.cur_iter = 0
        self.layer_id = layer_id

        self.layer_save_dir = SAVE_DIR + f"/layers_fwd/{layer_id}"

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        
        
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(x.to("cpu"), self.layer_save_dir + "/x_attn_in.pt")
            torch.save(xq.to("cpu"), self.layer_save_dir + "/xq.pt")
            torch.save(xk.to("cpu"), self.layer_save_dir + "/xk.pt")
            torch.save(xv.to("cpu"), self.layer_save_dir + "/xv.pt")

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(xq.reshape(bsz * seqlen, self.head_dim * self.n_local_heads).to("cpu"), self.layer_save_dir + "/xq_rope.pt")
            torch.save(xk.reshape(bsz * seqlen, self.head_dim * self.n_kv_heads).to("cpu"), self.layer_save_dir + "/xk_rope.pt")

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        
        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(xq.to("cpu"), self.layer_save_dir + "/pre_attn_xq.pt")
            torch.save(keys.transpose(2,3).to("cpu"),self.layer_save_dir + "/pre_attn_keys.pt")
            torch.save(values.to("cpu"), self.layer_save_dir + "/pre_attn_value.pt")

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
                torch.save(mask.to("cpu"), self.layer_save_dir + "/attn_mask.pt")
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)


        final_out = self.wo(output)
        
        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(output.to("cpu"),self.layer_save_dir + "/attn_scores_out.pt")

        self.cur_iter += 1

        return final_out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cur_iter = 0

        self.layer_save_dir = SAVE_DIR + f"/layers_fwd/{layer_id}"

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        
        norm = self.attention_norm(x)

        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(x.to("cpu"),self.layer_save_dir + "/block_inp.pt")
            torch.save(norm.to("cpu"), self.layer_save_dir + "/post_norm.pt")

        h = x + self.attention(norm, start_pos, freqs_cis, mask)

        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(h.to("cpu"), self.layer_save_dir + "/attn_final_out.pt")

        ffn_norm = self.ffn_norm(h)

        out = h + self.feed_forward(ffn_norm)

        if (self.cur_iter == SAVE_ITER and (SAVE_LAYER_ID == -1 or self.layer_id == SAVE_LAYER_ID)):
            torch.save(ffn_norm.to("cpu"), self.layer_save_dir + "/ffn_norm_out.pt")
            torch.save(out.to("cpu"), self.layer_save_dir + "/block_out.pt")

        self.cur_iter += 1
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        ## MODIFIED FOR LOGGING
        self.iters_to_print = 1
        self.cur_iter = 0

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        
        inp_device = tokens.device

        if (inp_device.type == "cuda"):
            torch.cuda.synchronize()

        orig_start = time.time_ns()

        _bsz, seqlen = tokens.shape

        if (self.cur_iter == SAVE_ITER):
            torch.save(tokens.to("cpu"), SAVE_DIR + "/token_ids.pt")
            print(f"Forward tokens: {tokens}")

        h = self.tok_embeddings(tokens)

        if (self.cur_iter == SAVE_ITER):
            torch.save(h.to("cpu"), SAVE_DIR + "/tok_embeddings.pt")

        self.freqs_cis = self.freqs_cis.to(h.device)

        #print(f"Start Pos: {start_pos}")

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if (self.cur_iter < TOKEN_ITERS_TO_PRINT):
            print(f"Input tokens:\n\t{tokens}\n\n")
            # print(f"Token embeddings:\n\t{h}\n\n")
            # print(f"Full freq cis shape:\n\t{self.freqs_cis.shape}\n\n")
            # print(f"Freq cis shape:\n\t{freqs_cis.shape}\n\n")
            # print(f"Freq cis:\n\t{freqs_cis}\n\n")

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=inp_device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=inp_device), mask]
            ).type_as(h)
        
        if (inp_device.type == "cuda"):
            torch.cuda.synchronize()

        stop = time.time_ns()
        time_ms = (stop - orig_start) / 1e6
        
        if (self.cur_iter < self.iters_to_print):
            print(f"Prepped for Layer 1\n\tBatch Size: {_bsz}\n\tSeq Len: {seqlen}\n\tRuntime: {time_ms} ms\n")
        
        i = 1
        for layer in self.layers:
            start = time.time_ns()
            h = layer(h, start_pos, freqs_cis, mask)
            
            if (inp_device.type == "cuda"):
                torch.cuda.synchronize()

            stop = time.time_ns()
            time_ms = (stop - start) / 1e6
            
            if (self.cur_iter < self.iters_to_print):
                print(f"Layer {i}\n\tRuntime: {time_ms} ms\n")
            i += 1

        if (inp_device.type == "cuda"):
            torch.cuda.synchronize()
        
        start = time.time_ns()

        if (self.cur_iter == SAVE_ITER):
            torch.save(h.to("cpu"), SAVE_DIR + "/head_fwd/head_inp.pt")

        h = self.norm(h)
        output = self.output(h)

        if (inp_device.type == "cuda"):
            torch.cuda.synchronize()

        if (self.cur_iter == SAVE_ITER):
            torch.save(h.to("cpu"), SAVE_DIR + "/head_fwd/head_norm.pt")
            torch.save(output.to("cpu"), SAVE_DIR + "/head_fwd/head_out.pt")

        if (inp_device.type == "cuda"):
            torch.cuda.synchronize()

        output = output.float()

        if (self.cur_iter == SAVE_ITER):
            torch.save(output.to("cpu"), SAVE_DIR + "/head_fwd/head_float.pt")

        logits = F.softmax(output, dim=-1).to(h.dtype)

        if (self.cur_iter == SAVE_ITER):
            torch.save(logits.to("cpu"), SAVE_DIR + "/head_fwd/logits.pt")

        final_stop = time.time_ns()

        time_ms = (final_stop - start) / 1e6
        
        if (self.cur_iter < self.iters_to_print):
            print(f"Finalized Output:\n\tBatch Size: {_bsz}\n\tSeq Len: {seqlen}\n\tRuntime: {time_ms} ms\n")


        overall_time = (final_stop - orig_start) / 1e6
        
        if (self.cur_iter < self.iters_to_print):
            print(f"OVERALL FORWARD RUNTIME:\n\tBatch Size {_bsz}\n\tSeq Len: {seqlen}\n\tRuntime: {overall_time} ms\n\n")
        
        self.cur_iter += 1

        return output
