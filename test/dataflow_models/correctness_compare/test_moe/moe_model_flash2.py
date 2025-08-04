# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import pickle
from flash_attn import flash_attn_varlen_func

TO_SAVE = False
SAVE_DIR = "/home/shein/Documents/grad_school/research/ml_dataflow_stuff/ml_dataflow/test/dataflow_models/correct_transformer_data"


@dataclass
class ModelArgs:
    embed_dtype: str = "bf16"
    attn_dtype: str = "bf16"
    router_dtype: str = "bf16"
    expert_dtype: str = "bf16"
    head_dtype: str = "bf16"
    vocab_size: int = 128256
    num_layers: int = 8
    model_dim: int = 1536
    num_q_heads: int = 24
    num_kv_heads: int = 3
    qk_norm_type: str = None
    qk_norm_weight_type: str = None
    num_shared_experts: int = 0
    num_routed_experts: int = 64
    top_k_routed_experts: int = 4
    expert_dim: int = 768
    expert_mlp_type: str = "swiglu"
    rope_theta: float = 500000
    rms_norm_epsilon: float = 1e-5
    max_seq_len: int = 1048576


def dtype_to_torch_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "fp32":
        return torch.float32
    elif dtype == "none":
        return None
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

class SeqlensInfo:
    def __init__(self, seqlens_q_np, seqlens_k_np, device):

        self.cu_seqlens_q = np.zeros(len(seqlens_q_np) + 1, dtype=np.int32)
        self.cu_seqlens_q[1:] = np.cumsum(seqlens_q_np)
        self.cu_seqlens_q = torch.from_numpy(self.cu_seqlens_q).to(device)
        self.cu_seqlens_k = np.zeros(len(seqlens_k_np) + 1, dtype=np.int32)
        self.cu_seqlens_k[1:] = np.cumsum(seqlens_k_np)
        self.cu_seqlens_k = torch.from_numpy(self.cu_seqlens_k).to(device)

        self.max_seqlen_q = np.max(seqlens_q_np)
        self.max_seqlen_k = np.max(seqlens_k_np)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)


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
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.num_kv_heads = args.num_q_heads if args.num_kv_heads is None else args.num_kv_heads
        self.n_local_heads = args.num_q_heads
        self.n_local_kv_heads = self.num_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.model_dim // args.num_q_heads

        self.attn_dtype = dtype_to_torch_dtype(args.attn_dtype)

        self.wq = nn.Linear(args.model_dim, args.num_q_heads * self.head_dim, bias=False, dtype=self.attn_dtype)
        self.wk = nn.Linear(args.model_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=self.attn_dtype)
        self.wv = nn.Linear(args.model_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=self.attn_dtype)
        
        self.wo = nn.Linear(args.num_q_heads * self.head_dim, args.model_dim, bias=False, dtype=self.attn_dtype)

        self.layer_id = layer_id

        self.layer_save_dir = SAVE_DIR + f"/layers_fwd/{layer_id}"

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seqlens_info: SeqlensInfo,
        mask: Optional[torch.Tensor],
        step_num: int,
    ):
        
        bsz, seqlen, _ = x.shape
        
        # if TO_SAVE:
        #     torch.save(x.cpu().view(bsz * seqlen, -1), f"flash_layers/{self.layer_id}/step_{step_num}_fwd_attn_x.pt")

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        output = flash_attn_varlen_func(
            xq, xk, xv,
            seqlens_info.cu_seqlens_q, seqlens_info.cu_seqlens_k,
            seqlens_info.max_seqlen_q, seqlens_info.max_seqlen_k,
            deterministic=True,
            causal=True
        )

        output = output.view(bsz, seqlen, -1)

        """
        # repeat k/v heads if num_kv_heads < num_q_heads
        keys = repeat_kv(
            xk, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            xv, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        """

        final_out = self.wo(output)

        return final_out
    
class MLP(nn.Module):
    def __init__(self, model_dim, expert_dim, expert_id, layer_id, expert_dtype=torch.bfloat16):
        super().__init__()
        self.expert_dtype = expert_dtype
        self.model_dim = model_dim
        self.expert_dim = expert_dim
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.gate_proj = nn.Linear(model_dim, expert_dim, bias=False, dtype=expert_dtype)
        self.up_proj = nn.Linear(model_dim, expert_dim, bias=False, dtype=expert_dtype)
        self.down_proj = nn.Linear(expert_dim, model_dim, bias=False, dtype=expert_dtype)

    def forward(self, x, step_num):

        x1 = self.gate_proj(x)
        x3 = self.up_proj(x)

        swiglu_out = F.silu(x1) * x3

        down_proj = self.down_proj(swiglu_out)

        if TO_SAVE and self.layer_id == 0 and self.expert_id == 0:

            print(f"w3 weight: {self.down_proj.weight}")
            torch.save(self.up_proj.weight.cpu().view(-1, self.expert_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{self.expert_id}_w3.pt")

            print(f"[Step {step_num}] Saving MLP inp for layer {self.layer_id} and expert {self.expert_id}...")
            torch.save(x.cpu().view(-1, self.model_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{self.expert_id}_inp.pt")

            print(f"[Step {step_num}] Saving MLP w1 for layer {self.layer_id} and expert {self.expert_id}...")
            torch.save(x1.cpu().view(-1, self.expert_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{self.expert_id}_w1.pt")

            print(f"[Step {step_num}] Saving MLP w3 for layer {self.layer_id} and expert {self.expert_id}...")
            torch.save(x3.cpu().view(-1, self.expert_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{self.expert_id}_w3.pt")

            print(f"[Step {step_num}] Saving MLP swiglu out for layer {self.layer_id} and expert {self.expert_id}...")
            torch.save(swiglu_out.cpu().view(-1, self.expert_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{self.expert_id}_swiglu.pt")
        
            print(f"[Step {step_num}] Saving MLP out for layer {self.layer_id} and expert {self.expert_id}...")
            torch.save(down_proj.cpu().view(-1, self.model_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{self.expert_id}_out.pt")

        return down_proj
    

def deterministic_topk(input_tensor: torch.Tensor, k: int, dim: int = -1, largest: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A robust and deterministic implementation of torch.topk that tie-breaks by
    choosing the element with the lower index.

    This version is numerically stable and handles broadcasting correctly.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        k (int): The number of top elements to retrieve.
        dim (int, optional): The dimension along which to find the top elements. Defaults to -1.
        largest (bool, optional): If True, finds the k largest elements.
                                  If False, finds the k smallest elements. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the top-k values and their indices.
    """
    # Ensure the dimension is non-negative
    if dim < 0:
        dim = input_tensor.dim() + dim

    # Promote to a higher-precision float for the composite key calculation
    input_float = input_tensor.to(torch.float32)
    
    # Create the indices tensor and shape it for broadcasting
    n = input_tensor.shape[dim]
    view_shape = [1] * input_tensor.dim()
    view_shape[dim] = n
    indices = torch.arange(n, device=input_tensor.device).view(view_shape)

    # --- Create the composite key ---
    # We add a tiny epsilon scaled by the index to break ties.
    # Epsilon must be small enough not to affect the original value order.
    epsilon = 1e-6

    if largest:
        # For largest=True, we want to minimize (-value + index * epsilon)
        composite_key = -input_float + indices * epsilon
    else:
        # For smallest=True, we want to minimize (value + index * epsilon)
        composite_key = input_float + indices * epsilon

    # Find the top-k elements of the composite key.
    # Since a "better" element always has a smaller key in our scheme,
    # we always use largest=False.
    _, topk_indices = torch.topk(composite_key, k, dim=dim, largest=False)

    # Gather the original values using the final, deterministic indices
    topk_values = torch.gather(input_tensor, dim, topk_indices)

    return topk_values, topk_indices


def create_fourier_feature_matrix(in_features, out_features, normalize=True):
    """
    Creates a matrix using Fourier feature initialization, with an option to normalize columns.
    """
    position = np.arange(in_features).reshape(-1, 1)
    d_model_term = np.arange(0, out_features, 2)
    div_term = 1 / (10000**(d_model_term / out_features))

    M = np.zeros((in_features, out_features))
    
    M[:, 0::2] = np.sin(position * div_term)
    M[:, 1::2] = np.cos(position * div_term)
    
    if normalize:
        # Normalize each column to have a unit L2 norm
        norms = np.linalg.norm(M, axis=0)
        M = M / norms
    
    return M

def create_scaled_qr_matrix(in_features, out_features, std_dev=1):
    """
    Creates an orthogonal matrix with a custom element-wise standard deviation.

    Args:
        in_features (int): The number of rows (e.g., model_dim).
        out_features (int): The number of columns (e.g., num_experts).
        std_dev (float): The desired standard deviation of the matrix elements.

    Returns:
        np.ndarray: An orthogonal matrix of shape (D, d) with the specified std dev.
    """
    # 1. Create the orthonormal matrix Q from a random Gaussian matrix.
    # The columns of Q are perfectly orthogonal and have a unit norm.
    random_matrix = np.random.randn(in_features, out_features)
    q_matrix, _ = np.linalg.qr(random_matrix, mode="reduced")

    # 2. Calculate the scaling factor needed to achieve the target std_dev.
    # The elements of q_matrix have a variance of ~1/D. To get a variance
    # of std_dev**2, we need to scale the matrix by std_dev * sqrt(in_features).
    scale_factor = std_dev * np.sqrt(in_features)

    # 3. Apply the scaling factor to the orthonormal matrix.
    scaled_q_matrix = q_matrix * scale_factor

    return scaled_q_matrix

class MoEMLP(nn.Module):
    def __init__(self, num_experts, top_k, model_dim, expert_dim, layer_id, router_dtype=torch.bfloat16, expert_dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_dtype = router_dtype
        self.expert_dtype = expert_dtype
        self.layer_id = layer_id

        # gating
        self.gate = nn.Linear(model_dim, num_experts, bias=False, dtype=router_dtype)

        ## give more std for the router than usual
        
        ## default init would be:
        #torch.nn.init.uniform_(self.gate.weight, -1 / math.sqrt(model_dim), 1 / math.sqrt(model_dim))

        """
        with torch.no_grad():
            # Create the matrix with shape (model_dim, num_experts)
            qr_matrix = create_scaled_qr_matrix(model_dim, num_experts, std_dev=1)
            
            # The gate weight requires shape (num_experts, model_dim), so we transpose.
            # Convert the NumPy array to a PyTorch tensor with the correct dtype.
            gate_weight = torch.from_numpy(qr_matrix).T.to(dtype=self.router_dtype)
            
            # Assign the new weights to the gate layer.
            self.gate.weight.copy_(gate_weight)
        """


        self.experts = nn.ModuleList(
            [MLP(model_dim, expert_dim, expert_id, self.layer_id, expert_dtype) for expert_id in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor, step_num: int) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states_temp = hidden_states.to(self.router_dtype)

        if TO_SAVE:
            print(f"[Step {step_num}] Saving pre-router for layer {self.layer_id}...")
            torch.save(hidden_states.cpu().view(-1, hidden_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/pre_router.pt")

            print(f"[Step {step_num}] Saving gate weights for layer {self.layer_id}...")
            torch.save(self.gate.weight.cpu().view(-1, self.num_experts), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/gate_weights.pt")

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states_temp)
        #routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        #routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights, selected_experts = deterministic_topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights.float(), dim=-1)

        if TO_SAVE:
            print(f"[Step {step_num}] Saving router logits for layer {self.layer_id}...")
            torch.save(router_logits.cpu().view(-1, self.num_experts), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/router_logits.pt")
        
            print(f"[Step {step_num}] Saving selected experts for layer {self.layer_id}...")
            torch.save(selected_experts.cpu().view(-1, self.top_k), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/selected_experts.pt")

            print(f"[Step {step_num}] Saving routing weights for layer {self.layer_id}...")
            torch.save(routing_weights.cpu().view(-1, self.top_k), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/routing_weights.pt")

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if TO_SAVE and self.layer_id == 0:
                print(f"[Step {step_num}] Saving chosen tokens for layer {self.layer_id} and expert {expert_idx}...")
                torch.save(top_x.cpu().view(-1), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/expert_{expert_idx.item()}_chosen_tokens.pt")

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (expert_layer(current_state, step_num).float() * routing_weights[top_x, idx, None]).to(hidden_states.dtype)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.num_q_heads = args.num_q_heads
        self.model_dim = args.model_dim
        self.head_dim = args.model_dim // args.num_q_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = MoEMLP(args.num_routed_experts, args.top_k_routed_experts, args.model_dim, args.expert_dim, layer_id, 
                                   router_dtype=dtype_to_torch_dtype(args.router_dtype), expert_dtype=dtype_to_torch_dtype(args.expert_dtype))
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.model_dim, eps=args.rms_norm_epsilon, dtype=dtype_to_torch_dtype(args.attn_dtype))
        self.ffn_norm = RMSNorm(args.model_dim, eps=args.rms_norm_epsilon, dtype=dtype_to_torch_dtype(args.attn_dtype))

        self.layer_save_dir = SAVE_DIR + f"/layers_fwd/{layer_id}"

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seqlens_info: SeqlensInfo,
        mask: Optional[torch.Tensor],
        step_num: int,
    ):
        if TO_SAVE:
            print(f"[Step {step_num}] Saving block inp for layer {self.layer_id}...")
            torch.save(x.cpu().view(-1, self.model_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/block_inp.pt")

        norm = self.attention_norm(x)

        h = x + self.attention(norm, freqs_cis, seqlens_info, mask, step_num)

        ffn_norm = self.ffn_norm(h)

        if TO_SAVE:
            print(f"[Step {step_num}] Saving FFN Norm out for layer {self.layer_id}...")
            torch.save(ffn_norm.cpu().view(-1, self.model_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/ffn_norm_out.pt")

        ffn_out, router_logits = self.feed_forward(ffn_norm, step_num)

        out = h + ffn_out

        if TO_SAVE:
            print(f"[Step {step_num}] Saving block out for layer {self.layer_id}...\n\n\n")
            torch.save(ffn_norm.cpu().view(-1, self.model_dim), f"{SAVE_DIR}/layers_fwd/{self.layer_id}/block_out.pt")

        return out


class MoETransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.num_layers = params.num_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.model_dim, dtype=dtype_to_torch_dtype(params.embed_dtype))

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.model_dim, eps=params.rms_norm_epsilon, dtype=dtype_to_torch_dtype(params.head_dtype))

        self.output = nn.Linear(params.model_dim, params.vocab_size, bias=False, dtype=dtype_to_torch_dtype(params.head_dtype))

        self.freqs_cis = precompute_freqs_cis(
            params.model_dim // params.num_q_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )


    def forward(self, tokens: torch.Tensor, seqlens_info: SeqlensInfo, step_num: int):
        
        inp_device = tokens.device

        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)

        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=inp_device)
            mask = torch.triu(mask, diagonal=1)
        
        for layer in self.layers:
            h = layer(h, freqs_cis, seqlens_info, mask, step_num)
        
        h = self.norm(h)
        output = self.output(h)

        if TO_SAVE:
            print(f"[Step {step_num}] Saving final head output...\n\n\n")
            torch.save(output.cpu().view(-1, self.vocab_size), f"{SAVE_DIR}/head_fwd/final_out.pt")

        return output
