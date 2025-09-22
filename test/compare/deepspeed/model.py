# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch.cuda.nvtx as nvtx # Import NVTX

# Import FlashAttention
import flash_attn_interface
#from flash_attn import flash_attn_func

from torch.utils.checkpoint import checkpoint

from liger_kernel.transformers.rms_norm import LigerRMSNorm

from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb as LigerRope

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from scattermoe.mlp import MLP, GLUMLP

from attention import do_attention

from select_bins import select_bins

@dataclass
class ModelArgs:
    """Configuration for the model."""
    embed_dtype: torch.dtype = torch.bfloat16
    attn_dtype: torch.dtype = torch.bfloat16
    router_dtype: str = "none"
    expert_dtype: torch.dtype = torch.bfloat16
    head_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 128256
    model_dim: int = 4096
    head_dim: int = 128
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    qk_norm_type: str = "none"
    qk_norm_weight_type: str = "none"
    num_shared_experts: int = 1
    num_routed_experts: int = 0
    top_k_routed_experts: int = 0
    expert_dim: int = 14336
    expert_mlp_type: str = "swiglu"
    rope_theta: float = 500000
    rms_norm_epsilon: float = 1e-5
    rand_seed: int = 42

@dataclass
class SwiGLUConfig:
    """Configuration for the LigerSwigluMLP"""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    hidden_act: str = "silu"


def precompute_theta_pos_frequencies(head_dim: int, max_seq_len: int, theta: float, device = "cpu"):
    """Precomputes the rotary frequencies for RoPE."""
    theta_base = torch.tensor(theta, device=device)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex6

class Attention(nn.Module):
    """Multi-Head Attention updated to use FlashAttention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads
        self.model_dim = args.model_dim
        self.dtype = args.attn_dtype
        self.wq = nn.Linear(self.model_dim, self.n_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wk = nn.Linear(self.model_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wv = nn.Linear(self.model_dim, self.n_kv_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.model_dim, bias=False, dtype=self.dtype)

    def forward(self, x: torch.Tensor, freqs):
        nvtx.range_push("Attention") # NVTX Start
        
        batch_size, seq_len, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
       
        xq, xk = LigerRope(xq, xk, freqs)

        attn_result = do_attention(xq, xk, xv, causal=True, deterministic=True)

        attn_result = attn_result.view(batch_size, seq_len, -1)
        
        attn_out_proj = self.wo(attn_result)
        
        nvtx.range_pop() # NVTX End
        return attn_out_proj

class DenseFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        swiglu_config = SwiGLUConfig()
        swiglu_config.hidden_size = args.model_dim
        swiglu_config.intermediate_size = args.expert_dim
        
        self.swiglu = LigerSwiGLUMLP(swiglu_config)
        

    def forward(self, x: torch.Tensor):
        nvtx.range_push("FeedForward") # NVTX Start
        
        result = self.swiglu(x)
        
        nvtx.range_pop() # NVTX End
        return result


class SparseFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.top_k = args.top_k_routed_experts
        self.router_dtype = args.router_dtype
        self.model_dim = args.model_dim
        self.expert_dim = args.expert_dim
        self.num_routed_experts = args.num_routed_experts
        self.router = nn.Linear(self.model_dim, self.num_routed_experts, bias=False, dtype=self.router_dtype)
        self.moe_layer = GLUMLP(input_size=self.model_dim, hidden_size=self.expert_dim, activation=nn.SiLU(), num_experts=self.num_routed_experts, top_k=self.top_k_routed_experts)


    def forward(self, x: torch.Tensor):
        nvtx.range_push("MoE") # NVTX Start

        batch_size, sequence_length, hidden_dim = x.shape

        x = x.view(-1, hidden_dim)
        routed_x = self.router(x)

        routing_weights = F.softmax(routed_x, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = self.moe_layer(x, routing_weights, selected_experts)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        nvtx.range_pop() # NVTX End
        return final_hidden_states

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id):
        super().__init__()
        self.model_dim = args.model_dim
        self.layer_id = layer_id
        self.attention = Attention(args)

        if args.num_routed_experts == 0:
            self.feed_forward = DenseFeedForward(args)
        else:
            self.feed_forward = SparseFeedForward(args)
        
        self.attention_norm = LigerRMSNorm(self.model_dim, eps=args.rms_norm_epsilon)
        self.ffn_norm = LigerRMSNorm(self.model_dim, eps=args.rms_norm_epsilon)

    def forward(self, x: torch.Tensor, freqs):
        nvtx.range_push("Attention Sub-Block")
        h = x + self.attention(x, freqs)
        nvtx.range_pop()
        
        nvtx.range_push("FeedForward Sub-Block")
        out = h + self.feed_forward(self.ffn_norm(h))
        nvtx.range_pop()
        
        return out

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.embed_dtype = args.embed_dtype
        self.head_dtype = args.head_dtype

        self.model_dim = args.model_dim
        self.head_dim = args.head_dim
        self.rope_theta = args.rope_theta

        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.model_dim, dtype=self.embed_dtype)
        self.layers = nn.ModuleList([DecoderBlock(args, i) for i in range(self.n_layers)])
        self.norm = LigerRMSNorm(self.model_dim, eps=args.rms_norm_epsilon)
        self.output = nn.Linear(self.model_dim, self.vocab_size, bias=False, dtype=self.head_dtype)

        self.max_seq_len = 2 ** 20
        self.freqs_complex = precompute_theta_pos_frequencies(self.head_dim, self.max_seq_len, self.rope_theta)
        
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()
       
    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, save_act_layer_frac = 0.0):
        batch_size, seq_len = tokens.shape
        
        act_layers_saved = select_bins(self.n_layers, save_act_layer_frac)

        nvtx.range_push("Token Embeddings")
        h = self.tok_embeddings(tokens)
        nvtx.range_pop()

        freqs = self.freqs_complex[:seq_len].to(h.device)

        nvtx.range_push("Decoder Layers")
        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Layer {i}")
            if i in act_layers_saved:
                h = layer(h, freqs)
            else:
                h = checkpoint(layer, h, freqs, use_reentrant=False) # Call the layer directly
            nvtx.range_pop()
        nvtx.range_pop()
            
        nvtx.range_push("Final Norm")
        h = self.norm(h)
        nvtx.range_pop()

        nvtx.range_push("Output Projection and Cross Entropy Loss")
        loss = self.loss_fn(self.output.weight, h.view(-1, self.model_dim), labels.view(-1))
        nvtx.range_pop()
        
        return loss
