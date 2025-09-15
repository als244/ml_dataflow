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

from liger_kernel.transformers.rope import liger_rotary_pos_emb as LigerRope

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from scattermoe.mlp import MLP, GLUMLP



@dataclass
class ModelArgs:
    """Configuration for the Llama3 model."""
    dtype: torch.dtype = torch.bfloat16
    dim: int = 1536
    n_layers: int = 24
    n_heads: int = 12
    n_kv_heads: int = 4
    vocab_size: int = 81920
    expert_dim: int = 384
    num_experts: int = 256
    top_k: int = 16
    norm_eps: float = 1e-5
    max_seq_len: int = 1048576
    rope_theta: float = 500000


def precompute_rope_embeddings(head_dim: int, max_seq_len: int, rope_theta: float = 500000.0):
    """
    Precomputes the cosine and sine embeddings for RoPE.
    """
    dim_indices = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / (rope_theta ** (dim_indices / head_dim))
    seq_positions = torch.arange(max_seq_len).float()

    angles = torch.outer(seq_positions, inv_freq)

    angles_expanded = torch.cat((angles, angles), dim=-1)
    cos_emb = torch.cos(angles_expanded)
    sin_emb = torch.sin(angles_expanded)

    return cos_emb, sin_emb


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """Precomputes the rotary frequencies for RoPE."""
    theta_base = torch.tensor(theta, device=device)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """Applies rotary positional embeddings to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class Attention(nn.Module):
    """Multi-Head Attention updated to use FlashAttention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dtype = args.dtype
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False, dtype=self.dtype)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, dtype=self.dtype)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        nvtx.range_push("Attention") # NVTX Start
        
        batch_size, seq_len, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        #xq, xk = LigerRope(xq, xk, self.cos_vals[:seq_len], self.sin_vals[:seq_len])

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        output = flash_attn_interface.flash_attn_func(xq, xk, xv, causal=True)
        
        output = output.view(batch_size, seq_len, -1)
        
        result = self.wo(output)
        
        nvtx.range_pop() # NVTX End
        return result

class MoEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs, layer_id):
        super().__init__()
         
        self.has_saved = False
        self.layer_id = layer_id
        self.top_k = args.top_k
        self.router = nn.Linear(args.dim, args.num_experts, bias=False, dtype=args.dtype)
        self.moe_layer = GLUMLP(input_size=args.dim, hidden_size=args.expert_dim, activation=nn.SiLU(), num_experts=args.num_experts, top_k=args.top_k)
        

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
        self.dtype = args.dtype
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.moe_feed_forward = MoEFeedForward(args, layer_id)
        self.attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        # The main range for this block is pushed in the Llama3Model loop
        h = x + self.attention(x, freqs_complex)
        
        out = h + self.moe_feed_forward(self.ffn_norm(h))

        return out

class Llama3Model(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.dtype = params.dtype
        self.params = params
        self.vocab_size = params.vocab_size

        self.sin_embeddings, self.cos_embeddings = precompute_rope_embeddings(params.dim // params.n_heads, params.max_seq_len, params.rope_theta)
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim, dtype=self.dtype)
        self.layers = nn.ModuleList([DecoderBlock(params, i) for i in range(params.n_layers)])
        self.norm = LigerRMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False, dtype=self.dtype)

        self.freqs_complex = precompute_theta_pos_frequencies(
            params.dim // params.n_heads, params.max_seq_len, "cpu"
        )

        self.loss_fn = LigerFusedLinearCrossEntropyLoss()
        
       

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor):
        batch_size, seq_len = tokens.shape

        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        nvtx.range_push("Token Embeddings")
        h = self.tok_embeddings(tokens)
        nvtx.range_pop()

        self.freqs_complex = self.freqs_complex.to(h.device)
        freqs = self.freqs_complex[:seq_len]
        

        nvtx.range_push("Decoder Layers")
        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Layer {i}") # Push a specific range for each layer
            h = checkpoint(layer, h, freqs, use_reentrant=False)
            nvtx.range_pop()
        nvtx.range_pop()
            
        nvtx.range_push("Final Norm")
        h = self.norm(h)
        nvtx.range_pop()

        nvtx.range_push("Output Projection and Cross Entropy Loss")
        loss = self.loss_fn(self.output.weight, h.view(-1, self.params.dim), labels.view(-1))
        nvtx.range_pop()
        
        return loss
