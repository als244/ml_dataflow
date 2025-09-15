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
    
class ExpertMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.dim
        self.expert_dim = args.expert_dim
        self.gate_proj = nn.Linear(self.hidden_size, self.expert_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.expert_dim, bias=False)
        self.down_proj = nn.Linear(self.expert_dim, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs, layer_id):
        super().__init__()
         
     
        self.layer_id = layer_id
        self.top_k = args.top_k
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False, dtype=args.dtype)
        self.experts = nn.ModuleList(
            [ExpertMLP(args) for _ in range(args.num_experts)]
        )
        
        

    def forward(self, x: torch.Tensor):
        nvtx.range_push("MoE") # NVTX Start

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
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
        # Calculate the attention block output (attention + residual connection)
        h = x + self.attention(self.attention_norm(x), freqs_complex) # Corrected: Norm before attention
        
        # Calculate the final output of the block
        out = h + self.moe_feed_forward(self.ffn_norm(h))

        # Return both the final output and the intermediate attention output
        return out, h

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
        
        # Attribute to store the attention outputs after a forward pass
        self.attention_outputs = []
       
    def forward(self, tokens: torch.Tensor, labels: torch.Tensor):
        # Clear previous outputs at the start of a new forward pass
        self.attention_outputs = []
        
        batch_size, seq_len = tokens.shape
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        nvtx.range_push("Token Embeddings")
        h = self.tok_embeddings(tokens)
        nvtx.range_pop()

        self.freqs_complex = self.freqs_complex.to(h.device)
        freqs = self.freqs_complex[:seq_len]
        
        nvtx.range_push("Decoder Layers")
        for i, layer in enumerate(self.layers):
            nvtx.range_push(f"Layer {i}")
            # Unpack the two return values from the checkpointed function
            h, attn_out = checkpoint(layer, h, freqs, use_reentrant=False)
            
            # Save the attention output
            self.attention_outputs.append(attn_out)
            
            nvtx.range_pop()
        nvtx.range_pop()
            
        nvtx.range_push("Final Norm")
        h = self.norm(h)
        nvtx.range_pop()

        nvtx.range_push("Output Projection and Cross Entropy Loss")
        loss = self.loss_fn(self.output.weight, h.view(-1, self.params.dim), labels.view(-1))
        nvtx.range_pop()
        
        return loss
