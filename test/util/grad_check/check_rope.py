import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math # Added for sqrt
import numpy as np
# --- Helper Functions (Copied from your code) ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency tensor for complex exponentials (cis) with given dimensions and sequence length.

    Args:
        dim: Dimension of the frequency tensor.
        end: End index for the sequence length.
        theta: Scaling factor for frequency computation.

    Returns:
        Tensor of complex exponentials with shape (end, dim // 2).
    """
    # Calculate frequencies based on theta and dimension
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create sequence indices
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # Calculate outer product to get frequencies for each position
    freqs = torch.outer(t, freqs)
    # Convert frequencies to complex exponentials using polar coordinates
    # Magnitude is 1, angle is the frequency
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes the frequency tensor to match the target tensor's shape for broadcasting.

    Args:
        freqs_cis: Complex exponential tensor with shape (sequence_length, dim // 2).
        x: Target tensor to reshape for broadcasting.

    Returns:
        Reshaped frequency tensor compatible with the target tensor.
    """
    ndim = x.ndim
    # Ensure dimensions are valid for broadcasting
    assert 0 <= 1 < ndim # Make sure sequence length dimension exists
    # Expected shape: (seqlen, features)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape} at dim 1 and -1"

    # Create a shape that broadcasts correctly:
    # Match sequence length (dim 1) and feature dim (-1), others are 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embedding to query (xq) and key (xk) tensors.

    Args:
        xq: Query tensor with shape (batch_size, sequence_length, n_heads, head_dim).
        xk: Key tensor with shape (batch_size, sequence_length, n_kv_heads, head_dim).
        freqs_cis: Precomputed complex exponential tensor.

    Returns:
        Tuple containing the modified query and key tensors after applying rotary embedding.
    """
    # Reshape xq and xk to view the last dimension as pairs for complex number representation
    # Convert to float for complex view, then reshape
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting with xq_
    # Note: Broadcasting assumes xq_ and xk_ have compatible shapes after complex view
    freqs_cis_q = reshape_for_broadcast(freqs_cis, xq_)
    # Apply the rotation by multiplying with complex exponentials
    xq_out_complex = xq_ * freqs_cis_q
    # Convert back to real view and flatten the last two dimensions
    xq_out = torch.view_as_real(xq_out_complex).flatten(3)

    # Reshape freqs_cis for broadcasting with xk_
    freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_)
     # Apply the rotation by multiplying with complex exponentials
    xk_out_complex = xk_ * freqs_cis_k
    # Convert back to real view and flatten the last two dimensions
    xk_out = torch.view_as_real(xk_out_complex).flatten(3)

    # Cast outputs back to the original input types
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- Script Setup ---
# Define some example dimensions
batch_size = 1
seq_len = 2048
n_heads = 32
n_kv_heads = 8 # Example GQA
head_dim = 64 # Must be even for RoPE complex view
dim = n_heads * head_dim
theta = 500000.0
dtype = torch.bfloat16 # Or torch.float16, torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}, dtype: {dtype}")

# --- Load Inputs ---
# 1. Create input tensors xq and xk (BEFORE RoPE)
# These represent the outputs of the wq and wk linear layers

np_xq = np.fromfile("test_rope/xq.dat", dtype=np.uint16).reshape(1, seq_len, n_heads, head_dim)
np_xk = np.fromfile("test_rope/xk.dat", dtype=np.uint16).reshape(1, seq_len, n_kv_heads, head_dim)

temp_xq = torch.tensor(np_xq, dtype=torch.uint16, device=device)
xq_input = temp_xq.view(torch.bfloat16).clone().detach().requires_grad_(True)

temp_xk = torch.tensor(np_xk, dtype=torch.uint16, device=device)
xk_input = temp_xk.view(torch.bfloat16).clone().detach().requires_grad_(True)

# 2. Precompute freqs_cis
# The dimension passed to precompute_freqs_cis should be the head_dim
freqs_cis = precompute_freqs_cis(head_dim, seq_len, theta=theta).to(device)
# freqs_cis shape will be (seq_len, head_dim // 2)

# --- Load Upstream Gradients ---

np_dxq = np.fromfile("test_rope/dxq.dat", dtype=np.uint16).reshape(1, seq_len, n_heads, head_dim)
np_dxk = np.fromfile("test_rope/dxk.dat", dtype=np.uint16).reshape(1, seq_len, n_kv_heads, head_dim)

grad_xq_out = torch.from_numpy(np_dxq).view(torch.bfloat16).to(device)
grad_xk_out = torch.from_numpy(np_dxk).view(torch.bfloat16).to(device)

# --- Forward Pass ---
# Apply the RoPE function
xq_out, xk_out = apply_rotary_emb(xq_input, xk_input, freqs_cis)

print("\n--- Shapes ---")
print(f"xq_input shape: {xq_input.shape}")
print(f"xk_input shape: {xk_input.shape}")
print(f"freqs_cis shape: {freqs_cis.shape}")
print(f"xq_out shape:   {xq_out.shape}")
print(f"xk_out shape:   {xk_out.shape}")
print(f"grad_xq_out shape: {grad_xq_out.shape}")
print(f"grad_xk_out shape: {grad_xk_out.shape}")


# --- Backward Pass ---
# Compute gradients dL/dxq_input and dL/dxk_input
# Pass the upstream gradients via the 'grad_tensors' argument
torch.autograd.backward(
    tensors=[xq_out, xk_out],      # Tensors we want to backpropagate from
    grad_tensors=[grad_xq_out, grad_xk_out] # The upstream gradients dL/dxq_out, dL/dxk_out
)

xq_grad_re = xq_input.grad.view(-1, int(head_dim * n_heads))
xk_grad_re = xk_input.grad.view(-1, int(head_dim * n_kv_heads))

# --- Results ---
# The desired gradients are now stored in .grad attributes of the input tensors
print("\n--- Computed Gradients ---")
torch.save(xq_grad_re, "test_rope/torch_dxq_input.pt")
torch.save(xk_grad_re, "test_rope/torch_dxk_input.pt")

print(f"------Rope X_q Gradient Check------")
print(f"xq_input.grad: {xq_grad_re}\n\n")
print(f"------Rope X_k Gradient Check------")
print(f"xk_input.grad: {xk_grad_re}\n\n")
