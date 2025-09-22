import torch

class FlashAttentionNotAvailableError(Exception):
    """Raised when neither Flash Attention 2 nor Flash Attention 3 is available"""
    pass

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def flash3_attention(q, k, v, causal=True, deterministic=True, rope_cos=None, rope_sin=None, rope_interleaved=False):
    return flash_attn_interface.flash_attn_func(q, k, v, causal=causal, deterministic=deterministic, rotary_cos=rope_cos, rotary_sin=rope_sin, rotary_interleaved=rope_interleaved)

def flash2_attention(q, k, v, causal=True, deterministic=True, rope_cos=None, rope_sin=None, rope_interleaved=False):
    return flash_attn_func(q, k, v, causal=causal, deterministic=deterministic)

def create_rope_embeddings(seq_len, head_dim, rope_theta=10000.0, device='cuda'):
    """
    Create RoPE (Rotary Position Embedding) cos and sin tensors.
    
    Args:
        seq_len: sequence length
        head_dim: head dimension (should be even)
        rope_theta: RoPE theta parameter
        device: device to create tensors on
    
    Returns:
        cos, sin tensors of shape (seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    
    # Create frequency tensor
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    # Create position indices
    position = torch.arange(seq_len, dtype=torch.float32, device=device)
    
    # Create frequency matrix
    freqs = torch.outer(position, inv_freq)
    
    # Get cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    return cos, sin

def apply_rope(x, cos, sin, interleaved=False):
    """
    Apply RoPE to input tensor using sin/cos with proper casting semantics.
    
    Args:
        x: input tensor of shape (..., seq_len, head_dim)
        cos: cosine tensor of shape (seq_len, head_dim // 2)
        sin: sine tensor of shape (seq_len, head_dim // 2)
        interleaved: whether to use interleaved RoPE (True) or GPT-NeoX style (False)
    
    Returns:
        tensor with RoPE applied, maintaining input dtype
    """
    seq_len = x.size(-2)
    head_dim = x.size(-1)
    
    # Ensure cos/sin match sequence length
    if cos.size(0) < seq_len:
        raise ValueError(f"cos tensor seq_len {cos.size(0)} < input seq_len {seq_len}")
    
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    
    # Convert to float for computation (following reference casting semantics)
    x_float = x.float()
    
    if interleaved:
        # Interleaved RoPE: combine adjacent dimensions
        x1, x2 = x_float[..., 0::2], x_float[..., 1::2]
        # Reshape cos/sin to match x1, x2 dimensions
        cos = cos.unsqueeze(-2)  # (seq_len, 1, head_dim // 2)
        sin = sin.unsqueeze(-2)  # (seq_len, 1, head_dim // 2)
        
        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Interleave back
        x_rot = torch.empty_like(x_float)
        x_rot[..., 0::2] = x1_rot
        x_rot[..., 1::2] = x2_rot
    else:
        # GPT-NeoX style: first half and second half
        half_dim = head_dim // 2
        x1, x2 = x_float[..., :half_dim], x_float[..., half_dim:]
        
        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
    
    # Cast back to original dtype using type_as (matching reference semantics)
    return x_rot.type_as(x)

def flash3_attention(q, k, v, causal=True, deterministic=True, rope_cos=None, rope_sin=None, rope_interleaved=False):
    if (rope_cos is not None) and (rope_sin is not None):
        q = apply_rope(q, rope_cos, rope_sin, rope_interleaved)
        k = apply_rope(k, rope_cos, rope_sin, rope_interleaved)
    
    return flash_attn_interface.flash_attn_func(q, k, v, causal=causal, deterministic=deterministic)

def flash2_attention(q, k, v, causal=True, deterministic=True, rope_cos=None, rope_sin=None, rope_interleaved=False):
    if (rope_cos is not None) and (rope_sin is not None):
        q = apply_rope(q, rope_cos, rope_sin, rope_interleaved)
        k = apply_rope(k, rope_cos, rope_sin, rope_interleaved)

    return flash_attn_func(q, k, v, causal=causal, deterministic=deterministic)

def do_attention(q, k, v, causal=True, deterministic=True, rope_cos=None, rope_sin=None, rope_interleaved=False):
    if FLASH_ATTN_3_AVAILABLE:
        return flash3_attention(q, k, v, causal=causal, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin, rope_interleaved=rope_interleaved)
    elif FLASH_ATTN_2_AVAILABLE:
        return flash2_attention(q, k, v, causal=causal, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin, rope_interleaved=rope_interleaved)
    else:
        raise FlashAttentionNotAvailableError(
            "Neither Flash Attention 2 nor Flash Attention 3 is available. "
            "Please install flash-attn package or ensure you have a compatible GPU (H100 for Flash Attention 3)."
        ) 
