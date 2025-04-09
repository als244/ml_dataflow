import numpy as np
import torch
import torch.nn.functional as F
import math

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

def create_causal_mask(seq_len, device, dtype):
    # Create an n x n matrix filled with zeros in fp16
    mat = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    # Create a boolean mask for the upper triangular region (above the diagonal)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # Assign -inf (as fp16) to positions in the upper triangular region
    mat[mask] = torch.tensor(float('-inf'), dtype=dtype)
    return mat

# Q: [B, T, h_q, head_dim]
# K: [B, T, h_k, head_dim]
# V: [B, T, h_k, head_dim]
def gqa(Q, K, V, h_q, h_k, head_dim, mask=None):
        
        bsz = Q.shape[0]
        seqlen = Q.shape[1]

        rep_factor = int(h_q / h_k)
    

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(K, rep_factor) # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(V, rep_factor)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = Q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, h_q, head_dim)
        return output

def main():
    
    num_seqs = 1
    seq_len = 512

    data_dir = f"test_attention"

    h_q = 64
    h_k = 8
    head_dim = 128

    model_dim = int(h_q * head_dim)



    # Load your numpy arrays.
    # Replace these with the actual file paths or loading mechanism.
    np_Q = np.fromfile(f"{data_dir}/x_q.dat", dtype=np.float16).reshape(num_seqs, seq_len, h_q, head_dim)       # Expected shape: [B, T, 64, 128]
    np_K = np.fromfile(f"{data_dir}/x_k.dat", dtype=np.float16).reshape(num_seqs, seq_len, h_k, head_dim)       # Expected shape: [B, S, 8, 128]
    np_V = np.fromfile(f"{data_dir}/x_v.dat", dtype=np.float16).reshape(num_seqs, seq_len, h_k, head_dim)       # Expected shape: [B, S, 8, 128]
    np_dX_out = np.fromfile(f"{data_dir}/dx_out.dat", dtype=np.float16).reshape(num_seqs, seq_len, h_q, head_dim)  # Expected shape: [B, T, 64, 128]

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert numpy arrays to FP16 PyTorch tensors and set requires_grad=True for inputs.
    Q = torch.tensor(np_Q, dtype=torch.float16, device=device, requires_grad=True)
    K = torch.tensor(np_K, dtype=torch.float16, device=device, requires_grad=True)
    V = torch.tensor(np_V, dtype=torch.float16, device=device, requires_grad=True)
    dX_out = torch.tensor(np_dX_out, dtype=torch.float16, device=device)

    # Compute attention output

    # causal mask
    causal_mask = create_causal_mask(seq_len, device, torch.float16)

    X_out = gqa(Q, K, V, h_q, h_k, head_dim, causal_mask)

    X_out_re = X_out.view(-1, model_dim)
    
    np_X_out = X_out_re.cpu().detach().numpy()
    np_X_out.tofile(f"{data_dir}/torch_x_out.dat")

    # Compute gradients using the provided upstream gradient dX_out.
    # This will backpropagate through the attention calculation.
    X_out.backward(dX_out)

    # Retrieve gradients
    dX_Q = Q.grad  # Gradient w.r.t. Q
    dX_K = K.grad  # Gradient w.r.t. K
    dX_V = V.grad  # Gradient w.r.t. V
    
    dX_Q_re = dX_Q.view(-1, 8192)
    print(f"dX_Q: {dX_Q_re}\n\n")

if __name__ == "__main__":
    main()