#ifndef FLASH3_WRAPPER_H
#define FLASH3_WRAPPER_H

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

// FOR FLASH3 ATTENTION:

// Only support for FP16, BF16, and FP8
// if TYPE FP8, output must be BF16
// Softmax LSE is of type FP32 and has length total_q * num_q_heads

// To compute required size of attn_workspace:

// attn_workspace_size = 0

// Occum and LSE accum:
// If num_splits > 1:
//      attn_workspace_size += num_splits * sizeof(float) * num_q_heads * total_q * (1 + head_dim)

// Tile count sem: 
// If arch >= 90 || num_splits > 1:
//      attn_workspace_size += sizeof(int)

// Dynamic split ptr for each seq:
// If num_seqs <= 992:
//      attn_workspace_size += num_seqs * sizeof(int)


// ASSUME CAUSAL

// - cum_q_seqlens should be of length num_seqs + 1, starting with 0
//      - cumsum of # of queries in each sequence
// - k_seqlens should be of length num_seqs
//      - total number of keys in sequence (should be >= # of queries) 
//          - (assumes that if sequence has Q queries and K keys, the starting position of Q_0
//              occurs at position K - Q)

int flash3_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                        int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, void * softmax_lse,
                        uint64_t workspaceBytes, void * workspace);



// inputs: same as fwd + dx_out (upstream gradient) and possibly different sized workspace

// purpose is to compute dx_q, dx_k, dx_v



// attn_bwd_workspace:

    /*

    int const head_size_rounded = round_up_headdim(head_size);

    int const kBlockM_sm90 = head_size_rounded <= 64 ? (is_causal && softcap > 0.0 ? 96 : 128)
        : (head_size_rounded <= 96 ? 64
           : (head_size_rounded <= 128 ? (is_causal || is_local || softcap > 0.0 ? 64 : 80)
              : 64));
    int const kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
    int const kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
    int const kBlockM = arch >= 90 ? kBlockM_sm90 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
    int const kBlockN_sm90 = head_size_rounded <= 128
        ? 128
        : (head_size_rounded <= 192 ? 96 : 80);
    int const kBlockN_sm80 = head_size_rounded <= 128
        ? 128
        : (head_size_rounded <= 192 ? 80 : 64);
    int const kBlockN_sm86 = head_size_rounded <= 64 ? 128
        : (head_size_rounded <= 96 ? 128
           : (head_size_rounded <= 128 ? 96
              : (head_size_rounded <= 192 ? 64 : 64)));
    int const kBlockN = arch >= 90 ? kBlockN_sm90 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
    int const seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
    int const total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM);
    int const total_k_padded_rounded = round_multiple(total_k + batch_size * kBlockN, kBlockN);
    */

    // - softmax_d: (num_q_heads, total_q_padded_rounded), dtype=float32
    // - softmax_lse_log2: (num_q_heads, total_q_padded_rounded), dtype=float32

    // - dq_accum: (num_q_heads, total_q_padded_rounded * head_size_rounded), dtype=float32
    
    // if (num_q_heads != num_kv_heads):
    //      - dk_accum: (num_kv_heads, total_k_padded_rounded, head_size_rounded), dtype=float32
    //      - dv_accum: (num_kv_heads, total_k_padded_rounded, head_size_rounded), dtype=float32
    

    // - dq_semaphore: ( (max_seqlen_q + kBlockM - 1) / (kBlockM), num_seqs, num_q_heads), dtype=int32
    // if (num_q_heads != num_kv_heads) & deterministic:
    //      - dk_semaphore: (max_seqlen_k + kBlockN - 1) / kBlockN, num_seqs, num_heads_kv), dtype=int32
    //      - dv_semaphore: (max_seqlen_k + kBlockN - 1) / kBlockN, num_seqs, num_heads_kv), dtype=int32
int flash3_bwd_wrapper(CUstream stream, int arch, int num_sm,
                            int flash_dtype_as_int, 
                            int num_seqs, int total_q, int total_k, 
                            int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                            int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, void * softmax_lse, 
                            void * dx_out, 
                            void * dx_q, void * dx_k, void * dx_v,
                            uint64_t workspaceBytes, void * workspace);

#endif