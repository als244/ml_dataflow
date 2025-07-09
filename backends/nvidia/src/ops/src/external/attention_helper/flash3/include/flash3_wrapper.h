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


uint64_t flash3_get_fwd_workspace_size(DataflowDatatype dtype, int arch, int num_sm, 
                                        int num_q_heads, int num_kv_heads, int head_dim, 
                                        int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                        int is_causal);

uint64_t flash3_get_bwd_workspace_size(DataflowDatatype dtype, int arch, int num_sm, 
                                        int num_q_heads, int num_kv_heads, int head_dim, 
                                        int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                        int is_causal);

int flash3_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                        int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, float * softmax_lse,
                        int is_causal,
                        uint64_t workspaceBytes, void * workspace);



// inputs: same as fwd + dx_out (upstream gradient) and possibly different sized workspace

// purpose is to compute dx_q, dx_k, dx_v
int flash3_bwd_wrapper(CUstream stream, int arch, int num_sm,
                            int flash_dtype_as_int, 
                            int num_seqs, int total_q, int total_k, 
                            int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                            int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, float * softmax_lse, 
                            void * dx_out, 
                            void * dx_q, void * dx_k, void * dx_v,
                            int is_causal,
                            uint64_t workspaceBytes, void * workspace);

#endif