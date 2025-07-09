#ifndef FLASH2_WRAPPER_H
#define FLASH2_WRAPPER_H

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

// Max chunk size refers to the max chunk size in tokens (i.e. the maxmium number of query tokens)
// Max seq len refers to the longest sequence part of round (in tokens)
// Max seqs per chunk refers the maximum number of unique sequences that are processed in a single chunk

// Current forward workspace is 0
uint64_t flash2_get_fwd_workspace_size(DataflowDatatype dtype, int arch, int num_sm, 
                                        int num_q_heads, int num_kv_heads, int head_dim, 
                                        int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                        int is_causal);

// In order to do autoconfiguration it is important for system to ensure that attention can run
// so it needs to know the minimmum workspace size
// For long seqs using GQA this number can be quite large
uint64_t flash2_get_bwd_workspace_size(DataflowDatatype dtype, int arch, int num_sm, 
                                        int num_q_heads, int num_kv_heads, int head_dim, 
                                        int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                        int is_causal);

int flash2_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                        int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, float * softmax_lse,
                        int is_causal,
                        uint64_t workspaceBytes, void * workspace);



int flash2_bwd_wrapper(CUstream stream, int arch, int num_sm,
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
