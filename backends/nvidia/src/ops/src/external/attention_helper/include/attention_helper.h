#ifndef ATTENTION_HELPER_H
#define ATTENTION_HELPER_H

#include "dataflow.h"

// to get device info
#include "cuda_dataflow_handle.h"


// calling flash3_fwd_wrapper from libflash3.so
#include "flash3_wrapper.h"

// calling flash2_bwd_wrapper from libflash2.so
#include "flash2_wrapper.h"

// functions to export

// No need
// int flash3_attention_init(Dataflow_Handle * dataflow_handle, void * op_table_value)





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
//		- cumsum of # of queries in each sequence
// - k_seqlens should be of length num_seqs
//		- total number of keys in sequence (should be >= # of queries) 
//			- (assumes that if sequence has Q queries and K keys, the starting position of Q_0
//				occurs at position K - Q)

int flash_attention_fwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);

// ^ Calls function from libflash3.so:

/* int flash3_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * cum_q_seqlens, int max_seqlen_q,
                        int * cum_k_seqlens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, void * softmax_lse,
                        void * attn_workspace) */

int flash_attention_bwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);


/*
int flash3_bwd_wrapper(CUstream stream, int arch, int num_sm,
                            int flash_dtype_as_int, 
                            int num_seqs, int total_q, int total_k, 
                            int * cum_q_seqlens, int max_seqlen_q,
                            int * cum_k_seqlens, int max_seqlen_k,
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, void * softmax_lse, 
                            void * dx_out, 
                            void * dx_q, void * dx_k, void * dx_v,
                            void * attn_bwd_workspace);
*/

#endif