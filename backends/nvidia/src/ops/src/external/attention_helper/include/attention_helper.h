#ifndef ATTENTION_HELPER_H
#define ATTENTION_HELPER_H

#include "dataflow.h"

// to get device info
#include "cuda_dataflow_handle.h"


// calling flash2_[fwd|bwd]_wrapper from libflash2.so
#include "flash2_wrapper.h"

// calling flash3_[fwd|bwd]_wrapper from libflash3.so
#include "flash3_wrapper.h"


// set higher flags that determine which archs use flash3

// seems like there is a slight numerical difference between flash2 and flash3 bwd
// not sure which is "better"

// performance seems about same on ampere for smallish seqs...
#define USE_FLASH3_AMPERE 1

#define USE_FLASH3_HOPPER 1



// functions to export

// If !is_training, returns the required workspace size for forward pass
// If is_training, returns the MAX(fwd_workspace_size, bwd_workspace_size

// Used in order to help autoconfigure memory during dataflow preparation...
// Stream is irrelevant here, but just using same format as other external ops
// for portability reasons...

int flash_attention_get_workspace_size(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);


/*
int flash[2|3]_get_[fwd|bwd]_workspace_size(Dataflow_Handle * dataflow_handle, int flash_dtype_as_int, int is_training, 
                                            int num_q_heads, int num_kv_heads, int head_dim, 
                                            int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                            int is_causal,
                                            uint64_t * ret_workspace_size);
*/

int flash_attention_fwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);

// ^ Calls function from libflash3.so:

/* int flash[2|3]_fwd_wrapper(CUstream stream, int arch, int num_sm,
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
int flash[2|3]_bwd_wrapper(CUstream stream, int arch, int num_sm,
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