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

// seems like there is a numerical difference between flash2 and flash3 bwd
// not sure which is "better", will set defaults to use flash3 for performance
// and will fall back to flash2 for other archs (blackwell)
#define USE_FLASH3_AMPERE 1
#define USE_FLASH3_HOPPER 1



// functions to export

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