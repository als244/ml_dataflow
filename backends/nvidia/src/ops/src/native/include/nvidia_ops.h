#ifndef NVIDIA_OPS_H
#define NVIDIA_OPS_H

#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include "cuda_ptx_macros.h"


#define WARP_SIZE 32
#define MAX_BLOCK_DIM 1024

#define CONST_ZERO_FP16 0x0000U
#define CONST_ONE_FP16 0x3C00U

#define CONST_ZERO_DEV_FP16 __ushort_as_half((unsigned short)0x0000U)
#define CONST_ONE_DEV_FP16 __ushort_as_half((unsigned short) 0x3C00U)
#define POS_INF_DEV_FP16 __ushort_as_half((unsigned short) 0x7C00U)
#define NEG_INF_DEV_FP16 __ushort_as_half((unsigned short) 0xFC00U)

#define CONST_ZERO_DEV_BF16 __ushort_as_bfloat16((unsigned short) 0x0000U)
#define CONST_ONE_DEV_BF16 __ushort_as_bfloat16((unsigned short) 0x3F80U)
#define POS_INF_DEV_BF16 __ushort_as_bfloat16((unsigned short) 0x7F80U)
#define NEG_INF_DEV_BF16 __ushort_as_bfloat16((unsigned short) 0xFF80U)



#define CONST_FLOAT_INF 0x7f800000
#define CONST_FLOAT_NEG_INF 0xff800000

#define CONST_DEV_FLOAT_INF __int_as_float(0x7f800000)
#define CONST_DEV_FLOAT_NEG_INF __int_as_float(0xff800000)

#define ROUND_UP_TO_MULTIPLE(x, multiple) (((x + multiple - 1) / multiple) * multiple)




// Embedding
extern "C" __global__ void embedding_fp32_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, float* embedding_table, float * output);
extern "C" __global__ void embedding_fp16_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __half * embedding_table, __half * output);
extern "C" __global__ void embedding_bf16_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __nv_bfloat16 * embedding_table, __nv_bfloat16 * output);
extern "C" __global__ void embedding_fp8e4m3_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __nv_fp8_e4m3 * embedding_table, __nv_fp8_e4m3 * output);
extern "C" __global__ void embedding_fp8e5m2_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __nv_fp8_e5m2 * embedding_table, __nv_fp8_e5m2 * output);


// Norm 
// Forward
extern "C" __global__ void rms_norm_fp32_kernel(int n_rows, int n_cols, float eps, float * rms_weight, float * X, float * out, float * weighted_sums, float * rms_vals);
extern "C" __global__ void rms_norm_fp16_kernel(int n_rows, int n_cols, float eps, __half * rms_weight, __half * X, __half * out, float * weighted_sums, float * rms_vals);
extern "C" __global__ void rms_norm_bf16_kernel(int n_rows, int n_cols, float eps,__nv_bfloat16 * rms_weight, __nv_bfloat16 * X, __nv_bfloat16 * out, float * weighted_sums, float * rms_vals);
extern "C" __global__ void rms_norm_fp8e4m3_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X, __nv_fp8_e4m3 * out, float * weighted_sums, float * rms_vals);
extern "C" __global__ void rms_norm_fp8e5m2_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X, __nv_fp8_e5m2 * out, float * weighted_sums, float * rms_vals);

// Backward Activation
extern "C" __global__ void rms_norm_bwd_x_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, float * rms_weight, float * X_inp, float * upstream_dX, float * dX);
extern "C" __global__ void rms_norm_bwd_x_fp16_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __half * rms_weight, __half * X_inp, __half * upstream_dX, __half * dX);
extern "C" __global__ void rms_norm_bwd_x_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX);
extern "C" __global__ void rms_norm_bwd_x_fp8e4m3_fp16_kernel(int n_rows, int n_cols,float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X_inp, __half * upstream_dX, __half * dX);
extern "C" __global__ void rms_norm_bwd_x_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX);
extern "C" __global__ void rms_norm_bwd_x_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X_inp, __half * upstream_dX, __half * dX);
extern "C" __global__ void rms_norm_bwd_x_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX);

// Backward Weights
extern "C" __global__ void rms_norm_bwd_w_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, float * X_inp, float * upstream_dX, float * dW);
extern "C" __global__ void rms_norm_bwd_w_fp16_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __half * X_inp, __half * upstream_dX, __half * dW);
extern "C" __global__ void rms_norm_bwd_w_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dW);
extern "C" __global__ void rms_norm_bwd_w_fp8e4m3_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __half * upstream_dX, __half * dW);
extern "C" __global__ void rms_norm_bwd_w_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dW);
extern "C" __global__ void rms_norm_bwd_w_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __half * upstream_dX, __half * dW);
extern "C" __global__ void rms_norm_bwd_w_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dW);





// Attention Misc

// Forward
extern "C" __global__ void rope_fp32_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, float * X_q, float * X_k);
extern "C" __global__ void rope_fp16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __half * X_q, __half * X_k);
extern "C" __global__ void rope_bf16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_bfloat16 * X_q, __nv_bfloat16 * X_k);
extern "C" __global__ void rope_fp8e4m3_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e4m3 * X_q, __nv_fp8_e4m3 * X_k);
extern "C" __global__ void rope_fp8e5m2_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e5m2 * X_q, __nv_fp8_e5m2 * X_k);


extern "C" __global__ void rope_bwd_x_fp32_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int *seq_positions, float *dX_q, float *dX_k);
extern "C" __global__ void rope_bwd_x_fp16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int *seq_positions, __half *dX_q, __half *dX_k);
extern "C" __global__ void rope_bwd_x_bf16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int *seq_positions, __nv_bfloat16 *dX_q, __nv_bfloat16 *dX_k);



extern "C" __global__ void copy_to_seq_context_fp32_kernel(uint64_t N, int total_tokens, int kv_dim, float * keys, float * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);
extern "C" __global__ void copy_to_seq_context_fp16_kernel(uint64_t N, int total_tokens, int kv_dim, __half * keys, __half * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);
extern "C" __global__ void copy_to_seq_context_bf16_kernel(uint64_t N, int total_tokens, int kv_dim, __nv_bfloat16 * keys, __nv_bfloat16 * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);
extern "C" __global__ void copy_to_seq_context_fp8e4m3_kernel(uint64_t N, int total_tokens, int kv_dim, __nv_fp8_e4m3 * keys, __nv_fp8_e4m3 * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);
extern "C" __global__ void copy_to_seq_context_fp8e5m2_kernel(uint64_t N, int total_tokens, int kv_dim, __nv_fp8_e5m2 * keys, __nv_fp8_e5m2 * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);





// MOE

// Selection
extern "C" __global__ void select_experts_fp32_kernel(int total_tokens, int n_experts, int top_k_experts,  float * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert);
extern "C" __global__ void select_experts_fp16_kernel(int total_tokens, int n_experts, int top_k_experts,  __half * X_routed, __half * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert);
extern "C" __global__ void select_experts_bf16_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_bfloat16 * X_routed, __nv_bfloat16 * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert);
extern "C" __global__ void select_experts_fp8e4m3_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_fp8_e4m3 * X_routed, __nv_fp8_e4m3 * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert);
extern "C" __global__ void select_experts_fp8e5m2_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_fp8_e5m2 * X_routed, __nv_fp8_e5m2 * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert);






// Activations

// Forward
extern "C" __global__ void swiglu_fp32_kernel(int num_rows, int num_cols, float * x_w1, float * x_w3, float * out);
extern "C" __global__ void swiglu_fp16_kernel(int num_rows, int num_cols, __half * x_w1, __half * x_w3, __half * out);
extern "C" __global__ void swiglu_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * out);
extern "C" __global__ void swiglu_fp8e4m3_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __nv_fp8_e4m3 * out);
extern "C" __global__ void swiglu_fp8e5m2_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __nv_fp8_e5m2 * out);

// Backward
extern "C" __global__ void swiglu_bwd_x_fp32_fp32_kernel(int num_rows, int num_cols, float * x_w1, float * x_w3, float * upstream_dX, float * dX_w1, float * dX_w3);
extern "C" __global__ void swiglu_bwd_x_fp16_fp16_kernel(int num_rows, int num_cols, __half * x_w1, __half * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3);
extern "C" __global__ void swiglu_bwd_x_bf16_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3);
extern "C" __global__ void swiglu_bwd_x_fp8e4m3_fp16_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3);
extern "C" __global__ void swiglu_bwd_x_fp8e4m3_bf16_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3);
extern "C" __global__ void swiglu_bwd_x_fp8e5m2_fp16_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3);
extern "C" __global__ void swiglu_bwd_x_fp8e5m2_bf16_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3);




// Loss Stuff


// Softmax
extern "C" __global__ void softmax_fp32_fp32_kernel(int n_rows, int n_cols, float * X, float * out);
extern "C" __global__ void softmax_fp16_fp16_kernel(int n_rows, int n_cols, __half * X, __half * out);
extern "C" __global__ void softmax_bf16_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * X, __nv_bfloat16 * out);
extern "C" __global__ void softmax_fp8e4m3_fp16_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * X, __half * out);
extern "C" __global__ void softmax_fp8e4m3_bf16_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * X, __nv_bfloat16 * out);
extern "C" __global__ void softmax_fp8e5m2_fp16_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * X, __half * out);
extern "C" __global__ void softmax_fp8e5m2_bf16_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * X, __nv_bfloat16 * out);


// Loss Functions Over Logits
// Inplace
extern "C" __global__ void cross_entropy_loss_fp32_kernel(int n_rows, int n_cols, float * pred_logits, uint32_t * labels, float * loss_vec);
extern "C" __global__ void cross_entropy_loss_fp16_kernel(int n_rows, int n_cols, __half * pred_logits, uint32_t * labels, float * loss_vec);
extern "C" __global__ void cross_entropy_loss_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * pred_logits, uint32_t * labels, float * loss_vec);



#endif