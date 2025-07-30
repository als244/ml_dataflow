#include "nvidia_ops.h"

// We now use float4 to represent 8 bfloat16 values (16 bytes total).
// This is a larger vector for potentially higher memory throughput.
typedef float4 bfloat16_8;

#define PREPARE_EXP_VEC_SIZE (sizeof(bfloat16_8) / sizeof(__nv_bfloat16))

// This is where casting to expert dtype would occur...
extern "C" __global__ void default_prepare_expert_zone_bf16_bf16_kernel(int model_dim, __nv_bfloat16 * X, int expert_id, int * expert_counts, int* expert_counts_cumsum, int* expert_mapping, __nv_bfloat16 * expert_zone) {

    int num_tokens = expert_counts[expert_id];

    if (num_tokens == 0) {
        return;
    }

    int new_row_id = blockIdx.x;

    int expert_base = expert_counts_cumsum[expert_id] - num_tokens;

    int orig_token_id;
    
    const bfloat16_8* X_vec = (const bfloat16_8*)X;
    bfloat16_8* expert_zone_vec = (bfloat16_8*)expert_zone;

    // Calculate the model dimension in terms of our vector type.
    int model_dim_vec = model_dim / PREPARE_EXP_VEC_SIZE;

    const bfloat16_8* src_ptr;
    bfloat16_8* dst_ptr;

    while (new_row_id < num_tokens) {
    
        orig_token_id = expert_mapping[expert_base + new_row_id];

        // Get pointers to the start of the source and destination rows.
        src_ptr = X_vec + (long long)orig_token_id * model_dim_vec;
        dst_ptr = expert_zone_vec + (long long)new_row_id * model_dim_vec;

        // Threads within the block cooperate to copy the row vector-by-vector.
        for (int i = threadIdx.x; i < model_dim_vec; i += blockDim.x) {
            dst_ptr[i] = src_ptr[i];
        }

        new_row_id += gridDim.x;
    }
}