#include "nvidia_ops.h"

extern "C" __global__ void default_merge_expert_result_bf16_bf16_kernel(int num_tokens, int model_dim, int top_k_experts, __nv_bfloat16 * expert_zone, int expert_id, int * expert_counts_cumsum, int * expert_mapping, float * token_expert_weights, uint16_t * chosen_experts, __nv_bfloat16 * X_combined){
    
    int new_token_ind = blockIdx.x;

    if (new_token_ind >= num_tokens) {
        return;
    }

    __shared__ float smem_token_expert_weight;

    // Initialize shared memory to prevent reading an uninitialized value
    // if the expert is not found (which shouldn't happen in correct logic).
    if (threadIdx.x == 0) {
        smem_token_expert_weight = 0.0f;
    }
    __syncthreads();
    
    int expert_base = expert_counts_cumsum[expert_id] - num_tokens;

    int orig_token_ind = expert_mapping[expert_base + new_token_ind];

    for (int i = threadIdx.x; i < top_k_experts; i += blockDim.x) {
        if (chosen_experts[orig_token_ind * top_k_experts + i] == (uint16_t) expert_id) {
            smem_token_expert_weight = token_expert_weights[orig_token_ind * top_k_experts + i];
            break;
        }
    }

    __syncthreads();

    float token_expert_weight = smem_token_expert_weight;

    if (token_expert_weight == 0.0f) {
        if (threadIdx.x == 0) {
            printf("Error: token expert weight is 0.0f for orig token %d and expert %d...\n", orig_token_ind, expert_id);
        }
        return;
    }

    __nv_bfloat16 * expert_processed_token = expert_zone + new_token_ind * model_dim;

    __nv_bfloat16 * token_final_home = X_combined + orig_token_ind * model_dim;

    /*
    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        token_final_home[i] += __float2bfloat16(token_expert_weight * __bfloat162float(expert_processed_token[i]));
    }
    */

    // Cast pointers to float4 for vectorized access. This is the core of the optimization.
    float4* expert_processed_token_f4 = (float4*)expert_processed_token;
    float4* token_final_home_f4 = (float4*)token_final_home;

    // The loop now strides over the vector in chunks of 8 bfloat16s (1 float4).
    for (int i = threadIdx.x; i < model_dim / 8; i += blockDim.x) {
        // Load 8 bfloat16s from the expert output and the destination buffer in a single transaction.
        float4 expert_val_f4 = expert_processed_token_f4[i];
        float4 home_val_f4 = token_final_home_f4[i];

        // A float4 load of bfloat16 data packs two bfloat16s into each 32-bit float component.
        // We need to unpack them into float2 vectors to work with them as floats.
        float2 expert_vals_01 = __bfloat1622float2(*((__nv_bfloat162*)&expert_val_f4.x));
        float2 expert_vals_23 = __bfloat1622float2(*((__nv_bfloat162*)&expert_val_f4.y));
        float2 expert_vals_45 = __bfloat1622float2(*((__nv_bfloat162*)&expert_val_f4.z));
        float2 expert_vals_67 = __bfloat1622float2(*((__nv_bfloat162*)&expert_val_f4.w));

        float2 home_vals_01 = __bfloat1622float2(*((__nv_bfloat162*)&home_val_f4.x));
        float2 home_vals_23 = __bfloat1622float2(*((__nv_bfloat162*)&home_val_f4.y));
        float2 home_vals_45 = __bfloat1622float2(*((__nv_bfloat162*)&home_val_f4.z));
        float2 home_vals_67 = __bfloat1622float2(*((__nv_bfloat162*)&home_val_f4.w));

        // Perform the weighted sum on all 8 values.
        home_vals_01.x += token_expert_weight * expert_vals_01.x;
        home_vals_01.y += token_expert_weight * expert_vals_01.y;
        home_vals_23.x += token_expert_weight * expert_vals_23.x;
        home_vals_23.y += token_expert_weight * expert_vals_23.y;
        home_vals_45.x += token_expert_weight * expert_vals_45.x;
        home_vals_45.y += token_expert_weight * expert_vals_45.y;
        home_vals_67.x += token_expert_weight * expert_vals_67.x;
        home_vals_67.y += token_expert_weight * expert_vals_67.y;

        // Pack the 8 resulting floats back into bfloat16 pairs.
        __nv_bfloat162 res_01 = __floats2bfloat162_rn(home_vals_01.x, home_vals_01.y);
        __nv_bfloat162 res_23 = __floats2bfloat162_rn(home_vals_23.x, home_vals_23.y);
        __nv_bfloat162 res_45 = __floats2bfloat162_rn(home_vals_45.x, home_vals_45.y);
        __nv_bfloat162 res_67 = __floats2bfloat162_rn(home_vals_67.x, home_vals_67.y);

        // Assemble the float4 for writing back to global memory.
        float4 result_f4;
        *((__nv_bfloat162*)&result_f4.x) = res_01;
        *((__nv_bfloat162*)&result_f4.y) = res_23;
        *((__nv_bfloat162*)&result_f4.z) = res_45;
        *((__nv_bfloat162*)&result_f4.w) = res_67;

        // Store 8 bfloat16s back to global memory in a single transaction.
        token_final_home_f4[i] = result_f4;
    }
    
}