#include "nvidia_ops.h"

extern "C" __global__ void default_merge_expert_result_bf16_bf16_kernel(int num_tokens, int model_dim, int top_k_experts, __nv_bfloat16 * expert_zone, int expert_id, int * expert_counts_cumsum, int * expert_mapping, float * token_expert_weights, uint16_t * chosen_experts, __nv_bfloat16 * X_combined){
    
    int new_token_ind = blockIdx.x;

    if (new_token_ind >= num_tokens) {
        return;
    }

    int expert_base = expert_counts_cumsum[expert_id] - num_tokens;

    int orig_token_ind = expert_mapping[expert_base + new_token_ind];

    __shared__ float smem_token_expert_weight;

    // IF NO WEIGHTS PASSED IN, ASSUME JUST ADDING with 1.0 (occurs during bwd x accumulation)
    if (!token_expert_weights) {
        if (threadIdx.x == 0) {
            smem_token_expert_weight = 1.0f;
        }
    }
    else{
        // Initialize shared memory to prevent reading an uninitialized value
        // if the expert is not found (which shouldn't happen in correct logic).
        if (threadIdx.x == 0) {
            smem_token_expert_weight = 0.0f;
        }
    }


    __syncthreads();
    

    // IF weights are passed in, find the weight for the current token and expert
    if (token_expert_weights){
        for (int i = threadIdx.x; i < top_k_experts; i += blockDim.x) {
            if (chosen_experts[orig_token_ind * top_k_experts + i] == (uint16_t) expert_id) {
                smem_token_expert_weight = token_expert_weights[orig_token_ind * top_k_experts + i];
                break;
            }
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

#define MERGE_EXPERT_RESULT_VEC_SIZE 8
extern "C" __global__ void default_merge_experts_bf16_bf16_kernel(
    int total_tokens,
    int model_dim,
    int num_selected_experts,
    const __nv_bfloat16* __restrict__ expert_zones,
    int* __restrict__ token_mapping,
    float* __restrict__ token_expert_weights,
    __nv_bfloat16* __restrict__ X_combined) {

    int orig_token_ind = blockIdx.x;

    if (orig_token_ind >= total_tokens) {
        return;
    }

    int model_dim_vec = model_dim / MERGE_EXPERT_RESULT_VEC_SIZE;

    const float4* X_combined_vec = (const float4*)X_combined;
    const float4* expert_zones_vec = (const float4*)expert_zones;
    float4* X_combined_out_vec = (float4*)X_combined;

    extern __shared__ uint8_t smem[];
    float* orig_token_smem = (float*)smem;

    /* Step 1: Vectorized load and unpack */
    int i;
    for (i = threadIdx.x; i < model_dim_vec; i += blockDim.x) {
        float4 bfloat16_in_f4 = X_combined_vec[orig_token_ind * model_dim_vec + i];
        
        /*
         * FIX: Cast the address of the loaded float4 to a pointer to an array of
         * __nv_bfloat162. This provides the correct type for the intrinsic.
         */
        __nv_bfloat162* bfloat16_in_vec2 = (__nv_bfloat162*)&bfloat16_in_f4;

        float2 f1 = __bfloat1622float2(bfloat16_in_vec2[0]);
        float2 f2 = __bfloat1622float2(bfloat16_in_vec2[1]);
        float2 f3 = __bfloat1622float2(bfloat16_in_vec2[2]);
        float2 f4 = __bfloat1622float2(bfloat16_in_vec2[3]);

        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 0] = f1.x;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 1] = f1.y;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 2] = f2.x;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 3] = f2.y;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 4] = f3.x;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 5] = f3.y;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 6] = f4.x;
        orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 7] = f4.y;
    }

    __syncthreads();

    /* Step 2: Accumulate expert outputs */
    for (i = threadIdx.x; i < model_dim_vec; i += blockDim.x) {
        float acc[MERGE_EXPERT_RESULT_VEC_SIZE];
        int j, k;

        #pragma unroll
        for(j = 0; j < MERGE_EXPERT_RESULT_VEC_SIZE; ++j) {
            acc[j] = orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + j];
        }

        for (k = 0; k < num_selected_experts; k++) {
            int expert_zone_row = token_mapping[orig_token_ind * num_selected_experts + k];
            float token_expert_weight = token_expert_weights[orig_token_ind * num_selected_experts + k];
            
            float4 expert_in_f4 = expert_zones_vec[expert_zone_row * model_dim_vec + i];
            __nv_bfloat162* expert_in_vec2 = (__nv_bfloat162*)&expert_in_f4;

            float2 ef1 = __bfloat1622float2(expert_in_vec2[0]);
            float2 ef2 = __bfloat1622float2(expert_in_vec2[1]);
            float2 ef3 = __bfloat1622float2(expert_in_vec2[2]);
            float2 ef4 = __bfloat1622float2(expert_in_vec2[3]);

            acc[0] += token_expert_weight * ef1.x;
            acc[1] += token_expert_weight * ef1.y;
            acc[2] += token_expert_weight * ef2.x;
            acc[3] += token_expert_weight * ef2.y;
            acc[4] += token_expert_weight * ef3.x;
            acc[5] += token_expert_weight * ef3.y;
            acc[6] += token_expert_weight * ef4.x;
            acc[7] += token_expert_weight * ef4.y;
        }

        #pragma unroll
        for(j = 0; j < MERGE_EXPERT_RESULT_VEC_SIZE; ++j) {
            orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + j] = acc[j];
        }
    }

    __syncthreads();

    /* Step 3: Pack and vectorized write back */
    for (i = threadIdx.x; i < model_dim_vec; i += blockDim.x) {
        float2 f1 = {orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 0], orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 1]};
        float2 f2 = {orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 2], orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 3]};
        float2 f3 = {orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 4], orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 5]};
        float2 f4 = {orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 6], orig_token_smem[i * MERGE_EXPERT_RESULT_VEC_SIZE + 7]};

        /*
         * FIX: Create a temporary array of the correct __nv_bfloat162 type to hold
         * the intrinsic's output. Then, cast this array to a float4 for the final write.
         */
        __nv_bfloat162 bfloat16_out_vec2[4];
        bfloat16_out_vec2[0] = __float22bfloat162_rn(f1);
        bfloat16_out_vec2[1] = __float22bfloat162_rn(f2);
        bfloat16_out_vec2[2] = __float22bfloat162_rn(f3);
        bfloat16_out_vec2[3] = __float22bfloat162_rn(f4);
        
        X_combined_out_vec[orig_token_ind * model_dim_vec + i] = *((float4*)bfloat16_out_vec2);
    }
}


extern "C" __global__ void naive_default_merge_experts_bf16_bf16_kernel(
    int total_tokens,
    int model_dim,
    int num_selected_experts,
    const __nv_bfloat16* __restrict__ expert_zones,
    int* __restrict__ token_mapping,
    float* __restrict__ token_expert_weights,
    __nv_bfloat16* __restrict__ X_combined){

    int orig_token_ind = blockIdx.x;

    if (orig_token_ind >= total_tokens) {
        return;
    }

    extern __shared__ uint8_t smem[];

    float * orig_token = (float *) smem;

    for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
        orig_token[i] = __bfloat162float(X_combined[orig_token_ind * model_dim + i]);
    }

    __syncthreads();

    int expert_zone_row;
    float token_expert_weight;

    for (int k = 0; k < num_selected_experts; k++){
        expert_zone_row = token_mapping[orig_token_ind * num_selected_experts + k];
        token_expert_weight = token_expert_weights[orig_token_ind * num_selected_experts + k];
        for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
            orig_token[i] += token_expert_weight * __bfloat162float(expert_zones[expert_zone_row * model_dim + i]);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
        X_combined[orig_token_ind * model_dim + i] = __float2bfloat16(orig_token[i]);
    }
  
}