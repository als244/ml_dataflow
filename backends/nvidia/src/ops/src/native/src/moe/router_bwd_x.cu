#include "nvidia_ops.h"


extern "C" __global__ void default_router_bwd_x_bf16_bf16_kernel(
    int num_expert_tokens, int model_dim, int num_routed_experts, int top_k_active,
    int expert_id, int *expert_counts_cumsum, int *expert_mapping,
    uint16_t *chosen_experts, float *token_expert_weights,
    __nv_bfloat16 *expert_out, __nv_bfloat16 *upstream_dX,
    __nv_bfloat16 *dX_routed, __nv_bfloat16 *dX_expert_out) {

    extern __shared__ char smem[];
    __nv_bfloat16* smem_upstream_dX = (__nv_bfloat16*)smem;
    float* warp_sums = (float*)(smem + model_dim * sizeof(__nv_bfloat16));
    float* smem_token_weight = (float*)(smem + model_dim * sizeof(__nv_bfloat16) + 32 * sizeof(float));

    const int VEC2_SIZE = 2;
    const int VEC8_SIZE = 8;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int new_row_ind = blockIdx.x;

    while (new_row_ind < num_expert_tokens) {
        const int expert_base_ind = expert_counts_cumsum[expert_id] - num_expert_tokens;
        const int orig_row_ind = expert_mapping[expert_base_ind + new_row_ind];
        const int model_dim_vec8 = model_dim / VEC8_SIZE;
        float4 *upstream_dX_f4 = (float4*)&upstream_dX[orig_row_ind * model_dim];
        float4 *smem_upstream_dX_f4 = (float4*)smem_upstream_dX;

        for (int i = tid; i < model_dim_vec8; i += blockDim.x) {
            smem_upstream_dX_f4[i] = upstream_dX_f4[i];
        }
        __syncthreads();

        float thread_sum = 0.0f;
        const int model_dim_vec2 = model_dim / VEC2_SIZE;
        const __nv_bfloat162* expert_out_bf162 = (const __nv_bfloat162*)&expert_out[new_row_ind * model_dim];
        const __nv_bfloat162* smem_upstream_dX_bf162 = (const __nv_bfloat162*)smem_upstream_dX;

        // CORRECTED: Manually compute the dot product for bfloat16.
        // This is the correct and performant method for all architectures.
        for (int i = tid; i < model_dim_vec2; i += blockDim.x) {
            float2 e_f2 = __bfloat1622float2(expert_out_bf162[i]);
            float2 u_f2 = __bfloat1622float2(smem_upstream_dX_bf162[i]);
            thread_sum += (e_f2.x * u_f2.x) + (e_f2.y * u_f2.y);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        }
        if (lane_id == 0) {
            warp_sums[warp_id] = thread_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            const int num_warps = blockDim.x / 32;
            thread_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
            }

            if (lane_id == 0) {
                dX_routed[orig_row_ind * num_routed_experts + expert_id] = __float2bfloat16(thread_sum);
                *smem_token_weight = 0.0f;
            }
             __syncwarp();

            for (int i = lane_id; i < top_k_active; i += 32) {
                if (chosen_experts[orig_row_ind * top_k_active + i] == (uint16_t)expert_id) {
                    *smem_token_weight = token_expert_weights[orig_row_ind * top_k_active + i];
                    break;
                }
            }
        }
        __syncthreads();

        const float token_weight = *smem_token_weight;

        if (token_weight != 0.0f) {
            __nv_bfloat162* dX_expert_out_bf162 = (__nv_bfloat162*)&dX_expert_out[new_row_ind * model_dim];
            // The __hmul2 intrinsic for bfloat16 IS valid and provides a speedup.
            const __nv_bfloat16 token_weight_bf16 = __float2bfloat16(token_weight);
            const __nv_bfloat162 token_weight_bf162 = __halves2bfloat162(token_weight_bf16, token_weight_bf16);
            for (int i = tid; i < model_dim_vec2; i += blockDim.x) {
                dX_expert_out_bf162[i] = __hmul2(smem_upstream_dX_bf162[i], token_weight_bf162);
            }
        }

        new_row_ind += gridDim.x;
    }
}

// TODO: vectorize this kernel, think about if block-per row is ideal, and handle smem more efficiently...
extern "C" __global__ void naive_default_router_bwd_x_bf16_bf16_kernel(int num_expert_tokens, int model_dim, int num_routed_experts, int top_k_active,
                                                int expert_id, int * expert_counts_cumsum, int * expert_mapping, 
                                                uint16_t * chosen_experts, float * token_expert_weights,
                                                __nv_bfloat16 * expert_out, __nv_bfloat16 * upstream_dX,
                                                __nv_bfloat16 * dX_routed,
                                                __nv_bfloat16 * dX_expert_out){


    int new_row_ind = blockIdx.x;

    if (new_row_ind >= num_expert_tokens){
        return;
    }

    int expert_base_ind = expert_counts_cumsum[expert_id] - num_expert_tokens;

    float thread_sum = 0;
    
    __shared__ float warp_sums[32];

    __shared__ float smem_token_weight;

      // could load the upstream_dX into shared memory for efficiency as we need to copy it to dX_expert_out...

    float token_weight;

    int orig_row_ind;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    while (new_row_ind < num_expert_tokens){

        if (threadIdx.x < 32){
            warp_sums[threadIdx.x] = 0;
        }
        
        // 1.) get orig row ind corresponding to this token
        orig_row_ind = expert_mapping[expert_base_ind + new_row_ind];

        // 2.) compute dot product of expert_out and upstream dX

        thread_sum = 0;
        
        for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
            thread_sum += __bfloat162float(expert_out[new_row_ind * model_dim + i]) * __bfloat162float(upstream_dX[orig_row_ind * model_dim + i]);
        }

        __syncwarp();

        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        }

        if (lane_id == 0) {
            warp_sums[warp_id] = thread_sum;
        }

        __syncthreads();

        // 3.) complete the dot-product and populate dX_routed with this scalar value

        if (warp_id == 0){
            thread_sum = warp_sums[lane_id];

            for (int offset = 16; offset > 0; offset >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
            }

            if (lane_id == 0){
                dX_routed[orig_row_ind * num_routed_experts + expert_id] = thread_sum;
            }

            __syncwarp();

            // figure out the token_weight is
            for (int i = lane_id; i < top_k_active; i += 32){
                if (chosen_experts[orig_row_ind * top_k_active + i] == (uint16_t) expert_id){
                    smem_token_weight = token_expert_weights[orig_row_ind * top_k_active + i];
                    break;
                }
            }
            
        }

        __syncthreads();

        token_weight = smem_token_weight;

        if (token_weight == 0.0f){
            if (threadIdx.x == 0){
                printf("Error: token_weight is 0.0f for orig token %d and expert %d...\n", orig_row_ind, expert_id);
            }
            return;
        }

        // 4.) repopulate dX_expert_out with the rows from inp_grad_stream -> X * weight assoicated with this expert (for each token)...
        
        // TODO: OPTIMIZE THESE STORES WITH VECTORIZATION
        for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
            dX_expert_out[new_row_ind * model_dim + i] = __float2bfloat16(__bfloat162float(upstream_dX[orig_row_ind * model_dim + i]) * token_weight);
        }

        new_row_ind += gridDim.x;
    }
}







