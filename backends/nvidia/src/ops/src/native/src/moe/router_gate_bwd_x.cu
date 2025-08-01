#include "nvidia_ops.h"


// only launch 1 warp per token

// doing derivative of softmax gate
extern "C" __global__ void default_router_gate_bwd_x_bf16_bf16_kernel(int num_tokens, int num_routed_experts, int top_k_active,
                                                    uint16_t * chosen_experts, float * token_expert_weights,
                                                    __nv_bfloat16 * dX_routed){

    int token_id = blockIdx.x;

    if (token_id >= num_tokens){
        return;
    }

    __shared__ float smem_avg_upstream_grad;

    float avg_upstream_grad;

    float thread_sum = 0;
    
    uint16_t chosen_expert;

    while (token_id < num_tokens){

        thread_sum = 0;

        // 1.) first get sum of dL/dp_j * p_j for each selected expert

        for (int i = threadIdx.x; i < top_k_active; i += blockDim.x){
            chosen_expert = chosen_experts[token_id * top_k_active + i];
            thread_sum += __bfloat162float(dX_routed[token_id * num_routed_experts + chosen_expert]) * token_expert_weights[token_id * top_k_active + i];
        }

        for (int offset = 16; offset > 0; offset >>= 1){
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        }

        if (threadIdx.x == 0){
            smem_avg_upstream_grad = thread_sum;
        }

        __syncthreads();

        avg_upstream_grad = smem_avg_upstream_grad;

        // 2.) now compute dL/dg_i = p_i * (dL/dp_i - avg_upstream_grad)

        for (int i = threadIdx.x; i < top_k_active; i += blockDim.x){
            chosen_expert = chosen_experts[token_id * top_k_active + i];
            dX_routed[token_id * num_routed_experts + chosen_expert] = __float2bfloat16(__bfloat162float(dX_routed[token_id * num_routed_experts + chosen_expert]) * (token_expert_weights[token_id * top_k_active + i] - avg_upstream_grad));
        }

        token_id += gridDim.x;
    }
}