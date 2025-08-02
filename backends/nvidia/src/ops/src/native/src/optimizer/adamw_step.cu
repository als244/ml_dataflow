#include "nvidia_ops.h"


extern "C" __global__ void default_adamw_step_bf16_bf16_bf16_bf16_kernel(uint64_t num_els, int step_num, float lr, float beta1, float beta2, float weight_decay, float epsilon, 
                                                                                __nv_bfloat16 * param, __nv_bfloat16 * grad, __nv_bfloat16 * mean, __nv_bfloat16 * var){

    // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];                                                                                

    // every block will handle a single layer of the transformer...
    uint64_t num_blocks = (uint64_t) gridDim.x;

    uint64_t num_els_per_block = num_els / num_blocks;

    uint64_t els_remain = num_els % num_blocks;
    uint64_t els_offset;
    if (blockIdx.x < els_remain){
        num_els_per_block += 1;
        els_offset = (uint64_t) blockIdx.x * num_els_per_block;
    }
    else{
        els_offset = (uint64_t) blockIdx.x * num_els_per_block + (uint64_t) els_remain;
    }
    

    float param_val;
    float grad_val;
    float mean_val;
    float var_val;

    float m_t;
    float v_t;
    float m_t_hat;
    float v_t_hat;
    float param_out;


    for (uint64_t i = els_offset + threadIdx.x; i < els_offset + num_els_per_block; i += blockDim.x) {
        param_val = __bfloat162float(param[i]);
        grad_val = __bfloat162float(grad[i]);
        mean_val = __bfloat162float(mean[i]);
        var_val = __bfloat162float(var[i]);

        // AdamW update: m_t and v_t are calculated with the original gradient
        m_t = beta1 * mean_val + (1.0f - beta1) * grad_val;
        v_t = beta2 * var_val + (1.0f - beta2) * grad_val * grad_val;

        // Bias correction
        // Ensure step_num is at least 1 for powf to behave as expected (1-beta^0 = 0 -> div by zero)
        // Adam typically uses step_num starting from 1.
        m_t_hat = m_t / (1.0f - powf(beta1, (float)step_num));
        v_t_hat = v_t / (1.0f - powf(beta2, (float)step_num));

        // AdamW parameter update
        // param_out = param_val - lr * (gradient_update_term + weight_decay_term)
        // gradient_update_term = m_t_hat / (sqrtf(v_t_hat) + epsilon)
        // weight_decay_term    = weight_decay * param_val
        param_out = param_val - lr * (m_t_hat / (sqrtf(v_t_hat) + epsilon) + weight_decay * param_val);

        param[i] = __float2bfloat16(param_out);
        mean[i] = __float2bfloat16(m_t);
        var[i] = __float2bfloat16(v_t);

        if (isnan(param_out)){
            printf("AdamW Step ERROR: param_out is nan at index %lu\n", i);
            return;
        }

        if (isinf(param_out)){
            printf("AdamW Step ERROR: param_out is inf at index %lu\n", i);
            return;
        }
    }
}