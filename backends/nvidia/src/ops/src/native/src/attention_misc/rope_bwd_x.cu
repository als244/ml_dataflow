#include "nvidia_ops.h"


extern "C" __global__ void default_rope_bwd_x_fp32_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, float * dX_q, float * dX_k){
    
    int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    int seq_pos = seq_positions[row_ind];

    // assert blockDim.x == head_dim / 2

    // each thread updates two positions within each head
    // for each head in row

    // e.g. if head_dim = 128, then blockDim.x = 64, and base_dim_in_head = 0, 2, 4, 6, 8, 10, ... 126
    int base_dim_in_head = 2 * threadIdx.x;

    float head_dim_frac = (float) base_dim_in_head / (float) head_dim;
    float angle = powf(theta, -1 * head_dim_frac);
    float cos_val = cosf(seq_pos * angle);
    float sin_val = sinf(seq_pos * angle);

    float dx_even;
    float dx_odd;

    float drope_even;
    float drope_odd;


    // first do the queries
    float * dX_q_row = dX_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        dx_even = dX_q_row[cur_dim];
        dx_odd = dX_q_row[cur_dim + 1];

        // the signs of the sin components flip during bwd pass
        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_q_row[cur_dim] = drope_even;
        dX_q_row[cur_dim + 1] = drope_odd;
    }

    
    int kv_dim = num_kv_heads * head_dim;
    float * dX_k_row = dX_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        dx_even = dX_k_row[cur_dim];
        dx_odd = dX_k_row[cur_dim + 1];

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_k_row[cur_dim] = drope_even;
        dX_k_row[cur_dim + 1] = drope_odd;
    }
}


extern "C" __global__ void default_rope_bwd_x_fp16_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __half * dX_q, __half * dX_k){
    
    int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    int seq_pos = seq_positions[row_ind];

    // assert blockDim.x == head_dim / 2

    // each thread updates two positions within each head
    // for each head in row

    // e.g. if head_dim = 128, then blockDim.x = 64, and base_dim_in_head = 0, 2, 4, 6, 8, 10, ... 126
    int base_dim_in_head = 2 * threadIdx.x;

    float head_dim_frac = (float) base_dim_in_head / (float) head_dim;
    float angle = powf(theta, -1 * head_dim_frac);
    float cos_val = cosf(seq_pos * angle);
    float sin_val = sinf(seq_pos * angle);

    float dx_even;
    float dx_odd;

    float drope_even;
    float drope_odd;


    // first do the queries
    __half * dX_q_row = dX_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        dx_even = __half2float(dX_q_row[cur_dim]);
        dx_odd = __half2float(dX_q_row[cur_dim + 1]);

        // the signs of the sin components flip during bwd pass
        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_q_row[cur_dim] = __float2half(drope_even);
        dX_q_row[cur_dim + 1] = __float2half(drope_odd);
    }

    
    int kv_dim = num_kv_heads * head_dim;
    __half * dX_k_row = dX_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        dx_even = __half2float(dX_k_row[cur_dim]);
        dx_odd = __half2float(dX_k_row[cur_dim + 1]);

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_k_row[cur_dim] = __float2half(drope_even);
        dX_k_row[cur_dim + 1] = __float2half(drope_odd);
    }
}

extern "C" __global__ void default_rope_bwd_x_bf16_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_bfloat16 * dX_q, __nv_bfloat16 * dX_k){
    
    int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    int seq_pos = seq_positions[row_ind];

    // assert blockDim.x == head_dim / 2

    // each thread updates two positions within each head
    // for each head in row

    // e.g. if head_dim = 128, then blockDim.x = 64, and base_dim_in_head = 0, 2, 4, 6, 8, 10, ... 126
    int base_dim_in_head = 2 * threadIdx.x;

    float head_dim_frac = (float) base_dim_in_head / (float) head_dim;
    float angle = powf(theta, -1 * head_dim_frac);
    float cos_val = cosf(seq_pos * angle);
    float sin_val = sinf(seq_pos * angle);

    float dx_even;
    float dx_odd;

    float drope_even;
    float drope_odd;


    // first do the queries
    __nv_bfloat16 * dX_q_row = dX_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        dx_even = __bfloat162float(dX_q_row[cur_dim]);
        dx_odd = __bfloat162float(dX_q_row[cur_dim + 1]);

        // the signs of the sin components flip during bwd pass
        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_q_row[cur_dim] = __float2bfloat16(drope_even);
        dX_q_row[cur_dim + 1] = __float2bfloat16(drope_odd);
    }

    
    int kv_dim = num_kv_heads * head_dim;
    __nv_bfloat16 * dX_k_row = dX_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        dx_even = __bfloat162float(dX_k_row[cur_dim]);
        dx_odd = __bfloat162float(dX_k_row[cur_dim + 1]);

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_k_row[cur_dim] = __float2bfloat16(drope_even);
        dX_k_row[cur_dim + 1] = __float2bfloat16(drope_odd);
    }
}

extern "C" __global__ void default_rope_bwd_x_fp8e4m3_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e4m3 * dX_q, __nv_fp8_e4m3 * dX_k){
    
    int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    int seq_pos = seq_positions[row_ind];

    // assert blockDim.x == head_dim / 2

    // each thread updates two positions within each head
    // for each head in row

    // e.g. if head_dim = 128, then blockDim.x = 64, and base_dim_in_head = 0, 2, 4, 6, 8, 10, ... 126
    int base_dim_in_head = 2 * threadIdx.x;

    float head_dim_frac = (float) base_dim_in_head / (float) head_dim;
    float angle = powf(theta, -1 * head_dim_frac);
    float cos_val = cosf(seq_pos * angle);
    float sin_val = sinf(seq_pos * angle);

    float dx_even;
    float dx_odd;

    float drope_even;
    float drope_odd;


    // first do the queries
    __nv_fp8_e4m3 * dX_q_row = dX_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        dx_even = float(dX_q_row[cur_dim]);
        dx_odd = float(dX_q_row[cur_dim + 1]);

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;

        dX_q_row[cur_dim] = __nv_fp8_e4m3(drope_even);
        dX_q_row[cur_dim + 1] = __nv_fp8_e4m3(drope_odd);
    }

    int kv_dim = num_kv_heads * head_dim;
    __nv_fp8_e4m3 * dX_k_row = dX_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        dx_even = float(dX_k_row[cur_dim]);
        dx_odd = float(dX_k_row[cur_dim + 1]);

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_k_row[cur_dim] = __nv_fp8_e4m3(drope_even);
        dX_k_row[cur_dim + 1] = __nv_fp8_e4m3(drope_odd);
    }
}

extern "C" __global__ void default_rope_bwd_x_fp8e5m2_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e5m2 * dX_q, __nv_fp8_e5m2 * dX_k){
    
    int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    int seq_pos = seq_positions[row_ind];

    // assert blockDim.x == head_dim / 2

    // each thread updates two positions within each head
    // for each head in row

    // e.g. if head_dim = 128, then blockDim.x = 64, and base_dim_in_head = 0, 2, 4, 6, 8, 10, ... 126
    int base_dim_in_head = 2 * threadIdx.x;

    float head_dim_frac = (float) base_dim_in_head / (float) head_dim;
    float angle = powf(theta, -1 * head_dim_frac);
    float cos_val = cosf(seq_pos * angle);
    float sin_val = sinf(seq_pos * angle);

    float dx_even;
    float dx_odd;

    float drope_even;
    float drope_odd;


    // first do the queries
    __nv_fp8_e5m2 * dX_q_row = dX_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        dx_even = float(dX_q_row[cur_dim]);
        dx_odd = float(dX_q_row[cur_dim + 1]);

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_q_row[cur_dim] = __nv_fp8_e5m2(drope_even);
        dX_q_row[cur_dim + 1] = __nv_fp8_e5m2(drope_odd);
    }

    int kv_dim = num_kv_heads * head_dim;
    __nv_fp8_e5m2 * dX_k_row = dX_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        dx_even = float(dX_k_row[cur_dim]);
        dx_odd = float(dX_k_row[cur_dim + 1]);

        drope_even = cos_val * dx_even + sin_val * dx_odd;
        drope_odd = cos_val * dx_odd - sin_val * dx_even;
        
        dX_k_row[cur_dim] = __nv_fp8_e5m2(drope_even);
        dX_k_row[cur_dim + 1] = __nv_fp8_e5m2(drope_odd);
    }
}