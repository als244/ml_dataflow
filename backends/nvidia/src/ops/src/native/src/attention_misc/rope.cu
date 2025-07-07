#include "nvidia_ops.h"


extern "C" __global__ void default_rope_fp32_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, float * X_q, float * X_k){
    
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

    float x_even;
    float x_odd;

    float rope_even;
    float rope_odd;


    // first do the queries
     float * X_q_row = X_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        x_even = X_q_row[cur_dim];
        x_odd = X_q_row[cur_dim + 1];

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        X_q_row[cur_dim] = rope_even;
        X_q_row[cur_dim + 1] = rope_odd;
    }

    // during recompute X_k is NULL
    if (!X_k){
        return;
    }

    
    int kv_dim = num_kv_heads * head_dim;
    float * X_k_row = X_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        x_even = X_k_row[cur_dim];
        x_odd = X_k_row[cur_dim + 1];

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        
        X_k_row[cur_dim] = rope_even;
        X_k_row[cur_dim + 1] = rope_odd;
    }
}


extern "C" __global__ void default_rope_fp16_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __half * X_q, __half * X_k){
    
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

    float x_even;
    float x_odd;

    float rope_even;
    float rope_odd;


    // first do the queries
     __half * X_q_row = X_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        x_even = X_q_row[cur_dim];
        x_odd = X_q_row[cur_dim + 1];

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        X_q_row[cur_dim] = rope_even;
        X_q_row[cur_dim + 1] = rope_odd;
    }

    // during recompute X_k is NULL
    if (!X_k){
        return;
    }

    
    int kv_dim = num_kv_heads * head_dim;
    __half * X_k_row = X_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        x_even = __half2float(X_k_row[cur_dim]);
        x_odd = __half2float(X_k_row[cur_dim + 1]);

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        
        X_k_row[cur_dim] = __float2half(rope_even);
        X_k_row[cur_dim + 1] = __float2half(rope_odd);
    }
}

extern "C" __global__ void default_rope_bf16_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_bfloat16 * X_q, __nv_bfloat16 * X_k){
    
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

    float x_even;
    float x_odd;

    float rope_even;
    float rope_odd;


    // first do the queries
     __nv_bfloat16 * X_q_row = X_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        x_even = X_q_row[cur_dim];
        x_odd = X_q_row[cur_dim + 1];

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        X_q_row[cur_dim] = rope_even;
        X_q_row[cur_dim + 1] = rope_odd;
    }

    // during recompute X_k is NULL
    if (!X_k){
        return;
    }
    
    int kv_dim = num_kv_heads * head_dim;
    __nv_bfloat16 * X_k_row = X_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        x_even = __bfloat162float(X_k_row[cur_dim]);
        x_odd = __bfloat162float(X_k_row[cur_dim + 1]);

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        
        X_k_row[cur_dim] = __float2bfloat16(rope_even);
        X_k_row[cur_dim + 1] = __float2bfloat16(rope_odd);
    }
}


extern "C" __global__ void default_rope_fp8e4m3_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e4m3 * X_q, __nv_fp8_e4m3 * X_k){
    
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

    float x_even;
    float x_odd;

    float rope_even;
    float rope_odd;


    // first do the queries
     __nv_fp8_e4m3 * X_q_row = X_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        x_even = float(X_q_row[cur_dim]);
        x_odd = float(X_q_row[cur_dim + 1]);

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        X_q_row[cur_dim] = __nv_fp8_e4m3(rope_even);
        X_q_row[cur_dim + 1] = __nv_fp8_e4m3(rope_odd);
    }

    // during recompute X_k is NULL
    if (!X_k){
        return;
    }
    
    int kv_dim = num_kv_heads * head_dim;
    __nv_fp8_e4m3 * X_k_row = X_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        x_even = float(X_k_row[cur_dim]);
        x_odd = float(X_k_row[cur_dim + 1]);

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        
        X_k_row[cur_dim] = __nv_fp8_e4m3(rope_even);
        X_k_row[cur_dim + 1] = __nv_fp8_e4m3(rope_odd);
    }
}

extern "C" __global__ void default_rope_fp8e5m2_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e5m2 * X_q, __nv_fp8_e5m2 * X_k){
    
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

    float x_even;
    float x_odd;

    float rope_even;
    float rope_odd;


    // first do the queries
     __nv_fp8_e5m2 * X_q_row = X_q + ((uint64_t)row_ind * (uint64_t) model_dim);

    // advance through each head
    // we set blockDim.x to be head_dim / 2, so we know  no threads will step on each other
    // and that all values in the row are covered
    for (int cur_dim = base_dim_in_head; cur_dim < model_dim; cur_dim += head_dim){
        x_even = float(X_q_row[cur_dim]);
        x_odd = float(X_q_row[cur_dim + 1]);

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        X_q_row[cur_dim] = __nv_fp8_e5m2(rope_even);
        X_q_row[cur_dim + 1] = __nv_fp8_e5m2(rope_odd);
    }

    // during recompute X_k is NULL
    if (!X_k){
        return;
    }
    
    int kv_dim = num_kv_heads * head_dim;
    __nv_fp8_e5m2 * X_k_row = X_k + ((uint64_t)row_ind * (uint64_t)kv_dim);

    // now do the keys
    for (int cur_dim = base_dim_in_head; cur_dim < kv_dim; cur_dim += head_dim){
        x_even = float(X_k_row[cur_dim]);
        x_odd = float(X_k_row[cur_dim + 1]);

        rope_even = cos_val * x_even - sin_val * x_odd;
        rope_odd = cos_val * x_odd + sin_val * x_even;
        
        X_k_row[cur_dim] = __nv_fp8_e5m2(rope_even);
        X_k_row[cur_dim + 1] = __nv_fp8_e5m2(rope_odd);
    }
}