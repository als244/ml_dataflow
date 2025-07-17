#include "nvidia_ops.h"

// Define constants for vectorization dimensions
#define ROPE_BWD_VEC_SIZE 8
#define ROPE_BWD_PAIRS (ROPE_BWD_VEC_SIZE / 2)

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

extern "C" __global__ void default_rope_bwd_x_bf16_kernel(
    int num_tokens,
    int model_dim,
    int head_dim,
    int num_kv_heads,
    int theta,
    int* __restrict__ seq_positions,
    __nv_bfloat16* __restrict__ dX_q,
    __nv_bfloat16* __restrict__ dX_k) {

    // One block per token (row)
    const int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    const int seq_pos = seq_positions[row_ind];

    // The starting dimension within a head for this thread's vector.
    // Note: Launch with blockDim.x = head_dim / ROPE_BWD_VEC_SIZE.
    const int base_dim_in_head = ROPE_BWD_VEC_SIZE * threadIdx.x;

    // Pre-calculate the cosine and sine values for the 4 ROPE_BWD_PAIRS.
    float cos_vals[ROPE_BWD_PAIRS];
    float sin_vals[ROPE_BWD_PAIRS];

    for (int i = 0; i < ROPE_BWD_PAIRS; ++i) {
        const int dim_offset = 2 * i;
        const float current_dim = (float)(base_dim_in_head + dim_offset);
        const float inv_freq = __powf((float)theta, -current_dim / (float)head_dim);
        __sincosf((float)seq_pos * inv_freq, &sin_vals[i], &cos_vals[i]);
    }

    // --- Process Query Gradients (dX_q) ---
    __nv_bfloat16* dX_q_row = dX_q + (uint64_t)row_ind * model_dim;

    for (int vec_start_dim = base_dim_in_head; vec_start_dim < model_dim; vec_start_dim += head_dim) {
        f4_bf162_converter data;

        // Vectorized load
        data.f4 = *( (float4*)(&dX_q_row[vec_start_dim]) );

        #pragma unroll
        for (int i = 0; i < ROPE_BWD_PAIRS; ++i) {
            const float2 input_grads = __bfloat1622float2(data.bf162[i]);
            const float dx_even = input_grads.x;
            const float dx_odd = input_grads.y;

            // Backward pass transformation (inverse rotation)
            const float drope_even = cos_vals[i] * dx_even + sin_vals[i] * dx_odd;
            const float drope_odd  = cos_vals[i] * dx_odd - sin_vals[i] * dx_even;

            data.bf162[i] = __floats2bfloat162_rn(drope_even, drope_odd);
        }

        // Vectorized store
        *( (float4*)(&dX_q_row[vec_start_dim]) ) = data.f4;
    }

    // --- Process Key Gradients (dX_k) ---
    if (!dX_k) {
        return;
    }
    
    const int kv_dim = num_kv_heads * head_dim;
    __nv_bfloat16* dX_k_row = dX_k + (uint64_t)row_ind * kv_dim;

    for (int vec_start_dim = base_dim_in_head; vec_start_dim < kv_dim; vec_start_dim += head_dim) {
        f4_bf162_converter data;

        data.f4 = *( (float4*)(&dX_k_row[vec_start_dim]) );

        #pragma unroll
        for (int i = 0; i < ROPE_BWD_PAIRS; ++i) {
            const float2 input_grads = __bfloat1622float2(data.bf162[i]);
            const float dx_even = input_grads.x;
            const float dx_odd = input_grads.y;

            // Backward pass transformation
            const float drope_even = cos_vals[i] * dx_even + sin_vals[i] * dx_odd;
            const float drope_odd  = cos_vals[i] * dx_odd - sin_vals[i] * dx_even;
            
            data.bf162[i] = __floats2bfloat162_rn(drope_even, drope_odd);
        }
        
        *( (float4*)(&dX_k_row[vec_start_dim]) ) = data.f4;
    }
    
}

extern "C" __global__ void naive_default_rope_bwd_x_bf16_kernel(int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_bfloat16 * dX_q, __nv_bfloat16 * dX_k){
    
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