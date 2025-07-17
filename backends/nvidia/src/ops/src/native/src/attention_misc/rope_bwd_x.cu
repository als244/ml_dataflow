#include "nvidia_ops.h"

// Define constants for vectorization dimensions
#define ROPE_BWD_VEC_SIZE 8
#define ROPE_BWD_PAIRS (ROPE_BWD_VEC_SIZE / 2)

/* * A union to convert between a float4 vector and an array of four 
 * __nv_bfloat162 ROPE_BWD_PAIRS. This facilitates efficient 16-byte memory 
 * operations while allowing easy access to individual data ROPE_BWD_PAIRS.
 */
 typedef union {
    float4 f4;
    __nv_bfloat162 bf162[ROPE_BWD_PAIRS];
} rope_bwd_f4_bf162_converter;

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

    // Shared memory for caching sin/cos values.
    __shared__ float smem_cos[MAX_HEAD_DIM_ROPE / 2];
    __shared__ float smem_sin[MAX_HEAD_DIM_ROPE / 2];

    // One block per token (row)
    const int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    const int seq_pos = seq_positions[row_ind];
    const int num_pairs = head_dim / 2;

    // --- Phase 1: Pre-compute sin/cos values into shared memory ---
    // This section is identical to the optimized forward pass.
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
        const float current_dim = (float)(2 * i);
        const float inv_freq = powf(theta, -current_dim / (float)head_dim);
        sincosf((float)seq_pos * inv_freq, &smem_sin[i], &smem_cos[i]);
    }
    __syncthreads(); // Ensure all threads have finished writing to smem.

    // --- Phase 2: Apply inverse RoPE using a grid-stride loop ---

    // --- Process Query Gradients (dX_q) ---
    const int num_q_vectors = model_dim / ROPE_VEC_SIZE;
    __nv_bfloat16* dX_q_row = dX_q + (uint64_t)row_ind * model_dim;

    for (int vec_idx = threadIdx.x; vec_idx < num_q_vectors; vec_idx += blockDim.x) {
        const int vec_start_dim = vec_idx * ROPE_VEC_SIZE;
        const int base_dim_in_head = vec_start_dim % head_dim;

        rope_bwd_f4_bf162_converter data;
        data.f4 = *( (float4*)(&dX_q_row[vec_start_dim]) );

        #pragma unroll
        for (int i = 0; i < ROPE_PAIRS; ++i) {
            const int pair_idx = (base_dim_in_head / 2) + i;

            const float2 input_grads = __bfloat1622float2(data.bf162[i]);
            const float dx_even = input_grads.x;
            const float dx_odd = input_grads.y;

            const float cos_val = smem_cos[pair_idx];
            const float sin_val = smem_sin[pair_idx];

            // Apply the backward transformation (inverse rotation)
            const float drope_even = cos_val * dx_even + sin_val * dx_odd;
            const float drope_odd  = cos_val * dx_odd  - sin_val * dx_even;

            data.bf162[i] = __floats2bfloat162_rn(drope_even, drope_odd);
        }
        *( (float4*)(&dX_q_row[vec_start_dim]) ) = data.f4;
    }

    // --- Process Key Gradients (dX_k) ---
    if (!dX_k) {
        return;
    }

    __syncthreads(); // Sync to ensure all Q-gradient writes are complete.

    const int kv_dim = num_kv_heads * head_dim;
    const int num_k_vectors = kv_dim / ROPE_VEC_SIZE;
    __nv_bfloat16* dX_k_row = dX_k + (uint64_t)row_ind * kv_dim;

    for (int vec_idx = threadIdx.x; vec_idx < num_k_vectors; vec_idx += blockDim.x) {
        const int vec_start_dim = vec_idx * ROPE_VEC_SIZE;
        const int base_dim_in_head = vec_start_dim % head_dim;

        rope_bwd_f4_bf162_converter data;
        data.f4 = *( (float4*)(&dX_k_row[vec_start_dim]) );

        #pragma unroll
        for (int i = 0; i < ROPE_PAIRS; ++i) {
            const int pair_idx = (base_dim_in_head / 2) + i;

            const float2 input_grads = __bfloat1622float2(data.bf162[i]);
            const float dx_even = input_grads.x;
            const float dx_odd = input_grads.y;
            
            const float cos_val = smem_cos[pair_idx];
            const float sin_val = smem_sin[pair_idx];

            // Apply the backward transformation (inverse rotation)
            const float drope_even = cos_val * dx_even + sin_val * dx_odd;
            const float drope_odd  = cos_val * dx_odd  - sin_val * dx_even;
            
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