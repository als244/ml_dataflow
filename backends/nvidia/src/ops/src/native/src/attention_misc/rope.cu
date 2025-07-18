#include "nvidia_ops.h"

// Define constants for vectorization dimensions
#define ROPE_VEC_SIZE 8
#define ROPE_PAIRS (ROPE_VEC_SIZE / 2)
#define ROPE_MAX_HEAD_DIM 256
#define ROPE_BANKED_SMEM_COLS (ROPE_MAX_HEAD_DIM / 2 + NUM_BANKS - 1) / NUM_BANKS

/* * A union to convert between a float4 vector and an array of four 
 * __nv_bfloat162 ROPE_BWD_PAIRS. This facilitates efficient 16-byte memory 
 * operations while allowing easy access to individual data ROPE_BWD_PAIRS.
 */
 typedef union {
    float4 f4;
    __nv_bfloat162 bf162[ROPE_PAIRS];
} rope_f4_bf162_converter;

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

extern "C" __global__ void default_rope_bf16_kernel(
    int num_tokens,
    int model_dim,
    int head_dim,
    int num_kv_heads,
    int theta,
    int* __restrict__ seq_positions, // Can also be restricted
    __nv_bfloat16* __restrict__ X_q,
    __nv_bfloat16* __restrict__ X_k) {
    
    // Banked shared memory to prevent bank conflicts
    __shared__ float smem_cos[NUM_BANKS][ROPE_BANKED_SMEM_COLS];
    __shared__ float smem_sin[NUM_BANKS][ROPE_BANKED_SMEM_COLS];

    const int row_ind = blockIdx.x;
    if (row_ind >= num_tokens) {
        return;
    }

    const int seq_pos = seq_positions[row_ind];
    const int num_pairs = head_dim / 2;

    // --- Phase 1: Write to shared memory using the banked layout ---
    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
        const int bank = i % NUM_BANKS;
        const int col = i / NUM_BANKS;
        const float current_dim = (float)(2 * i);
        const float inv_freq = powf(theta, -current_dim / (float)head_dim);
        sincosf((float)seq_pos * inv_freq, &smem_sin[bank][col], &smem_cos[bank][col]);
    }
    __syncthreads();

    // --- Phase 2: Apply RoPE, reading from banked shared memory ---
    
    // Process Queries
    const int num_q_vectors = model_dim / ROPE_VEC_SIZE;
    __nv_bfloat16* X_q_row = X_q + (uint64_t)row_ind * model_dim;

    for (int vec_idx = threadIdx.x; vec_idx < num_q_vectors; vec_idx += blockDim.x) {
        const int vec_start_dim = vec_idx * ROPE_VEC_SIZE;
        const int base_dim_in_head = vec_start_dim % head_dim;

        rope_f4_bf162_converter data;
        data.f4 = *( (float4*)(&X_q_row[vec_start_dim]) );

        #pragma unroll
        for (int i = 0; i < ROPE_PAIRS; ++i) {
            const int pair_idx = (base_dim_in_head / 2) + i;
            // Map the pair_idx to the banked 2D layout to read
            const int bank = pair_idx % NUM_BANKS;
            const int col = pair_idx / NUM_BANKS;

            const float2 vals_fp32 = __bfloat1622float2(data.bf162[i]);
            const float cos_val = smem_cos[bank][col];
            const float sin_val = smem_sin[bank][col];

            const float rope_even = cos_val * vals_fp32.x - sin_val * vals_fp32.y;
            const float rope_odd = cos_val * vals_fp32.y + sin_val * vals_fp32.x;

            data.bf162[i] = __floats2bfloat162_rn(rope_even, rope_odd);
        }
        *( (float4*)(&X_q_row[vec_start_dim]) ) = data.f4;
    }

    // Process Keys
    if (!X_k) return;
    __syncthreads();

    const int kv_dim = num_kv_heads * head_dim;
    const int num_k_vectors = kv_dim / ROPE_VEC_SIZE;
    __nv_bfloat16* X_k_row = X_k + (uint64_t)row_ind * kv_dim;

    for (int vec_idx = threadIdx.x; vec_idx < num_k_vectors; vec_idx += blockDim.x) {
        const int vec_start_dim = vec_idx * ROPE_VEC_SIZE;
        const int base_dim_in_head = vec_start_dim % head_dim;

        rope_f4_bf162_converter data;
        data.f4 = *( (float4*)(&X_k_row[vec_start_dim]) );

        #pragma unroll
        for (int i = 0; i < ROPE_PAIRS; ++i) {
            const int pair_idx = (base_dim_in_head / 2) + i;
            const int bank = pair_idx % NUM_BANKS;
            const int col = pair_idx / NUM_BANKS;

            const float2 vals_fp32 = __bfloat1622float2(data.bf162[i]);
            const float cos_val = smem_cos[bank][col];
            const float sin_val = smem_sin[bank][col];
            
            const float rope_even = cos_val * vals_fp32.x - sin_val * vals_fp32.y;
            const float rope_odd = cos_val * vals_fp32.y + sin_val * vals_fp32.x;

            data.bf162[i] = __floats2bfloat162_rn(rope_even, rope_odd);
        }
        *( (float4*)(&X_k_row[vec_start_dim]) ) = data.f4;
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