#include "nvidia_ops.h"

#include "nvidia_ops.h"

// Use float4 to represent 8 bfloat16 values (16 bytes) for vectorized memory access.
typedef float4 bfloat16_8;

// Calculate the number of __nv_bfloat16 elements per vector.
#define PREPARE_EXPERT_ZONE_VEC_SIZE (sizeof(bfloat16_8) / sizeof(__nv_bfloat16))

/**
 * @brief Gathers expert rows using a cooperative, block-level approach for maximum memory bandwidth.
 *
 * This optimized kernel assigns one thread block per row to be gathered. All threads
 * in the block work together to copy the data for that row, ensuring that memory
 * accesses to the source and destination tensors are fully coalesced. This design
 * is crucial for maximizing memory bandwidth, especially when the kernel is only
 * launched on a subset of the GPU's SMs.
 *
 * @param model_dim The dimension of the model (i.e., the length of a row).
 * @param X The source tensor containing all token data.
 * @param expert_id The ID of the current expert being processed.
 * @param expert_counts An array holding the number of tokens assigned to each expert.
 * @param expert_counts_cumsum A cumulative sum of expert_counts, used to find offsets.
 * @param expert_mapping An array that maps each expert token to its original token ID.
 * @param expert_zone The destination buffer for the gathered rows for this expert.
 */
extern "C" __global__ void default_prepare_expert_zone_bf16_bf16_kernel(
    int model_dim,
    const __nv_bfloat16* __restrict__ X,
    int expert_id,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_counts_cumsum,
    const int* __restrict__ expert_mapping,
    __nv_bfloat16* __restrict__ expert_zone)
{
    const int num_tokens = expert_counts[expert_id];
    if (num_tokens == 0) {
        return;
    }

    // --- Pointers and Dimensions Setup ---
    const bfloat16_8* X_vec = (const bfloat16_8*)X;
    bfloat16_8* expert_zone_vec = (bfloat16_8*)expert_zone;
    const int model_dim_vec = model_dim / PREPARE_EXPERT_ZONE_VEC_SIZE;
    const int expert_base = expert_counts_cumsum[expert_id] - num_tokens;

    // --- Grid-Stride Loop for Cooperative Gathering (Block-per-Row) ---

    // Each BLOCK iterates through the rows assigned to this expert.
    for (int row_idx = blockIdx.x; row_idx < num_tokens; row_idx += gridDim.x) {
        // Find the original row ID. This is the only uncoalesced "gather" read.
        // Since it's one read per block, its performance impact is minimal.
        const int orig_token_id = expert_mapping[expert_base + row_idx];

        // Pointers to the start of the source and destination rows.
        const bfloat16_8* src_ptr = X_vec + (long long)orig_token_id * model_dim_vec;
        bfloat16_8* dst_ptr = expert_zone_vec + (long long)row_idx * model_dim_vec;

        // Threads within the block cooperatively copy the columns of the row.
        // Each thread starts at its threadIdx.x and strides by the block size.
        for (int col_vec_idx = threadIdx.x; col_vec_idx < model_dim_vec; col_vec_idx += blockDim.x) {
            // This access pattern is now COALESCED!
            dst_ptr[col_vec_idx] = src_ptr[col_vec_idx];
        }
    }
}

extern "C" __global__ void default_prepare_experts_bf16_bf16_kernel(
    int total_tokens,
    int model_dim,
    int num_selected_experts,
    const __nv_bfloat16* __restrict__ X,
    int* __restrict__ token_mapping,
    __nv_bfloat16* __restrict__ expert_zones) {

    // Each float4 vector holds 8 __nv_bfloat16 elements (16 bytes).
    // Accesses by a full warp (32 threads) will cover 32 * 16 = 512 bytes.
    // The hardware efficiently breaks this down into four 128-byte memory transactions.
    const int VEC_SIZE = sizeof(float4) / sizeof(__nv_bfloat16);

    // This vectorized kernel assumes model_dim is a multiple of 8.
    const int model_dim_vec = model_dim / VEC_SIZE;
    const int orig_token_ind = blockIdx.x;

    if (orig_token_ind >= total_tokens) {
        return;
    }

    // --- 1. Set up Shared Memory ---
    extern __shared__ uint8_t smem[];
    // Cast shared memory pointer to our vector type
    float4* orig_token = (float4*)smem;

    // --- 2. Reinterpret Global Pointers for Vector Access ---
    const float4* X_vec = (const float4*)X;
    float4* expert_zones_vec = (float4*)expert_zones;

    // --- 3. Vectorized Load from Global to Shared Memory ---
    // Each thread now loads one float4 (8 bfloat16s) instead of one bfloat16.
    for (int i = threadIdx.x; i < model_dim_vec; i += blockDim.x) {
        orig_token[i] = X_vec[orig_token_ind * model_dim_vec + i];
    }

    __syncthreads();

    // --- 4. Vectorized Store from Shared to Global Memory ---
    for (int k = 0; k < num_selected_experts; k++) {
        // Find the destination row for this token and expert
        int expert_zone_row = token_mapping[orig_token_ind * num_selected_experts + k];
        
        // Each thread now stores one float4 (8 bfloat16s).
        for (int i = threadIdx.x; i < model_dim_vec; i += blockDim.x) {
            expert_zones_vec[expert_zone_row * model_dim_vec + i] = orig_token[i];
        }
    }
}


extern "C" __global__ void naive_default_prepare_experts_bf16_bf16_kernel(
    int total_tokens,
    int model_dim,
    int num_selected_experts,
    const __nv_bfloat16* __restrict__ X,
    int* __restrict__ token_mapping,
    __nv_bfloat16* __restrict__ expert_zones){

    int orig_token_ind = blockIdx.x;

    if (orig_token_ind >= total_tokens){
        return;
    }

    extern __shared__ uint8_t smem[];

    __nv_bfloat16 * orig_token = (__nv_bfloat16 *) smem;

    for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
        orig_token[i] = X[orig_token_ind * model_dim + i];
    }

    __syncthreads();

    int expert_zone_row;

    for (int k = 0; k < num_selected_experts; k++){
        expert_zone_row = token_mapping[orig_token_ind * num_selected_experts + k];
        for (int i = threadIdx.x; i < model_dim; i += blockDim.x){
            expert_zones[expert_zone_row * model_dim + i] = orig_token[i];
        }
    }
}