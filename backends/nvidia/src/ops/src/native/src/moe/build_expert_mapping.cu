#include "nvidia_ops.h"

#define MAX_NUM_EXPERTS 4096

/**
 * @brief Fused kernel that performs counting, offset calculation, and writing
 * in a single launch.
 *
 * @param total_tokens        Total number of tokens in the input matrix
 * @param top_k_experts       Number of experts in the input matrix
 * @param chosen_experts           Input M x K matrix of uint16 values.
 * @param expert_counts_cumsum  A K-length array of atomic counters, PRE-INITIALIZED
 * with the starting offsets for each value category 
 * => expert_counts_cumsum is the argument pssed in
 * @param expert_mapping      The final M x K output array for row indices.

 */
extern "C" __global__ void default_build_expert_mapping_kernel(int num_tokens, int num_routed_experts, int num_selected_experts, 
                                        const uint16_t* chosen_experts,
                                       int* expert_counts_cumsum,
                                       int* expert_mapping)
{
    // --- Shared Memory Declaration ---
    // Stores counts for this block (Phase 1). Re-used for local write offsets (Phase 3).
    __shared__ int s_counts[MAX_NUM_EXPERTS];
    // Stores the base global memory write position for this block (Phase 2).
    __shared__ int s_block_write_base[MAX_NUM_EXPERTS];


    // ========================================================================
    // == PHASE 1: Local Counting in Shared Memory
    // ========================================================================

    // Each thread initializes one counter in shared memory.
    for (int j = threadIdx.x; j < num_routed_experts; j += blockDim.x) {
        s_counts[j] = 0;
    }
    __syncthreads(); // Ensure all counters are zero before proceeding.

    // Use a grid-stride loop to have each thread process one or more rows.
    // This ensures that the kernel works correctly for any M.
    for (int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
         row_idx < num_tokens;
         row_idx += gridDim.x * blockDim.x)
    {
        // Each thread processes all K elements in its assigned row.
        for (int k = 0; k < num_selected_experts; ++k) {
            uint16_t val = chosen_experts[row_idx * num_selected_experts + k];
            // ensure val [0, num_routed_experts)
            atomicAdd(&s_counts[val], 1);
        }
    }

    // Wait for all threads to finish counting. s_counts now holds the
    // total count for each value within this block.
    __syncthreads();


    // ========================================================================
    // == PHASE 2: Reserve Global Write Space
    // ========================================================================

    for (int j = threadIdx.x; j < num_routed_experts; j += blockDim.x) {
        // Atomically get the starting position for this block's contribution
        // and update the global head for the next block.
        s_block_write_base[j] = atomicAdd(&expert_counts_cumsum[j], s_counts[j]);
    }

    // Synchronize to make the block's base write addresses visible to all threads.
    __syncthreads();


    // ========================================================================
    // == PHASE 3: Parallel Write to Global Memory
    // ========================================================================

    // Repurpose s_counts to be the local write offset within the block.
    // It must be reset to zero.
    for (int j = threadIdx.x; j < num_routed_experts; j += blockDim.x) {
        s_counts[j] = 0;
    }
    __syncthreads();

    // Re-iterate over the same rows to perform the writes.
    // The previous counts/values are not stored, so we re-read.
    for (int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
         row_idx < num_tokens;
         row_idx += gridDim.x * blockDim.x)
    {
        for (int k = 0; k < num_selected_experts; ++k) {
            uint16_t val = chosen_experts[row_idx * num_selected_experts + k];
            // Get a unique local offset for this value within this block.
            int local_offset = atomicAdd(&s_counts[val], 1);

            // Calculate the final global write position.
            int write_pos = s_block_write_base[val] + local_offset;

            // Write the row index to its final sorted position.
            expert_mapping[write_pos] = row_idx;
        }
    }
}