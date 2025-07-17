#include "nvidia_ops.h"

/**
 * @brief A union to facilitate type-punning between a float4 vector and 
 * an array of 8 __nv_bfloat16 values.
 * * The size of a float4 (4 * 4 bytes = 16 bytes) is the same as the size of
 * eight __nv_bfloat16 values (8 * 2 bytes = 16 bytes). This union allows
 * us to load 128 bits of data from memory using a single vectorized load
 * and then access it as individual bfloat16 elements for computation.
 */
 typedef union {
    float4 f4;
    __nv_bfloat16 bf16[8];
} b16x8_f4_union;

extern "C" __global__ void default_softmax_fp32_fp32_kernel(int n_rows, int n_cols, float * X, float * out) {
	 // Each block processes one row
	 uint64_t row_ind = blockIdx.x;
	 if (row_ind >= n_rows) {
		 return;
	 }
 
	 uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	 float* row_start = X + row_offset;
	 float* out_row_start = out + row_offset;
 
	 // Use float for intermediate computations for precision
	 float thread_max = -FLT_MAX;
	 float thread_sum = 0.0f;
 
	 // =========================================================================
	 // Pass 1: Online calculation of max and sum for each thread's data chunk
	 // =========================================================================
	 for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
		 float val = row_start[i];
		 
		 float old_max = thread_max;
		 thread_max = fmaxf(thread_max, val);
 
		 thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
	 }
 
	 // =========================================================================
	 // Block-wide reduction of (max, sum) pairs
	 // =========================================================================
	 
	 // Step 1: Warp-level reduction using shuffle instructions
	 unsigned mask = 0xFFFFFFFFU;
	 for (int offset = 16; offset > 0; offset >>= 1) {
		 // Get partner thread's (max, sum)
		 float partner_max = __shfl_down_sync(mask, thread_max, offset);
		 float partner_sum = __shfl_down_sync(mask, thread_sum, offset);
 
		 // Combine the pairs using the same online logic
		 float old_max = thread_max;
		 thread_max = fmaxf(thread_max, partner_max);
		 float scale = expf(old_max - thread_max);
		 
		 thread_sum = thread_sum * scale + partner_sum * expf(partner_max - thread_max);
	 }
 
	 // Step 2: Inter-warp reduction using shared memory
	 // Each warp's lane 0 now holds the warp's (max, sum)
	 __shared__ float smem_max[32]; // Max warps per block = 1024 / 32 = 32
	 __shared__ float smem_sum[32];
 
	 int warp_id = threadIdx.x / 32;
	 int lane_id = threadIdx.x % 32;
 
	 if (lane_id == 0) {
		 smem_max[warp_id] = thread_max;
		 smem_sum[warp_id] = thread_sum;
	 }
 
	 __syncthreads();
 
	 // Step 3: Final reduction by the first warp
	 // Load warp-reduced values into first warp's registers
	 if (warp_id == 0) {
		 // Only load if there's a valid corresponding warp
		 if (lane_id * 32 < blockDim.x) {
			 thread_max = smem_max[lane_id];
			 thread_sum = smem_sum[lane_id];
		 } else {
			 thread_max = -FLT_MAX;
			 thread_sum = 0.0f;
		 }
 
		 // Final reduction within the first warp
		 for (int offset = 16; offset > 0; offset >>= 1) {
			  float partner_max = __shfl_down_sync(mask, thread_max, offset);
			  float partner_sum = __shfl_down_sync(mask, thread_sum, offset);
			  
			  float old_max = thread_max;
			  thread_max = fmaxf(thread_max, partner_max);
			  float scale = expf(old_max - thread_max);
			  thread_sum = thread_sum * scale + partner_sum * expf(partner_max - thread_max);
		 }
	 }
 
	 // The final (max, sum) is in lane 0 of warp 0. Broadcast to all threads.
	 if (warp_id == 0 && lane_id == 0) {
		 smem_max[0] = thread_max;
		 smem_sum[0] = thread_sum;
	 }
	 
	 __syncthreads();
	 
	 float block_max = smem_max[0];
	 float block_sum = smem_sum[0];
 
	 // =========================================================================
	 // Pass 2: Apply normalization and write output
	 // =========================================================================
	 for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
		 float val = row_start[i];
		 float final_val = expf(val - block_max) / block_sum;
		 out_row_start[i] = final_val;
	 }
}


extern "C" __global__ void default_softmax_fp16_fp16_kernel(int n_rows, int n_cols, __half * X, __half * out) {
	 // Each block processes one row
	 uint64_t row_ind = blockIdx.x;
	 if (row_ind >= n_rows) {
		 return;
	 }
 
	 uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	 __half* row_start = X + row_offset;
	 __half* out_row_start = out + row_offset;
 
	 // Use float for intermediate computations for precision
	 float thread_max = -FLT_MAX;
	 float thread_sum = 0.0f;
 
	 // =========================================================================
	 // Pass 1: Online calculation of max and sum for each thread's data chunk
	 // =========================================================================
	 for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
		 float val = __half2float(row_start[i]);
		 
		 float old_max = thread_max;
		 thread_max = fmaxf(thread_max, val);
 
		 thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
	 }
 
	 // =========================================================================
	 // Block-wide reduction of (max, sum) pairs
	 // =========================================================================
	 
	 // Step 1: Warp-level reduction using shuffle instructions
	 unsigned mask = 0xFFFFFFFFU;
	 for (int offset = 16; offset > 0; offset >>= 1) {
		 // Get partner thread's (max, sum)
		 float partner_max = __shfl_down_sync(mask, thread_max, offset);
		 float partner_sum = __shfl_down_sync(mask, thread_sum, offset);
 
		 // Combine the pairs using the same online logic
		 float old_max = thread_max;
		 thread_max = fmaxf(thread_max, partner_max);
		 float scale = expf(old_max - thread_max);
		 
		 thread_sum = thread_sum * scale + partner_sum * expf(partner_max - thread_max);
	 }
 
	 // Step 2: Inter-warp reduction using shared memory
	 // Each warp's lane 0 now holds the warp's (max, sum)
	 __shared__ float smem_max[32]; // Max warps per block = 1024 / 32 = 32
	 __shared__ float smem_sum[32];
 
	 int warp_id = threadIdx.x / 32;
	 int lane_id = threadIdx.x % 32;
 
	 if (lane_id == 0) {
		 smem_max[warp_id] = thread_max;
		 smem_sum[warp_id] = thread_sum;
	 }
 
	 __syncthreads();
 
	 // Step 3: Final reduction by the first warp
	 // Load warp-reduced values into first warp's registers
	 if (warp_id == 0) {
		 // Only load if there's a valid corresponding warp
		 if (lane_id * 32 < blockDim.x) {
			 thread_max = smem_max[lane_id];
			 thread_sum = smem_sum[lane_id];
		 } else {
			 thread_max = -FLT_MAX;
			 thread_sum = 0.0f;
		 }
 
		 // Final reduction within the first warp
		 for (int offset = 16; offset > 0; offset >>= 1) {
			  float partner_max = __shfl_down_sync(mask, thread_max, offset);
			  float partner_sum = __shfl_down_sync(mask, thread_sum, offset);
			  
			  float old_max = thread_max;
			  thread_max = fmaxf(thread_max, partner_max);
			  float scale = expf(old_max - thread_max);
			  thread_sum = thread_sum * scale + partner_sum * expf(partner_max - thread_max);
		 }
	 }
 
	 // The final (max, sum) is in lane 0 of warp 0. Broadcast to all threads.
	 if (warp_id == 0 && lane_id == 0) {
		 smem_max[0] = thread_max;
		 smem_sum[0] = thread_sum;
	 }
	 
	 __syncthreads();
	 
	 float block_max = smem_max[0];
	 float block_sum = smem_sum[0];
 
	 // =========================================================================
	 // Pass 2: Apply normalization and write output
	 // =========================================================================
	 for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
		 float val = __half2float(row_start[i]);
		 float final_val = expf(val - block_max) / block_sum;
		 out_row_start[i] = __float2half(final_val);
	 }
}

/**
 * @brief Computes the softmax function for each row of a matrix using float4 vectorization.
 * * This kernel is optimized for memory bandwidth by loading and storing 
 * __nv_bfloat16 data in 128-bit chunks using the float4 vector type.
 * It uses a numerically stable online algorithm for softmax calculation.
 * * The computation proceeds in two main passes over the data, with a block-wide
 * reduction in between.
 * * - Pass 1: Calculates the max value and a scaled sum for the row in a
 * single pass. This is done in a vectorized manner for the bulk of the data,
 * with a scalar loop to handle any remaining elements.
 * - Reduction: A highly efficient block-wide reduction combines the max/sum pairs
 * from each thread. It uses warp-shuffle instructions for intra-warp
 * reduction and shared memory for inter-warp reduction.
 * - Pass 2: Normalizes each element using the final block-wide max and sum,
 * and writes the result back to global memory, again using vectorization.
 * * @param n_rows The number of rows in the input matrix.
 * @param n_cols The number of columns in the input matrix.
 * @param X      Pointer to the input matrix in global memory (bfloat16).
 * @param out    Pointer to the output matrix in global memory (bfloat16).
 */
 extern "C" __global__ void opt_default_softmax_bf16_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * X, __nv_bfloat16 * out) {
    // Each block processes one row of the input matrix.
    uint64_t row_ind = blockIdx.x;
    if (row_ind >= n_rows) {
        return;
    }

    // Calculate pointers to the start of the row for input and output.
    uint64_t row_offset = row_ind * ((uint64_t) n_cols);
    __nv_bfloat16* row_start = X + row_offset;
    __nv_bfloat16* out_row_start = out + row_offset;

    // Use float for intermediate computations for better precision and performance.
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    // The number of columns that can be processed in a vectorized fashion (8 elements per float4).
    int n_cols_vec = n_cols / 8;

    // =========================================================================
    // Pass 1: Online calculation of max and sum.
    // =========================================================================
    
    // --- Vectorized Part ---
    // Each thread processes chunks of 8 elements at a time.
    for (int i = threadIdx.x; i < n_cols_vec; i += blockDim.x) {
        // Load 8 bfloat16 elements using a single float4 load instruction.
        b16x8_f4_union converter;
        converter.f4 = ((float4*)row_start)[i];

        // Process the 8 elements loaded. Unrolling this loop reduces overhead.
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __bfloat162float(converter.bf16[j]);
            
            float old_max = thread_max;
            thread_max = fmaxf(thread_max, val);

            // This online update is numerically stable.
            thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
        }
    }

    // --- Scalar Tail Part ---
    // Process any remaining elements if n_cols is not a multiple of 8.
    int tail_start = n_cols_vec * 8;
    for (int i = tail_start + threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = __bfloat162float(row_start[i]);
        
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);

        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    // =========================================================================
    // Block-wide reduction of (max, sum) pairs. This logic is unchanged.
    // =========================================================================
    
    // Step 1: Warp-level reduction using shuffle instructions.
    unsigned mask = 0xFFFFFFFFU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float partner_max = __shfl_down_sync(mask, thread_max, offset);
        float partner_sum = __shfl_down_sync(mask, thread_sum, offset);

        float old_max = thread_max;
        thread_max = fmaxf(thread_max, partner_max);
        
        thread_sum = thread_sum * expf(old_max - thread_max) + partner_sum * expf(partner_max - thread_max);
    }

    // Step 2: Inter-warp reduction using shared memory.
    __shared__ float smem_max[32]; // Max warps per block = 1024 / 32 = 32
    __shared__ float smem_sum[32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        smem_max[warp_id] = thread_max;
        smem_sum[warp_id] = thread_sum;
    }

    __syncthreads();

    // Step 3: Final reduction by the first warp.
    if (warp_id == 0) {
        if (lane_id * 32 < blockDim.x) {
            thread_max = smem_max[lane_id];
            thread_sum = smem_sum[lane_id];
        } else {
            thread_max = -FLT_MAX;
            thread_sum = 0.0f;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
             float partner_max = __shfl_down_sync(mask, thread_max, offset);
             float partner_sum = __shfl_down_sync(mask, thread_sum, offset);
             
             float old_max = thread_max;
             thread_max = fmaxf(thread_max, partner_max);
             thread_sum = thread_sum * expf(old_max - thread_max) + partner_sum * expf(partner_max - thread_max);
        }
    }

    // Broadcast the final (max, sum) to all threads in the block via shared memory.
    if (warp_id == 0 && lane_id == 0) {
        smem_max[0] = thread_max;
        smem_sum[0] = thread_sum;
    }
    
    __syncthreads();
    
    float block_max = smem_max[0];
    float block_sum = smem_sum[0];

    // =========================================================================
    // Pass 2: Apply normalization and write output.
    // =========================================================================

    // --- Vectorized Part ---
    for (int i = threadIdx.x; i < n_cols_vec; i += blockDim.x) {
        b16x8_f4_union in_converter;
        in_converter.f4 = ((float4*)row_start)[i];

        b16x8_f4_union out_converter;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __bfloat162float(in_converter.bf16[j]);
            float final_val = expf(val - block_max) / block_sum;
            out_converter.bf16[j] = __float2bfloat16(final_val);
        }
        
        // Write 8 bfloat16 elements using a single float4 store instruction.
        ((float4*)out_row_start)[i] = out_converter.f4;
    }

    // --- Scalar Tail Part ---
    for (int i = tail_start + threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = __bfloat162float(row_start[i]);
        float final_val = expf(val - block_max) / block_sum;
        out_row_start[i] = __float2bfloat16(final_val);
    }
}

extern "C" __global__ void default_softmax_bf16_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * X, __nv_bfloat16 * out) {
    // Each block processes one row
    uint64_t row_ind = blockIdx.x;
    if (row_ind >= n_rows) {
        return;
    }

    uint64_t row_offset = row_ind * ((uint64_t) n_cols);
    __nv_bfloat16* row_start = X + row_offset;
    __nv_bfloat16* out_row_start = out + row_offset;

    // Use float for intermediate computations for precision
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    // =========================================================================
    // Pass 1: Online calculation of max and sum for each thread's data chunk
    // =========================================================================
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = __bfloat162float(row_start[i]);
        
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);

		thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    // =========================================================================
    // Block-wide reduction of (max, sum) pairs
    // =========================================================================
    
    // Step 1: Warp-level reduction using shuffle instructions
    unsigned mask = 0xFFFFFFFFU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Get partner thread's (max, sum)
        float partner_max = __shfl_down_sync(mask, thread_max, offset);
        float partner_sum = __shfl_down_sync(mask, thread_sum, offset);

        // Combine the pairs using the same online logic
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, partner_max);
        float scale = expf(old_max - thread_max);
        
        thread_sum = thread_sum * scale + partner_sum * expf(partner_max - thread_max);
    }

    // Step 2: Inter-warp reduction using shared memory
    // Each warp's lane 0 now holds the warp's (max, sum)
    __shared__ float smem_max[32]; // Max warps per block = 1024 / 32 = 32
    __shared__ float smem_sum[32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        smem_max[warp_id] = thread_max;
        smem_sum[warp_id] = thread_sum;
    }

    __syncthreads();

    // Step 3: Final reduction by the first warp
    // Load warp-reduced values into first warp's registers
    if (warp_id == 0) {
        // Only load if there's a valid corresponding warp
        if (lane_id * 32 < blockDim.x) {
            thread_max = smem_max[lane_id];
            thread_sum = smem_sum[lane_id];
        } else {
            thread_max = -FLT_MAX;
            thread_sum = 0.0f;
        }

        // Final reduction within the first warp
        for (int offset = 16; offset > 0; offset >>= 1) {
             float partner_max = __shfl_down_sync(mask, thread_max, offset);
             float partner_sum = __shfl_down_sync(mask, thread_sum, offset);
             
			 float old_max = thread_max;
             thread_max = fmaxf(thread_max, partner_max);
             float scale = expf(old_max - thread_max);
             thread_sum = thread_sum * scale + partner_sum * expf(partner_max - thread_max);
        }
    }

    // The final (max, sum) is in lane 0 of warp 0. Broadcast to all threads.
    if (warp_id == 0 && lane_id == 0) {
        smem_max[0] = thread_max;
        smem_sum[0] = thread_sum;
    }
    
    __syncthreads();
    
    float block_max = smem_max[0];
    float block_sum = smem_sum[0];

    // =========================================================================
    // Pass 2: Apply normalization and write output
    // =========================================================================
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = __bfloat162float(row_start[i]);
        float final_val = expf(val - block_max) / block_sum;
        out_row_start[i] = __float2bfloat16(final_val);
    }
}


extern "C" __global__ void default_softmax_fp8e4m3_fp16_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * X, __half * out) {
	return;
}


extern "C" __global__ void default_softmax_fp8e4m3_bf16_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * X, __nv_bfloat16 * out) {
	return;
}


extern "C" __global__ void default_softmax_fp8e5m2_fp16_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * X, __half * out) {
	return;
}


extern "C" __global__ void default_softmax_fp8e5m2_bf16_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * X, __nv_bfloat16 * out) {
	return;
}
