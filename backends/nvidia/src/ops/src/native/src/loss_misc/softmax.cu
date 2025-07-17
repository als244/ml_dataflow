#include "nvidia_ops.h"

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
