#include "nvidia_ops.h"

// Define vector size using a macro. float4 holds 8 bf16 values.
#define RMS_NORM_VEC_SIZE (sizeof(float4) / sizeof(__nv_bfloat16))

extern "C" __global__ void default_rms_norm_fp32_kernel(int n_rows, int n_cols, float eps, float * rms_weight, float * X, float * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * row = (float *) sdata;

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	unsigned warp_mask = 0xFFFFFFFFU;

	float running_sq_sum = 0;

	if (thread_id < 32){
		reduction_data_sq[0] = 0;
	}

	__syncthreads();

	float float_val;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		float_val = X[row_base + i];
		row[i] = float_val;
		running_sq_sum += float_val * float_val;
	}

	for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
		running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
	}

	if (lane_id == 0){
		reduction_data_sq[warp_id] = running_sq_sum;
	}

	__syncthreads();

	if (warp_id == 0){
		running_sq_sum = reduction_data_sq[lane_id];

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[0] = rsqrtf((running_sq_sum / (float) n_cols) + eps);

			// Save down the squared sums of this row
			// so we can easilly compute the backpass...

			// During inference this should be null and not needed
			if (rms_vals){
				rms_vals[row_ind] = reduction_data_sq[0];
			}
		}
	}

	__syncthreads();

	float recip_avg = reduction_data_sq[0];

	float rms_val;

	float weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = __ldg(rms_weight + i);

		out[row_base + i] = rms_val * weight_val;
	}
}



extern "C" __global__ void default_rms_norm_fp16_kernel(int n_rows, int n_cols, float eps, __half * rms_weight, __half * X, __half * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * row = (float *) sdata;

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	unsigned warp_mask = 0xFFFFFFFFU;

	float running_sq_sum = 0;

	if (thread_id < 32){
		reduction_data_sq[0] = 0;
	}

	__syncthreads();

	float float_val;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		float_val = __half2float(X[row_base + i]);
		row[i] = float_val;
		running_sq_sum += float_val * float_val;
	}

	for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
		running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
	}

	if (lane_id == 0){
		reduction_data_sq[warp_id] = running_sq_sum;
	}

	__syncthreads();

	if (warp_id == 0){
		running_sq_sum = reduction_data_sq[lane_id];

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[0] = rsqrtf((running_sq_sum / (float) n_cols) + eps);

			// Save down the squared sums of this row
			// so we can easilly compute the backpass...

			// During inference this should be null and not needed
			if (rms_vals){
				rms_vals[row_ind] = reduction_data_sq[0];
			}
		}
	}

	__syncthreads();

	float recip_avg = reduction_data_sq[0];

	float rms_val;

	__half weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = __ldg(rms_weight + i);

		out[row_base + i] = __float2half(rms_val) * weight_val;
	}
}

/**
 * @brief Optimized RMS Normalization kernel for bfloat16 data types.
 *
 * This kernel calculates the RMS norm for each row of an input matrix 'X',
 * normalizes the row, and applies a learned weight. It is optimized for
 * performance by using vectorized memory operations (uint4) to load and
 * store 8 bfloat16 elements at a time, maximizing memory bandwidth.
 *
 * Each CUDA block processes a single row of the input matrix.
 *
 * Assumptions:
 * - The number of columns, 'n_cols', must be a multiple of 8.
 * - Input pointers 'rms_weight', 'X', and 'out' are guaranteed by the
 * caller to be aligned to at least 16 bytes (float4/uint4 alignment).
 * The user specified 256-byte alignment, which is more than sufficient.
 * - The kernel is launched with enough shared memory to hold one row of
 * data as float32s: `n_cols * sizeof(float)`.
 *
 * The kernel performs the following steps:
 * 1. Vectorized Load & Sum of Squares:
 * - Threads cooperatively load a row from global memory into shared memory.
 * - Loads are done in 128-bit (8x bfloat16) chunks using uint4.
 * - bfloat16 values are converted to float32 for computation.
 * - Each thread calculates a partial sum of squares for the elements it handles.
 * - The float32 representation of the row is stored in shared memory.
 *
 * 2. Parallel Reduction:
 * - A highly efficient two-stage reduction is performed to find the total
 * sum of squares for the row.
 * - Stage 1: An intra-warp reduction using __shfl_down_sync.
 * - Stage 2: An inter-warp reduction using shared memory, performed by the
 * first warp.
 *
 * 3. Normalization and Scaling:
 * - The final thread (thread 0) calculates the inverse square root of the
 * mean of squares (`rsqrtf`). This value is broadcast to all threads
 * via shared memory.
 * - Threads read the float32 values from shared memory, apply the normalization,
 * and scale by the corresponding bfloat16 weights.
 *
 * 4. Vectorized Store:
 * - The final results are converted back to bfloat16, packed into uint4
 * vectors, and written to global memory in 128-bit chunks.
 */
extern "C" __global__ void default_rms_norm_bf16_kernel(
    int n_rows,
    int n_cols,
    float eps,
    const __nv_bfloat16* __restrict__ rms_weight,
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ rms_vals) {
	
	/*
     * Dynamically allocated shared memory. The size must be at least
     * n_cols * sizeof(float).
     */
	 extern __shared__ uint8_t sdata[];
	 float* row_smem = (float*)sdata;
 
	 /*
	  * Shared memory for the parallel reduction of sum of squares.
	  * One float per warp. Max 32 warps (1024 threads) supported.
	  */
	 __shared__ float reduction_data_sq[32];
 
	 const int row_ind = blockIdx.x;
	 if (row_ind >= n_rows) {
		 return;
	 }
 
	 const int thread_id = threadIdx.x;
	 const int warp_id = thread_id / 32;
	 const int lane_id = thread_id % 32;
	 const unsigned warp_mask = 0xFFFFFFFFU;
 
	 /* Base pointers for the current row, cast for vectorized access */
	 const uint64_t row_offset = (uint64_t)row_ind * (uint64_t)n_cols;
	 const uint4* x_vec = (const uint4*)(X + row_offset);
	 uint4* out_vec = (uint4*)(out + row_offset);
	 const uint4* w_vec = (const uint4*)rms_weight;
 
	 /*
	  * Step 1: Vectorized load, convert to float, compute sum of squares,
	  * and store float values to shared memory.
	  * Each thread processes one uint4 (8 bfloat16s) per iteration.
	  */
	 float running_sq_sum = 0.0f;
	 const int n_cols_div_8 = n_cols / 8;
 
	 for (int i = thread_id; i < n_cols_div_8; i += blockDim.x) {
		 /* Load 8 bfloat16s as one 128-bit uint4 vector */
		 const uint4 packed_x = __ldg(&x_vec[i]);
 
		 /* Unpack the 8 bfloat16s into 8 floats using bfloat162 intrinsics */
		 const float2 f2_0 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.x));
		 const float2 f2_1 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.y));
		 const float2 f2_2 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.z));
		 const float2 f2_3 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.w));
 
		 /* Store the 8 float values into shared memory */
		 /* Casting the shared memory pointer to float4 for a vectorized store */
		 ((float4*)row_smem)[i * 2]     = make_float4(f2_0.x, f2_0.y, f2_1.x, f2_1.y);
		 ((float4*)row_smem)[i * 2 + 1] = make_float4(f2_2.x, f2_2.y, f2_3.x, f2_3.y);
 
		 /* Accumulate sum of squares */
		 running_sq_sum += f2_0.x * f2_0.x + f2_0.y * f2_0.y;
		 running_sq_sum += f2_1.x * f2_1.x + f2_1.y * f2_1.y;
		 running_sq_sum += f2_2.x * f2_2.x + f2_2.y * f2_2.y;
		 running_sq_sum += f2_3.x * f2_3.x + f2_3.y * f2_3.y;
	 }
 
	 /*
	  * Step 2: Parallel reduction to get the total sum of squares for the row.
	  */
	 /* Intra-warp reduction */
	 for (int offset = 16; offset > 0; offset >>= 1) {
		 running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, offset);
	 }
 
	 if (lane_id == 0) {
		 reduction_data_sq[warp_id] = running_sq_sum;
	 }
 
	 __syncthreads();
 
	 /* Inter-warp reduction (performed by the first warp) */
	 if (warp_id == 0) {
		 /* Ensure the reduction array is initialized before reading from it */
		 running_sq_sum = (lane_id < blockDim.x / 32) ? reduction_data_sq[lane_id] : 0.0f;
 
		 for (int offset = 16; offset > 0; offset >>= 1) {
			 running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, offset);
		 }
 
		 /*
		  * Final calculation and broadcast. Thread 0 computes the reciprocal
		  * of the RMS value and writes it to a known location in shared memory
		  * for all other threads to read.
		  */
		 if (lane_id == 0) {
			 const float inv_n_cols = 1.0f / (float)n_cols;
			 const float rsqrt_val = rsqrtf(running_sq_sum * inv_n_cols + eps);
			 reduction_data_sq[0] = rsqrt_val;
 
			 /* Optionally save the rsqrt value for the backward pass */
			 if (rms_vals) {
				 rms_vals[row_ind] = rsqrt_val;
			 }
		 }
	 }
 
	 __syncthreads();
 
	 /*
	  * Step 3 & 4: Apply normalization, scale by weights, and vectorized store.
	  */
	 const float recip_avg = reduction_data_sq[0];
 
	 for (int i = thread_id; i < n_cols_div_8; i += blockDim.x) {
		 /* Load 8 float values from shared memory */
		 const float4 f4_0 = ((float4*)row_smem)[i * 2];
		 const float4 f4_1 = ((float4*)row_smem)[i * 2 + 1];
 
		 /* Load 8 bfloat16 weights and convert to float */
		 const uint4 packed_w = __ldg(&w_vec[i]);
		 const float2 w2_0 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.x));
		 const float2 w2_1 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.y));
		 const float2 w2_2 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.z));
		 const float2 w2_3 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.w));
 
		 /* Apply normalization and element-wise weight scaling */
		 const float f_out[8] = {
			 f4_0.x * recip_avg * w2_0.x,
			 f4_0.y * recip_avg * w2_0.y,
			 f4_0.z * recip_avg * w2_1.x,
			 f4_0.w * recip_avg * w2_1.y,
			 f4_1.x * recip_avg * w2_2.x,
			 f4_1.y * recip_avg * w2_2.y,
			 f4_1.z * recip_avg * w2_3.x,
			 f4_1.w * recip_avg * w2_3.y,
		 };
 
		 /* Pack 8 floats back into 8 bfloat16s in a uint4 vector */
		 uint4 packed_out;
		 const __nv_bfloat162 bf162_0 = __float22bfloat162_rn(make_float2(f_out[0], f_out[1]));
		 const __nv_bfloat162 bf162_1 = __float22bfloat162_rn(make_float2(f_out[2], f_out[3]));
		 const __nv_bfloat162 bf162_2 = __float22bfloat162_rn(make_float2(f_out[4], f_out[5]));
		 const __nv_bfloat162 bf162_3 = __float22bfloat162_rn(make_float2(f_out[6], f_out[7]));
 
		 packed_out.x = *((const unsigned int*)&bf162_0);
		 packed_out.y = *((const unsigned int*)&bf162_1);
		 packed_out.z = *((const unsigned int*)&bf162_2);
		 packed_out.w = *((const unsigned int*)&bf162_3);
 
		 /* Store the 128-bit vector to global memory */
		 out_vec[i] = packed_out;
	 }
}

extern "C" __global__ void naive_default_rms_norm_bf16_kernel(int n_rows, int n_cols, float eps, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X, __nv_bfloat16 * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * row = (float *) sdata;

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	unsigned warp_mask = 0xFFFFFFFFU;

	float running_sq_sum = 0;

	if (thread_id < 32){
		reduction_data_sq[0] = 0;
	}

	__syncthreads();

	float float_val;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		float_val = __bfloat162float(X[row_base + i]);
		row[i] = float_val;
		running_sq_sum += float_val * float_val;
	}

	for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
		running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
	}

	if (lane_id == 0){
		reduction_data_sq[warp_id] = running_sq_sum;
	}

	__syncthreads();

	if (warp_id == 0){
		running_sq_sum = reduction_data_sq[lane_id];

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[0] = rsqrtf((running_sq_sum / (float) n_cols) + eps);

			// Save down the squared sums of this row
			// so we can easilly compute the backpass...

			// During inference this should be null and not needed
			if (rms_vals){
				rms_vals[row_ind] = reduction_data_sq[0];
			}
		}
	}

	__syncthreads();

	float recip_avg = reduction_data_sq[0];

	float rms_val;

	__nv_bfloat16 weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = __ldg(rms_weight + i);

		out[row_base + i] = __float2bfloat16(rms_val) * weight_val;
	}
}


extern "C" __global__ void default_rms_norm_fp8e4m3_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X, __nv_fp8_e4m3 * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * row = (float *) sdata;

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	unsigned warp_mask = 0xFFFFFFFFU;

	float running_sq_sum = 0;

	if (thread_id < 32){
		reduction_data_sq[0] = 0;
	}

	__syncthreads();

	float float_val;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		float_val = float(X[row_base + i]);
		row[i] = float_val;
		running_sq_sum += float_val * float_val;
	}

	for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
		running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
	}

	if (lane_id == 0){
		reduction_data_sq[warp_id] = running_sq_sum;
	}

	__syncthreads();

	if (warp_id == 0){
		running_sq_sum = reduction_data_sq[lane_id];

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[0] = rsqrtf((running_sq_sum / (float) n_cols) + eps);

			// Save down the squared sums of this row
			// so we can easilly compute the backpass...

			// During inference this should be null and not needed
			if (rms_vals){
				rms_vals[row_ind] = reduction_data_sq[0];
			}
		}
	}

	__syncthreads();

	float recip_avg = reduction_data_sq[0];

	float rms_val;

	__nv_fp8_e4m3 weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = rms_weight[i];

		out[row_base + i] = __nv_fp8_e4m3(rms_val * float(weight_val));
	}
}


extern "C" __global__ void default_rms_norm_fp8e5m2_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X, __nv_fp8_e5m2 * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * row = (float *) sdata;

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	unsigned warp_mask = 0xFFFFFFFFU;

	float running_sq_sum = 0;

	if (thread_id < 32){
		reduction_data_sq[0] = 0;
	}

	__syncthreads();

	float float_val;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		float_val = float(X[row_base + i]);
		row[i] = float_val;
		running_sq_sum += float_val * float_val;
	}

	for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
		running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
	}

	if (lane_id == 0){
		reduction_data_sq[warp_id] = running_sq_sum;
	}

	__syncthreads();

	if (warp_id == 0){
		running_sq_sum = reduction_data_sq[lane_id];

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[0] = rsqrtf((running_sq_sum / (float) n_cols) + eps);

			// Save down the squared sums of this row
			// so we can easilly compute the backpass...

			// During inference this should be null and not needed
			if (rms_vals){
				rms_vals[row_ind] = reduction_data_sq[0];
			}
		}
	}

	__syncthreads();

	float recip_avg = reduction_data_sq[0];

	float rms_val;

	__nv_fp8_e5m2 weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = rms_weight[i];

		out[row_base + i] = __nv_fp8_e5m2(rms_val * float(weight_val));
	}
}