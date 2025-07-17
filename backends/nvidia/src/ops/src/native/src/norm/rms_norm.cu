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
	
		extern __shared__ uint8_t sdata[];
		float* row_smem = (float*)sdata;
	
		__shared__ float reduction_data_sq[32];
	
		const int row_ind = blockIdx.x;
		if (row_ind >= n_rows) {
			return;
		}
	
		const int thread_id = threadIdx.x;
		const int warp_id = thread_id / 32;
		const int lane_id = thread_id % 32;
		const unsigned warp_mask = 0xFFFFFFFFU;
	
		const uint64_t row_offset = (uint64_t)row_ind * (uint64_t)n_cols;
		const uint4* x_vec = (const uint4*)(X + row_offset);
		uint4* out_vec = (uint4*)(out + row_offset);
		const uint4* w_vec = (const uint4*)rms_weight;
	
		float running_sq_sum = 0.0f;
		const int n_cols_div_8 = n_cols / 8;
	
		// Step 1: Vectorized load, compute sum of squares, and store to shared memory (transposed)
		for (int i = thread_id; i < n_cols_div_8; i += blockDim.x) {
			const uint4 packed_x = __ldg(&x_vec[i]);
			
			const float2 f2_0 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.x));
			const float2 f2_1 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.y));
			const float2 f2_2 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.z));
			const float2 f2_3 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_x.w));
	
			const float f_in[8] = {
				f2_0.x, f2_0.y, f2_1.x, f2_1.y,
				f2_2.x, f2_2.y, f2_3.x, f2_3.y
			};
			
			// Write to shared memory with a transposed layout to ensure coalesced access
			#pragma unroll
			for (int k = 0; k < 8; ++k) {
				row_smem[k * n_cols_div_8 + i] = f_in[k];
			}
	
			running_sq_sum += f_in[0] * f_in[0] + f_in[1] * f_in[1];
			running_sq_sum += f_in[2] * f_in[2] + f_in[3] * f_in[3];
			running_sq_sum += f_in[4] * f_in[4] + f_in[5] * f_in[5];
			running_sq_sum += f_in[6] * f_in[6] + f_in[7] * f_in[7];
		}
	
		// Step 2: Parallel reduction (unchanged)
		for (int offset = 16; offset > 0; offset >>= 1) {
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, offset);
		}
		if (lane_id == 0) {
			reduction_data_sq[warp_id] = running_sq_sum;
		}
		__syncthreads();
		if (warp_id == 0) {
			running_sq_sum = (lane_id < blockDim.x / 32) ? reduction_data_sq[lane_id] : 0.0f;
			for (int offset = 16; offset > 0; offset >>= 1) {
				running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, offset);
			}
			if (lane_id == 0) {
				const float inv_n_cols = 1.0f / (float)n_cols;
				const float rsqrt_val = rsqrtf(running_sq_sum * inv_n_cols + eps);
				reduction_data_sq[0] = rsqrt_val;
				if (rms_vals) {
					rms_vals[row_ind] = rsqrt_val;
				}
			}
		}
		__syncthreads();
	
		// Step 3 & 4: Load from shared, normalize, scale, and vectorized store
		const float recip_avg = reduction_data_sq[0];
	
		for (int i = thread_id; i < n_cols_div_8; i += blockDim.x) {
			// Load from shared memory using the same transposed layout
			float f_smem[8];
			#pragma unroll
			for (int k = 0; k < 8; ++k) {
				f_smem[k] = row_smem[k * n_cols_div_8 + i];
			}
	
			const uint4 packed_w = __ldg(&w_vec[i]);
			const float2 w2_0 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.x));
			const float2 w2_1 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.y));
			const float2 w2_2 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.z));
			const float2 w2_3 = __bfloat1622float2(*((const __nv_bfloat162*)&packed_w.w));
			
			const float f_out[8] = {
				f_smem[0] * recip_avg * w2_0.x, f_smem[1] * recip_avg * w2_0.y,
				f_smem[2] * recip_avg * w2_1.x, f_smem[3] * recip_avg * w2_1.y,
				f_smem[4] * recip_avg * w2_2.x, f_smem[5] * recip_avg * w2_2.y,
				f_smem[6] * recip_avg * w2_3.x, f_smem[7] * recip_avg * w2_3.y,
			};
	
			uint4 packed_out;
			*((__nv_bfloat162*)&packed_out.x) = __float22bfloat162_rn(make_float2(f_out[0], f_out[1]));
			*((__nv_bfloat162*)&packed_out.y) = __float22bfloat162_rn(make_float2(f_out[2], f_out[3]));
			*((__nv_bfloat162*)&packed_out.z) = __float22bfloat162_rn(make_float2(f_out[4], f_out[5]));
			*((__nv_bfloat162*)&packed_out.w) = __float22bfloat162_rn(make_float2(f_out[6], f_out[7]));
	
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