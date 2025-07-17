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

extern "C" __global__ void default_rms_norm_bf16_kernel(
    int n_rows, int n_cols, float eps,
    const __nv_bfloat16 * __restrict__ rms_weight,
    const __nv_bfloat16 * __restrict__ X,
    __nv_bfloat16 * __restrict__ out,
    float * __restrict__ rms_vals)
{
    // --- Kernel Setup ---
    extern __shared__ uint8_t sdata[];

    // Shared memory for caching the row as float32 and for the reduction.
    float* row_data = (float*)sdata;
    size_t row_bytes = n_cols * sizeof(float);
    float* reduction_data_sq = (float*)(sdata + ((row_bytes + 3) & ~3));

    const int row_ind = blockIdx.x;
    if (row_ind >= n_rows) {
        return;
    }

    const uint64_t row_base = (uint64_t)row_ind * (uint64_t)n_cols;
    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const unsigned warp_mask = 0xFFFFFFFFU;

    const int vec_n_cols = n_cols / RMS_NORM_VEC_SIZE;
    const int remainder_start = vec_n_cols * RMS_NORM_VEC_SIZE;

    // --- 1. Load data, cache in shared memory, and calculate sum of squares ---
    float running_sq_sum = 0.0f;

    // Vectorized part
    for (int i = thread_id; i < vec_n_cols; i += blockDim.x) {
        // Load 8 bf16s from global memory
        float4 x_vec = __ldg(((const float4*)(X + row_base)) + i);
        __nv_bfloat162* x_b162_ptr = (__nv_bfloat162*)(&x_vec);

        // Pointer to float2 in shared memory for storing two floats at a time
        float2* row_f2_ptr = (float2*)(&row_data[i * RMS_NORM_VEC_SIZE]);

        #pragma unroll
        for (int k = 0; k < RMS_NORM_VEC_SIZE / 2; k++) {
            float2 f_vals = __bfloat1622float2(x_b162_ptr[k]);
            row_f2_ptr[k] = f_vals; // Cache as float32
            running_sq_sum += f_vals.x * f_vals.x + f_vals.y * f_vals.y;
        }
    }

    // Scalar part for the tail
    if (remainder_start < n_cols) {
        for (int i = remainder_start + thread_id; i < n_cols; i += blockDim.x) {
            float float_val = __bfloat162float(X[row_base + i]);
            row_data[i] = float_val;
            running_sq_sum += float_val * float_val;
        }
    }
    __syncthreads();

    // --- 2. Block-wide reduction to get the final sum of squares ---
    for (int offset = 16; offset > 0; offset >>= 1) running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, offset);
    if (lane_id == 0) reduction_data_sq[warp_id] = running_sq_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        running_sq_sum = (lane_id < (blockDim.x / 32)) ? reduction_data_sq[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, offset);
        
        if (lane_id == 0) {
            float recip_avg = rsqrtf((running_sq_sum / (float)n_cols) + eps);
            reduction_data_sq[0] = recip_avg; // Broadcast rsqrt value
            if (rms_vals) {
                rms_vals[row_ind] = recip_avg;
            }
        }
    }
    __syncthreads();

    // --- 3. Apply normalization and write to output ---
    const float recip_avg = reduction_data_sq[0];

    // Vectorized part
    for (int i = thread_id; i < vec_n_cols; i += blockDim.x) {
        // Load 8 bf16 weights and 8 float data points
        float4 w_vec = __ldg(((const float4*)rms_weight) + i);
        float4 row_vec = ((const float4*)row_data)[i];

        // Unpack vectors to perform calculations
        __nv_bfloat162* w_b162_ptr = (__nv_bfloat162*)(&w_vec);
        float2* row_f2_ptr = (float2*)(&row_vec);
        
        float4 out_vec;
        __nv_bfloat162* out_b162_ptr = (__nv_bfloat162*)(&out_vec);

        #pragma unroll
        for (int k = 0; k < RMS_NORM_VEC_SIZE / 2; k++) {
            float2 rms_vals_f2;
            rms_vals_f2.x = row_f2_ptr[k].x * recip_avg;
            rms_vals_f2.y = row_f2_ptr[k].y * recip_avg;

            // Cast to bf16 before multiplying by weights to match original kernel's precision
            __nv_bfloat162 rms_vals_b162 = __float22bfloat162_rn(rms_vals_f2);
            
            // Perform two bf16 multiplications at once and store in the output vector
            out_b162_ptr[k] = __hmul2(rms_vals_b162, w_b162_ptr[k]);
        }
        // Store 8 bf16 results to global memory
        ((float4*)(out + row_base))[i] = out_vec;
    }

    // Scalar part for the tail
    if (remainder_start < n_cols) {
        for (int i = remainder_start + thread_id; i < n_cols; i += blockDim.x) {
            float rms_val = row_data[i] * recip_avg;
            __nv_bfloat16 weight_val = __ldg(rms_weight + i);
            out[row_base + i] = __float2bfloat16(rms_val) * weight_val;
        }
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