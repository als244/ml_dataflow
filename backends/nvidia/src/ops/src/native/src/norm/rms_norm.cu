#include "nvidia_ops.h"


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


extern "C" __global__ void default_rms_norm_bf16_kernel(int n_rows, int n_cols, float eps, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X, __nv_bfloat16 * out, float * rms_vals) {

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

	__nv_bfloat16 weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = __ldg(rms_weight + i);

		out[row_base + i] = __nv_fp8_e4m3(rms_val) * weight_val;
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

	__nv_bfloat16 weight_val;

	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		// copying casting locations as in llama3
		rms_val =  row[i] * recip_avg;

		weight_val = __ldg(rms_weight + i);

		out[row_base + i] = __nv_fp8_e5m2(rms_val) * weight_val;
	}
}