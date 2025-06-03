#include "nvidia_ops.h"


extern "C" __global__ void default_rms_norm_fp32_kernel(int n_rows, int n_cols, float eps, float * rms_weight, float * X, float * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * row = (float *) sdata;
	float * weights = row + n_cols;

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}
	__syncthreads();

	float cur_row_val;
	float float_val;
	float running_sq_sum;
	uint64_t row_ind_start;

	// can assume model dim is a multiple of 32...
	unsigned warp_mask = 0xFFFFFFFFU;

	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		running_sq_sum = 0;

		// 1.) do a per thread loading an initial reduction on max_smem
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			cur_row_val = X[row_ind_start + i];
			// save for re-scaling
			row[i] = cur_row_val;
			float_val = cur_row_val;
			running_sq_sum += float_val * float_val;
			
		}

		// add this warp's result and place in smem
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[warp_id] = running_sq_sum;
		}

		__syncthreads();


		// 2.) now combine all the reductions from each thread
		
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
					rms_vals[row_id] = reduction_data_sq[0];
				}
			}

		}

		__syncthreads();

		
		// now reduction_data[0] has float32 representing total squared sum
		float recip_avg = reduction_data_sq[0];

		// 3.) now need to store back all of the row values and mutliply with rms_weight
		float rms_val;

		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			// copying casting locations as in llama3
			rms_val =  row[i] * recip_avg;

			out[row_ind_start + i] = rms_val * weights[i];
		}

		// ensure all threads are complete before we start overwriting row in smem
		__syncthreads();
	}
}



// num_stages is defined by amount of smem avail, so needs to be passed in as arg
extern "C" __global__ void default_rms_norm_fp16_kernel(int n_rows, int n_cols, float eps, __half * rms_weight, __half * X, __half * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__half * row = (__half *) sdata;
	__half * weights = (row + n_cols);

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}
	__syncthreads();

	__half cur_row_val;
	float float_val;
	float running_sq_sum;
	uint64_t row_ind_start;

	// can assume model dim is a multiple of 32...
	unsigned warp_mask = 0xFFFFFFFFU;

	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		running_sq_sum = 0;

		// 1.) do a per thread loading an initial reduction on max_smem
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			cur_row_val = X[row_ind_start + i];
			// save for re-scaling
			row[i] = cur_row_val;
			float_val = __half2float(cur_row_val);
			running_sq_sum += float_val * float_val;
			
		}

		// add this warp's result and place in smem
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[warp_id] = running_sq_sum;
		}

		__syncthreads();


		// 2.) now combine all the reductions from each thread
		
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
					rms_vals[row_id] = reduction_data_sq[0];
				}
			}

		}

		__syncthreads();

		
		// now reduction_data[0] has float32 representing total squared sum
		float recip_avg = reduction_data_sq[0];

		// 3.) now need to store back all of the row values and mutliply with rms_weight
		float rms_val;

		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			// copying casting locations as in llama3
			rms_val =  __half2float(row[i]) * recip_avg;

			out[row_ind_start + i] = __float2half(rms_val) * weights[i];
		}

		// ensure all threads are complete before we start overwriting row in smem
		__syncthreads();
	}
}


extern "C" __global__ void default_rms_norm_bf16_kernel(int n_rows, int n_cols, float eps, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X, __nv_bfloat16 * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__nv_bfloat16 * row = (__nv_bfloat16 *) sdata;
	__nv_bfloat16 * weights = (row + n_cols);

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}
	__syncthreads();

	__nv_bfloat16 cur_row_val;
	float float_val;
	float running_sq_sum;
	uint64_t row_ind_start;

	// can assume model dim is a multiple of 32...
	unsigned warp_mask = 0xFFFFFFFFU;

	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		running_sq_sum = 0;

		// 1.) do a per thread loading an initial reduction on max_smem
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			cur_row_val = X[row_ind_start + i];
			// save for re-scaling
			row[i] = cur_row_val;
			float_val = __bfloat162float(cur_row_val);
			running_sq_sum += float_val * float_val;
			
		}

		// add this warp's result and place in smem
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[warp_id] = running_sq_sum;
		}

		__syncthreads();


		// 2.) now combine all the reductions from each thread
		
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
					rms_vals[row_id] = reduction_data_sq[0];
				}
			}

		}

		__syncthreads();

		
		// now reduction_data[0] has float32 representing total squared sum
		float recip_avg = reduction_data_sq[0];

		// 3.) now need to store back all of the row values and mutliply with rms_weight
		float rms_val;

		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			// copying casting locations as in llama3
			rms_val =  __bfloat162float(row[i]) * recip_avg;

			out[row_ind_start + i] = __float2bfloat16(rms_val) * weights[i];
		}

		// ensure all threads are complete before we start overwriting row in smem
		__syncthreads();
	}
}


extern "C" __global__ void default_rms_norm_fp8e4m3_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X, __nv_fp8_e4m3 * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__nv_fp8_e4m3 * row = (__nv_fp8_e4m3 *) sdata;
	float * weights = (float *) (row + n_cols);

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = float(rms_weight[i]);
	}
	__syncthreads();

	__nv_fp8_e4m3 cur_row_val;
	float float_val;
	float running_sq_sum;
	uint64_t row_ind_start;

	// can assume model dim is a multiple of 32...
	unsigned warp_mask = 0xFFFFFFFFU;

	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		running_sq_sum = 0;
		// 1.) do a per thread loading an initial reduction on max_smem
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			cur_row_val = X[row_ind_start + i];
			// save for re-scaling
			row[i] = cur_row_val;
			float_val = float(cur_row_val);
			running_sq_sum += float_val * float_val;
			
		}

		// add this warp's result and place in smem
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[warp_id] = running_sq_sum;
		}

		__syncthreads();


		// 2.) now combine all the reductions from each thread
		
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
					rms_vals[row_id] = reduction_data_sq[0];
				}
			}

		}

		__syncthreads();

		
		// now reduction_data[0] has float32 representing total squared sum
		float recip_avg = reduction_data_sq[0];

		// 3.) now need to store back all of the row values and mutliply with rms_weight
		float rms_val;

		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			// copying casting locations as in llama3
			rms_val =  float(row[i]) * recip_avg;

			out[row_ind_start + i] = __nv_fp8_e4m3(rms_val * weights[i]);
		}

		// ensure all threads are complete before we start overwriting row in smem
		__syncthreads();
	}
}


extern "C" __global__ void default_rms_norm_fp8e5m2_kernel(int n_rows, int n_cols, float eps, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X, __nv_fp8_e5m2 * out, float * rms_vals) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__nv_fp8_e5m2 * row = (__nv_fp8_e5m2 *) sdata;
	float * weights = (float *) (row + n_cols);

	// every warp will have a reduced value
	__shared__ float reduction_data_sq[32];

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

	int rows_per_block = n_rows / gridDim.x;
	
	int rows_remain = n_rows % gridDim.x;
	int row_offset;
	if (blockIdx.x < rows_remain){
		// this block will need to do an extra row
		rows_per_block += 1;
		// all prior blocks also had an extra row
		row_offset = row_base * rows_per_block;
	}
	else{
		row_offset = row_base * rows_per_block + rows_remain;
	}

	int thread_id = threadIdx.x;

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = float(rms_weight[i]);
	}
	__syncthreads();

	__nv_fp8_e5m2 cur_row_val;
	float float_val;
	float running_sq_sum;
	uint64_t row_ind_start;

	// can assume model dim is a multiple of 32...
	unsigned warp_mask = 0xFFFFFFFFU;

	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		running_sq_sum = 0;

		// 1.) do a per thread loading an initial reduction on max_smem
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			cur_row_val = X[row_ind_start + i];
			// save for re-scaling
			row[i] = cur_row_val;
			float_val = float(cur_row_val);
			running_sq_sum += float_val * float_val;
			
		}

		// add this warp's result and place in smem
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			running_sq_sum += __shfl_down_sync(warp_mask, running_sq_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_sq[warp_id] = running_sq_sum;
		}

		__syncthreads();


		// 2.) now combine all the reductions from each thread
		
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
					rms_vals[row_id] = reduction_data_sq[0];
				}
			}

		}

		__syncthreads();

		
		// now reduction_data[0] has float32 representing total squared sum
		float recip_avg = reduction_data_sq[0];

		// 3.) now need to store back all of the row values and mutliply with rms_weight
		float rms_val;

		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			// copying casting locations as in llama3
			rms_val =  float(row[i]) * recip_avg;

			out[row_ind_start + i] = __nv_fp8_e5m2(rms_val * weights[i]);
		}

		// ensure all threads are complete before we start overwriting row in smem
		__syncthreads();
	}
}