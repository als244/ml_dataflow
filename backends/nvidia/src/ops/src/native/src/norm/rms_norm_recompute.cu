#include "nvidia_ops.h"


extern "C" __global__ void default_rms_norm_recompute_fp32_kernel(int n_rows, int n_cols, float * rms_weight, float * rms_vals, float * X, float * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	float * weights = (float *) sdata;

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
    int num_warps = blockDim.x / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}

	__syncthreads();

	float cur_row_val;
	uint64_t row_ind_start;

    float cur_recip_avg;
    float rms_val;

	for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){

        // 1.) load in new rms val for this warp...

        cur_recip_avg = rms_vals[row_id];
		
        row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		
        // 2. do streaming update based on prior cur_recip_avg
		for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
			cur_row_val = X[row_ind_start + i];
			rms_val = cur_row_val * cur_recip_avg;
            out[row_ind_start + i] = rms_val * weights[i];
		}
	}
}



// num_stages is defined by amount of smem avail, so needs to be passed in as arg
extern "C" __global__ void default_rms_norm_recompute_fp16_kernel(int n_rows, int n_cols, __half * rms_weight, float * rms_vals, __half * X, __half * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__half * weights = (__half *) sdata;

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
    int num_warps = blockDim.x / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}

	__syncthreads();

	float cur_row_val;
	uint64_t row_ind_start;

    float cur_recip_avg;
    float rms_val;

	for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){

        // 1.) load in new rms val for this warp...

        cur_recip_avg = rms_vals[row_id];
		
        row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		
        // 2. do streaming update based on prior cur_recip_avg
		for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
			cur_row_val = __half2float(X[row_ind_start + i]);
			rms_val = cur_row_val * cur_recip_avg;
            out[row_ind_start + i] = __float2half(rms_val) * weights[i];
		}
	}
}


extern "C" __global__ void default_rms_norm_recompute_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * rms_weight, float * rms_vals, __nv_bfloat16 * X, __nv_bfloat16 * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__nv_bfloat16 * weights = (__nv_bfloat16 *) sdata;

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
    int num_warps = blockDim.x / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}

	__syncthreads();

	float cur_row_val;
	uint64_t row_ind_start;

    float cur_recip_avg;
    float rms_val;

	for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){

        // 1.) load in new rms val for this warp...

        cur_recip_avg = rms_vals[row_id];
		
        row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		
        // 2. do streaming update based on prior cur_recip_avg
		for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
			cur_row_val = __bfloat162float(X[row_ind_start + i]);
			rms_val = cur_row_val * cur_recip_avg;
            out[row_ind_start + i] = __float2bfloat16(rms_val) * weights[i];
		}
	}
}


extern "C" __global__ void default_rms_norm_recompute_fp8e4m3_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * rms_weight, float * rms_vals, __nv_fp8_e4m3 * X, __nv_fp8_e4m3 * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__nv_fp8_e4m3 * weights = (__nv_fp8_e4m3 *) sdata;

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
    int num_warps = blockDim.x / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}

	__syncthreads();

	float cur_row_val;
	uint64_t row_ind_start;

    float cur_recip_avg;
    float rms_val;

	for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){

        // 1.) load in new rms val for this warp...

        cur_recip_avg = rms_vals[row_id];
		
        row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		
        // 2. do streaming update based on prior cur_recip_avg
		for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
			cur_row_val = float(X[row_ind_start + i]);
			rms_val = cur_row_val * cur_recip_avg;
            out[row_ind_start + i] = __nv_fp8_e4m3(rms_val * float(weights[i]));
		}
	}
}


extern "C" __global__ void default_rms_norm_recompute_fp8e5m2_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * rms_weight,  float * rms_vals, __nv_fp8_e5m2 * X, __nv_fp8_e5m2 * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	__nv_fp8_e5m2 * weights = (__nv_fp8_e5m2 *) sdata;

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
    int num_warps = blockDim.x / 32;
	int lane_id = thread_id % 32;

	// Load weights which are shared between all rows (when doing output in item 3...)
	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weights[i] = rms_weight[i];
	}

	__syncthreads();

	float cur_row_val;
	uint64_t row_ind_start;

    float cur_recip_avg;
    float rms_val;

	for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){

        // 1.) load in new rms val for this warp...

        cur_recip_avg = rms_vals[row_id];
		
        row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		
        // 2. do streaming update based on prior cur_recip_avg
		for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
			cur_row_val = float(X[row_ind_start + i]);
			rms_val = cur_row_val * cur_recip_avg;
            out[row_ind_start + i] = __nv_fp8_e5m2(rms_val * float(weights[i]));
		}
	}
}