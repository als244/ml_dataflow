#include "nvidia_ops.h"


extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, float * X_inp, float * upstream_dX, float * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	float * shared_fwd_weighted_sums = (float *) (sdata); 
	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block);  



	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

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


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = X_inp[row_ind_start + i];
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * deriv;

		}
	}
}


// Here dX is (N, model_dim) and contains the backprop loss flow that we will update in-place
// This needs to be called after the bwd_weight because the weight we use the updstream dL/dX and this function will
// modify the same pointer...
extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp16_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __half * X_inp, __half * upstream_dX, __half * dX){
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	float * shared_fwd_weighted_sums = (float *) (sdata); 

	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

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


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __half2float(X_inp[row_ind_start + i]);
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * __float2half(deriv);

		}
	}
}


extern "C" __global__ void default_rms_norm_noscale_bwd_x_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);


	float * shared_fwd_weighted_sums = (float *) (sdata); 
	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block); 

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

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


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __bfloat162float(X_inp[row_ind_start + i]);
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * __float2bfloat16(deriv);

		}
	}
}


extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e4m3_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __half * upstream_dX, __half * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);


	float * shared_fwd_weighted_sums = (float *) (sdata); 
	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block);  

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}


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


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * __float2half(deriv);

		}
	}
}

extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);


	float * shared_fwd_weighted_sums = (float *) (sdata); 
	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block);   

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}


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


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * __float2bfloat16(deriv);

		}
	}
}


extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __half * upstream_dX, __half * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);


	float * shared_fwd_weighted_sums = (float *) (sdata); 
	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block);  

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

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

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * __float2half(deriv);

		}
	}
}

extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_weighted_sums, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	float * shared_fwd_weighted_sums = (float *) (sdata); 
	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * shared_fwd_rms_vals = (float *) (shared_fwd_weighted_sums + max_rows_per_block);

	int row_base = blockIdx.x;

	if (row_base >= n_rows){
		return;
	}

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

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_fwd_weighted_sums[i - row_offset] = fwd_weighted_sums[i];
		shared_fwd_rms_vals[i - row_offset] = fwd_rms_vals[i];

	}

	__syncthreads();

	float deriv;
	float cur_weighted_sum;
	float cur_rms_val;
	float cur_rms_val_cub;

	float inp_val;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;
		
		cur_weighted_sum = shared_fwd_weighted_sums[row_id - row_offset];
		cur_rms_val = shared_fwd_rms_vals[row_id - row_offset];
		cur_rms_val_cub = cur_rms_val * cur_rms_val * cur_rms_val;
		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(X_inp[row_ind_start + i]);
			deriv = (cur_rms_val) - (((inp_val * cur_rms_val_cub) / n_cols) * cur_weighted_sum);

			// now update dX
			dX[row_id * n_cols + i] += upstream_dX[row_id * n_cols + i] * __float2bfloat16(deriv);

		}
	}
}