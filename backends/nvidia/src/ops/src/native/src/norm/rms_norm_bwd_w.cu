#include "nvidia_ops.h"


extern "C" __global__ void default_rms_norm_bwd_w_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, float * X_inp, float * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched){
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols);

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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}


	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += upstream_dX[cur_row * n_cols + cur_dim_offset] * X_inp[cur_row * n_cols + cur_dim_offset] * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}



// Because X_inp is in row-major order we should be clever about doing column-wise dot products...

// at the end will do atomicAdds to dW because other blocks will have partial dot products as well

// cannot launch with more threads and n_cols otherwise will be bugs
// # blocks launched is a performance optimization and might be better with less due to less atomicAdds...
// definitely shouldn't launch with more than n_rows
extern "C" __global__ void default_rms_norm_bwd_w_fp16_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __half * X_inp, __half * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched){
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols);

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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += __half2float(upstream_dX[cur_row * n_cols + cur_dim_offset]) * __half2float(X_inp[cur_row * n_cols + cur_dim_offset]) * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}


extern "C" __global__ void default_rms_norm_bwd_w_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched){
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols);


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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += __bfloat162float(upstream_dX[cur_row * n_cols + cur_dim_offset]) * __bfloat162float(X_inp[cur_row * n_cols + cur_dim_offset]) * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}

extern "C" __global__ void default_rms_norm_bwd_w_fp8e4m3_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __half * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched){
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols); 

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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += __half2float(upstream_dX[cur_row * n_cols + cur_dim_offset]) * float(X_inp[cur_row * n_cols + cur_dim_offset]) * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}


extern "C" __global__ void default_rms_norm_bwd_w_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __nv_bfloat16 * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched) {
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols);

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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += __bfloat162float(upstream_dX[cur_row * n_cols + cur_dim_offset]) * float(X_inp[cur_row * n_cols + cur_dim_offset]) * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}

extern "C" __global__ void default_rms_norm_bwd_w_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __half * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched){
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols); 

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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += __half2float(upstream_dX[cur_row * n_cols + cur_dim_offset]) * float(X_inp[cur_row * n_cols + cur_dim_offset]) * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}


extern "C" __global__ void default_rms_norm_bwd_w_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __nv_bfloat16 * upstream_dX, float * dW_workspace, int * ret_num_blocks_launched){
	
	if (blockIdx.x == 0 && threadIdx.x == 0){
		*ret_num_blocks_launched = gridDim.x;
	}

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];



	// working space when computing weight derivs...
	// the dot products will be updated here and when complete
	// will accumulate in dW

	// length equal to the number of columns
	float * weight_derivs = (float *) sdata; 

	// length should be equal to max number of rows per block
	// load in squared sums and then divide by n_cols and take sqrt
	float * recip_avgs = (float *) (weight_derivs + n_cols);

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

	

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	for (uint64_t i = thread_id; i < n_cols; i+=blockDim.x){
		weight_derivs[i] = 0;
	}

	__syncthreads();

	
	// ensure that # threads launched is less than n_cols
	int num_warps = blockDim.x / 32;
	int dims_per_warp = ceilf((float) n_cols / (float) num_warps);

	int warp_iter;
	int cur_dim_offset;

	float cur_recip_avg;

	for (int cur_row = row_offset; cur_row < row_offset + rows_per_block; cur_row++){

		cur_recip_avg = recip_avgs[cur_row - row_offset];

		// each warp within threadblock will have a different dim_offset
		// and only be respno
		warp_iter = 0;
		cur_dim_offset = dims_per_warp * warp_id + lane_id;
		while ((warp_iter * 32) < (dims_per_warp) && (cur_dim_offset < n_cols)){

			// portion of dot product to update weight at cur_dim_offset
			// because each warp has their own section of dims some can run ahead
			// vs. others and ensure that the shared memory weigth_derivs (portions of column-wise dot product)
			// are still OK...

			// apply chain rule by multiplying with the upstream value...
			weight_derivs[cur_dim_offset] += __bfloat162float(upstream_dX[cur_row * n_cols + cur_dim_offset]) * float(X_inp[cur_row * n_cols + cur_dim_offset]) * cur_recip_avg;
			cur_dim_offset += 32;
			warp_iter++;
		}
	}

	// ensure all warps finish their portion of block
	__syncthreads();

	// now need to do atomic add into the global dW for this section of rows

	
	for (uint64_t dim = thread_id; dim < n_cols; dim+=blockDim.x){
		dW_workspace[blockIdx.x * n_cols + dim] = weight_derivs[dim];
	}
}