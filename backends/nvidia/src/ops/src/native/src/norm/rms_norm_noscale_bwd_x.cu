#include "nvidia_ops.h"


extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, float * X_inp, float * upstream_dX, float * dX, float * X_out) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * inp_row = (float *) sdata;

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (inp_row + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			out_val = (inp_row[i] * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = out_val;
			}

			cur_upstream_sum += upstream_dX[row_ind_start + i] * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = inp_row[i];
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((upstream_dX[row_ind_start + i]) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += deriv;
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}


// Here dX is (N, model_dim) and contains the backprop loss flow that we will update in-place
// This needs to be called after the bwd_weight because the weight we use the updstream dL/dX and this function will
// modify the same pointer...
extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp16_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __half * X_inp,  __half * upstream_dX, __half * dX, __half * X_out){
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	__half * inp_row = (__half *) sdata;

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (inp_row + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __half2float(inp_row[i]);
			out_val = (inp_val * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = __float2half(out_val);
			}

			cur_upstream_sum += __half2float(upstream_dX[row_ind_start + i]) * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __half2float(inp_row[i]);
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((__half2float(upstream_dX[row_ind_start + i])) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += __float2half(deriv);
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}


extern "C" __global__ void default_rms_norm_noscale_bwd_x_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_bfloat16 * X_out) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	__nv_bfloat16 * inp_row = (__nv_bfloat16 *) sdata;

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (inp_row + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __bfloat162float(inp_row[i]);
			out_val = (inp_val * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = __float2bfloat16(out_val);
			}

			cur_upstream_sum += __bfloat162float(upstream_dX[row_ind_start + i]) * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = __bfloat162float(inp_row[i]);
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((__bfloat162float(upstream_dX[row_ind_start + i])) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += __float2bfloat16(deriv);
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}


extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e4m3_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __half * upstream_dX, __half * dX, __nv_fp8_e4m3 * X_out) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	__nv_fp8_e4m3 * inp_row = (__nv_fp8_e4m3 *) sdata;

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (inp_row + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val = (inp_val * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = __nv_fp8_e4m3(out_val);
			}

			cur_upstream_sum += __half2float(upstream_dX[row_ind_start + i]) * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((__half2float(upstream_dX[row_ind_start + i])) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += __float2half(deriv);
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}

extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_fp8_e4m3 * X_out) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	__nv_fp8_e4m3 * inp_row = (__nv_fp8_e4m3 *) sdata;
	float * weights = (float *) (inp_row + n_cols);

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (weights + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val = (inp_val * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = __nv_fp8_e4m3(out_val);
			}

			cur_upstream_sum += __bfloat162float(upstream_dX[row_ind_start + i]) * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((__bfloat162float(upstream_dX[row_ind_start + i])) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += __float2bfloat16(deriv);
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}


extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __half * upstream_dX, __half * dX, __nv_fp8_e5m2 * X_out) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	__nv_fp8_e5m2 * inp_row = (__nv_fp8_e5m2 *) sdata;

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (inp_row + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val = (inp_val * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = __nv_fp8_e5m2(out_val);
			}

			cur_upstream_sum += __half2float(upstream_dX[row_ind_start + i]) * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((__half2float(upstream_dX[row_ind_start + i])) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += __float2half(deriv);
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}

extern "C" __global__ void default_rms_norm_noscale_bwd_x_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_fp8_e5m2 * X_out) {
		
	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];


	int rows_per_block = n_rows / gridDim.x;
	int rows_remain = n_rows % gridDim.x;
	int max_rows_per_block = rows_per_block + (rows_remain > 0);

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	__nv_fp8_e5m2 * inp_row = (__nv_fp8_e5m2 *) sdata;
	float * weights = (float *) (inp_row + n_cols);

	// length equal to the number of columns
	float * shared_recip_avgs = (float *) (weights + max_rows_per_block);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	//

	// then we compute 
	// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))
	

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

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;


	// retrieve back the recip squared avgs
	// corresponding to this blocks rows
	for (int i = row_offset + thread_id; i < row_offset + rows_per_block; i+=blockDim.x){
		shared_recip_avgs[i - row_offset] = fwd_rms_vals[i];
	}

	__syncthreads();

	float deriv;
	
	float cur_recip_avg;
	float cur_upstream_sum;

	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;

	uint64_t row_ind_start;
	for (int row_id = row_offset; row_id < row_offset + rows_per_block; row_id++){
		row_ind_start = (uint64_t) (row_id) * (uint64_t) n_cols;

		// first save down the input row, so we can can recompute the output
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_row[i] = X_inp[row_ind_start + i];
		}

		__syncthreads();

		cur_recip_avg = shared_recip_avgs[row_id - row_offset];

		
		// first get the value C which is the sum of the upstream_dX * X_out
		cur_upstream_sum = 0;
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val = (inp_val * cur_recip_avg);

			// if we want to recompute the forward pass, we already have done all the work
			// and can save it down here...
			if (X_out){
				X_out[row_id * n_cols + i] = __nv_fp8_e5m2(out_val);
			}

			cur_upstream_sum += __bfloat162float(upstream_dX[row_ind_start + i]) * out_val;
		}

		// now have each warp reduce their results
		__syncwarp();

		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
		}

		if (lane_id == 0){
			reduction_data_dout[warp_id] = cur_upstream_sum;
		}

		__syncthreads();

		// now we reduce among the warps, by using threads within the first warp

		if (warp_id == 0){

			// reassigning starting value to be the value reduced among each of the warps
			cur_upstream_sum = reduction_data_dout[lane_id];

			for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
				cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, warp_offset);
			}

			// now thread 0 contains the full sum
			// can reset to be within the first index, that other threads can access
			if (lane_id == 0){
				reduction_data_dout[0] = cur_upstream_sum;
			}
		}

		__syncthreads();

		// now reset the upstream sum variable to be the entire row's sum
		// that thread 0 (warp id 0, lane id 0) has just published
		cur_upstream_sum = reduction_data_dout[0];

		// now can compute:
		// dX[i] = recip_avg * (upstream_dX[i] * rms_weight[i] - C * ((x_inp[i] * recip_avg) / n_cols))

		
		for (int i = thread_id; i < n_cols; i+=blockDim.x){
			inp_val = float(inp_row[i]);
			out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
			deriv = (cur_recip_avg * ((__bfloat162float(upstream_dX[row_ind_start + i])) - out_val_scaled));

			// now update dX
			dX[row_id * n_cols + i] += __float2bfloat16(deriv);
		}

		// ensure sync before next row which will overwrite the input row
		__syncthreads();
		
	}
}