#include "nvidia_ops.h"

extern "C" __global__ void default_rms_norm_bwd_x_fp32_fp32_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, float * rms_weight, float * X_inp, float * upstream_dX, float * dX, float * X_out) {
		
	// this gets dynamically allocated the correct size
	extern __shared__ uint8_t sdata[];

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * inp_row = (float *) sdata;
	float * upstream_row = (inp_row + n_cols);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;
	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	float cur_recip_avg = fwd_rms_vals[row_ind];

	float cur_weight;

	float cur_upstream_sum;

	float deriv;
	
	float inp_val;
	float out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;


	// first save down the input row, so we can can recompute the output
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		inp_row[i] = X_inp[row_base + i];
		upstream_row[i] = upstream_dX[row_base + i];
	}

	__syncthreads();

		
	// first get the value C which is the sum of the upstream_dX * X_out
	cur_upstream_sum = 0;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		cur_weight = __ldg(rms_weight + i);
		out_val = cur_weight * (inp_row[i] * cur_recip_avg);

		// if we want to recompute the forward pass, we already have done all the work
		// and can save it down here...
		if (X_out){
			X_out[row_base + i] = out_val;
		}

		cur_upstream_sum += upstream_row[i] * out_val;
	}

	// now have each warp reduce their results
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
		cur_weight = __ldg(rms_weight + i);
		inp_val = inp_row[i];
		out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
		deriv = (cur_recip_avg * (cur_weight * upstream_row[i] - out_val_scaled));

		// now update dX
		// (assume we want to accumulate)
		dX[row_base + i] += deriv;
	}
}


extern "C" __global__ void default_rms_norm_bwd_x_fp16_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __half * rms_weight, __half * X_inp, __half * upstream_dX, __half * dX, __half * X_out) {
		
	// this gets dynamically allocated the correct size
	extern __shared__ uint8_t sdata[];

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * inp_row = (float *) sdata;
	__half * upstream_row = (__half *) (inp_row + n_cols);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;
	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	float cur_recip_avg = fwd_rms_vals[row_ind];

	__half cur_weight;

	float cur_upstream_sum;

	float deriv;
	
	float inp_val;
	__half out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;


	// first save down the input row, so we can can recompute the output
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		inp_row[i] = __half2float(X_inp[row_base + i]);
		upstream_row[i] = upstream_dX[row_base + i];
	}

	__syncthreads();

		
	// first get the value C which is the sum of the upstream_dX * X_out
	cur_upstream_sum = 0;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		cur_weight = __ldg(rms_weight + i);
		out_val = cur_weight * __float2half(inp_row[i] * cur_recip_avg);

		// if we want to recompute the forward pass, we already have done all the work
		// and can save it down here...
		if (X_out){
			X_out[row_base + i] = out_val;
		}

		cur_upstream_sum += __half2float(upstream_row[i] * out_val);
	}

	// now have each warp reduce their results
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
		cur_weight = __ldg(rms_weight + i);
		inp_val = inp_row[i];
		out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
		deriv = (cur_recip_avg * (__half2float(cur_weight * upstream_row[i]) - out_val_scaled));

		// now update dX
		// (assume we want to accumulate)
		dX[row_base + i] += __float2half(deriv);
	}
}


extern "C" __global__ void default_rms_norm_bwd_x_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_bfloat16 * X_out) {
		
	// this gets dynamically allocated the correct size
	extern __shared__ uint8_t sdata[];

	// length should be equal to number of rows
	// load in squared sums and then divide by n_cols and take sqrt
	float * inp_row = (float *) sdata;
	__nv_bfloat16 * upstream_row = (__nv_bfloat16 *) (inp_row + n_cols);

	// every warp will have a reduced value

	// for every row we need to compute C = sum_j(upstream_dX[j] * x_out[j])
	// as prerequisite to computing the RMS norm...
	__shared__ float reduction_data_dout[32];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

	int thread_id = threadIdx.x;
	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	float cur_recip_avg = fwd_rms_vals[row_ind];

	__nv_bfloat16 cur_weight;

	float cur_upstream_sum;

	float deriv;
	
	float inp_val;
	__nv_bfloat16 out_val;
	float out_val_scaled;

	unsigned warp_mask = 0xFFFFFFFFU;


	// first save down the input row, so we can can recompute the output
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		inp_row[i] = __bfloat162float(X_inp[row_base + i]);
		upstream_row[i] = upstream_dX[row_base + i];
	}

	__syncthreads();

		
	// first get the value C which is the sum of the upstream_dX * X_out
	cur_upstream_sum = 0;
	for (int i = thread_id; i < n_cols; i+=blockDim.x){
		cur_weight = __ldg(rms_weight + i);
		out_val = cur_weight * __float2bfloat16(inp_row[i] * cur_recip_avg);

		// if we want to recompute the forward pass, we already have done all the work
		// and can save it down here...
		if (X_out){
			X_out[row_base + i] = out_val;
		}

		cur_upstream_sum += __bfloat162float(upstream_row[i] * out_val);
	}

	// now have each warp reduce their results
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
		cur_weight = __ldg(rms_weight + i);
		inp_val = inp_row[i];
		out_val_scaled = cur_upstream_sum * ((inp_val * cur_recip_avg) / n_cols);
		deriv = (cur_recip_avg * (__bfloat162float(cur_weight * upstream_row[i]) - out_val_scaled));

		// now update dX
		// (assume we want to accumulate)
		dX[row_base + i] += __float2bfloat16(deriv);
	}
}


extern "C" __global__ void default_rms_norm_bwd_x_fp8e4m3_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X_inp, __half * upstream_dX, __half * dX, __nv_fp8_e4m3 * X_out) {
	return;
}

extern "C" __global__ void default_rms_norm_bwd_x_fp8e4m3_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e4m3 * rms_weight, __nv_fp8_e4m3 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_fp8_e4m3 * X_out) {
	return;
}

extern "C" __global__ void default_rms_norm_bwd_x_fp8e5m2_fp16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X_inp, __half * upstream_dX, __half * dX, __nv_fp8_e5m2 * X_out) {
	return;
}

extern "C" __global__ void default_rms_norm_bwd_x_fp8e5m2_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_fp8_e5m2 * rms_weight, __nv_fp8_e5m2 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_fp8_e5m2 * X_out) {
	return;
}