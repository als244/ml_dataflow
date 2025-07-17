#include "nvidia_ops.h"

// Define vector size using a macro for C-style compatibility
#define VEC_SIZE (sizeof(float4) / sizeof(__nv_bfloat16))

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

extern "C" __global__ void default_rms_norm_bwd_x_bf16_bf16_kernel(
    int n_rows, int n_cols, float eps,
    const float * __restrict__ fwd_rms_vals,
    const __nv_bfloat16 * __restrict__ rms_weight,
    const __nv_bfloat16 * __restrict__ X_inp,
    const __nv_bfloat16 * __restrict__ upstream_dX,
    __nv_bfloat16 * __restrict__ dX,
    __nv_bfloat16 * __restrict__ X_out){

	// --- Kernel Setup ---
    extern __shared__ uint8_t sdata[];

    // Pointers to shared memory with C-style casts
    __nv_bfloat16* s_X_inp = (__nv_bfloat16*)sdata;
    __nv_bfloat16* s_upstream = s_X_inp + n_cols;
    size_t s_bf16_bytes = 2 * n_cols * sizeof(__nv_bfloat16);
    float* reduction_data_dout = (float*)(sdata + ((s_bf16_bytes + 3) & ~3));

    const int row_ind = blockIdx.x;
    if (row_ind >= n_rows) {
        return;
    }

    const uint64_t row_base = (uint64_t)row_ind * (uint64_t)n_cols;
    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const unsigned warp_mask = 0xFFFFFFFFU;

    // --- 1. Load data into Shared Memory (Vectorized) ---
    const int vec_n_cols = n_cols / VEC_SIZE;
    const int remainder_start = vec_n_cols * VEC_SIZE;

    // Vectorized load for the bulk of the data
    for (int i = thread_id; i < vec_n_cols; i += blockDim.x) {
        ((float4*)s_X_inp)[i] = __ldg(((const float4*)(X_inp + row_base)) + i);
        ((float4*)s_upstream)[i] = __ldg(((const float4*)(upstream_dX + row_base)) + i);
    }
    // Scalar load for the remaining "tail" elements
    if (remainder_start < n_cols) {
        for (int i = remainder_start + thread_id; i < n_cols; i += blockDim.x) {
            s_X_inp[i] = __ldg(X_inp + row_base + i);
            s_upstream[i] = __ldg(upstream_dX + row_base + i);
        }
    }
    __syncthreads();

    // --- 2. Compute C = sum(upstream_dX * X_out) and optionally write X_out ---
    float cur_upstream_sum = 0.0f;
    const float cur_recip_avg = fwd_rms_vals[row_ind];

    // Vectorized computation
    for (int i = thread_id; i < vec_n_cols; i += blockDim.x) {
        float4 x_vec = ((float4*)s_X_inp)[i];
        float4 up_vec = ((float4*)s_upstream)[i];
        float4 w_vec = __ldg(((const float4*)rms_weight) + i);
        
        bfloat162* x_b162_ptr = (bfloat162*)(&x_vec);
        bfloat162* up_b162_ptr = (bfloat162*)(&up_vec);
        bfloat162* w_b162_ptr = (bfloat162*)(&w_vec);
        
        float4 out_vec;
        bfloat162* out_b162_ptr = (bfloat162*)(&out_vec);

        #pragma unroll
        for (int k = 0; k < VEC_SIZE / 2; ++k) {
            float2 x_f2 = __bfloat1622float2(x_b162_ptr[k]);
            float2 up_f2 = __bfloat1622float2(up_b162_ptr[k]);
            float2 w_f2 = __bfloat1622float2(w_b162_ptr[k]);
            
            float2 out_f2;
            out_f2.x = x_f2.x * cur_recip_avg * w_f2.x;
            out_f2.y = x_f2.y * cur_recip_avg * w_f2.y;

            out_b162_ptr[k] = __float22bfloat162_rn(out_f2);
            cur_upstream_sum += out_f2.x * up_f2.x + out_f2.y * up_f2.y;
        }

        if (X_out) {
            ((float4*)(X_out + row_base))[i] = out_vec;
        }
    }
    // Scalar computation for the tail
    if (remainder_start < n_cols) {
        for (int i = remainder_start + thread_id; i < n_cols; i += blockDim.x) {
            float out_val = __bfloat162float(s_X_inp[i]) * cur_recip_avg * __bfloat162float(__ldg(rms_weight + i));
            if (X_out) X_out[row_base + i] = __float2bfloat16(out_val);
            cur_upstream_sum += out_val * __bfloat162float(s_upstream[i]);
        }
    }

    // --- 3. Block-wide Reduction for C ---
    for (int offset = 16; offset > 0; offset >>= 1) cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, offset);
    if (lane_id == 0) reduction_data_dout[warp_id] = cur_upstream_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        cur_upstream_sum = (lane_id < (blockDim.x / 32)) ? reduction_data_dout[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) cur_upstream_sum += __shfl_down_sync(warp_mask, cur_upstream_sum, offset);
    }
    if (thread_id == 0) reduction_data_dout[0] = cur_upstream_sum;
    __syncthreads();

    const float C = reduction_data_dout[0];
    const float C_div_n = C / (float)n_cols;

    // --- 4. Compute and Accumulate dX ---
    // Vectorized update
    for (int i = thread_id; i < vec_n_cols; i += blockDim.x) {
        float4 x_vec = ((float4*)s_X_inp)[i];
        float4 up_vec = ((float4*)s_upstream)[i];
        float4 w_vec = __ldg(((const float4*)rms_weight) + i);

        bfloat162* x_b162_ptr = (bfloat162*)(&x_vec);
        bfloat162* up_b162_ptr = (bfloat162*)(&up_vec);
        bfloat162* w_b162_ptr = (bfloat162*)(&w_vec);
        
        float4 dX_new_vec;
        bfloat162* dX_new_b162_ptr = (bfloat162*)(&dX_new_vec);

        float4 dX_old_vec = __ldcg(((const float4*)(dX + row_base)) + i);
        bfloat162* dX_old_b162_ptr = (bfloat162*)(&dX_old_vec);
        
        #pragma unroll
        for (int k = 0; k < VEC_SIZE / 2; k++) {
            float2 x_f2 = __bfloat1622float2(x_b162_ptr[k]);
            float2 up_f2 = __bfloat1622float2(up_b162_ptr[k]);
            float2 w_f2 = __bfloat1622float2(w_b162_ptr[k]);
            
            float2 deriv_f2;
            deriv_f2.x = cur_recip_avg * (up_f2.x * w_f2.x - C_div_n * (x_f2.x * cur_recip_avg));
            deriv_f2.y = cur_recip_avg * (up_f2.y * w_f2.y - C_div_n * (x_f2.y * cur_recip_avg));
            
            float2 dX_old_f2 = __bfloat1622float2(dX_old_b162_ptr[k]);
            dX_old_f2.x += deriv_f2.x;
            dX_old_f2.y += deriv_f2.y;
            
            dX_new_b162_ptr[k] = __float22bfloat162_rn(dX_old_f2);
        }
        ((float4*)(dX + row_base))[i] = dX_new_vec;
    }
    // Scalar update for the tail
    if (remainder_start < n_cols) {
        for (int i = remainder_start + thread_id; i < n_cols; i += blockDim.x) {
            float inp_val = __bfloat162float(s_X_inp[i]);
            float upstream_val = __bfloat162float(s_upstream[i]);
            float weight_val = __bfloat162float(__ldg(rms_weight + i));
            float deriv = cur_recip_avg * (upstream_val * weight_val - C_div_n * (inp_val * cur_recip_avg));
            dX[row_base + i] += __float2bfloat16(deriv);
        }
    }
}


extern "C" __global__ void naive_default_rms_norm_bwd_x_bf16_bf16_kernel(int n_rows, int n_cols, float eps, float * fwd_rms_vals, __nv_bfloat16 * rms_weight, __nv_bfloat16 * X_inp, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX, __nv_bfloat16 * X_out) {
		
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