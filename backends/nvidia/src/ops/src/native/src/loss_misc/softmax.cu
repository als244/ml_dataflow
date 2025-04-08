#include "nvidia_ops.h"

extern "C" __global__ void default_softmax_fp32_fp32_kernel(int n_rows, int n_cols, float * X, float * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	float * row_start = X + row_offset;
	float * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ float warp_maxs[32];
	__shared__ float warp_sums[32];
	__shared__ float global_max[1];
	__shared__ float global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	if (warp_id == 0){
		warp_maxs[lane_id] = CONST_DEV_FLOAT_NEG_INF;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	float other_val;

	float new_max = CONST_DEV_FLOAT_NEG_INF;

	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = max(new_max, row_start[cur_ind]);

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = max(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = max(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	float overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(row_start[cur_ind] - overall_max);

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = expf(row_start[cur_ind] - overall_max) / overall_sum;
		cur_ind += num_warps * 32;
	}
}


extern "C" __global__ void default_softmax_fp16_fp16_kernel(int n_rows, int n_cols, __half * X, __half * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__half * row_start = X + row_offset;
	__half * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __half warp_maxs[32];
	__shared__ __half warp_sums[32];
	__shared__ __half global_max[1];
	__shared__ __half global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_FP16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__half other_val;

	__half new_max = NEG_INF_DEV_FP16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, row_start[cur_ind]);

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__half overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(__half2float(row_start[cur_ind] - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = __float2half(expf(__half2float(row_start[cur_ind] - overall_max)) / overall_sum);
		cur_ind += num_warps * 32;
	}
}

extern "C" __global__ void default_softmax_bf16_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * X, __nv_bfloat16 * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__nv_bfloat16 * row_start = X + row_offset;
	__nv_bfloat16 * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __nv_bfloat16 warp_maxs[32];
	__shared__ __nv_bfloat16 warp_sums[32];
	__shared__ __nv_bfloat16 global_max[1];
	__shared__ __nv_bfloat16 global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_BF16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__nv_bfloat16 other_val;

	__nv_bfloat16 new_max = NEG_INF_DEV_BF16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, row_start[cur_ind]);

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__nv_bfloat16 overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(__bfloat162float(row_start[cur_ind] - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = __float2bfloat16(expf(__bfloat162float(row_start[cur_ind] - overall_max)) / overall_sum);
		cur_ind += num_warps * 32;
	}
}

extern "C" __global__ void default_softmax_fp8e4m3_fp16_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * X, __half * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__nv_fp8_e4m3 * row_start = X + row_offset;
	__half * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __half warp_maxs[32];
	__shared__ __half warp_sums[32];
	__shared__ __half global_max[1];
	__shared__ __half global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_FP16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__half other_val;

	__half new_max = NEG_INF_DEV_FP16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, __half(row_start[cur_ind]));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__half overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(__half2float(__half(row_start[cur_ind]) - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = __float2half(expf(__half2float(__half(row_start[cur_ind]) - overall_max)) / overall_sum);
		cur_ind += num_warps * 32;
	}
}


extern "C" __global__ void default_softmax_fp8e4m3_bf16_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * X, __nv_bfloat16 * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__nv_fp8_e4m3 * row_start = X + row_offset;
	__nv_bfloat16 * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __nv_bfloat16 warp_maxs[32];
	__shared__ __nv_bfloat16 warp_sums[32];
	__shared__ __nv_bfloat16 global_max[1];
	__shared__ __nv_bfloat16 global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_BF16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__nv_bfloat16 other_val;

	__nv_bfloat16 new_max = NEG_INF_DEV_BF16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, __nv_bfloat16(row_start[cur_ind]));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__nv_bfloat16 overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(__bfloat162float(__nv_bfloat16(row_start[cur_ind]) - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = __float2bfloat16(expf(__bfloat162float(__nv_bfloat16(row_start[cur_ind]) - overall_max)) / overall_sum);
		cur_ind += num_warps * 32;
	}
}


extern "C" __global__ void default_softmax_fp8e5m2_fp16_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * X, __half * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__nv_fp8_e5m2 * row_start = X + row_offset;
	__half * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __half warp_maxs[32];
	__shared__ __half warp_sums[32];
	__shared__ __half global_max[1];
	__shared__ __half global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_FP16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__half other_val;

	__half new_max = NEG_INF_DEV_FP16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, __half(row_start[cur_ind]));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__half overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(__half2float(__half(row_start[cur_ind]) - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = __float2half(expf(__half2float(__half(row_start[cur_ind]) - overall_max)) / overall_sum);
		cur_ind += num_warps * 32;
	}
}


extern "C" __global__ void default_softmax_fp8e5m2_bf16_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * X, __nv_bfloat16 * out) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__nv_fp8_e5m2 * row_start = X + row_offset;
	__nv_bfloat16 * out_row_start = out + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __nv_bfloat16 warp_maxs[32];
	__shared__ __nv_bfloat16 warp_sums[32];
	__shared__ __nv_bfloat16 global_max[1];
	__shared__ __nv_bfloat16 global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_BF16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__nv_bfloat16 other_val;

	__nv_bfloat16 new_max = NEG_INF_DEV_BF16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, __nv_bfloat16(row_start[cur_ind]));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__nv_bfloat16 overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	while (cur_ind < n_cols){

		new_sum = expf(__bfloat162float(__nv_bfloat16(row_start[cur_ind]) - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_row_start[cur_ind] = __float2bfloat16(expf(__bfloat162float(__nv_bfloat16(row_start[cur_ind]) - overall_max)) / overall_sum);
		cur_ind += num_warps * 32;
	}
}






// Assumes N = # columns
// And block_idx is the row

// very naive implementation for now....
extern "C" __global__ void default_softmax_fp16_fp32_kernel(int n_rows, int n_cols, __half * X_in, float * out, uint32_t * arg_maxs) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__half * row_start = X_in + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __half warp_maxs[32];
	__shared__ __half warp_sums[32];
	__shared__ __half global_max[1];
	__shared__ __half global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_FP16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__half other_val;

	__half new_max = NEG_INF_DEV_FP16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, row_start[cur_ind]);

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__half overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	__half cur_val;
	while (cur_ind < n_cols){

		cur_val = row_start[cur_ind];
		if (arg_maxs && (cur_val == overall_max)){
			arg_maxs[row_ind] = (uint32_t) cur_ind;
		}

		new_sum = expf(__half2float(cur_val - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	float * out_start = out + row_offset;

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_start[cur_ind] = expf(__half2float(row_start[cur_ind] - overall_max)) / overall_sum;
		cur_ind += num_warps * 32;
	}
}

extern "C" __global__ void default_softmax_bf16_fp32_kernel(int n_rows, int n_cols, __nv_bfloat16 * X_in, float * out, uint32_t * arg_maxs) {

	uint64_t row_ind = blockIdx.x;

	uint64_t row_offset = row_ind * ((uint64_t) n_cols);
	__nv_bfloat16 * row_start = X_in + row_offset;

	int thread_id = threadIdx.x;

	__shared__ __nv_bfloat16 warp_maxs[32];
	__shared__ __nv_bfloat16 warp_sums[32];
	__shared__ __nv_bfloat16 global_max[1];
	__shared__ __nv_bfloat16 global_sum[1];

	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;
	int num_warps = blockDim.x / 32;

	

	if (warp_id == 0){
		warp_maxs[lane_id] = NEG_INF_DEV_BF16;
		warp_sums[lane_id] = 0;
	}

	__syncthreads();

	__nv_bfloat16 other_val;

	__nv_bfloat16 new_max = NEG_INF_DEV_BF16;


	unsigned warp_mask = 0xFFFFFFFFU;

	int cur_ind = thread_id;

	// Assuming N is a multiple of 32 for simplicity...
	while (cur_ind < n_cols){

		new_max = __hmax(new_max, row_start[cur_ind]);

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_maxs[warp_id] = new_max;
	}

	__syncthreads();


	if (warp_id == 0){

		new_max = warp_maxs[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			other_val = __shfl_down_sync(warp_mask, new_max, warp_offset);
			new_max = __hmax(new_max, other_val);
		}

		if (lane_id == 0){
			global_max[0] = new_max;
		}
	}

	__syncthreads();


	// now do sums

	cur_ind = thread_id;

	__nv_bfloat16 overall_max = global_max[0];

	float total_sum = 0;
	float new_sum;
	__nv_bfloat16 cur_val;
	while (cur_ind < n_cols){

		cur_val = row_start[cur_ind];
		if (arg_maxs && (cur_val == overall_max)){
			arg_maxs[row_ind] = (uint32_t) cur_ind;
		}

		new_sum = expf(__bfloat162float(cur_val - overall_max));

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			new_sum += __shfl_down_sync(warp_mask, new_sum, warp_offset);
		}

		if (lane_id == 0){
			total_sum += new_sum;
		}

		cur_ind += num_warps * 32;
	}

	if (lane_id == 0){
		warp_sums[warp_id] = total_sum;
	}

	__syncthreads();

	if (warp_id == 0){

		total_sum = warp_sums[lane_id];

		#pragma unroll
		for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
			total_sum += __shfl_down_sync(warp_mask, total_sum, warp_offset);
		}

		if (lane_id == 0){
			global_sum[0] = total_sum;
		}
	}

	__syncthreads();


	// now do output

	float overall_sum = global_sum[0];

	float * out_start = out + row_offset;

	cur_ind = thread_id;

	while (cur_ind < n_cols){

		out_start[cur_ind] = expf(__bfloat162float(row_start[cur_ind] - overall_max)) / overall_sum;
		cur_ind += num_warps * 32;
	}
}



