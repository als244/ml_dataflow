#include "nvidia_ops.h"


// subtracts 1 from correct values
extern "C" __global__ void default_cross_entropy_loss_fp32_kernel(int n_rows, int n_cols, float * pred_logits, uint32_t * labels, float * loss_vec) {

	uint64_t row_start = blockIdx.x * blockDim.x;

	__shared__ float thread_loss[1024];

	for (int i = threadIdx.x; i < 1024; i += blockDim.x){
		thread_loss[i] = 0.0f;
	}

	float thread_loss_val = 0.0f;

	int iter_size = gridDim.x * blockDim.x;

	int thread_row_start = row_start + threadIdx.x;

	uint32_t correct_ind;
	float loss_val;
	for (int row_ind = thread_row_start; row_ind < n_rows; row_ind += iter_size){
		correct_ind = labels[row_ind];
		loss_val = -1 * logf(pred_logits[row_ind * n_cols + correct_ind]);
		loss_vec[row_ind] = loss_val;
		thread_loss_val += loss_val;
		pred_logits[row_ind * n_cols + correct_ind] -= 1.0f;
	}

	thread_loss[threadIdx.x] = thread_loss_val;

	__syncwarp();

	// First reduce within each warp
	for (int offset = 16; offset > 0; offset /= 2) {
		thread_loss[threadIdx.x] += __shfl_down_sync(0xffffffff, thread_loss[threadIdx.x], offset);
	}

	__syncwarp();

	// Have first thread in each warp write its result
	if ((threadIdx.x & 31) == 0) {
		thread_loss[threadIdx.x / 32] = thread_loss[threadIdx.x];
	}

	// Make sure all warps have finished their local reductions

	__syncthreads();

	// Final reduction across warps using first warp
	if (threadIdx.x < 32) {
		float warp_sum = (threadIdx.x < (blockDim.x / 32)) ? thread_loss[threadIdx.x] : 0.0f;
		
		// Reduce across the first warp
		for (int offset = 16; offset > 0; offset /= 2) {
			warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
		}

		if (threadIdx.x == 0) {
			atomicAdd(&loss_vec[n_rows], warp_sum);
		}
	}

	return;
}

// launched with number of rows (total tokens to predict)
extern "C" __global__ void default_cross_entropy_loss_fp16_kernel(int n_rows, int n_cols, __half * pred_logits, uint32_t * labels, float * loss_vec){

	uint64_t row_start = blockIdx.x * blockDim.x;

	__shared__ float thread_loss[1024];

	for (int i = threadIdx.x; i < 1024; i += blockDim.x){
		thread_loss[i] = 0.0f;
	}

	float thread_loss_val = 0.0f;

	int iter_size = gridDim.x * blockDim.x;

	int thread_row_start = row_start + threadIdx.x;

	uint32_t correct_ind;
	float loss_val;
	for (int row_ind = thread_row_start; row_ind < n_rows; row_ind += iter_size){
		correct_ind = labels[row_ind];
		loss_val = -1 * logf(__half2float(pred_logits[row_ind * n_cols + correct_ind]));
		loss_vec[row_ind] = loss_val;
		thread_loss_val += loss_val;
		pred_logits[row_ind * n_cols + correct_ind] -= CONST_ONE_DEV_FP16;
	}

	thread_loss[threadIdx.x] = thread_loss_val;

	__syncwarp();

	// First reduce within each warp
	for (int offset = 16; offset > 0; offset /= 2) {
		thread_loss[threadIdx.x] += __shfl_down_sync(0xffffffff, thread_loss[threadIdx.x], offset);
	}

	__syncwarp();

	// Have first thread in each warp write its result
	if ((threadIdx.x & 31) == 0) {
		thread_loss[threadIdx.x / 32] = thread_loss[threadIdx.x];
	}

	// Make sure all warps have finished their local reductions

	__syncthreads();

	// Final reduction across warps using first warp
	if (threadIdx.x < 32) {
		float warp_sum = (threadIdx.x < (blockDim.x / 32)) ? thread_loss[threadIdx.x] : 0.0f;
		
		// Reduce across the first warp
		for (int offset = 16; offset > 0; offset /= 2) {
			warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
		}

		if (threadIdx.x == 0) {
			atomicAdd(&loss_vec[n_rows], warp_sum);
		}
	}

	return;
}


extern "C" __global__ void default_cross_entropy_loss_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * pred_logits, uint32_t * labels, float * loss_vec){

	uint64_t row_start = blockIdx.x * blockDim.x;

	__shared__ float thread_loss[1024];

	for (int i = threadIdx.x; i < 1024; i += blockDim.x){
		thread_loss[i] = 0.0f;
	}

	float thread_loss_val = 0.0f;

	int iter_size = gridDim.x * blockDim.x;

	int thread_row_start = row_start + threadIdx.x;

	uint32_t correct_ind;
	float loss_val;
	for (int row_ind = thread_row_start; row_ind < n_rows; row_ind += iter_size){
		correct_ind = labels[row_ind];
		loss_val = -1 * logf(__bfloat162float(pred_logits[row_ind * n_cols + correct_ind]));
		loss_vec[row_ind] = loss_val;
		thread_loss_val += loss_val;
		pred_logits[row_ind * n_cols + correct_ind] -= CONST_ONE_DEV_BF16;
	}

	thread_loss[threadIdx.x] = thread_loss_val;

	__syncwarp();

	// First reduce within each warp
	for (int offset = 16; offset > 0; offset /= 2) {
		thread_loss[threadIdx.x] += __shfl_down_sync(0xffffffff, thread_loss[threadIdx.x], offset);
	}

	__syncwarp();

	// Have first thread in each warp write its result
	if ((threadIdx.x & 31) == 0) {
		thread_loss[threadIdx.x / 32] = thread_loss[threadIdx.x];
	}

	// Make sure all warps have finished their local reductions

	__syncthreads();

	// Final reduction across warps using first warp
	if (threadIdx.x < 32) {
		float warp_sum = (threadIdx.x < (blockDim.x / 32)) ? thread_loss[threadIdx.x] : 0.0f;
		
		// Reduce across the first warp
		for (int offset = 16; offset > 0; offset /= 2) {
			warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
		}

		if (threadIdx.x == 0) {
			atomicAdd(&loss_vec[n_rows], warp_sum);
		}
	}

	return;
}