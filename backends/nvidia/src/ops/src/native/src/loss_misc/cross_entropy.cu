#include "nvidia_ops.h"


// subtracts 1 from correct values
extern "C" __global__ void default_cross_entropy_loss_fp32_kernel(int n_rows, int n_cols, float * pred_logits, uint32_t * labels, float * loss_vec) {

	uint64_t row_start = blockIdx.x * blockDim.x;

	int iter_size = gridDim.x * blockDim.x;

	int thread_row_start = row_start + threadIdx.x;

	uint32_t correct_ind;
	float loss_val;
	for (int row_ind = thread_row_start; row_ind < n_rows; row_ind += iter_size){
		correct_ind = labels[row_ind];
		loss_val = -1 * logf(pred_logits[row_ind * n_cols + correct_ind]);
		loss_vec[row_ind] = loss_val;
		pred_logits[row_ind * n_cols + correct_ind] -= 1.0f;
	}

	return;
}

// launched with number of rows (total tokens to predict)
extern "C" __global__ void default_cross_entropy_loss_fp16_kernel(int n_rows, int n_cols, __half * pred_logits, uint32_t * labels, float * loss_vec){

	uint64_t row_start = blockIdx.x * blockDim.x;


	int iter_size = gridDim.x * blockDim.x;

	int thread_row_start = row_start + threadIdx.x;

	uint32_t correct_ind;
	float loss_val;
	for (int row_ind = thread_row_start; row_ind < n_rows; row_ind += iter_size){
		correct_ind = labels[row_ind];
		loss_val = -1 * logf(__half2float(pred_logits[row_ind * n_cols + correct_ind]));
		loss_vec[row_ind] = loss_val;
		pred_logits[row_ind * n_cols + correct_ind] -= CONST_ONE_DEV_FP16;
	}

	return;
}


extern "C" __global__ void default_cross_entropy_loss_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * pred_logits, uint32_t * labels, float * loss_vec){

	uint64_t row_start = blockIdx.x * blockDim.x;


	int iter_size = gridDim.x * blockDim.x;

	int thread_row_start = row_start + threadIdx.x;

	uint32_t correct_ind;
	float loss_val;
	for (int row_ind = thread_row_start; row_ind < n_rows; row_ind += iter_size){
		correct_ind = labels[row_ind];
		loss_val = -1 * logf(__bfloat162float(pred_logits[row_ind * n_cols + correct_ind]));
		loss_vec[row_ind] = loss_val;
		pred_logits[row_ind * n_cols + correct_ind] -= CONST_ONE_DEV_BF16;
	}

	return;
}

// LAUNCHES WITH ONLY 1 BLOCK!
extern "C" __global__ void default_set_average_loss_kernel(int num_tokens, float * loss_vec, float * ret_avg_loss){


	__shared__ float warp_loss[WARP_SIZE];

	if (threadIdx.x < WARP_SIZE){
		warp_loss[threadIdx.x] = 0.0f;
	}

	int token_start = threadIdx.x;

	float thread_loss_val = 0.0f;

	unsigned mask = 0xFFFFFFFFU;

	int warp_id = threadIdx.x / 32;
	int lane_id = threadIdx.x % 32;

	for (int i = token_start; i < num_tokens; i += blockDim.x){
		thread_loss_val += loss_vec[i];
	}

	__syncwarp();

	for (int offset = 16; offset > 0; offset >>= 1) {
		thread_loss_val += __shfl_down_sync(mask, thread_loss_val, offset);
	}

	if (lane_id == 0){
		warp_loss[warp_id] = thread_loss_val;
	}

	__syncthreads();

	if (warp_id == 0){
		thread_loss_val = warp_loss[lane_id];

		for (int offset = 16; offset > 0; offset >>= 1) {
			thread_loss_val += __shfl_down_sync(mask, thread_loss_val, offset);
		}

		// set the average loss
		if (lane_id == 0){
			*ret_avg_loss = thread_loss_val / (float)num_tokens;
		}
	}

	return;
}