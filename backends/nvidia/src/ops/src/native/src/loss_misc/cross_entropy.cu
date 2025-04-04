#include "nvidia_ops.h"


// subtracts 1 from correct values
extern "C" __global__ void cross_entropy_loss_fp32_kernel(int n_rows, int n_cols, float * pred_logits, uint32_t * labels) {

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	int row_ind;
	uint32_t correct_ind;
	if (i < n_rows){
		row_ind = i / n_cols;
		correct_ind = labels[row_ind];
		pred_logits[row_ind * n_cols + correct_ind] -= 1.0f;
	}
}

// launched with number of rows (total tokens to predict)
extern "C" __global__ void cross_entropy_loss_fp16_kernel(int n_rows, int n_cols, __half * pred_logits, uint32_t * labels){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	int row_ind;
	uint32_t correct_ind;
	if (i < n_rows){
		row_ind = i / n_cols;
		correct_ind = labels[row_ind];
		pred_logits[row_ind * n_cols + correct_ind] -= CONST_ONE_DEV_FP16;
	}
}


extern "C" __global__ void cross_entropy_loss_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * pred_logits, uint32_t * labels){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	int row_ind;
	uint32_t correct_ind;
	if (i < n_rows){
		row_ind = i / n_cols;
		correct_ind = labels[row_ind];
		pred_logits[row_ind * n_cols + correct_ind] -= CONST_ONE_DEV_BF16;
	}
}