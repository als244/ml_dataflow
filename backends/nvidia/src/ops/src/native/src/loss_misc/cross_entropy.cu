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
extern "C" __global__ void default_set_average_loss_kernel(int num_tokens, float * loss_vec) {

    // It's good practice to use a template or an assertion to ensure
    // blockDim.x is a multiple of 32 and does not exceed 1024.
    // For this example, we assume blockDim.x <= 1024.
    
    // Size the shared memory array for the number of warps.
    // Using a fixed size of 32 is okay if you enforce blockDim.x <= 1024.
    __shared__ float partial_sums[32];

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int num_warps = blockDim.x / 32;

    // Each thread calculates its local sum
    float thread_loss_val = 0.0f;
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
        thread_loss_val += loss_vec[i];
    }

    // 1. Intra-warp reduction
    // The sum for each warp is now in its lane-0 thread
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_loss_val += __shfl_down_sync(0xFFFFFFFFU, thread_loss_val, offset);
    }

    // 2. Store partial sums from each warp into shared memory
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_loss_val;
    }

    // Wait for all warps to write to shared memory
    __syncthreads();

    // 3. Final reduction using the first warp
    if (warp_id == 0) {
        // Safely load the partial sums. Load 0.0f if the warp doesn't exist.
        // This is more explicit and robust than relying on prior initialization.
        float final_sum = (lane_id < num_warps) ? partial_sums[lane_id] : 0.0f;

        // Reduce the partial sums within the first warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xFFFFFFFFU, final_sum, offset);
        }

        // 4. Thread 0 writes the final average loss
        if (lane_id == 0) {
            loss_vec[num_tokens] = final_sum / (float)num_tokens;
        }
    }
}