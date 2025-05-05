#include "nvidia_ops.h"

extern "C" __global__ void default_rope_bwd_x_fp32_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int *seq_positions, float *dX_q, float *dX_k) {
	
	// Each thread handles two consecutive half elements.
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N) {
		// ------------------------
		// Process the query branch
		// ------------------------
		// Compute token row and corresponding position/dimension.
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// Compute the angle for the current dimension.
		float angle = powf(theta, -1.0f * ((float)cur_dim / head_dim));
		float cos_val = cosf(cur_pos * angle);
		float sin_val = sinf(cur_pos * angle);

		// Load the upstream gradients from dX_q.
		float grad_even = dX_q[i];
		float grad_odd  = dX_q[i + 1];

		// Apply the transpose of the rotation matrix.
		float updated_grad_even = cos_val * grad_even + sin_val * grad_odd;
		float updated_grad_odd  = -sin_val * grad_even + cos_val * grad_odd;

		// Update the gradients in place.
		dX_q[i]     = updated_grad_even;
		dX_q[i + 1] = updated_grad_odd;

		// ------------------------
		// Process the key branch
		// ------------------------
		int kv_dim = num_kv_heads * head_dim;
		// Adjust token indexing for key tensor layout.
		token_row = i / kv_dim;
		int total_tokens = N / model_dim;
		if (token_row < total_tokens) {
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;
			angle = powf(theta, -1.0f * ((float)cur_dim / head_dim));
			cos_val = cosf(cur_pos * angle);
			sin_val = sinf(cur_pos * angle);

			grad_even = dX_k[i];
			grad_odd  = dX_k[i + 1];

			updated_grad_even = cos_val * grad_even + sin_val * grad_odd;
			updated_grad_odd  = -sin_val * grad_even + cos_val * grad_odd;

			dX_k[i]     = updated_grad_even;
			dX_k[i + 1] = updated_grad_odd;
		}
	}
}



extern "C" __global__ void default_rope_bwd_x_fp16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int *seq_positions, __half *dX_q, __half *dX_k) {
	
	// Each thread handles two consecutive half elements.
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N) {
		// ------------------------
		// Process the query branch
		// ------------------------
		// Compute token row and corresponding position/dimension.
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// Compute the angle for the current dimension.
		float angle = powf(theta, -1.0f * ((float)cur_dim / head_dim));
		float cos_val = cosf(cur_pos * angle);
		float sin_val = sinf(cur_pos * angle);

		// Load the upstream gradients from dX_q.
		float grad_even = __half2float(dX_q[i]);
		float grad_odd  = __half2float(dX_q[i + 1]);

		// Apply the transpose of the rotation matrix.
		float updated_grad_even = cos_val * grad_even + sin_val * grad_odd;
		float updated_grad_odd  = -sin_val * grad_even + cos_val * grad_odd;

		// Update the gradients in place.
		dX_q[i]     = __float2half(updated_grad_even);
		dX_q[i + 1] = __float2half(updated_grad_odd);

		// ------------------------
		// Process the key branch
		// ------------------------
		int kv_dim = num_kv_heads * head_dim;
		// Adjust token indexing for key tensor layout.
		token_row = i / kv_dim;
		int total_tokens = N / model_dim;
		if (token_row < total_tokens) {
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;
			angle = powf(theta, -1.0f * ((float)cur_dim / head_dim));
			cos_val = cosf(cur_pos * angle);
			sin_val = sinf(cur_pos * angle);

			grad_even = __half2float(dX_k[i]);
			grad_odd  = __half2float(dX_k[i + 1]);

			updated_grad_even = cos_val * grad_even + sin_val * grad_odd;
			updated_grad_odd  = -sin_val * grad_even + cos_val * grad_odd;

			dX_k[i]     = __float2half(updated_grad_even);
			dX_k[i + 1] = __float2half(updated_grad_odd);
		}
	}
}


extern "C" __global__ void default_rope_bwd_x_bf16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int *seq_positions, __nv_bfloat16 *dX_q, __nv_bfloat16 *dX_k) {
	
	// Each thread handles two consecutive half elements.
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N) {
		// ------------------------
		// Process the query branch
		// ------------------------
		// Compute token row and corresponding position/dimension.
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// Compute the angle for the current dimension.
		float angle = powf(theta, -1.0f * ((float)cur_dim / head_dim));
		float cos_val = cosf(cur_pos * angle);
		float sin_val = sinf(cur_pos * angle);

		// Load the upstream gradients from dX_q.
		float grad_even = __bfloat162float(dX_q[i]);
		float grad_odd  = __bfloat162float(dX_q[i + 1]);

		// Apply the transpose of the rotation matrix.
		float updated_grad_even = cos_val * grad_even + sin_val * grad_odd;
		float updated_grad_odd  = -sin_val * grad_even + cos_val * grad_odd;

		// Update the gradients in place.
		dX_q[i]     = __float2bfloat16(updated_grad_even);
		dX_q[i + 1] = __float2bfloat16(updated_grad_odd);

		// ------------------------
		// Process the key branch
		// ------------------------
		int kv_dim = num_kv_heads * head_dim;
		// Adjust token indexing for key tensor layout.
		token_row = i / kv_dim;
		int total_tokens = N / model_dim;
		if (token_row < total_tokens) {
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;
			angle = powf(theta, -1.0f * ((float)cur_dim / head_dim));
			cos_val = cosf(cur_pos * angle);
			sin_val = sinf(cur_pos * angle);

			grad_even = __bfloat162float(dX_k[i]);
			grad_odd  = __bfloat162float(dX_k[i + 1]);

			updated_grad_even = cos_val * grad_even + sin_val * grad_odd;
			updated_grad_odd  = -sin_val * grad_even + cos_val * grad_odd;

			dX_k[i]     = __float2bfloat16(updated_grad_even);
			dX_k[i + 1] = __float2bfloat16(updated_grad_odd);
		}
	}
}
