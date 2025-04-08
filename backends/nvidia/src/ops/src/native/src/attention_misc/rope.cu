#include "nvidia_ops.h"

extern "C" __global__ void default_rope_fp32_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, float * X_q, float * X_k) {

	// launched with half the number of threads as output positions because each thread updates two spots
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	// N = total_tokens * model_dim
	if (i < N){
		// ASSUMING model_dim > kv_dim

		// If performance is issue with this divides and modulus
		// we could either use bit tricks or hard-code constants...
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// probably faster (& simpler) to just use arithmetic functions and recompute
		// instead of loading in from global device memory
		float angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
		float cos_val = cosf((float) cur_pos * angle);
		float sin_val = sinf((float) cur_pos * angle);

		float x_even, x_odd; 

		// first do X_q
		x_even = X_q[i];
		x_odd = X_q[i + 1];
		X_q[i] = cos_val * x_even - sin_val * x_odd;
		X_q[i + 1] = cos_val * x_odd + sin_val * x_even;

		// Now reassign this thread to update x_k
		int kv_dim = num_kv_heads * head_dim;
		token_row = i / (kv_dim);
		int total_tokens = N / model_dim;
		if (token_row < total_tokens){
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;

			angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
			cos_val = cosf((float) cur_pos * angle);
			sin_val = sinf((float) cur_pos * angle);

			// now do X_k in same manner but obtaining different x vals
			x_even = X_k[i];
			x_odd = X_k[i + 1];
			X_k[i] = cos_val * x_even - sin_val * x_odd;
			X_k[i + 1] = cos_val * x_odd + sin_val * x_even;

			// Optimization: Could store in kv cache when already in register instead of 
			// reloading again within kv cache kernel.

			// But the cost of these kernels is minimal compared to the matmuls and attention,
			// so not that big a deal and cleaner to seperate.
		}
	}
}

// THIS COULD REALLY BE PART OF ATTN KERNEL...
extern "C" __global__ void default_rope_fp16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __half * X_q, __half * X_k) {

	// launched with half the number of threads as output positions because each thread updates two spots
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	// N = total_tokens * model_dim
	if (i < N){



		// ASSUMING model_dim > kv_dim

		// If performance is issue with this divides and modulus
		// we could either use bit tricks or hard-code constants...
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// probably faster (& simpler) to just use arithmetic functions and recompute
		// instead of loading in from global device memory
		float angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
		float cos_val = cosf((float) cur_pos * angle);
		float sin_val = sinf((float) cur_pos * angle);

		float x_even, x_odd; 

		// first do X_q
		x_even = __half2float(X_q[i]);
		x_odd = __half2float(X_q[i + 1]);
		X_q[i] = __float2half(cos_val * x_even - sin_val * x_odd);
		X_q[i + 1] = __float2half(cos_val * x_odd + sin_val * x_even);

		// Now reassign this thread to update x_k
		int kv_dim = num_kv_heads * head_dim;
		token_row = i / (kv_dim);
		int total_tokens = N / model_dim;
		if (token_row < total_tokens){
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;

			angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
			cos_val = cosf((float) cur_pos * angle);
			sin_val = sinf((float) cur_pos * angle);

			// now do X_k in same manner but obtaining different x vals
			x_even = __half2float(X_k[i]);
			x_odd = __half2float(X_k[i + 1]);
			X_k[i] = __float2half(cos_val * x_even - sin_val * x_odd);
			X_k[i + 1] = __float2half(cos_val * x_odd + sin_val * x_even);

			// Optimization: Could store in kv cache when already in register instead of 
			// reloading again within kv cache kernel.

			// But the cost of these kernels is minimal compared to the matmuls and attention,
			// so not that big a deal and cleaner to seperate.
		}
	}
}




// THIS COULD REALLY BE PART OF ATTN KERNEL...
extern "C" __global__ void default_rope_bf16_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_bfloat16 * X_q, __nv_bfloat16 * X_k) {

	// launched with half the number of threads as output positions because each thread updates two spots
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	// N = total_tokens * model_dim
	if (i < N){



		// ASSUMING model_dim > kv_dim

		// If performance is issue with this divides and modulus
		// we could either use bit tricks or hard-code constants...
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// probably faster (& simpler) to just use arithmetic functions and recompute
		// instead of loading in from global device memory
		float angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
		float cos_val = cosf((float) cur_pos * angle);
		float sin_val = sinf((float) cur_pos * angle);

		float x_even, x_odd; 

		// first do X_q
		x_even = __bfloat162float(X_q[i]);
		x_odd = __bfloat162float(X_q[i + 1]);
		X_q[i] = __float2bfloat16(cos_val * x_even - sin_val * x_odd);
		X_q[i + 1] = __float2bfloat16(cos_val * x_odd + sin_val * x_even);

		// Now reassign this thread to update x_k
		int kv_dim = num_kv_heads * head_dim;
		token_row = i / (kv_dim);
		int total_tokens = N / model_dim;
		if (token_row < total_tokens){
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;

			angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
			cos_val = cosf((float) cur_pos * angle);
			sin_val = sinf((float) cur_pos * angle);

			// now do X_k in same manner but obtaining different x vals
			x_even = __bfloat162float(X_k[i]);
			x_odd = __bfloat162float(X_k[i + 1]);
			X_k[i] = __float2bfloat16(cos_val * x_even - sin_val * x_odd);
			X_k[i + 1] = __float2bfloat16(cos_val * x_odd + sin_val * x_even);

			// Optimization: Could store in kv cache when already in register instead of 
			// reloading again within kv cache kernel.

			// But the cost of these kernels is minimal compared to the matmuls and attention,
			// so not that big a deal and cleaner to seperate.
		}
	}
}


extern "C" __global__ void default_rope_fp8e4m3_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e4m3 * X_q, __nv_fp8_e4m3 * X_k) {

	// launched with half the number of threads as output positions because each thread updates two spots
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	// N = total_tokens * model_dim
	if (i < N){



		// ASSUMING model_dim > kv_dim

		// If performance is issue with this divides and modulus
		// we could either use bit tricks or hard-code constants...
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// probably faster (& simpler) to just use arithmetic functions and recompute
		// instead of loading in from global device memory
		float angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
		float cos_val = cosf((float) cur_pos * angle);
		float sin_val = sinf((float) cur_pos * angle);

		float x_even, x_odd; 

		// first do X_q
		x_even = float(X_q[i]);
		x_odd = float(X_q[i + 1]);
		X_q[i] = __nv_fp8_e4m3(cos_val * x_even - sin_val * x_odd);
		X_q[i + 1] = __nv_fp8_e4m3(cos_val * x_odd + sin_val * x_even);

		// Now reassign this thread to update x_k
		int kv_dim = num_kv_heads * head_dim;
		token_row = i / (kv_dim);
		int total_tokens = N / model_dim;
		if (token_row < total_tokens){
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;

			angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
			cos_val = cosf((float) cur_pos * angle);
			sin_val = sinf((float) cur_pos * angle);

			// now do X_k in same manner but obtaining different x vals
			x_even = float(X_k[i]);
			x_odd = float(X_k[i + 1]);
			X_k[i] = __nv_fp8_e4m3(cos_val * x_even - sin_val * x_odd);
			X_k[i + 1] = __nv_fp8_e4m3(cos_val * x_odd + sin_val * x_even);

			// Optimization: Could store in kv cache when already in register instead of 
			// reloading again within kv cache kernel.

			// But the cost of these kernels is minimal compared to the matmuls and attention,
			// so not that big a deal and cleaner to seperate.
		}
	}
}


extern "C" __global__ void default_rope_fp8e5m2_kernel(uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta, int * seq_positions, __nv_fp8_e5m2 * X_q, __nv_fp8_e5m2 * X_k) {

	// launched with half the number of threads as output positions because each thread updates two spots
	uint64_t i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

	// N = total_tokens * model_dim
	if (i < N){



		// ASSUMING model_dim > kv_dim

		// If performance is issue with this divides and modulus
		// we could either use bit tricks or hard-code constants...
		int token_row = i / model_dim;
		int cur_pos = seq_positions[token_row];
		int cur_dim = i % head_dim;

		// probably faster (& simpler) to just use arithmetic functions and recompute
		// instead of loading in from global device memory
		float angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
		float cos_val = cosf((float) cur_pos * angle);
		float sin_val = sinf((float) cur_pos * angle);

		float x_even, x_odd; 

		// first do X_q
		x_even = float(X_q[i]);
		x_odd = float(X_q[i + 1]);
		X_q[i] = __nv_fp8_e5m2(cos_val * x_even - sin_val * x_odd);
		X_q[i + 1] = __nv_fp8_e5m2(cos_val * x_odd + sin_val * x_even);

		// Now reassign this thread to update x_k
		int kv_dim = num_kv_heads * head_dim;
		token_row = i / (kv_dim);
		int total_tokens = N / model_dim;
		if (token_row < total_tokens){
			cur_pos = seq_positions[token_row];
			cur_dim = i % head_dim;

			angle = powf(theta, -1 * ((float) cur_dim / (float) head_dim));
			cos_val = cosf((float) cur_pos * angle);
			sin_val = sinf((float) cur_pos * angle);

			// now do X_k in same manner but obtaining different x vals
			x_even = float(X_k[i]);
			x_odd = float(X_k[i + 1]);
			X_k[i] = __nv_fp8_e5m2(cos_val * x_even - sin_val * x_odd);
			X_k[i + 1] = __nv_fp8_e5m2(cos_val * x_odd + sin_val * x_even);

			// Optimization: Could store in kv cache when already in register instead of 
			// reloading again within kv cache kernel.

			// But the cost of these kernels is minimal compared to the matmuls and attention,
			// so not that big a deal and cleaner to seperate.
		}
	}
}
