#include "nvidia_ops.h"

extern "C" __global__ void copy_to_seq_context_fp32_kernel(uint64_t N, int total_tokens, int kv_dim, float * keys, float * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N){

		uint64_t token_ind = i / kv_dim;

		float * seq_context = (float *) seq_context_ptrs[token_ind];

		uint64_t seq_pos = seq_positions[token_ind];

		uint64_t cur_dim = i % kv_dim;
		
		seq_context[seq_pos * kv_dim + cur_dim] = keys[token_ind * kv_dim + cur_dim];
		seq_context[(seq_context_sizes[token_ind] * kv_dim) + seq_pos * kv_dim + cur_dim] = values[token_ind * kv_dim + cur_dim];
	}
}


// N = total_tokens * kv_dim
extern "C" __global__ void copy_to_seq_context_fp16_kernel(uint64_t N, int total_tokens, int kv_dim, __half * keys, __half * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N){

		uint64_t token_ind = i / kv_dim;

		__half * seq_context = (__half *) seq_context_ptrs[token_ind];

		uint64_t seq_pos = seq_positions[token_ind];

		uint64_t cur_dim = i % kv_dim;
		
		seq_context[seq_pos * kv_dim + cur_dim] = keys[token_ind * kv_dim + cur_dim];
		seq_context[(seq_context_sizes[token_ind] * kv_dim) + seq_pos * kv_dim + cur_dim] = values[token_ind * kv_dim + cur_dim];
	}
}


extern "C" __global__ void copy_to_seq_context_bf16_kernel(uint64_t N, int total_tokens, int kv_dim, __nv_bfloat16 * keys, __nv_bfloat16 * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N){

		uint64_t token_ind = i / kv_dim;

		__nv_bfloat16 * seq_context = (__nv_bfloat16 *) seq_context_ptrs[token_ind];

		uint64_t seq_pos = seq_positions[token_ind];

		uint64_t cur_dim = i % kv_dim;
		
		seq_context[seq_pos * kv_dim + cur_dim] = keys[token_ind * kv_dim + cur_dim];
		seq_context[(seq_context_sizes[token_ind] * kv_dim) + seq_pos * kv_dim + cur_dim] = values[token_ind * kv_dim + cur_dim];
	}
}


extern "C" __global__ void copy_to_seq_context_fp8e4m3_kernel(uint64_t N, int total_tokens, int kv_dim, __nv_fp8_e4m3 * keys, __nv_fp8_e4m3 * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N){

		uint64_t token_ind = i / kv_dim;

		__nv_fp8_e4m3 * seq_context = (__nv_fp8_e4m3 *) seq_context_ptrs[token_ind];

		uint64_t seq_pos = seq_positions[token_ind];

		uint64_t cur_dim = i % kv_dim;
		
		seq_context[seq_pos * kv_dim + cur_dim] = keys[token_ind * kv_dim + cur_dim];
		seq_context[(seq_context_sizes[token_ind] * kv_dim) + seq_pos * kv_dim + cur_dim] = values[token_ind * kv_dim + cur_dim];
	}
}


extern "C" __global__ void copy_to_seq_context_fp8e5m2_kernel(uint64_t N, int total_tokens, int kv_dim, __nv_fp8_e5m2 * keys, __nv_fp8_e5m2 * values, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes){

	uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

	if (i < N){

		uint64_t token_ind = i / kv_dim;

		__nv_fp8_e5m2 * seq_context = (__nv_fp8_e5m2 *) seq_context_ptrs[token_ind];

		uint64_t seq_pos = seq_positions[token_ind];

		uint64_t cur_dim = i % kv_dim;
		
		seq_context[seq_pos * kv_dim + cur_dim] = keys[token_ind * kv_dim + cur_dim];
		seq_context[(seq_context_sizes[token_ind] * kv_dim) + seq_pos * kv_dim + cur_dim] = values[token_ind * kv_dim + cur_dim];
	}
}