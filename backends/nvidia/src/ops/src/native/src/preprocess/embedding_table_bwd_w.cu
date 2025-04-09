#include "nvidia_ops.h"

extern "C" __global__ void default_embedding_table_bwd_w_fp32_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, float * grad_stream, float * grad_embedding_table) {

     // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    float * grad_agg = (float *) (sdata);

    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

   

    // read embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        grad_agg[i] = 0.0f;
    }

    __syncthreads();

    // now accumulate gradients for all rows that correspond to this unique toke
    int grad_row_id;
    for (int i = 0; i < num_repeated_tokens; i++) {
        grad_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x) {
            grad_agg[j] += grad_stream[grad_row_id * embed_dim + j];
        }
    }

    __syncthreads();
    
    // now write back to grad embedding table
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        grad_embedding_table[token_id * embed_dim + i] += grad_agg[i];
    }

    return;

}


extern "C" __global__ void default_embedding_table_bwd_w_fp16_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, __half * grad_stream, __half * grad_embedding_table) {

    // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    __half * grad_agg = (__half *) (sdata);
    
    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

   

    // read embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        grad_agg[i] = CONST_ZERO_DEV_FP16;
    }

    __syncthreads();

    // now accumulate gradients for all rows that correspond to this unique toke
    int grad_row_id;
    for (int i = 0; i < num_repeated_tokens; i++) {
        grad_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x) {
            grad_agg[j] += grad_stream[grad_row_id * embed_dim + j];
        }
    }

    __syncthreads();
    
    // now write back to grad embedding table
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        grad_embedding_table[token_id * embed_dim + i] += grad_agg[i];
    }

    return;
}

extern "C" __global__ void default_embedding_table_bwd_w_bf16_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, __nv_bfloat16 * grad_stream, __nv_bfloat16 * grad_embedding_table) {

    // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    __nv_bfloat16 * grad_agg = (__nv_bfloat16 *) (sdata);
    
    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

   

    // read embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        grad_agg[i] = CONST_ZERO_DEV_BF16;
    }

    __syncthreads();

    // now accumulate gradients for all rows that correspond to this unique toke
    int grad_row_id;
    for (int i = 0; i < num_repeated_tokens; i++) {
        grad_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x) {
            grad_agg[j] += grad_stream[grad_row_id * embed_dim + j];
        }
    }

    __syncthreads();
    
    // now write back to grad embedding table
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        grad_embedding_table[token_id * embed_dim + i] += grad_agg[i];
    }

    return;
  
}
