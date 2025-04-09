#include "nvidia_ops.h"


extern "C" __global__ void default_embedding_table_fp32_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, float * embedding_table, float * output) {

     // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    float * embed_row = (float *) (sdata);

    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

    // load in embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x){
        embed_row[i] = embedding_table[token_id * embed_dim + i];
    }

    __syncthreads();

    // now store the embedding table row into output
    int out_row_id;
    for (int i = 0; i < num_repeated_tokens; i++){
        out_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x){
            output[out_row_id * embed_dim + j] = embed_row[j];
        }
    }

    return;
}


extern "C" __global__ void default_embedding_table_fp16_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, __half * embedding_table, __half * output){

    // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    __half * embed_row = (__half *) (sdata);

    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

    // load in embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x){
        embed_row[i] = embedding_table[token_id * embed_dim + i];
    }

    __syncthreads();

    // now store the embedding table row into output
    for (int i = 0; i < num_repeated_tokens; i++){
        int out_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x){
            output[out_row_id * embed_dim + j] = embed_row[j];
        }
    }

    return;

}

extern "C" __global__ void default_embedding_table_bf16_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, __nv_bfloat16 * embedding_table, __nv_bfloat16 * output){

    // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    __nv_bfloat16 * embed_row = (__nv_bfloat16 *) (sdata);

    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

    // load in embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x){
        embed_row[i] = embedding_table[token_id * embed_dim + i];
    }

    __syncthreads();

    // now store the embedding table row into output
    for (int i = 0; i < num_repeated_tokens; i++){
        int out_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x){
            output[out_row_id * embed_dim + j] = embed_row[j];
        }
    }

    return;
}

extern "C" __global__ void default_embedding_table_fp8e4m3_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, __nv_fp8_e4m3  * embedding_table, __nv_fp8_e4m3  * output){

     // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    __nv_fp8_e4m3 * embed_row = (__nv_fp8_e4m3 *) (sdata);

    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

    // load in embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x){
        embed_row[i] = embedding_table[token_id * embed_dim + i];
    }

    __syncthreads();

    // now store the embedding table row into output
    for (int i = 0; i < num_repeated_tokens; i++){
        int out_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x){
            output[out_row_id * embed_dim + j] = embed_row[j];
        }
    }

    return;

}

extern "C" __global__ void default_embedding_table_fp8e5m2_kernel(int num_unique_tokens, int embed_dim, uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, __nv_fp8_e5m2  * embedding_table, __nv_fp8_e5m2  * output){

     // this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

    __nv_fp8_e5m2 * embed_row = (__nv_fp8_e5m2 *) (sdata);

    int unique_token_num = blockIdx.x;

    if (unique_token_num >= num_unique_tokens){
        return;
    }

    int unique_token_start = unique_token_sorted_inds_start[unique_token_num];

    int num_repeated_tokens = unique_token_sorted_inds_start[unique_token_num + 1] - unique_token_start;

    int token_id = sorted_token_ids[unique_token_start];

    // load in embedding table row into shared memory
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x){
        embed_row[i] = embedding_table[token_id * embed_dim + i];
    }

    __syncthreads();

    // now store the embedding table row into output
    for (int i = 0; i < num_repeated_tokens; i++){
        int out_row_id = sorted_token_mapping[unique_token_start + i];
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x){
            output[out_row_id * embed_dim + j] = embed_row[j];
        }
    }

    return;
}
