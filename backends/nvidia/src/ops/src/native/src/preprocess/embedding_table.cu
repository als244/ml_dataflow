#include "nvidia_ops.h"

extern "C" __global__ void default_embedding_table_fp32_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, float* embedding_table, float * output){

    int token_num = blockIdx.x;

    if (token_num >= nums_tokens){
        return;
    }

    uint64_t dtype_size = sizeof(float);

    uint64_t token_id = (uint64_t) token_ids[token_num];

    uint64_t embedding_table_offset = token_id * embed_dim * dtype_size;
    
    uint64_t output_offset = token_num * embed_dim * dtype_size;
    
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x){
        output[output_offset + d] = embedding_table[embedding_table_offset + d];
    }

    return;
}


extern "C" __global__ void default_embedding_table_fp16_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __half * embedding_table, __half * output){

    int token_num = blockIdx.x;

    if (token_num >= nums_tokens){
        return;
    }   

    uint64_t dtype_size = sizeof(__half);

    uint64_t token_id = (uint64_t) token_ids[token_num];

    uint64_t embedding_table_offset = token_id * embed_dim * dtype_size;    
    
    uint64_t output_offset = token_num * embed_dim * dtype_size;

    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x){
        output[output_offset + d] = embedding_table[embedding_table_offset + d];
    }

    return;

}

extern "C" __global__ void default_embedding_table_bf16_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __nv_bfloat16 * embedding_table, __nv_bfloat16 * output){

    int token_num = blockIdx.x;

    if (token_num >= nums_tokens){
        return;
    }
    
    uint64_t dtype_size = sizeof(__nv_bfloat16);

    uint64_t token_id = (uint64_t) token_ids[token_num];

    uint64_t embedding_table_offset = token_id * embed_dim * dtype_size;

    uint64_t output_offset = token_num * embed_dim * dtype_size;

    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x){
        output[output_offset + d] = embedding_table[embedding_table_offset + d];
    }

    return;
}

extern "C" __global__ void default_embedding_table_fp8e4m3_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __nv_fp8_e4m3  * embedding_table, __nv_fp8_e4m3  * output){

    int token_num = blockIdx.x;

    if (token_num >= nums_tokens){
        return;
    }
    
    uint64_t dtype_size = sizeof(__nv_fp8_e4m3);

    uint64_t token_id = (uint64_t) token_ids[token_num];

    uint64_t embedding_table_offset = token_id * embed_dim * dtype_size;

    uint64_t output_offset = token_num * embed_dim * dtype_size;
    
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x){
        output[output_offset + d] = embedding_table[embedding_table_offset + d];
    }

    return;

}

extern "C" __global__ void default_embedding_table_fp8e5m2_kernel(int nums_tokens, int embed_dim, uint32_t * token_ids, __nv_fp8_e5m2  * embedding_table, __nv_fp8_e5m2  * output){

    int token_num = blockIdx.x; 

    if (token_num >= nums_tokens){
        return;
    }

    uint64_t dtype_size = sizeof(__nv_fp8_e5m2);

    uint64_t token_id = (uint64_t) token_ids[token_num];

    uint64_t embedding_table_offset = token_id * embed_dim * dtype_size;

    uint64_t output_offset = token_num * embed_dim * dtype_size;
    
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x){
        output[output_offset + d] = embedding_table[embedding_table_offset + d];
    }

    return;
}
