#include "nvidia_ops.h"

// extern "C" __global__ void default_cast_bf16_fp32_kernel(uint64_t num_elements, const __nv_bfloat16 * src, float * dst){
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= num_elements) return;

//     dst[tid] = __bfloat162float(src[tid]);
// }


extern "C" __global__ void default_cast_bf16_fp32_kernel (uint64_t num_elements, __nv_bfloat16 * src, float * dst) 
{
    // Vectorization factor (4 elements per thread)
    const int VEC_SIZE = 4;

    // Global thread ID and total number of threads in the grid
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // ====== 1. Vectorized Main Loop ======
    uint64_t num_vec_elements = num_elements / VEC_SIZE;

    // Use C-style casts instead of reinterpret_cast
    const __nv_bfloat162* src_v2 = (const __nv_bfloat162*)src;
    float4* dst_v4 = (float4*)dst;
    
    // Grid-stride loop for vectorized part
    for (uint64_t i = tid; i < num_vec_elements; i += stride) {
        // Load four bfloat16s (as two __nv_bfloat162)
        __nv_bfloat162 b2_0 = src_v2[i * 2];
        __nv_bfloat162 b2_1 = src_v2[i * 2 + 1];

        // Convert to two float2s
        float2 f2_0 = __bfloat1622float2(b2_0);
        float2 f2_1 = __bfloat1622float2(b2_1);

        // Combine and store as one float4
        dst_v4[i] = make_float4(f2_0.x, f2_0.y, f2_1.x, f2_1.y);
    }

    // ====== 2. Scalar Cleanup Loop ======
    uint64_t remainder_start_idx = num_vec_elements * VEC_SIZE;

    // Grid-stride loop for scalar remainder
    for (uint64_t i = remainder_start_idx + tid; i < num_elements; i += stride) {
        dst[i] = __bfloat162float(src[i]);
    }
}


// extern "C" __global__ void default_cast_and_add_fp32_bf16_bf16_kernel(uint64_t num_elements, float alpha, float * A, float beta, __nv_bfloat16 * B, __nv_bfloat16 * C){
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= num_elements) return;

//     __nv_bfloat16 beta_bf16 = __float2bfloat16(beta);

//     C[tid] = __float2bfloat16(alpha * A[tid]) + beta_bf16 * B[tid];
// }


extern "C" __global__ void default_cast_and_add_fp32_bf16_bf16_kernel(int64_t num_elements, float alpha, float * A, float beta, __nv_bfloat16 * B, __nv_bfloat16 * C)
{
    // Vectorization factor (4 elements per thread)
    const int VEC_SIZE = 4;

    // Global thread ID and grid stride
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // ====== 1. Vectorized Main Loop ======
    uint64_t num_vec_elements = num_elements / VEC_SIZE;

    // Cast pointers for vectorized access
    const float4* a_v4 = (const float4*)A;
    const __nv_bfloat162* b_v2 = (const __nv_bfloat162*)B;
    __nv_bfloat162* c_v2 = (__nv_bfloat162*)C;

    // Pre-convert scalar beta to a bfloat16 vector
    __nv_bfloat16 beta_bf16_scalar = __float2bfloat16(beta);
    __nv_bfloat162 beta_b2 = __bfloat162bfloat162(beta_bf16_scalar);

    // Grid-stride loop processes VEC_SIZE elements per iteration
    for (uint64_t i = tid; i < num_vec_elements; i += stride) {
        // Load 4 elements from A (as float4) and B (as two __nv_bfloat162)
        float4 a_val = a_v4[i];
        __nv_bfloat162 b_val_0 = b_v2[i * 2];
        __nv_bfloat162 b_val_1 = b_v2[i * 2 + 1];

        // --- Process the first pair of elements (0, 1) ---
        float2 a_f2_0 = make_float2(a_val.x, a_val.y);
        float2 term1_f2_0 = make_float2(alpha * a_f2_0.x, alpha * a_f2_0.y);
        __nv_bfloat162 term1_b2_0 = __float22bfloat162_rn(term1_f2_0);
        __nv_bfloat162 term2_b2_0 = __hmul2(beta_b2, b_val_0);
        __nv_bfloat162 c_val_0 = __hadd2(term1_b2_0, term2_b2_0);

        // --- Process the second pair of elements (2, 3) ---
        float2 a_f2_1 = make_float2(a_val.z, a_val.w);
        float2 term1_f2_1 = make_float2(alpha * a_f2_1.x, alpha * a_f2_1.y);
        __nv_bfloat162 term1_b2_1 = __float22bfloat162_rn(term1_f2_1);
        __nv_bfloat162 term2_b2_1 = __hmul2(beta_b2, b_val_1);
        __nv_bfloat162 c_val_1 = __hadd2(term1_b2_1, term2_b2_1);
        
        // Store the 4 resulting elements (as two __nv_bfloat162)
        c_v2[i * 2] = c_val_0;
        c_v2[i * 2 + 1] = c_val_1;
    }

    // ====== 2. Scalar Cleanup Loop ======
    uint64_t remainder_start_idx = num_vec_elements * VEC_SIZE;

    // Grid-stride loop for remaining elements (up to 3)
    for (uint64_t i = remainder_start_idx + tid; i < num_elements; i += stride) {
        C[i] = __float2bfloat16(alpha * A[i]) + beta_bf16_scalar * B[i];
    }
}