#include "nvidia_ops.h"


extern "C" __global__ void default_swiglu_fp32_kernel(int num_rows, int num_cols, float * x_w1, float * x_w3, float * out){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float silu_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = x_w1[row_num * num_cols + d];
                        x_w3_val = x_w3[row_num * num_cols + d];

                        // overwrite contents in x_w1
                        silu_x_w1 = x_w1_val / (1 + expf(-1 * x_w1_val));

                        // normally would set out to be x_w1...
                        out[row_num * num_cols + d] = silu_x_w1 * x_w3_val;
                }
        }
}

extern "C" __global__ void default_swiglu_fp16_kernel(int num_rows, int num_cols, __half * x_w1, __half * x_w3, __half * out){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float silu_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = __half2float(x_w1[row_num * num_cols + d]);
                        x_w3_val = __half2float(x_w3[row_num * num_cols + d]);

                        // overwrite contents in x_w1
                        silu_x_w1 = x_w1_val / (1 + expf(-1 * x_w1_val));

                        // normally would set out to be x_w1...
                        out[row_num * num_cols + d] = __float2half(silu_x_w1 * x_w3_val);
                }
        }
}

/**
 * Highly optimized SwiGLU activation kernel for bfloat16 data using wide, 64-bit loads.
 *
 * This version uses C-style casts instead of C++ reinterpret_cast for pointer type conversion.
 * The underlying logic and performance are identical to the C++ version.
 * It reinterprets the bfloat16 pointers to perform 64-bit memory transactions, fetching
 * four bfloat16 values per load. This maximizes memory bandwidth utilization.
 *
 * @param num_rows The number of rows in the input tensors.
 * @param num_cols The number of columns in the input tensors. This MUST be divisible by 4.
 * @param x_w1 Pointer to the first input tensor (global memory).
 * @param x_w3 Pointer to the second input tensor (global memory).
 * @param out Pointer to the output tensor (global memory).
 */
 extern "C" __global__ void default_swiglu_bf16_kernel(int num_rows, int num_cols, const __nv_bfloat16 * __restrict__ x_w1, const __nv_bfloat16 * __restrict__ x_w3, __nv_bfloat16 * __restrict__ out) {
        
        // This kernel requires the column dimension to be divisible by 4 to use
        // 64-bit (float2) vectorized loads.
        const int num_cols_vec = num_cols / 4;
    
        // Reinterpret the bfloat16 pointers as float2 pointers using C-style casts.
        // sizeof(float2) is 8 bytes, so this allows us to load/store four
        // bfloat16 values (4 * 2 = 8 bytes) in a single instruction.
        const float2 *x_w1_vec = (const float2*)(x_w1);
        const float2 *x_w3_vec = (const float2*)(x_w3);
        float2 *out_vec = (float2*)(out);
    
        // Each block processes one row.
        const int row_num = blockIdx.x;
    
        // Early exit for blocks outside the valid row range.
        if (row_num >= num_rows) {
            return;
        }
    
        // Grid-stride loop. Each thread processes multiple sets of 4 elements.
        for (int i = threadIdx.x; i < num_cols_vec; i += blockDim.x) {
            // Calculate the linear index for the 64-bit vectorized access.
            const int idx = row_num * num_cols_vec + i;
    
            // --- Step 1: Vectorized 64-bit Load ---
            // Perform a single 64-bit load to get four bfloat16 values for each input.
            const float2 x_w1_packed = x_w1_vec[idx];
            const float2 x_w3_packed = x_w3_vec[idx];
    
            // --- Step 2: Unpack and Compute ---
            // To compute, we must unpack the four bfloat16 values into floats.
            // We can reinterpret our loaded float2 (8 bytes) as an array of two
            // __nv_bfloat162s (2 * 4 bytes) using C-style casts.
            const __nv_bfloat162* x_w1_b162s = (const __nv_bfloat162*)(&x_w1_packed);
            const __nv_bfloat162* x_w3_b162s = (const __nv_bfloat162*)(&x_w3_packed);
    
            // Convert the two bfloat162s into two float2s for each input.
            const float2 x_w1_f2_0 = __bfloat1622float2(x_w1_b162s[0]); // Elements 0, 1
            const float2 x_w1_f2_1 = __bfloat1622float2(x_w1_b162s[1]); // Elements 2, 3
    
            const float2 x_w3_f2_0 = __bfloat1622float2(x_w3_b162s[0]);
            const float2 x_w3_f2_1 = __bfloat1622float2(x_w3_b162s[1]);
    
            // Apply SiLU and element-wise multiply for all four elements.
            const float silu_0 = x_w1_f2_0.x / (1.0f + __expf(-x_w1_f2_0.x));
            const float silu_1 = x_w1_f2_0.y / (1.0f + __expf(-x_w1_f2_0.y));
            const float silu_2 = x_w1_f2_1.x / (1.0f + __expf(-x_w1_f2_1.x));
            const float silu_3 = x_w1_f2_1.y / (1.0f + __expf(-x_w1_f2_1.y));
    
            const float out_0 = silu_0 * x_w3_f2_0.x;
            const float out_1 = silu_1 * x_w3_f2_0.y;
            const float out_2 = silu_2 * x_w3_f2_1.x;
            const float out_3 = silu_3 * x_w3_f2_1.y;
    
            // --- Step 3: Pack and Vectorized 64-bit Store ---
            // Convert the four float results back into two bfloat162s.
            const __nv_bfloat162 out_b162_0 = __float22bfloat162_rn(make_float2(out_0, out_1));
            const __nv_bfloat162 out_b162_1 = __float22bfloat162_rn(make_float2(out_2, out_3));
    
            // Pack the two bfloat162s back into a single float2 for the 64-bit store.
            float2 out_packed;
            ((__nv_bfloat162*)(&out_packed))[0] = out_b162_0;
            ((__nv_bfloat162*)(&out_packed))[1] = out_b162_1;
    
            // Perform a single 64-bit store to write four bfloat16 results.
            out_vec[idx] = out_packed;
        }
}

extern "C" __global__ void naive_default_swiglu_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * out){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float silu_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = __bfloat162float(x_w1[row_num * num_cols + d]);
                        x_w3_val = __bfloat162float(x_w3[row_num * num_cols + d]);

                        // overwrite contents in x_w1
                        silu_x_w1 = x_w1_val / (1 + expf(-1 * x_w1_val));

                        // normally would set out to be x_w1...
                        out[row_num * num_cols + d] = __float2bfloat16(silu_x_w1 * x_w3_val);
                }
        }
}

extern "C" __global__ void default_swiglu_fp8e4m3_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __nv_fp8_e4m3 * out){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float silu_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);

                        // overwrite contents in x_w1
                        silu_x_w1 = x_w1_val / (1 + expf(-1 * x_w1_val));

                        // normally would set out to be x_w1...
                        out[row_num * num_cols + d] = __nv_fp8_e4m3(silu_x_w1 * x_w3_val);
                }
        }
}

extern "C" __global__ void default_swiglu_fp8e5m2_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __nv_fp8_e5m2 * out){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float silu_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);

                        // overwrite contents in x_w1
                        silu_x_w1 = x_w1_val / (1 + expf(-1 * x_w1_val));

                        // normally would set out to be x_w1...
                        out[row_num * num_cols + d] = __nv_fp8_e5m2(silu_x_w1 * x_w3_val);
                }
        }
}


