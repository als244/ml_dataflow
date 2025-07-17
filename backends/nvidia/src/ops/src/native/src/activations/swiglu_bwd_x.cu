#include "nvidia_ops.h"


extern "C" __global__ void default_swiglu_bwd_x_fp32_fp32_kernel(int num_rows, int num_cols, float * x_w1, float * x_w3, float * upstream_dX, float * dX_w1, float * dX_w3) {

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = x_w1[row_num * num_cols + d];
                        x_w3_val = x_w3[row_num * num_cols + d];
                        upstream_deriv = upstream_dX[row_num * num_cols + d];

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = upstream_deriv * silu_x1;

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = upstream_deriv * x_w3_val * dsilu_dx1;
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp16_fp16_kernel(int num_rows, int num_cols, __half * x_w1, __half * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3) {

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = __half2float(x_w1[row_num * num_cols + d]);
                        x_w3_val = __half2float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __half2float(upstream_dX[row_num * num_cols + d]);

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = __float2half(upstream_deriv * silu_x1);

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = __float2half(upstream_deriv * x_w3_val * dsilu_dx1);
                }
        }
}

/**
 * Highly optimized backward pass for the SwiGLU activation function using
 * bfloat16 data and wide, 128-bit vectorized memory operations.
 *
 * This kernel calculates the gradients with respect to the inputs x_w1 and x_w3.
 * It uses float4 vectorized loads and stores to process eight bfloat16 elements
 * per thread per iteration, maximizing memory bandwidth.
 *
 * All pointer arguments are marked as __restrict__ under the assumption that
 * input, output, and upstream gradient tensors are all distinct.
 *
 * @param num_rows The number of rows in the tensors.
 * @param num_cols The number of columns in the tensors. MUST be divisible by 8.
 * @param x_w1 Pointer to the first input tensor from the forward pass.
 * @param x_w3 Pointer to the second input tensor from the forward pass.
 * @param upstream_dX Pointer to the upstream gradient tensor.
 * @param dX_w1 Pointer to the output gradient tensor for x_w1.
 * @param dX_w3 Pointer to the output gradient tensor for x_w3.
 */
 extern "C" __global__ void default_swiglu_bwd_x_bf16_bf16_kernel(
        int num_rows, int num_cols,
        const __nv_bfloat16 * __restrict__ x_w1,
        const __nv_bfloat16 * __restrict__ x_w3,
        const __nv_bfloat16 * __restrict__ upstream_dX,
        __nv_bfloat16 * __restrict__ dX_w1,
        __nv_bfloat16 * __restrict__ dX_w3) {
    
        // This kernel requires the column dimension to be divisible by 8.
        const int num_cols_vec = num_cols / 8;
    
        // Reinterpret bfloat16 pointers as float4 for 128-bit memory operations.
        const float4 *x_w1_vec = (const float4*)(x_w1);
        const float4 *x_w3_vec = (const float4*)(x_w3);
        const float4 *upstream_dX_vec = (const float4*)(upstream_dX);
        float4 *dX_w1_vec = (float4*)(dX_w1);
        float4 *dX_w3_vec = (float4*)(dX_w3);
    
        const int row_num = blockIdx.x;
        if (row_num >= num_rows) {
            return;
        }
    
        // Grid-stride loop to process 8 elements per thread per iteration.
        for (int i = threadIdx.x; i < num_cols_vec; i += blockDim.x) {
            const int idx = row_num * num_cols_vec + i;
    
            // --- Step 1: Vectorized 128-bit Loads ---
            const float4 x_w1_packed = x_w1_vec[idx];
            const float4 x_w3_packed = x_w3_vec[idx];
            const float4 upstream_dX_packed = upstream_dX_vec[idx];
    
            // --- Step 2: Unpack bfloat16x8 to floatx8 for computation ---
            const __nv_bfloat162* x_w1_b162s = (const __nv_bfloat162*)(&x_w1_packed);
            const float2 x_w1_f2_0 = __bfloat1622float2(x_w1_b162s[0]);
            const float2 x_w1_f2_1 = __bfloat1622float2(x_w1_b162s[1]);
            const float2 x_w1_f2_2 = __bfloat1622float2(x_w1_b162s[2]);
            const float2 x_w1_f2_3 = __bfloat1622float2(x_w1_b162s[3]);
    
            const __nv_bfloat162* x_w3_b162s = (const __nv_bfloat162*)(&x_w3_packed);
            const float2 x_w3_f2_0 = __bfloat1622float2(x_w3_b162s[0]);
            const float2 x_w3_f2_1 = __bfloat1622float2(x_w3_b162s[1]);
            const float2 x_w3_f2_2 = __bfloat1622float2(x_w3_b162s[2]);
            const float2 x_w3_f2_3 = __bfloat1622float2(x_w3_b162s[3]);
    
            const __nv_bfloat162* upstream_dX_b162s = (const __nv_bfloat162*)(&upstream_dX_packed);
            const float2 upstream_dX_f2_0 = __bfloat1622float2(upstream_dX_b162s[0]);
            const float2 upstream_dX_f2_1 = __bfloat1622float2(upstream_dX_b162s[1]);
            const float2 upstream_dX_f2_2 = __bfloat1622float2(upstream_dX_b162s[2]);
            const float2 upstream_dX_f2_3 = __bfloat1622float2(upstream_dX_b162s[3]);
    
            // --- Step 3: Perform gradient calculations for all 8 elements ---
            // Pre-calculate sigmoid and its derivative components
            const float inv_exp_0 = __expf(-x_w1_f2_0.x);
            const float inv_exp_1 = __expf(-x_w1_f2_0.y);
            const float inv_exp_2 = __expf(-x_w1_f2_1.x);
            const float inv_exp_3 = __expf(-x_w1_f2_1.y);
            const float inv_exp_4 = __expf(-x_w1_f2_2.x);
            const float inv_exp_5 = __expf(-x_w1_f2_2.y);
            const float inv_exp_6 = __expf(-x_w1_f2_3.x);
            const float inv_exp_7 = __expf(-x_w1_f2_3.y);
    
            const float s1_0 = 1.0f / (1.0f + inv_exp_0);
            const float s1_1 = 1.0f / (1.0f + inv_exp_1);
            const float s1_2 = 1.0f / (1.0f + inv_exp_2);
            const float s1_3 = 1.0f / (1.0f + inv_exp_3);
            const float s1_4 = 1.0f / (1.0f + inv_exp_4);
            const float s1_5 = 1.0f / (1.0f + inv_exp_5);
            const float s1_6 = 1.0f / (1.0f + inv_exp_6);
            const float s1_7 = 1.0f / (1.0f + inv_exp_7);
    
            const float silu_x1_0 = x_w1_f2_0.x * s1_0;
            const float silu_x1_1 = x_w1_f2_0.y * s1_1;
            const float silu_x1_2 = x_w1_f2_1.x * s1_2;
            const float silu_x1_3 = x_w1_f2_1.y * s1_3;
            const float silu_x1_4 = x_w1_f2_2.x * s1_4;
            const float silu_x1_5 = x_w1_f2_2.y * s1_5;
            const float silu_x1_6 = x_w1_f2_3.x * s1_6;
            const float silu_x1_7 = x_w1_f2_3.y * s1_7;
    
            // Calculate gradient w.r.t. x3 (dX_w3)
            const float dX_w3_f_0 = upstream_dX_f2_0.x * silu_x1_0;
            const float dX_w3_f_1 = upstream_dX_f2_0.y * silu_x1_1;
            const float dX_w3_f_2 = upstream_dX_f2_1.x * silu_x1_2;
            const float dX_w3_f_3 = upstream_dX_f2_1.y * silu_x1_3;
            const float dX_w3_f_4 = upstream_dX_f2_2.x * silu_x1_4;
            const float dX_w3_f_5 = upstream_dX_f2_2.y * silu_x1_5;
            const float dX_w3_f_6 = upstream_dX_f2_3.x * silu_x1_6;
            const float dX_w3_f_7 = upstream_dX_f2_3.y * silu_x1_7;
    
            // Calculate gradient w.r.t. x1 (dX_w1)
            const float ds1_dx1_0 = s1_0 * (1.0f - s1_0);
            const float ds1_dx1_1 = s1_1 * (1.0f - s1_1);
            const float ds1_dx1_2 = s1_2 * (1.0f - s1_2);
            const float ds1_dx1_3 = s1_3 * (1.0f - s1_3);
            const float ds1_dx1_4 = s1_4 * (1.0f - s1_4);
            const float ds1_dx1_5 = s1_5 * (1.0f - s1_5);
            const float ds1_dx1_6 = s1_6 * (1.0f - s1_6);
            const float ds1_dx1_7 = s1_7 * (1.0f - s1_7);
    
            const float dsilu_dx1_0 = s1_0 + x_w1_f2_0.x * ds1_dx1_0;
            const float dsilu_dx1_1 = s1_1 + x_w1_f2_0.y * ds1_dx1_1;
            const float dsilu_dx1_2 = s1_2 + x_w1_f2_1.x * ds1_dx1_2;
            const float dsilu_dx1_3 = s1_3 + x_w1_f2_1.y * ds1_dx1_3;
            const float dsilu_dx1_4 = s1_4 + x_w1_f2_2.x * ds1_dx1_4;
            const float dsilu_dx1_5 = s1_5 + x_w1_f2_2.y * ds1_dx1_5;
            const float dsilu_dx1_6 = s1_6 + x_w1_f2_3.x * ds1_dx1_6;
            const float dsilu_dx1_7 = s1_7 + x_w1_f2_3.y * ds1_dx1_7;
    
            const float dX_w1_f_0 = upstream_dX_f2_0.x * x_w3_f2_0.x * dsilu_dx1_0;
            const float dX_w1_f_1 = upstream_dX_f2_0.y * x_w3_f2_0.y * dsilu_dx1_1;
            const float dX_w1_f_2 = upstream_dX_f2_1.x * x_w3_f2_1.x * dsilu_dx1_2;
            const float dX_w1_f_3 = upstream_dX_f2_1.y * x_w3_f2_1.y * dsilu_dx1_3;
            const float dX_w1_f_4 = upstream_dX_f2_2.x * x_w3_f2_2.x * dsilu_dx1_4;
            const float dX_w1_f_5 = upstream_dX_f2_2.y * x_w3_f2_2.y * dsilu_dx1_5;
            const float dX_w1_f_6 = upstream_dX_f2_3.x * x_w3_f2_3.x * dsilu_dx1_6;
            const float dX_w1_f_7 = upstream_dX_f2_3.y * x_w3_f2_3.y * dsilu_dx1_7;
    
            // --- Step 4: Pack floatx8 results back to bfloat16x8 and store ---
            // Pack dX_w1
            const __nv_bfloat162 dX_w1_b162_0 = __float22bfloat162_rn(make_float2(dX_w1_f_0, dX_w1_f_1));
            const __nv_bfloat162 dX_w1_b162_1 = __float22bfloat162_rn(make_float2(dX_w1_f_2, dX_w1_f_3));
            const __nv_bfloat162 dX_w1_b162_2 = __float22bfloat162_rn(make_float2(dX_w1_f_4, dX_w1_f_5));
            const __nv_bfloat162 dX_w1_b162_3 = __float22bfloat162_rn(make_float2(dX_w1_f_6, dX_w1_f_7));
            float4 dX_w1_packed;
            ((__nv_bfloat162*)(&dX_w1_packed))[0] = dX_w1_b162_0;
            ((__nv_bfloat162*)(&dX_w1_packed))[1] = dX_w1_b162_1;
            ((__nv_bfloat162*)(&dX_w1_packed))[2] = dX_w1_b162_2;
            ((__nv_bfloat162*)(&dX_w1_packed))[3] = dX_w1_b162_3;
    
            // Pack dX_w3
            const __nv_bfloat162 dX_w3_b162_0 = __float22bfloat162_rn(make_float2(dX_w3_f_0, dX_w3_f_1));
            const __nv_bfloat162 dX_w3_b162_1 = __float22bfloat162_rn(make_float2(dX_w3_f_2, dX_w3_f_3));
            const __nv_bfloat162 dX_w3_b162_2 = __float22bfloat162_rn(make_float2(dX_w3_f_4, dX_w3_f_5));
            const __nv_bfloat162 dX_w3_b162_3 = __float22bfloat162_rn(make_float2(dX_w3_f_6, dX_w3_f_7));
            float4 dX_w3_packed;
            ((__nv_bfloat162*)(&dX_w3_packed))[0] = dX_w3_b162_0;
            ((__nv_bfloat162*)(&dX_w3_packed))[1] = dX_w3_b162_1;
            ((__nv_bfloat162*)(&dX_w3_packed))[2] = dX_w3_b162_2;
            ((__nv_bfloat162*)(&dX_w3_packed))[3] = dX_w3_b162_3;
    
            // --- Step 5: Vectorized 128-bit Stores ---
            dX_w1_vec[idx] = dX_w1_packed;
            dX_w3_vec[idx] = dX_w3_packed;
        }
}
 
extern "C" __global__ void naive_default_swiglu_bwd_x_bf16_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3) {

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = __bfloat162float(x_w1[row_num * num_cols + d]);
                        x_w3_val = __bfloat162float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __bfloat162float(upstream_dX[row_num * num_cols + d]);

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * silu_x1);

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * x_w3_val * dsilu_dx1);
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp8e4m3_fp16_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __half2float(upstream_dX[row_num * num_cols + d]);

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = __float2half(upstream_deriv * silu_x1);

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = __float2half(upstream_deriv * x_w3_val * dsilu_dx1);
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp8e4m3_bf16_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __bfloat162float(upstream_dX[row_num * num_cols + d]);

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * silu_x1);

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * x_w3_val * dsilu_dx1);
                }
        }
}


extern "C" __global__ void default_swiglu_bwd_x_fp8e5m2_fp16_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __half2float(upstream_dX[row_num * num_cols + d]);

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = __float2half(upstream_deriv * silu_x1);

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = __float2half(upstream_deriv * x_w3_val * dsilu_dx1);
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp8e5m2_bf16_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float upstream_deriv;
                float s1; // sigmoid(x1)
                float silu_x1; // silu(x1) = x1 * sigmoid(x1)
                float dsilu_dx1; // derivative of silu(x1) w.r.t x1
                float ds1_dx1; // derivative of sigmoid(x1) w.r.t x1
                float inv_exp_x_w1;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        // 1. Load inputs and upstream gradient, convert to float
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __bfloat162float(upstream_dX[row_num * num_cols + d]);

                        // 2. Calculate sigmoid(x1) and SiLU(x1)
                        // Compute expf only once for stability and efficiency
                        inv_exp_x_w1 = expf(-x_w1_val); // exp(-x1)
                        s1 = 1.0f / (1.0f + inv_exp_x_w1); // sigmoid(x1) = 1 / (1 + exp(-x1))
                        silu_x1 = x_w1_val * s1;           // silu(x1) = x1 * sigmoid(x1)

                        // 3. Calculate gradient w.r.t. x3 (dX_w3)
                        // dX_w3 = upstream_deriv * dOut/dx3 = upstream_deriv * silu(x1)
                        dX_w3[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * silu_x1);

                        // 4. Calculate gradient w.r.t. x1 (dX_w1) using the stable formulation
                        // dOut/dx1 = x3 * d(silu(x1))/dx1
                        // d(silu(x1))/dx1 = s1 + x1 * s1 * (1 - s1)
                        //                 = sigmoid(x1) + x1 * sigmoid_derivative(x1)

                        ds1_dx1 = s1 * (1.0f - s1); // Derivative of sigmoid(x1)
                        dsilu_dx1 = s1 + x_w1_val * ds1_dx1; // Derivative of silu(x1)

                        // dX_w1 = upstream_deriv * dOut/dx1 = upstream_deriv * x3 * dsilu_dx1
                        dX_w1[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * x_w3_val * dsilu_dx1);
                }
        }
}