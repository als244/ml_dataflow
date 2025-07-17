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
extern "C" __global__ void default_swiglu_bwd_x_bf16_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3) {

    // This kernel requires the column dimension to be divisible by 4.
    const int num_cols_vec = num_cols / 4;

    const float2 *x_w1_vec = (const float2*)(x_w1);
    const float2 *x_w3_vec = (const float2*)(x_w3);
    const float2 *upstream_dX_vec = (const float2*)(upstream_dX);
    float2 *dX_w1_vec = (float2*)(dX_w1);
    float2 *dX_w3_vec = (float2*)(dX_w3);

    const int row_num = blockIdx.x;
    if (row_num >= num_rows) {
        return;
    }

    // Grid-stride loop to process 4 elements per thread per iteration.
    for (int i = threadIdx.x; i < num_cols_vec; i += blockDim.x) {
        const int idx = row_num * num_cols_vec + i;

        // --- Step 1: Vectorized 64-bit Loads ---
        const float2 x_w1_packed = x_w1_vec[idx];
        const float2 x_w3_packed = x_w3_vec[idx];
        const float2 upstream_dX_packed = upstream_dX_vec[idx];

        // --- Step 2: Unpack bfloat16x4 to floatx4 for computation ---
        const __nv_bfloat162* x_w1_b162s = (const __nv_bfloat162*)(&x_w1_packed);
        const float2 x_w1_f2_0 = __bfloat1622float2(x_w1_b162s[0]);
        const float2 x_w1_f2_1 = __bfloat1622float2(x_w1_b162s[1]);

        const __nv_bfloat162* x_w3_b162s = (const __nv_bfloat162*)(&x_w3_packed);
        const float2 x_w3_f2_0 = __bfloat1622float2(x_w3_b162s[0]);
        const float2 x_w3_f2_1 = __bfloat1622float2(x_w3_b162s[1]);

        const __nv_bfloat162* upstream_dX_b162s = (const __nv_bfloat162*)(&upstream_dX_packed);
        const float2 upstream_dX_f2_0 = __bfloat1622float2(upstream_dX_b162s[0]);
        const float2 upstream_dX_f2_1 = __bfloat1622float2(upstream_dX_b162s[1]);

        // --- Step 3: Perform gradient calculations for all 4 elements ---
        // Pre-calculate sigmoid and its derivative components
        const float inv_exp_0 = __expf(-x_w1_f2_0.x);
        const float inv_exp_1 = __expf(-x_w1_f2_0.y);
        const float inv_exp_2 = __expf(-x_w1_f2_1.x);
        const float inv_exp_3 = __expf(-x_w1_f2_1.y);

        const float s1_0 = 1.0f / (1.0f + inv_exp_0);
        const float s1_1 = 1.0f / (1.0f + inv_exp_1);
        const float s1_2 = 1.0f / (1.0f + inv_exp_2);
        const float s1_3 = 1.0f / (1.0f + inv_exp_3);

        const float silu_x1_0 = x_w1_f2_0.x * s1_0;
        const float silu_x1_1 = x_w1_f2_0.y * s1_1;
        const float silu_x1_2 = x_w1_f2_1.x * s1_2;
        const float silu_x1_3 = x_w1_f2_1.y * s1_3;

        // Calculate gradient w.r.t. x3 (dX_w3)
        const float dX_w3_f_0 = upstream_dX_f2_0.x * silu_x1_0;
        const float dX_w3_f_1 = upstream_dX_f2_0.y * silu_x1_1;
        const float dX_w3_f_2 = upstream_dX_f2_1.x * silu_x1_2;
        const float dX_w3_f_3 = upstream_dX_f2_1.y * silu_x1_3;

        // Calculate gradient w.r.t. x1 (dX_w1)
        const float ds1_dx1_0 = s1_0 * (1.0f - s1_0);
        const float ds1_dx1_1 = s1_1 * (1.0f - s1_1);
        const float ds1_dx1_2 = s1_2 * (1.0f - s1_2);
        const float ds1_dx1_3 = s1_3 * (1.0f - s1_3);

        const float dsilu_dx1_0 = s1_0 + x_w1_f2_0.x * ds1_dx1_0;
        const float dsilu_dx1_1 = s1_1 + x_w1_f2_0.y * ds1_dx1_1;
        const float dsilu_dx1_2 = s1_2 + x_w1_f2_1.x * ds1_dx1_2;
        const float dsilu_dx1_3 = s1_3 + x_w1_f2_1.y * ds1_dx1_3;

        const float dX_w1_f_0 = upstream_dX_f2_0.x * x_w3_f2_0.x * dsilu_dx1_0;
        const float dX_w1_f_1 = upstream_dX_f2_0.y * x_w3_f2_0.y * dsilu_dx1_1;
        const float dX_w1_f_2 = upstream_dX_f2_1.x * x_w3_f2_1.x * dsilu_dx1_2;
        const float dX_w1_f_3 = upstream_dX_f2_1.y * x_w3_f2_1.y * dsilu_dx1_3;

        // --- Step 4: Pack floatx4 results back to bfloat16x4 and store ---
        // Pack dX_w1
        const __nv_bfloat162 dX_w1_b162_0 = __float22bfloat162_rn(make_float2(dX_w1_f_0, dX_w1_f_1));
        const __nv_bfloat162 dX_w1_b162_1 = __float22bfloat162_rn(make_float2(dX_w1_f_2, dX_w1_f_3));
        float2 dX_w1_packed;
        ((__nv_bfloat162*)(&dX_w1_packed))[0] = dX_w1_b162_0;
        ((__nv_bfloat162*)(&dX_w1_packed))[1] = dX_w1_b162_1;

        // Pack dX_w3
        const __nv_bfloat162 dX_w3_b162_0 = __float22bfloat162_rn(make_float2(dX_w3_f_0, dX_w3_f_1));
        const __nv_bfloat162 dX_w3_b162_1 = __float22bfloat162_rn(make_float2(dX_w3_f_2, dX_w3_f_3));
        float2 dX_w3_packed;
        ((__nv_bfloat162*)(&dX_w3_packed))[0] = dX_w3_b162_0;
        ((__nv_bfloat162*)(&dX_w3_packed))[1] = dX_w3_b162_1;

        // --- Step 5: Vectorized 64-bit Stores ---
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