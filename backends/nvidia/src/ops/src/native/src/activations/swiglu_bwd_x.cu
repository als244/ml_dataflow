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