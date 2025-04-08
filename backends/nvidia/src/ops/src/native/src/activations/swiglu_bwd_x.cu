#include "nvidia_ops.h"


extern "C" __global__ void default_swiglu_bwd_x_fp32_fp32_kernel(int num_rows, int num_cols, float * x_w1, float * x_w3, float * upstream_dX, float * dX_w1, float * dX_w3) {

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = x_w1[row_num * num_cols + d];
                        x_w3_val = x_w3[row_num * num_cols + d];
                        upstream_deriv = upstream_dX[row_num * num_cols + d];

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = upstream_deriv * silu_x_w1;

                        dX_w1[row_num * num_cols + d] = upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1)));
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp16_fp16_kernel(int num_rows, int num_cols, __half * x_w1, __half * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3) {

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = __half2float(x_w1[row_num * num_cols + d]);
                        x_w3_val = __half2float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __half2float(upstream_dX[row_num * num_cols + d]);

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = __float2half(upstream_deriv * silu_x_w1);

                        dX_w1[row_num * num_cols + d] = __float2half(upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1))));
                }
        }
}
 
extern "C" __global__ void default_swiglu_bwd_x_bf16_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3) {

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = __bfloat162float(x_w1[row_num * num_cols + d]);
                        x_w3_val = __bfloat162float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __bfloat162float(upstream_dX[row_num * num_cols + d]);

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * silu_x_w1);

                        dX_w1[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1))));
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp8e4m3_fp16_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __half2float(upstream_dX[row_num * num_cols + d]);

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = __float2half(upstream_deriv * silu_x_w1);

                        dX_w1[row_num * num_cols + d] = __float2half(upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1))));
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp8e4m3_bf16_kernel(int num_rows, int num_cols, __nv_fp8_e4m3 * x_w1, __nv_fp8_e4m3 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __bfloat162float(upstream_dX[row_num * num_cols + d]);

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * silu_x_w1);

                        dX_w1[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1))));
                }
        }
}


extern "C" __global__ void default_swiglu_bwd_x_fp8e5m2_fp16_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __half * upstream_dX, __half * dX_w1, __half * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __half2float(upstream_dX[row_num * num_cols + d]);

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = __float2half(upstream_deriv * silu_x_w1);

                        dX_w1[row_num * num_cols + d] = __float2half(upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1))));
                }
        }
}

extern "C" __global__ void default_swiglu_bwd_x_fp8e5m2_bf16_kernel(int num_rows, int num_cols, __nv_fp8_e5m2 * x_w1, __nv_fp8_e5m2 * x_w3, __nv_bfloat16 * upstream_dX, __nv_bfloat16 * dX_w1, __nv_bfloat16 * dX_w3){

        int row_num = blockIdx.x;
        int thread_id = threadIdx.x;

        if (row_num < num_rows){
                float x_w1_val;
                float x_w3_val;
                float inv_exp_x_w1;
                float silu_x_w1;
                float upstream_deriv;
                for (int d = thread_id; d < num_cols; d+=blockDim.x){
                        x_w1_val = float(x_w1[row_num * num_cols + d]);
                        x_w3_val = float(x_w3[row_num * num_cols + d]);
                        upstream_deriv = __bfloat162float(upstream_dX[row_num * num_cols + d]);

                        inv_exp_x_w1 = expf(-1 * x_w1_val);
                        silu_x_w1 = x_w1_val / (1 + inv_exp_x_w1);

                        dX_w3[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * silu_x_w1);

                        dX_w1[row_num * num_cols + d] = __float2bfloat16(upstream_deriv * ((x_w3_val * (1 + inv_exp_x_w1 + x_w1_val * inv_exp_x_w1)) / ((1 + inv_exp_x_w1) * (1 + inv_exp_x_w1))));
                }
        }
}