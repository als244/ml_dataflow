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

extern "C" __global__ void default_swiglu_bf16_kernel(int num_rows, int num_cols, __nv_bfloat16 * x_w1, __nv_bfloat16 * x_w3, __nv_bfloat16 * out){

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


