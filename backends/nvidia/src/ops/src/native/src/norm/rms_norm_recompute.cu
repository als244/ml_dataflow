#include "nvidia_ops.h"

#define RMS_NORM_RECOMPUTE_VEC_SIZE 8

extern "C" __global__ void default_rms_norm_recompute_fp32_kernel(int n_rows, int n_cols, float * rms_weight, float * rms_vals, float * X, float * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

    // 1.) load in new rms val
    float cur_recip_avg = rms_vals[row_ind];

	float weight_val;

	float cur_row_val;
	float rms_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = threadIdx.x; i < n_cols; i+=blockDim.x){
		cur_row_val = X[row_base + i];
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = __ldg(rms_weight + i);
        out[row_base + i] = rms_val * weight_val;
	}
}

extern "C" __global__ void default_rms_norm_recompute_fp16_kernel(int n_rows, int n_cols, __half * rms_weight, float * rms_vals, __half * X, __half * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

    // 1.) load in new rms val
    float cur_recip_avg = rms_vals[row_ind];

	__half weight_val;

	float cur_row_val;
	float rms_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = threadIdx.x; i < n_cols; i+=blockDim.x){
		cur_row_val = __half2float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = __ldg(rms_weight + i);
        out[row_base + i] = __float2half(rms_val) * weight_val;
	}
}

__global__ void default_rms_norm_recompute_bf16_kernel_c(
    int n_rows, 
    int n_cols, 
    const __nv_bfloat16* __restrict__ rms_weight, 
    const float* __restrict__ rms_vals, 
    const __nv_bfloat16* __restrict__ X, 
    __nv_bfloat16* __restrict__ out) {

    // Each block processes one row.
    int row_ind = blockIdx.x;
    if (row_ind >= n_rows) {
        return;
    }

    // Load the pre-calculated reciprocal for the entire row.
    const float cur_recip_avg = rms_vals[row_ind];
    
    // Use unsigned long long for 64-bit integers in C.
    unsigned long long row_base = (unsigned long long)row_ind * (unsigned long long)n_cols;

    // Pointers for vectorized access using C-style casts.
    // Assumes pointers are aligned to 16 bytes.
    const float4* x_vec = (const float4*)(X + row_base);
    const float4* w_vec = (const float4*)(rms_weight);
    float4* o_vec = (float4*)(out + row_base);

    const int n_cols_vec = n_cols / RMS_NORM_RECOMPUTE_VEC_SIZE;

    // --- Vectorized Loop ---
    // Each thread processes one vector (8 elements) per iteration.
    for (int i = threadIdx.x; i < n_cols_vec; i += blockDim.x) {
        // 1. Load 8 elements at once (16 bytes)
        float4 x_data = x_vec[i];
        float4 w_data = w_vec[i];
        float4 out_data;

        // Reinterpret float4 using C-style casts
        const __nv_bfloat162_t* x_h2 = (const __nv_bfloat162_t*)(&x_data);
        const __nv_bfloat162_t* w_h2 = (const __nv_bfloat162_t*)(&w_data);
        __nv_bfloat162_t* o_h2 = (__nv_bfloat162_t*)(&out_data);

        // 2. Process 4 pairs of bfloat16s
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            // Convert bf16 pairs to float pairs
            float2 x_f2 = __bfloat1622float2(x_h2[j]);
            float2 w_f2 = __bfloat1622float2(w_h2[j]);
            
            // Apply normalization and weighting
            float2 o_f2;
            o_f2.x = (x_f2.x * cur_recip_avg) * w_f2.x;
            o_f2.y = (x_f2.y * cur_recip_avg) * w_f2.y;
            
            // Convert back to bfloat16 pair
            o_h2[j] = __float22bfloat162_rn(o_f2);
        }

        // 3. Store 8 elements at once (16 bytes)
        o_vec[i] = out_data;
    }

    // --- Tail Loop for Remainder ---
    // Handle elements if n_cols is not perfectly divisible by VEC_SIZE.
    int tail_start = n_cols_vec * RMS_NORM_RECOMPUTE_VEC_SIZE;
    for (int i = tail_start + threadIdx.x; i < n_cols; i += blockDim.x) {
        float cur_row_val = __bfloat162float(X[row_base + i]);
        float rms_val = cur_row_val * cur_recip_avg;
        __nv_bfloat16 weight_val = rms_weight[i];
        out[row_base + i] = __float2bfloat16(rms_val) * weight_val;
    }
}

extern "C" __global__ void naive_default_rms_norm_recompute_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * rms_weight, float * rms_vals, __nv_bfloat16 * X, __nv_bfloat16 * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

    // 1.) load in new rms val
    float cur_recip_avg = rms_vals[row_ind];

	__nv_bfloat16 weight_val;

	float cur_row_val;
	float rms_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = threadIdx.x; i < n_cols; i+=blockDim.x){
		cur_row_val = __bfloat162float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = __ldg(rms_weight + i);
        out[row_base + i] = __float2bfloat16(rms_val) * weight_val;
	}
}

extern "C" __global__ void default_rms_norm_recompute_fp8e4m3_kernel(int n_rows, int n_cols, __nv_fp8_e4m3 * rms_weight, float * rms_vals, __nv_fp8_e4m3 * X, __nv_fp8_e4m3 * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

    // 1.) load in new rms val
    float cur_recip_avg = rms_vals[row_ind];

	__nv_fp8_e4m3 weight_val;

	float cur_row_val;
	float rms_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = threadIdx.x; i < n_cols; i+=blockDim.x){
		cur_row_val = float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = rms_weight[i];
        out[row_base + i] = __nv_fp8_e4m3(rms_val * float(weight_val));
	}
}


extern "C" __global__ void default_rms_norm_recompute_fp8e5m2_kernel(int n_rows, int n_cols, __nv_fp8_e5m2 * rms_weight, float * rms_vals, __nv_fp8_e5m2 * X, __nv_fp8_e5m2 * out) {

	// this gets dynamically allocated the size of model_dim
	extern __shared__ uint8_t sdata[];

	int row_ind = blockIdx.x;
	uint64_t row_base = (uint64_t) (row_ind) * (uint64_t) n_cols;

	if (row_ind >= n_rows){
		return;
	}

    // 1.) load in new rms val
    float cur_recip_avg = rms_vals[row_ind];

	__nv_fp8_e5m2 weight_val;

	float cur_row_val;
	float rms_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = threadIdx.x; i < n_cols; i+=blockDim.x){
		cur_row_val = float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = rms_weight[i];
        out[row_base + i] = __nv_fp8_e5m2(rms_val * float(weight_val));
	}
}