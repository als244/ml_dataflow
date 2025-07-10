#include "nvidia_ops.h"


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
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
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

	float weight_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
		cur_row_val = __half2float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = __ldg(rms_weight + i);
        out[row_base + i] = __float2half(rms_val) * weight_val;
	}
}

extern "C" __global__ void default_rms_norm_recompute_bf16_kernel(int n_rows, int n_cols, __nv_bfloat16 * rms_weight, float * rms_vals, __nv_bfloat16 * X, __nv_bfloat16 * out) {

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
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
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

	float weight_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
		cur_row_val = float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = __ldg(rms_weight + i);
        out[row_base + i] = __nv_fp8_e4m3(rms_val) * weight_val;
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

	float weight_val;
		
    // 2. do streaming update based on prior cur_recip_avg
	for (int i = lane_id; i < n_cols; i+=WARP_SIZE){
		cur_row_val = float(X[row_base + i]);
		rms_val = cur_row_val * cur_recip_avg;
		weight_val = __ldg(rms_weight + i);
        out[row_base + i] = __nv_fp8_e5m2(rms_val) * weight_val;
	}
}