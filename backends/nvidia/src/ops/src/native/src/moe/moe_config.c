#include "nvidia_ops_config.h"

int default_select_experts_set_attribute_config(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function) {

	int ret;

	Cuda_Device_Info * dev_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int dev_max_smem = dev_info -> optin_max_smem_per_block;

	int select_experts_max_mem = dev_max_smem - (1U << 11);

	ret = cu_func_set_attribute(cuda_function -> function_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, select_experts_max_mem);
	if (ret){
		fprintf(stderr, "Error: could not set select experts attribute for smem of size: %d...\n", select_experts_max_mem);
		return -1;
	}

	(cuda_function -> function_config).func_max_smem = select_experts_max_mem;

	ret = cu_func_get_attribute(&((cuda_function -> function_config).func_max_threads_per_block), cuda_function -> function_handle, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	if (ret){
		fprintf(stderr, "Error: could not get select experts attribute for max_threads_per_block...\n");
		return -1;
	}

	return 0;
}

int default_select_experts_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;


	DataflowDatatype temp_dt = DATAFLOW_FP32;

	DataflowDatatype top_k_dt = arg_dtypes[4];

	DataflowDatatype token_cnts_dt = arg_dtypes[6];

	size_t dtype_size_temp = dataflow_sizeof_element(temp_dt);

	size_t dtype_size_top_k = dataflow_sizeof_element(top_k_dt);

	size_t dtype_size_exp_token_cnts = dataflow_sizeof_element(token_cnts_dt);

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int sm_count = device_info -> sm_count;

	// NOTE: NEEDS TO BE LESS THAN OR EQUAL TO SM_COUNT
	// BECAUSE BLOCK IDX 0 IS BLOCKING UNTIL ALL OTHER BLOCKS FINISH!
	// thus all blocks need an sm to run on otherwise block 0 is waiting...
	cuda_launch_config -> gridDimX = sm_count;

	int num_experts = *((int *) op -> op_args[1]);
	int top_k_experts = *((int *) op -> op_args[2]);

	// need to determine the number of warps per block!

	// shared memory usage:
	// num_warps * WARP_SIZE * dtype_size_temp
	// num_warps * top_k_experts * dtype_size_temp
	// num_warps * top_k_experts * dtype_size_top_k
	// num_experts *  dtype_size_exp_token_cnts

	// Total smem usage (in terms of num_warps)

	int smem_per_warp = WARP_SIZE * dtype_size_temp + top_k_experts * dtype_size_temp + top_k_experts * dtype_size_top_k;


	int num_warps = (cuda_launch_config -> blockDimX + WARP_SIZE - 1) / WARP_SIZE;


	int select_experts_max_smem = (cuda_function -> function_config).func_max_smem;

	int select_experts_base_smem = num_experts * dtype_size_exp_token_cnts;
	int select_experts_remain_smem = select_experts_max_smem - select_experts_base_smem;

	int num_warps = select_experts_remain_smem / smem_per_warp;

	if (num_warps == 0){
		fprintf(stderr, "Error: not enough smem on device for default select experts. Requires at least %d bytes of smem (for one warp), but only %d bytes are available...\n", select_experts_base_smem + smem_per_warp, select_experts_max_smem);
		return -1;
	}

	int select_experts_chosen_warps = num_warps;

	int select_experts_max_threads_per_block = (cuda_function -> function_config).func_max_threads_per_block;
	
	if (num_warps * WARP_SIZE > select_experts_max_threads_per_block){
		select_experts_chosen_warps = select_experts_max_threads_per_block / WARP_SIZE;
	}

	int select_experts_chosen_smem = select_experts_base_smem + select_experts_chosen_warps * smem_per_warp;

	cuda_launch_config -> sharedMemBytes = select_experts_chosen_smem;

	cuda_launch_config -> blockDimX = select_experts_chosen_warps * WARP_SIZE;
	return 0;
}