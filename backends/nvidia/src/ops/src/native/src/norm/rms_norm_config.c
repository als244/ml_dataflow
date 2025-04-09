#include "nvidia_ops_config.h"

int default_rms_norm_set_attribute_config(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function) {

	int ret;

	Cuda_Device_Info * dev_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int dev_max_smem = dev_info -> optin_max_smem_per_block;

	int rms_max_mem = dev_max_smem - (1U << 11);

	ret = cu_func_set_attribute(cuda_function -> function_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, rms_max_mem);
	if (ret){
		fprintf(stderr, "Error: could not set rms norm attribute for smem of size: %d...\n", rms_max_mem);
		return -1;
	}

	(cuda_function -> function_config).func_max_smem = rms_max_mem;

	ret = cu_func_get_attribute(&((cuda_function -> function_config).func_max_threads_per_block), cuda_function -> function_handle, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	if (ret){
		fprintf(stderr, "Error: could not get rms norm attribute for max_threads_per_block...\n");
		return -1;
	}

	return 0;
}

int default_rms_norm_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

	DataflowDatatype norm_dt = arg_dtypes[3];

	size_t dtype_size = dataflow_sizeof_element(norm_dt);

	if (dtype_size == 0){
		fprintf(stderr, "Error: rms norm not available for dtype %s...\n", dataflow_datatype_as_string(norm_dt));
		return -1;
	}


	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int sm_count = device_info -> sm_count;

	cuda_launch_config -> gridDimX = sm_count;

	int rms_max_threads_per_block = (cuda_function -> function_config).func_max_threads_per_block;
	cuda_launch_config -> blockDimX = rms_max_threads_per_block;


	void ** op_args = op -> op_args;

	int model_dim = *((int *) op_args[1]);

	int rms_smem = 2 * dtype_size * model_dim;

	int rms_max_smem = (cuda_function -> function_config).func_max_smem;

	if (rms_smem > rms_max_smem){
		fprintf(stderr, "Error: rms norm will fail. Unable to support model dim of %d and dtype %s. Not enough smem on device, max for this func is %d bytes, but requires %d...\n", model_dim, dataflow_datatype_as_string(norm_dt), rms_max_smem, rms_smem);
		return -1;
	}

	cuda_launch_config -> sharedMemBytes = rms_smem;

	return 0;
}

int default_rms_norm_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

	// backward dtype
	DataflowDatatype helper_data_dt = arg_dtypes[3];

	size_t helper_data_dtype_size = dataflow_sizeof_element(helper_data_dt);

	if (helper_data_dtype_size == 0){
		fprintf(stderr, "Error: rms norm not available for dtype %s...\n", dataflow_datatype_as_string(helper_data_dt));
		return -1;
	}


	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	
	void ** op_args = op -> op_args;

	int num_rows = *((int *) op_args[0]);

	int model_dim = *((int *) op_args[1]);

	int rms_base_smem = helper_data_dtype_size * model_dim;

	int rms_max_smem = (cuda_function -> function_config).func_max_smem;

	int rms_remain_smem = rms_max_smem - rms_base_smem;

	// need to save the weighted sums and squared sums
	int rms_max_rows_per_block = rms_remain_smem / (2 * helper_data_dtype_size);

	int num_blocks = MY_CEIL(num_rows, rms_max_rows_per_block);

	int sm_count = device_info -> sm_count;
	if (num_blocks <= sm_count){
		cuda_launch_config -> gridDimX = sm_count;
	}
	else{
		cuda_launch_config -> gridDimX = num_blocks;
	}

	int max_rows_per_block = MY_CEIL(num_rows, (cuda_launch_config -> gridDimX));

	int rms_smem = helper_data_dtype_size * (max_rows_per_block + model_dim);	

	// Should never fail...
	if (rms_smem > rms_max_smem){
		fprintf(stderr, "Error: rms norm will fail. Unable to support model dim of %d with max rows per block of %d, and intermediate dtype %s. Not enough smem on device, max for this func is %d bytes, but requires %d...\n", model_dim, max_rows_per_block, dataflow_datatype_as_string(helper_data_dt), rms_max_smem, rms_smem);
		return -1;
	}

	cuda_launch_config -> sharedMemBytes = rms_smem;

	int rms_max_threads_per_block = (cuda_function -> function_config).func_max_threads_per_block;
	cuda_launch_config -> blockDimX = rms_max_threads_per_block;


	return 0;

}

// Same as bwd_x....
int default_rms_norm_bwd_w_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

	// backward dtype
	DataflowDatatype helper_data_dt = arg_dtypes[3];

	size_t helper_data_dtype_size = dataflow_sizeof_element(helper_data_dt);

	if (helper_data_dtype_size == 0){
		fprintf(stderr, "Error: rms norm not available for dtype %s...\n", dataflow_datatype_as_string(helper_data_dt));
		return -1;
	}


	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	
	void ** op_args = op -> op_args;

	int num_rows = *((int *) op_args[0]);

	int model_dim = *((int *) op_args[1]);

	int rms_base_smem = helper_data_dtype_size * model_dim;

	int rms_max_smem = (cuda_function -> function_config).func_max_smem;

	int rms_remain_smem = rms_max_smem - rms_base_smem;

	int rms_max_rows_per_block = rms_remain_smem / helper_data_dtype_size;

	int num_blocks = MY_CEIL(num_rows, rms_max_rows_per_block);

	int sm_count = device_info -> sm_count;
	if (num_blocks <= sm_count){
		cuda_launch_config -> gridDimX = sm_count;
	}
	else{
		cuda_launch_config -> gridDimX = num_blocks;
	}

	int max_rows_per_block = MY_CEIL(num_rows, (cuda_launch_config -> gridDimX));

	int rms_smem = helper_data_dtype_size * (max_rows_per_block + model_dim);	

	// Should never fail...
	if (rms_smem > rms_max_smem){
		fprintf(stderr, "Error: rms norm will fail. Unable to support model dim of %d with max rows per block of %d, and intermediate dtype %s. Not enough smem on device, max for this func is %d bytes, but requires %d...\n", model_dim, max_rows_per_block, dataflow_datatype_as_string(helper_data_dt), rms_max_smem, rms_smem);
		return -1;
	}

	cuda_launch_config -> sharedMemBytes = rms_smem;

	int rms_max_threads_per_block = (cuda_function -> function_config).func_max_threads_per_block;
	cuda_launch_config -> blockDimX = rms_max_threads_per_block;

	return 0;

}


int default_rms_norm_noscale_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

	DataflowDatatype norm_dt = arg_dtypes[3];

	size_t dtype_size = dataflow_sizeof_element(norm_dt);

	if (dtype_size == 0){
		fprintf(stderr, "Error: rms norm not available for dtype %s...\n", dataflow_datatype_as_string(norm_dt));
		return -1;
	}


	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int sm_count = device_info -> sm_count;

	cuda_launch_config -> gridDimX = sm_count;

	int rms_max_threads_per_block = (cuda_function -> function_config).func_max_threads_per_block;
	cuda_launch_config -> blockDimX = rms_max_threads_per_block;


	void ** op_args = op -> op_args;

	int model_dim = *((int *) op_args[1]);

	// no scale doens't need space for weights...
	int rms_smem = dtype_size * model_dim;

	int rms_max_smem = (cuda_function -> function_config).func_max_smem;

	if (rms_smem > rms_max_smem){
		fprintf(stderr, "Error: rms norm will fail. Unable to support model dim of %d and dtype %s. Not enough smem on device, max for this func is %d bytes, but requires %d...\n", model_dim, dataflow_datatype_as_string(norm_dt), rms_max_smem, rms_smem);
		return -1;
	}

	cuda_launch_config -> sharedMemBytes = rms_smem;

	return 0;
}
 
int default_rms_norm_noscale_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

	// backward dtype
	DataflowDatatype helper_data_dt = arg_dtypes[3];

	size_t helper_data_dtype_size = dataflow_sizeof_element(helper_data_dt);

	if (helper_data_dtype_size == 0){
		fprintf(stderr, "Error: rms norm not available for dtype %s...\n", dataflow_datatype_as_string(helper_data_dt));
		return -1;
	}


	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	
	void ** op_args = op -> op_args;

	int num_rows = *((int *) op_args[0]);

	int model_dim = *((int *) op_args[1]);

	int rms_base_smem = helper_data_dtype_size * model_dim;

	int rms_max_smem = (cuda_function -> function_config).func_max_smem;

	int rms_remain_smem = rms_max_smem - rms_base_smem;

	// need to save the weighted sums and squared sums
	int rms_max_rows_per_block = rms_remain_smem / (2 * helper_data_dtype_size);

	int num_blocks = MY_CEIL(num_rows, rms_max_rows_per_block);

	int sm_count = device_info -> sm_count;
	if (num_blocks <= sm_count){
		cuda_launch_config -> gridDimX = sm_count;
	}
	else{
		cuda_launch_config -> gridDimX = num_blocks;
	}

	int max_rows_per_block = MY_CEIL(num_rows, (cuda_launch_config -> gridDimX));

	int rms_smem = helper_data_dtype_size * (max_rows_per_block + model_dim);	

	// Should never fail...
	if (rms_smem > rms_max_smem){
		fprintf(stderr, "Error: rms norm will fail. Unable to support model dim of %d with max rows per block of %d, and intermediate dtype %s. Not enough smem on device, max for this func is %d bytes, but requires %d...\n", model_dim, max_rows_per_block, dataflow_datatype_as_string(helper_data_dt), rms_max_smem, rms_smem);
		return -1;
	}

	cuda_launch_config -> sharedMemBytes = rms_smem;

	int rms_max_threads_per_block = (cuda_function -> function_config).func_max_threads_per_block;
	cuda_launch_config -> blockDimX = rms_max_threads_per_block;


	return 0;

}