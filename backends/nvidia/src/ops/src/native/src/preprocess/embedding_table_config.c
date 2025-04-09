#include "nvidia_ops_config.h"


int default_embedding_table_set_attribute_config(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function) {

	int ret;

	Cuda_Device_Info * dev_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int dev_max_smem = dev_info -> optin_max_smem_per_block;

	int embedding_table_max_mem = dev_max_smem - (1U << 11);

	ret = cu_func_set_attribute(cuda_function -> function_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, embedding_table_max_mem);
	if (ret){
		fprintf(stderr, "Error: could not set embedding table attribute for smem of size: %d...\n", embedding_table_max_mem);
		return -1;
	}

	(cuda_function -> function_config).func_max_smem = embedding_table_max_mem;

	ret = cu_func_get_attribute(&((cuda_function -> function_config).func_max_threads_per_block), cuda_function -> function_handle, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	if (ret){
		fprintf(stderr, "Error: could not get embedding table attribute for max_threads_per_block...\n");
		return -1;
	}

	return 0;
}

int default_embedding_table_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op){

    Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

    DataflowDatatype embed_dt = arg_dtypes[5];

    size_t dtype_size = dataflow_sizeof_element(embed_dt);

    cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

    void ** op_args = op -> op_args;

    int num_unique_tokens = *((int *) op_args[0]);
    int embed_dim = *((int *) op_args[1]);

    cuda_launch_config -> gridDimX = num_unique_tokens;

    Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

    int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> blockDimX = max_threads_per_block;

    int shared_mem_bytes = embed_dim * dtype_size;
    
    int embedding_table_max_smem = (cuda_function -> function_config).func_max_smem;
    if (shared_mem_bytes > embedding_table_max_smem){
        fprintf(stderr, "Error: not enough smem to run embedding table... requires %d bytes, but only have %d...\n", shared_mem_bytes, embedding_table_max_smem);
        return -1;
    }

    cuda_launch_config -> sharedMemBytes = shared_mem_bytes;

    return 0;
}


int default_embedding_table_bwd_w_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

    Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

    DataflowDatatype embed_dt = arg_dtypes[5];

    size_t dtype_size = dataflow_sizeof_element(embed_dt);


    cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

    void ** op_args = op -> op_args;

    int num_unique_tokens = *((int *) op_args[0]);
    int embed_dim = *((int *) op_args[1]);
    cuda_launch_config -> gridDimX = num_unique_tokens;

    Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

    int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> blockDimX = max_threads_per_block;

    int shared_mem_bytes = embed_dim * dtype_size;

    int embedding_table_max_smem = (cuda_function -> function_config).func_max_smem;
    if (shared_mem_bytes > embedding_table_max_smem){
        fprintf(stderr, "Error: not enough smem to run embedding table bwd w... requires %d bytes, but only have %d...\n", shared_mem_bytes, embedding_table_max_smem);
        return -1;
    }

    cuda_launch_config -> sharedMemBytes = shared_mem_bytes;

    return 0;
}