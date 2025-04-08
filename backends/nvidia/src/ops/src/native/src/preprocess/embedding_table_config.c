#include "nvidia_ops_config.h"

int default_embedding_table_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op){

    Op_Skeleton * op_skeleton = &(cuda_function -> op_skeleton);

	Op_Skeleton_Header * op_skeleton_header = &(op_skeleton -> header);

	DataflowDatatype * arg_dtypes = op_skeleton_header -> arg_dtypes;

    // get index 3 or 4 which are embedding table and output
    DataflowDatatype input_dt = arg_dtypes[3];

    size_t dtype_size = dataflow_sizeof_element(input_dt);

    if (dtype_size == 0){
        fprintf(stderr, "Error: embedding not available for dtype %s...\n", dataflow_datatype_as_string(input_dt));
		return -1;
    }

    cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

    void ** op_args = op -> op_args;

    int num_tokens = *((int *) op_args[0]);
    cuda_launch_config -> gridDimX = num_tokens;

    Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

    int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> blockDimX = max_threads_per_block;

    cuda_launch_config -> sharedMemBytes = 0;

    return 0;
}