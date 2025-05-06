#include "nvidia_ops_config.h"

int default_rope_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;
	
	int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	cuda_launch_config -> sharedMemBytes = 0;

	void ** op_args = op -> op_args;

	// total tokens * model dim
	uint64_t N = *((uint64_t *) op_args[0]);

	cuda_launch_config -> gridDimX = MY_CEIL(N / 2, max_threads_per_block);
	cuda_launch_config -> blockDimX = max_threads_per_block;

	return 0;

}

// Same as rope_set_launch_config, but using a different symbol for clarity

int default_rope_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;
	
	int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	cuda_launch_config -> sharedMemBytes = 0;

	void ** op_args = op -> op_args;

	// total tokens * model dim
	uint64_t N = *((uint64_t *) op_args[0]);

	cuda_launch_config -> gridDimX = MY_CEIL(N / 2, max_threads_per_block);
	cuda_launch_config -> blockDimX = max_threads_per_block;

	return 0;

}


int default_copy_to_seq_context_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op){

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;
	
	int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	cuda_launch_config -> sharedMemBytes = 0;

	void ** op_args = op -> op_args;

	// total tokens * model dim
	uint64_t N = *((uint64_t *) op_args[0]);

	cuda_launch_config -> gridDimX = MY_CEIL(N, max_threads_per_block);
	cuda_launch_config -> blockDimX = max_threads_per_block;

	return 0;

}