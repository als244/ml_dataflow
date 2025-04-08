#include "nvidia_ops_config.h"

int default_softmax_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;
	
	int max_threads_per_block = device_info -> max_threads_per_block;

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	cuda_launch_config -> sharedMemBytes = 0;

	void ** op_args = op -> op_args;

	// total tokens * model dim
	int n_rows = *((int *) op_args[0]);

	cuda_launch_config -> gridDimX = n_rows;
	cuda_launch_config -> blockDimX = max_threads_per_block;

	return 0;

}


int default_cross_entropy_loss_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op) {

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;
	
	int max_threads_per_block = device_info -> max_threads_per_block;

	int num_sm = device_info -> sm_count;

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	cuda_launch_config -> sharedMemBytes = 0;

	void ** op_args = op -> op_args;

	int n_rows = *((int *) op_args[0]);

	int thread_per_block = MY_MIN(max_threads_per_block, ROUND_UP_TO_NEAREST_32(MY_CEIL(n_rows, num_sm)));

	cuda_launch_config -> gridDimX = num_sm;
	cuda_launch_config -> blockDimX = thread_per_block;

	return 0;

}