#include "nvidia_ops_config.h"

int default_adamw_step_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op){

    Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

    int sm_count = device_info -> sm_count;

    cuda_launch_config -> gridDimX = sm_count;
	
	int max_threads_per_block = device_info -> max_threads_per_block;

    cuda_launch_config -> blockDimX = max_threads_per_block;

	cuda_launch_config -> gridDimY = 1;
	cuda_launch_config -> gridDimZ = 1;
	cuda_launch_config -> blockDimY = 1;
	cuda_launch_config -> blockDimZ = 1;

	cuda_launch_config -> sharedMemBytes = 0;
    
	return 0;
}
