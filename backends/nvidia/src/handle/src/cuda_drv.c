#include "cuda_drv.h"

int cu_initialize_drv(){

	CUresult result;
	const char * err;

	unsigned long flags = 0;
	result = cuInit(flags);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not init driver: %s\n", err);
    	return -1;
	}
	return 0;
}

int cu_get_device(CUdevice * dev, int device_id){

	CUresult result;
	const char * err;

	result = cuDeviceGet(dev, device_id);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not get device id %d: %s\n", device_id, err);
    	return -1;
	}

	return 0;
}



int cu_initialize_ctx(CUcontext * ctx, CUdevice dev, unsigned int ctx_flags){

	CUresult result;
	const char * err;

	result = cuCtxCreate(ctx, ctx_flags, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not create context: %s\n", err);
    	return -1;
	}

	// automatically sets newly created contexxt...

	return 0;
}


int cu_get_dev_total_mem(size_t * ret_val, CUdevice dev) {

	CUresult result;
	const char * err;

	result = cuDeviceTotalMem(ret_val, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get device total for memory: %s\n", err);
		return -1;
	}

	return 0;
}

int cu_get_dev_attribute(int * ret_val, CUdevice dev, CUdevice_attribute attrib) {

	CUresult result;
	const char * err;

	result = cuDeviceGetAttribute(ret_val, attrib, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get device attribute %d: %s\n", attrib, err);
		return -1;
	}

	return 0;
}


int cu_load_module(CUmodule * module, char * module_filename){

	CUresult result;
	const char * err;

	result = cuModuleLoad(module, module_filename);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not load module from file: %s\n", err);
		return -1;
	}

	return 0;
}

int cu_module_get_function(CUfunction * function, CUmodule module, char * function_name) {

	CUresult result;
	const char * err;

	result = cuModuleGetFunction(function, module, function_name);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get function with name %s from module: %s\n", function_name, err);
		return -1;
	}
	return 0;
}

int cu_func_get_attribute(int * ret_val, CUfunction function, CUfunction_attribute attrib) {

	CUresult result;
	const char * err;

	result = cuFuncGetAttribute(ret_val, attrib, function);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get function attribute %d: %s\n", attrib, err);
		return -1;
	}
	return 0;


}

int cu_func_set_attribute(CUfunction function, CUfunction_attribute attrib, int val) {

	CUresult result;
	const char * err;

	result = cuFuncSetAttribute(function, attrib, val);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not set function attribute %d to %d: %s\n", attrib, val, err);
		return -1;
	}

	return 0;
}

int cu_func_load(CUfunction function) {

	CUresult result;
	const char * err;

	result = cuFuncLoad(function);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not load function: %s\n", err);
		return -1;
	}

	return 0;

}


int cu_func_launch(CUstream stream, CUfunction function, void ** func_params, 
					unsigned int sharedMemBytes, 
					unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
					unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
					void ** extra) {

	CUresult result;
	const char * err;

	result = cuLaunchKernel(function, 
							gridDimX, gridDimY, gridDimZ, 
							blockDimX, blockDimY, blockDimZ,
							sharedMemBytes,
							stream,
							func_params,
							extra);

	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to launch cuda function: %s\n", err);
		return -1;
	}

	return 0;
}

int cu_host_func_launch(CUstream stream, CUhostFn host_fn, void * userData) {

	CUresult result;
	const char * err;

	result = cuLaunchHostFunc(stream, host_fn, userData);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to launch host function in cuda stream: %s\n", err);
		return -1;
	}

	return 0;
}


int cu_initialize_stream(CUstream * stream, int prio){

	CUresult result;
	const char * err;

	result = cuStreamCreateWithPriority(stream, CU_STREAM_NON_BLOCKING, prio);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to create cuda stream: %s\n", err);
		return -1;
	}

	return 0;
}

int cu_initialize_event(CUevent * event){

	CUresult result;
	const char * err;

	result = cuEventCreate(event, CU_EVENT_DISABLE_TIMING);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to create cuda event: %s\n", err);
		return -1;
	}

	return 0;
}

int cu_record_event(CUevent event, CUstream stream){

	CUresult result;
	const char * err;

	result = cuEventRecord(event, stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to record cuda event: %s\n", err);
		return -1;
	}

	return 0;

}

int cu_stream_wait_event(CUstream cur_stream, CUevent event_to_wait_on){

	CUresult result;
	const char * err;

	result = cuStreamWaitEvent(cur_stream, event_to_wait_on, CU_EVENT_WAIT_DEFAULT);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to have stream wait event: %s\n", err);
		return -1;
	}

	return 0;
}





int cu_stream_add_callback(CUstream stream, CUstreamCallback callback_func, void * userData){

	CUresult result;
	const char * err;

	result = cuStreamAddCallback(stream, callback_func, userData, 0);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do cuStreamAddCallback...\n");
		return -1;
	}

	return 0;


}

int cu_stream_synchronize(CUstream stream){

	CUresult result;
	const char * err;

	result = cuStreamSynchronize(stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do cuStreamSynchronize: %s\n", err);
		return -1;
	}

	return 0;
}


int cu_ctx_synchronize() {

	CUresult result;
	const char * err;

	result = cuCtxSynchronize();
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do cuCtxSynchronize: %s\n", err);
		return -1;
	}

	return 0;


}





int cu_alloc_mem(void ** dev_ptr_ref, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuMemAlloc((CUdeviceptr *) dev_ptr_ref, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to alloc size %lu...\n", size_bytes);
		return -1;
	}

	return 0;
}


int cu_free_mem(void * dev_ptr){

	CUresult result;
	const char * err;

	result = cuMemFree((CUdeviceptr) dev_ptr);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to free dev mem...\n");
		return -1;
	}

	return 0;


}

int cu_set_mem_blocking(void * dev_ptr, uint8_t val, uint64_t size_bytes){

	CUresult result;
	const char * err;

	result = cuMemsetD8((CUdeviceptr) dev_ptr, val, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to call cuMemsetD8. Err: %s\n", err);
		return -1;
	}

	return 0;

}


int cu_set_mem(CUstream stream, void * dev_ptr, uint8_t val, uint64_t size_bytes){

	CUresult result;
	const char * err;

	result = cuMemsetD8Async((CUdeviceptr) dev_ptr, val, size_bytes, stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to call cuMemsetD8Async. Err: %s\n", err);
		return -1;
	}

	return 0;

}



int cu_register_host_mem(void * host_mem, uint64_t size_bytes, unsigned int flags) {

	CUresult result;
	const char * err;

	// Flags may be
	// - CU_MEMHOSTREGISTER_PORTABLE
	// - CU_MEMHOSTREGISTER_DEVICEMAP
	// - CU_MEMHOSTREGSITER_IOMEMORY
	// - CU_MEMHOSTREGSITER_READ_ONLY
	// - CU_M
	result = cuMemHostRegister(host_mem, size_bytes, flags);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to regsiter system memory buffer of size %lu with cuda. Err: %s\n", size_bytes, err);
		return -1;
	}

	return 0;
}



int cu_unregister_host_mem(void * host_mem){

	CUresult result;
	const char * err;

	result = cuMemHostUnregister(host_mem);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to unregister system memory buffer with cuda. Err: %s\n", err);
		return -1;
	}

	return 0;
}



int cu_enable_peer_ctx(CUcontext peer_ctx) {

	CUresult result;
	const char * err;

	result = cuCtxEnablePeerAccess(peer_ctx, 0);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to enable peer access...\n");
		return -1;
	}

	return 0;


}

int cu_disable_peer_ctx(CUcontext peer_ctx) {

	CUresult result;
	const char * err;

	result = cuCtxDisablePeerAccess(peer_ctx);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to disable peer access...\n");
		return -1;
	}

	return 0;


}







int cu_transfer_host_to_dev_blocking(void * dev_dest, void * host_src, uint64_t size_bytes){

	CUresult result;
	const char * err;

	result = cuMemcpyHtoD((CUdeviceptr) dev_dest, host_src, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do h to d blocking transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}

int cu_transfer_host_to_dev_async(CUstream stream, void * dev_dest, void * host_src, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuMemcpyHtoDAsync((CUdeviceptr) dev_dest, host_src, size_bytes, stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to submit h to d async transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}

int cu_transfer_dev_to_host_blocking(void * host_dest, void * dev_src, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuMemcpyDtoH(host_dest, (CUdeviceptr) dev_src, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do d to h blocking transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}

int cu_transfer_dev_to_host_async(CUstream stream, void * host_dest, void * dev_src, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuMemcpyDtoHAsync(host_dest, (CUdeviceptr) dev_src, size_bytes, stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to submit d to h async transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}


int cu_transfer_dev_to_dev_blocking(void * dev_dest, void * dev_src, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuMemcpyDtoD((CUdeviceptr) dev_src, (CUdeviceptr) dev_src, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do d to h blocking transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}


int cu_transfer_dev_to_dev_async(CUstream stream, void * dev_dest, void * dev_src, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuMemcpyDtoDAsync((CUdeviceptr) dev_src, (CUdeviceptr) dev_src, size_bytes, stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to submit d to d async transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}
