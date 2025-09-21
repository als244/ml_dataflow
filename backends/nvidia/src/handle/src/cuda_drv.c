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

	#if CUDA_VERSION >= 13000
		result = cuCtxCreate(ctx, NULL, ctx_flags, dev);
	#else
		result = cuCtxCreate(ctx, ctx_flags, dev);
	#endif
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not create context: %s\n", err);
    	return -1;
	}

	// automatically sets newly created contexxt...

	return 0;
}

int cu_initialize_ctx_compute_frac(CUcontext * ctx, CUdevice dev, unsigned int ctx_flags, float compute_frac, int * used_sms, bool to_set_ctx) {

	int ret;

	CUresult result;
	const char * err;

	int sm_count;

	ret = cu_get_dev_attribute(&sm_count, dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
	if (ret){
		fprintf(stderr, "Error: failed to get total sm count when setting device info...\n");
		return -1;
	}

	if ((compute_frac <= 0.0f) || (compute_frac >= 1.0f)) {
		*used_sms = sm_count;
		return cu_initialize_ctx(ctx, dev, ctx_flags);
	}


	int major_arch_num;

	ret = cu_get_dev_attribute(&major_arch_num, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
	if (ret){
		fprintf(stderr, "Error: failed to get major arch num for cuda device...\n");
		return -1;
	}

	int min_count = 1;
	int multiple_factor = 1;
	int green_avail = true;
	// if num_sms > 0 then using green context
	// 6.X min count is 1
	// 7.X min count is 2 and must be multiple of 2
	// 8.X min count is 4 and must be multiple of 2
	// 9.X min count is 8 and must be multiple of 8
	switch(major_arch_num){
		case 6:
			min_count = 1;
			multiple_factor = 1;
			break;
		case 7:
			min_count = 2;
			multiple_factor = 2;
			break;
		case 8:
			min_count = 4;
			multiple_factor = 2;
		default:
			min_count = 8;
			multiple_factor = 8;
			break;
	}

	

	int target_sm_count = (int) ((float) sm_count * compute_frac);


	// create green context
	CUdevResource sm_resource;
	result = cuDeviceGetDevResource(dev, &sm_resource, CU_DEV_RESOURCE_TYPE_SM);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
	    fprintf(stderr, "Error: Could not get available sms: %s\n", err);
	    return -1;
	}

	int cur_sm_cnt = sm_resource.sm.smCount;

	// No need for green context at this point, cant obtain more sms than available
	if (target_sm_count >= cur_sm_cnt){
		*used_sms = sm_count;
		result = cuCtxCreate(ctx, NULL, ctx_flags, dev);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
	    	fprintf(stderr, "Error: Could not create context: %s\n", err);
	    	return -1;
		}
		return 0;
	}

	// Abide by arch specs
	if (target_sm_count < min_count){
		target_sm_count = min_count;
	}
	else{
		int remain = target_sm_count % multiple_factor;
		target_sm_count = target_sm_count - remain;
	}

	unsigned int num_groups = cur_sm_cnt / target_sm_count;


	CUdevResource * result_sm_resources = malloc(num_groups * sizeof(CUdevResource));
	if (!result_sm_resources){
		fprintf(stderr, "Error: malloc failed to alloc container for result sm resources...\n");
		return -1;
	}

	// Split SM Resources to Generate Result Resource
	result = cuDevSmResourceSplitByCount(result_sm_resources, &num_groups, &sm_resource, NULL, 0, target_sm_count);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
	    fprintf(stderr, "Error: Could split sm resources: %s\n", err);
	    return -1;
	}

	if (num_groups == 0){
		fprintf(stderr, "Error: no groups available to split sm resources...\n");
		return -1;
	}

	// Generate resource desc
	CUdevResourceDesc sm_resource_desc;
	unsigned int nbResources = 1;

	*used_sms = result_sm_resources[0].sm.smCount;

	result = cuDevResourceGenerateDesc(&sm_resource_desc, result_sm_resources, nbResources);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
	    fprintf(stderr, "Error: Could not generate resource desc: %s\n", err);
	    return -1;
	}

	// Create green context
	CUgreenCtx green_ctx;
	unsigned int green_ctx_flags = CU_GREEN_CTX_DEFAULT_STREAM;
	result = cuGreenCtxCreate(&green_ctx, sm_resource_desc, dev, green_ctx_flags);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
	    fprintf(stderr, "Error: Could not create green ctx: %s\n", err);
	    return -1;
	}

	// Convert from green context to primary
	result = cuCtxFromGreenCtx(ctx, green_ctx);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
	    fprintf(stderr, "Error: Could not convert from green ctx to primary: %s\n", err);
	    return -1;
	}

	if (to_set_ctx){
		result = cuCtxPushCurrent(*ctx);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
		    fprintf(stderr, "Error: Could not push converted green context after creation: %s\n", err);
		    return -1;
		}
	}


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

int cu_get_dev_name(char * dev_name, int max_name_len,CUdevice dev){

	CUresult result;
	const char * err;

	result = cuDeviceGetName(dev_name, max_name_len, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get device name: %s\n", err);
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

	result = cuMemcpyDtoD((CUdeviceptr) dev_dest, (CUdeviceptr) dev_src, size_bytes);
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

	result = cuMemcpyDtoDAsync((CUdeviceptr) dev_dest, (CUdeviceptr) dev_src, size_bytes, stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to submit d to d async transfer of size %lu: %s...\n", size_bytes, err);
		return -1;
	}

	return 0;

}

int cu_transfer_batch_dev_to_dev_async(CUstream stream, int num_transfers, int dest_id, int src_id, void ** dev_dest, void ** dev_src, uint64_t * size_bytes) {

	CUresult result;
	const char * err;

	size_t numAttrs = 1;
	size_t attrsIdxs = 0;

	CUmemcpyAttributes attrs;

	attrs.srcLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
	attrs.srcLocHint.id = src_id;
	attrs.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
	attrs.dstLocHint.id = dest_id;
	attrs.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
	attrs.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;

	size_t failIdx;

	#if CUDA_VERSION >= 13000
		result = cuMemcpyBatchAsync((CUdeviceptr *) dev_dest, (CUdeviceptr *) dev_src, (size_t *) size_bytes, (size_t) num_transfers, &attrs, &attrsIdxs, numAttrs, stream);
	#else
		result = cuMemcpyBatchAsync((CUdeviceptr *) dev_dest, (CUdeviceptr *) dev_src, (size_t *) size_bytes, (size_t) num_transfers, &attrs, &attrsIdxs, numAttrs, &failIdx, stream);
	#endif

	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to submit batch d to d async with %d transfers. Failed at index %zu: %s...\n", num_transfers, failIdx, err);
		return -1;
	}

	return 0;

}