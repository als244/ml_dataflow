#include "cuda_profiler.h"

int profile_start(){

	CUresult result;
	const char * err;

	result = cuProfilerStart();
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to start cuda profiler: %s\n", err);
		return -1;
	}

	return 0;
}

int profile_stop(){

	CUresult result;
	const char * err;

	result = cuProfilerStop();
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to stop cuda profiler: %s\n", err);
		return -1;
	}

	return 0;
}

// TODO: IN REALITY SHOULD MAKE A BACKEND PROFILE STRUCT TO KNOW WHICH BACKEND PROFILE API TO CALL
//			- for now just doing Nvidia's NVTX

int profile_range_push(const char * message){
	return nvtxRangePush(message);
}

int profile_range_pop(){
	return nvtxRangePop();
}

void profile_name_stream(void * stream_ref, const char * stream_name){
	if (stream_ref){
		CUstream stream = *((CUstream *) stream_ref);
		nvtxNameCuStreamA(stream, stream_name);
	}
}

void profile_name_context(void * context_ref, const char * context_name){
	if (context_ref){
		CUcontext context = *((CUcontext *) context_ref);
		nvtxNameCuContextA(context, context_name);
	}
	return;
}

void profile_name_device(void * device_ref, const char * device_name){
	if (device_ref){
		CUdevice device = *((CUdevice *) device_ref);
		nvtxNameCuDeviceA(device, device_name);
	}
	return;
}

void profile_name_thread(const char * thread_name){
	nvtxNameOsThreadA(pthread_self(), thread_name);
	return;
}