#ifndef CUDA_PROFILER_H
#define CUDA_PROFILER_H


#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

#include <cudaProfiler.h> 

int profile_start();
int profile_stop();

int profile_range_push(const char * message);
int profile_range_pop();
void profile_name_stream(void * stream_ref, const char * stream_name);
void profile_name_context(void * context_ref, const char * context_name);
void profile_name_device(void * device_ref, const char * device_name);
void profile_name_thread(const char * thread_name);



#endif
