#ifndef CUDA_DRV_H
#define CUDA_DRV_H

#include "dataflow_common.h"

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

int cu_initialize_drv();

int cu_get_device(CUdevice * dev, int device_id);
	
int cu_initialize_ctx(CUcontext * ctx, CUdevice dev, unsigned int ctx_flags);
int cu_initialize_ctx_compute_frac(CUcontext * ctx, CUdevice dev, unsigned int ctx_flags, float compute_frac, int * used_sms, bool to_set_ctx);

// ALL FUNCTIONS BELOW ASSUME APPROPRIATE CONTEXT HAS BEEN SET...

int cu_get_dev_total_mem(size_t * ret_val, CUdevice dev);

int cu_get_dev_name(char * dev_name, int max_name_len, CUdevice dev);

int cu_get_dev_attribute(int * ret_val, CUdevice dev, CUdevice_attribute attrib);

int cu_load_module(CUmodule * module, char * module_filename);

int cu_module_get_function(CUfunction * function, CUmodule module, char * function_name);

int cu_func_get_attribute(int * ret_val, CUfunction function, CUfunction_attribute attrib);

int cu_func_set_attribute(CUfunction function, CUfunction_attribute attrib, int val);

int cu_func_load(CUfunction function);

int cu_func_launch(CUstream stream, CUfunction function, void ** func_params, 
					unsigned int sharedMemBytes, 
					unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
					unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
					void ** extra);

int cu_host_func_launch(CUstream stream, CUhostFn host_fn, void * userData);

int cu_initialize_stream(CUstream * stream, int prio);

int cu_initialize_event(CUevent * event);

int cu_record_event(CUevent event, CUstream stream);

int cu_stream_wait_event(CUstream cur_stream, CUevent event_to_wait_on);

int cu_stream_add_callback(CUstream stream, CUstreamCallback callback_func, void * userData);

int cu_stream_synchronize(CUstream stream);

int cu_ctx_synchronize();


int cu_alloc_mem(void ** dev_ptr_ref, uint64_t size_bytes);
int cu_free_mem(void * dev_ptr);

int cu_set_mem_blocking(void * dev_ptr, uint8_t val, uint64_t size_bytes);
int cu_set_mem(CUstream stream, void * dev_ptr, uint8_t val, uint64_t size_bytes);



int cu_register_host_mem(void * host_mem, uint64_t size_bytes, unsigned int flags);
int cu_unregister_host_mem(void * host_mem);

int cu_enable_peer_ctx(CUcontext peer_ctx);
int cu_disable_peer_ctx(CUcontext peer_ctx);

int cu_transfer_host_to_dev_blocking(void * dev_dest, void * host_src, uint64_t size_bytes);
int cu_transfer_host_to_dev_async(CUstream stream, void * dev_dest, void * host_src, uint64_t size_bytes);

int cu_transfer_dev_to_host_blocking(void * host_dest, void * dev_src, uint64_t size_bytes);
int cu_transfer_dev_to_host_async(CUstream stream, void * host_dest, void * dev_src, uint64_t size_bytes);

int cu_transfer_dev_to_dev_blocking(void * dev_dest, void * dev_src, uint64_t size_bytes);
int cu_transfer_dev_to_dev_async(CUstream stream, void * dev_dest, void * dev_src, uint64_t size_bytes);

int cu_transfer_batch_dev_to_dev_async(CUstream stream, int num_transfers, int dest_id, int src_id, void ** dev_dest, void ** dev_src, uint64_t * size_bytes);

#endif