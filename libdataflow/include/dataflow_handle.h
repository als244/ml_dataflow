#ifndef DATAFLOW_HANDLE_H
#define DATAFLOW_HANDLE_H

#include "dataflow_common.h"
#include "dataflow_table.h"
#include "dataflow_op_structs.h"

#define OP_TABLE_MIN_SIZE 1UL << 10
#define OP_TABLE_MAX_SIZE 1UL << 20

#define MAX_NATIVE_FUNCTION_LIBS 1024

#define MAX_STREAMS 256

#define FUNC_SYMBOL_MAX_LEN 512

typedef enum compute_type {
	COMPUTE_NONE,
	COMPUTE_CPU,
	COMPUTE_CUDA,
	COMPUTE_HSA,
	COMPUTE_LEVEL_ZERO
} ComputeType;


typedef struct dataflow_handle Dataflow_Handle;

struct dataflow_handle {

	// Breakdown of core components...
	// Could group these into more compact data structures, 
	// but clearer if each distinct component has its own field
	
	ComputeType compute_type;
	// should be -1 for CPU
	int device_id;
	void * device_handle;
	// backend specific device info maybe needed by attribute setting and launch config...
	void * device_info;
	// user defined id in case multiple handles are created on same device
	int ctx_id;
	// backend specific context handle
	void * ctx;
	// user defined number of streams created at init time
	int num_streams;
	// array of backend specific streams
	void * streams;
	// optional user defined stream priorities and names
	int stream_prios[MAX_STREAMS];
	char * stream_names[MAX_STREAMS];
	// array of backend specific events, with size of num streams created at init time
	// CUevents for cuda
	void * stream_states;

	// space to actually load native module and functions
	int num_native_function_libs;
	void * native_function_libs[MAX_NATIVE_FUNCTION_LIBS];

	// Table containing mapping of op_skeleton.fingerprint => backend specific function info/launch func pointers
	// initially populated using combination of native function lib (pre-compiled into native assembly)
	// and external function pointers from other shared libs
	int num_ops;
	Dataflow_Table op_table;
	
	// Backend Required Functions...

	// 0.) OPS Functionality

	// returns number of functions added to op table
	// or -1 if error loading native/external code filename, or for native code, if error loading native code config lib
	int (*register_native_code)(Dataflow_Handle * dataflow_handle, char * native_code_filename, 
								char * native_code_config_lib_filename, int num_funcs, 
								Op_Skeleton * func_op_skeletons, char ** func_symbols,
								 char ** func_set_launch_symbols, char ** func_init_symbols);
	int (*register_external_code)(Dataflow_Handle * dataflow_handle, char * external_code_filename, int num_funcs, 
								Op_Skeleton * func_op_skeletons, char ** func_symbols, char ** func_init_symbols);

	
	// 1.) COMPUTE Functionality
	
	int (*submit_op)(Dataflow_Handle * dataflow_handle, Op * op, int stream_id);
	int (*submit_host_op)(Dataflow_Handle * dataflow_handle, void * host_func, void * host_func_arg, int stream_id);


	
	// 2.) DEPENDENCIES Functionality 
	
	// records event and returns a reference to event that can be passed to same/different dataflow handle
	void * (*get_stream_state)(Dataflow_Handle * dataflow_handle, int stream_id);
	int (*submit_dependency)(Dataflow_Handle * dataflow_handle, int stream_id, void * other_stream_state);
	int (*sync_stream)(Dataflow_Handle * dataflow_handle, int stream_id);
	// Synchronizes all streams
	int (*sync_handle)(Dataflow_Handle * dataflow_handle);

	
	// 3.) MEMORY Functionality
	
	// Note: Memory allocs and frees are slow and globally synchronizing, so these should be embedded within
	// a higher layer of memory management. Bulk call to alloc that then can be oragnized elsewhere...
	void * (*alloc_mem)(Dataflow_Handle * dataflow_handle, uint64_t size_bytes);
	void (*free_mem)(Dataflow_Handle * dataflow_handle, void * dev_ptr);
	int (*set_mem)(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_ptr, uint8_t val, uint64_t size_bytes);
	// The host pointer should be on same numa node as device
	// ensures good transfer performance by page-locking host memory in way that streaming device driver can understand
	int (*enable_access_to_host_mem)(Dataflow_Handle * dataflow_handle, void * host_ptr, uint64_t size_bytes, unsigned int flags);
	// undoes the enable access step above
	int (*disable_access_to_host_mem)(Dataflow_Handle * dataflow_handle, void * host_ptr);
	// lets this context be able to access memory allocated on peer device
	int (*enable_access_to_peer_mem)(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle);
	int (*disable_access_to_peer_mem)(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle);
	

	// 4.) TRANSFER Functionality
	
	// From/to host memory
	int (*submit_inbound_transfer)(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * host_src, uint64_t size_bytes);
	int (*submit_outbound_transfer)(Dataflow_Handle * dataflow_handle, int stream_id, void * host_dest, void * dev_src, uint64_t size_bytes);
	// also works for self!
	int (*submit_peer_transfer)(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * dev_src, uint64_t size_bytes);
	
	// TODO: Network
	
};


// this is the function signature of optional init function
// that is called during handle initialization 
// The op_table_value argument can be casted to specific backend function structure (e.g. Cuda_Function)
// which is stored within the op table
// It should be able to hold an "extra" field that can be populated by init function and later retrieved
// during actual call to the external function
typedef int (*External_Lib_Func_Init)(Dataflow_Handle * dataflow_handle, void * op_table_value);

// this is the function signature of all external functions
// the extra argument allows for passing of library handles and other attributes
// the extra argument can be populated from the initialization function and saved within
typedef int (*External_Lib_Func)(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);


#endif