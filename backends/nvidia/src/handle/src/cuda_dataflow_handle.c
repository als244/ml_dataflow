#include "cuda_dataflow_handle.h"

void CUDA_CB cuda_post_sem_callback(CUstream stream, CUresult status, void * data) {

	sem_t * sem_to_post = (sem_t *) data;

	int ret = sem_post(sem_to_post);
	if (ret){
		fprintf(stderr, "Error: failed to post to semaphore within cuda sem callback...\n");
	}
}


// Can use this instead of callback functionality by launching it as a host function in appropriate stream...
void CUDA_CB post_sem_callback(void * _sem_to_post) {

	// this serves as the CUhostFn type 
	// within cu_host_func_launch

	sem_t * sem_to_post = (sem_t *) _sem_to_post;

	int ret = sem_post(sem_to_post);
	if (ret){
		fprintf(stderr, "Error: failed to post to semaphore within cuda sem callback...\n");
	}
}



// re-definition from within fingerprint.c to avoid dependency...
static uint64_t cuda_fingerprint_to_least_sig64(uint8_t * fingerprint, int fingerprint_num_bytes){
	uint8_t * least_sig_start = fingerprint + fingerprint_num_bytes - sizeof(uint64_t);
	uint64_t result = 0;
    for(int i = 0; i < 8; i++){
        result <<= 8;
        result |= (uint64_t)least_sig_start[i];
    }
    return result;
}

uint64_t cuda_op_table_hash_func(void * op_fingerprint, uint64_t table_size) {
	uint64_t least_sig_64bits = cuda_fingerprint_to_least_sig64((void *) op_fingerprint, DATAFLOW_OP_IDENTIFIER_FINGERPRINT_NUM_BYTES);
	return least_sig_64bits % table_size;
}


// the functions for the profiler

int cuda_profiler_start(){
	return profile_start();
}

int cuda_profiler_stop(){
	return profile_stop();
}

void cuda_profiler_name_device(Dataflow_Handle * dataflow_handle, const char * device_name){
	void * dev_ref = dataflow_handle -> device_handle;
	profile_name_device(dev_ref, device_name);
	return;
}

void cuda_profiler_name_handle(Dataflow_Handle * dataflow_handle, const char * handle_name){
	void * ctx_ref = dataflow_handle -> ctx;
	profile_name_context(ctx_ref, handle_name);
	return;
}

void cuda_profiler_name_stream(Dataflow_Handle * dataflow_handle, int stream_id, const char * stream_name){
	CUstream * streams = (CUstream *) dataflow_handle -> streams;
	CUstream * stream_ref;
	if ((stream_id >= 0) && (stream_id < dataflow_handle -> num_streams)){
		stream_ref = &(streams[stream_id]);
	}
	else{
		fprintf(stderr, "Error: cannot name stream with id %d, when device only has %d streams...\n", stream_id, dataflow_handle -> num_streams);
		return;
	}
	profile_name_stream(stream_ref, stream_name);
	return;
}

void cuda_profiler_name_thread(Dataflow_Handle * dataflow_handle, const char * thread_name){
	profile_name_thread(thread_name);
	return;
}

int cuda_profiler_range_push(const char * message){
	return profile_range_push(message);
}

int cuda_profiler_range_pop(){
	return profile_range_pop();
}







// FUFILLING ALL FUNCTION POINTERS WIHTIN COMPUTE_HANDLE INTERFACE...

// CUDA arch id
int cuda_get_arch_id(Dataflow_Handle * dataflow_handle){
	Cuda_Device_Info * device_info = dataflow_handle -> device_info;
	return device_info -> arch_num;
}

// CUDA hardware model id
int cuda_get_hardware_model_id(Dataflow_Handle * dataflow_handle){
	Cuda_Device_Info * device_info = dataflow_handle -> device_info;
	char * device_name = device_info -> device_name;	
	return device_info -> sm_count;
}

HardwareArchType cuda_get_hardware_arch_type(Dataflow_Handle * dataflow_handle){
	Cuda_Device_Info * device_info = dataflow_handle -> device_info;
	char * device_name = device_info -> device_name;

	if (strncmp(device_name, "NVIDIA A100", strlen("NVIDIA A100")) == 0) {
    return BACKEND_ARCH_A100;
}
else if (strncmp(device_name, "NVIDIA H100", strlen("NVIDIA H100")) == 0) {
    return BACKEND_ARCH_H100;
}
else if (strncmp(device_name, "NVIDIA GeForce RTX 3090", strlen("NVIDIA GeForce RTX 3090")) == 0) {
    return BACKEND_ARCH_RTX_3090;
}
else if (strncmp(device_name, "NVIDIA GeForce RTX 4090", strlen("NVIDIA GeForce RTX 4090")) == 0) {
    return BACKEND_ARCH_RTX_4090;
}
else if (strncmp(device_name, "NVIDIA GeForce RTX 5090", strlen("NVIDIA GeForce RTX 5090")) == 0) {
    return BACKEND_ARCH_RTX_5090;
}
	
	return UNKNOWN_HARDWARE_ARCH;
}


// num procs
int cuda_get_num_procs(Dataflow_Handle * dataflow_handle){
	Cuda_Device_Info * device_info = dataflow_handle -> device_info;
	return device_info -> sm_count;
}



/* 0. OPS FUNCTIONALITY */

int cuda_register_native_code(Dataflow_Handle * dataflow_handle, char * native_code_filename, char * native_code_config_lib_filename, 
								int num_funcs, Op_Skeleton * func_op_skeletons, char ** func_symbols, char ** func_set_launch_symbols, char ** func_init_symbols) {
	
	int ret;

	if (dataflow_handle -> num_native_function_libs >= MAX_NATIVE_FUNCTION_LIBS){
		fprintf(stderr, "Error: cannot register more than %d native function libs...\n", MAX_NATIVE_FUNCTION_LIBS);
		return -1;
	}

	CUmodule * cur_module = malloc(sizeof(CUmodule));
	if (!cur_module){
		fprintf(stderr, "Error: malloc failed to allocate space for cumodule container...\n");
		return -1;
	}

	ret = cu_load_module(cur_module, native_code_filename);
	if (ret){
		fprintf(stderr, "Error: unable to load function lib from path: %s...\n", native_code_filename);
		free(cur_module);
		return -1;
	}

	
	CUmodule module = *(cur_module);

	

	// Load the config lib
	void * native_config_lib_handle = dlopen(native_code_config_lib_filename, RTLD_LAZY);
	if (!native_config_lib_handle){
		fprintf(stderr, "Error: could not load function config shared lib with dlopen() from path: %s\n", native_code_config_lib_filename);
		free(cur_module);
		return -1;
	}

	Cuda_Function cur_cuda_function;

	int added_funcs = 0;

	char * cur_func_symbol;

	for (int i = 0; i < num_funcs; i++){

		cur_func_symbol = func_symbols[i];

		if (cur_func_symbol){

			if (!func_set_launch_symbols[i]){
				fprintf(stderr, "Error: set launch symbol not set for native function%s, but is required for native functions..\n", cur_func_symbol);
				continue;
			}

			cur_cuda_function.is_native = true;

			ret = cu_module_get_function(&(cur_cuda_function.function_handle), module, cur_func_symbol);
			if (ret){
				fprintf(stderr, "Error: failed to load function #%d with name %s from function lib...\n", i, cur_func_symbol);
				continue;
			}

			// call the set attribute function from shared library with symbol specified by func_config_lib_set_attribute_symbol_name
			if (func_init_symbols[i]){
				Cuda_Set_Func_Attribute set_func_attribute = dlsym(native_config_lib_handle, func_init_symbols[i]);
				if (!set_func_attribute){
					fprintf(stderr, "Error: failed to load symbol to initialize function #%d with name %s and attribute setting function as %s...\n", i, cur_func_symbol, func_init_symbols[i]);
					continue;
				}

				// now call set func attribute
				ret = set_func_attribute(dataflow_handle, &(cur_cuda_function));
				if (ret){
					fprintf(stderr, "Error: failed to set function attributes for function #%d with name %s and attribute setting function as %s\n", i, cur_func_symbol, func_init_symbols[i]);
					continue;
				}
			}


			ret = cu_func_load(cur_cuda_function.function_handle);
			if (ret){
				fprintf(stderr, "Error: could not load cuda function with name %s from function lib...\n", cur_func_symbol);
				continue;
			}

			
			// now set the pointer for launch config as part of 
			cur_cuda_function.set_launch_config = dlsym(native_config_lib_handle, func_set_launch_symbols[i]);
			if (!cur_cuda_function.set_launch_config){
				fprintf(stderr, "Error: failed to get function pointer to launch config for for function #%d with name %s and set_launch_config function name as %s\n", i, cur_func_symbol, func_set_launch_symbols[i]);
				continue;
			}
		
			// copy over the op skeleton
			memcpy(&(cur_cuda_function.op_skeleton), &(func_op_skeletons[i]), sizeof(Op_Skeleton));

			// set extra attributes to null
			cur_cuda_function.cuda_external_func = NULL;
			cur_cuda_function.op_extra = NULL;
		
			// now insert the cuda function into the table...

			// the table copies the contents of cur_cuda_function...
			ret = dataflow_insert_table(&(dataflow_handle -> op_table), &(cur_cuda_function.op_skeleton.identifier.fingerprint), &cur_cuda_function);
			if (ret){
				fprintf(stderr, "Error: failed to insert op for function #%d with name %s to op table...\n", i, cur_func_symbol);
				continue;
			}

			added_funcs++;

		}

		if (added_funcs == 0){
			fprintf(stderr, "Error: no functions were added to op table...\n");
			free(cur_module);
			return -1;
		}

	}

	dataflow_handle -> native_function_libs[dataflow_handle -> num_native_function_libs] = cur_module;
	dataflow_handle -> num_native_function_libs++;

	dataflow_handle -> num_ops += added_funcs;

	return added_funcs;
}

int cuda_register_external_code(Dataflow_Handle * dataflow_handle, char * external_code_filename, int num_funcs, 
								Op_Skeleton * func_op_skeletons, char ** func_symbols, char ** func_init_symbols) {
	int ret;
	void * external_lib_handle = dlopen(external_code_filename, RTLD_LAZY);
	if (!external_lib_handle) {
		fprintf(stderr, "Error: could not open external library with path: %s\n", external_code_filename);
		return -1;
	}

	External_Lib_Func_Init external_lib_func_init_ref;
	External_Lib_Func external_lib_func_ref;
	Cuda_Function cur_cuda_function;
	int added_funcs = 0;
	char * cur_func_symbol;

	for (int i = 0; i < num_funcs; i++) {
		cur_func_symbol = func_symbols[i];
		if (cur_func_symbol) {
			// Obtain the actual function
			external_lib_func_ref = dlsym(external_lib_handle, cur_func_symbol);
			if (!external_lib_func_ref) {
				fprintf(stderr, "Error: could not obtain function symbol %s from external lib %s...\n", cur_func_symbol, external_code_filename);
				continue;
			}

			// set the external function
			cur_cuda_function.cuda_external_func = external_lib_func_ref;

			// If function has an init coupled
			if (func_init_symbols[i]) {
				external_lib_func_init_ref = dlsym(external_lib_handle, func_init_symbols[i]);
				if (!external_lib_func_init_ref) {
					fprintf(stderr, "Error: could not get ref to initialization function %s, for external function %s\n", func_init_symbols[i], cur_func_symbol);
					continue;
				}

				// call the initialization function
				ret = (*external_lib_func_init_ref)(dataflow_handle, (void *) &cur_cuda_function);
				if (ret) {
					fprintf(stderr, "Error: initialization function %s, for external function %s returned an error...\n", func_init_symbols[i], cur_func_symbol);
					continue;
				}
			}

			// now insert the cuda function into the table...

			// copy over the op skeleton
			memcpy(&(cur_cuda_function.op_skeleton), &(func_op_skeletons[i]), sizeof(Op_Skeleton));

			// set native attributes to null
			memset(&cur_cuda_function.function_config, 0, sizeof(Cuda_Function_Config));
			cur_cuda_function.is_native = false;
			memset(&cur_cuda_function.function_handle, 0, sizeof(CUfunction));
			cur_cuda_function.set_launch_config = NULL;

			// the table copies the value
			ret = dataflow_insert_table(&(dataflow_handle -> op_table), &(cur_cuda_function.op_skeleton.identifier.fingerprint), &cur_cuda_function);
			if (ret){
				fprintf(stderr, "Error: failed to insert op for function #%d with name %s to op table...\n", i, cur_func_symbol);
				continue;
			}

			added_funcs++;
		}		
	}

	dataflow_handle -> num_ops += added_funcs;

	return added_funcs;
}



/* 1. COMPUTE FUNCTIONALITY */

int cuda_submit_op(Dataflow_Handle * dataflow_handle, Op * op, int stream_id){

	int ret;

	
	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream;

	if (stream_id == -1){
		stream = NULL;
	}
	else if (stream_id < dataflow_handle -> num_streams){
		stream = cu_streams[stream_id];
	}
	else{
		fprintf(stderr, "Error: cannot submit op with nickname %s to stream id %d, when device only has %d streams. Null stream specified by -1...\n", 
								(op -> op_skeleton).header.op_nickname, stream_id, dataflow_handle -> num_streams);
		return -1;
	}

	// lookup Cuda_Function in table based on op skeleton hash

	Dataflow_Table * op_table = &(dataflow_handle -> op_table);

	Op_Identifier * op_identifier = &((op -> op_skeleton).identifier);

	Cuda_Function * cuda_op_function_ref = NULL;

	long table_ind = dataflow_find_table(op_table, op_identifier -> fingerprint, false, (void **) &cuda_op_function_ref);

	if ((table_ind == -1) || (!cuda_op_function_ref)){
		fprintf(stderr, "Error: failed to find op with nickname %s and matching dtypes...\n", (op -> op_skeleton).header.op_nickname);
		return -1;
	}


	// now determine if native, or should just call external...

	if (cuda_op_function_ref -> is_native){

		

		// need launch config
		Cuda_Launch_Config cuda_launch_config;

		// if native then launch config function pointer must be set

		// get launch config using function pointer in Cuda Function
		ret = (cuda_op_function_ref -> set_launch_config)(&cuda_launch_config, dataflow_handle, cuda_op_function_ref, op);
		if (ret){
			fprintf(stderr, "Error: failed to correctly set launch config for op with nickname %s...\n", (op -> op_skeleton).header.op_nickname);
			return -1;
		}

		void ** func_params = op -> op_args;

		// for now ignoring extra..
		void ** op_extra = NULL;

		// can error check args here to set kernel params

		// call cuLaunchKernel
		ret = cu_func_launch(stream, cuda_op_function_ref -> function_handle, func_params, 
					cuda_launch_config.sharedMemBytes, 
					cuda_launch_config.gridDimX, cuda_launch_config.gridDimY, cuda_launch_config.gridDimZ, 
					cuda_launch_config.blockDimX, cuda_launch_config.blockDimY, cuda_launch_config.blockDimZ,
					op_extra);

		if (ret){
			fprintf(stderr, "Error: failed to launch kernel for op with nickname %s...\n", (op -> op_skeleton).header.op_nickname);
			return -1;
		}

		return 0;
	}
	
	// otherwise handle external lib function calls...

	if (!(cuda_op_function_ref -> cuda_external_func)){
		fprintf(stderr, "Error: if op is not native it must have a reference to external function (error with op having nickname %s)\n", (op -> op_skeleton).header.op_nickname);
		return -1;
	}
	
	void * op_extra = cuda_op_function_ref -> op_extra;

	ret = (cuda_op_function_ref -> cuda_external_func)(dataflow_handle, stream_id, op, op_extra);
	if (ret){
		fprintf(stderr, "Error: failed to do external cuda function for op with nickname %s...\n", (op -> op_skeleton).header.op_nickname);
		return -1;
	}

	return 0;

}

int cuda_submit_host_op(Dataflow_Handle * dataflow_handle, void * host_func, void * host_func_arg, int stream_id){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	ret = cu_host_func_launch(stream, (CUhostFn) host_func, host_func_arg);
	if (ret){
		fprintf(stderr, "Error: failed to launch host function...\n");
		return -1;
	}

	return 0;
}


/* 2. DEPENDENCY FUNCTIONALITY */

void * cuda_get_stream_state(Dataflow_Handle * dataflow_handle, int stream_id){
	
	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	CUevent * cu_events = (CUevent *) dataflow_handle -> stream_states;

	CUevent event_to_wait_on = cu_events[stream_id];

	ret = cu_record_event(event_to_wait_on, stream);
	if (ret){
		fprintf(stderr, "Error: failed to record event when trying to get stream state, cannot return ref...\n");
		return NULL;
	}

	return &(cu_events[stream_id]);
}



int cuda_submit_dependency(Dataflow_Handle * dataflow_handle, int stream_id, void * other_stream_state){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	CUevent event_to_wait_on = *((CUevent *) other_stream_state);

	ret = cu_stream_wait_event(stream, event_to_wait_on);
	if (ret){
		fprintf(stderr, "Error: failed to have current stream wait on event during submitting dependency...\n");
		return -1;
	}

	return 0;
}

int cuda_submit_stream_post_sem_callback(Dataflow_Handle * dataflow_handle, int stream_id, sem_t * sem_to_post){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	ret = cu_stream_add_callback(stream, cuda_post_sem_callback, (void *) sem_to_post);
	if (ret){
		fprintf(stderr, "Error: failed to add post callback...\n");
		return -1;
	}

	return 0;
}

int cuda_sync_stream(Dataflow_Handle * dataflow_handle, int stream_id){

	int ret;

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	ret = cu_stream_synchronize(stream);
	if (ret){
		fprintf(stderr, "Error: unable ot do cuda stream synchronize...\n");
		return -1;
	}

	return 0;
}


int cuda_sync_handle(Dataflow_Handle * dataflow_handle){

	int ret;	

	int device_id = dataflow_handle -> device_id;
	int ctx_id = dataflow_handle -> ctx_id;

	ret = cu_ctx_synchronize();
	if (ret){
		fprintf(stderr, "Error: was unable to sync ctx id: %d on device id: %d...\n", ctx_id, device_id);
		return -1;
	}

	return 0;

}


/* 3. MEMORY FUNCTIONALITY */

void * cuda_alloc_mem(Dataflow_Handle * dataflow_handle, uint64_t size_bytes){

	int ret;

	int device_id = dataflow_handle -> device_id;

	void * dev_ptr = NULL;

	ret = cu_alloc_mem(&dev_ptr, size_bytes);
	if (ret || !dev_ptr){
		fprintf(stderr, "Error: failed to alloc memory on device id %d of size %lu bytes...\n", device_id, size_bytes);
		return NULL;
	}

	return dev_ptr;
}

void cuda_free_mem(Dataflow_Handle * dataflow_handle, void * dev_ptr){

	int ret;

	int device_id = dataflow_handle -> device_id;

	ret = cu_free_mem(dev_ptr);
	if (ret){
		fprintf(stderr, "Error: was unable to free memory on device id %d with ptr %p...\n", device_id, dev_ptr);
	}

	return;
}

int cuda_set_mem(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_ptr, uint8_t val, uint64_t size_bytes){

	int ret;

	if (stream_id == -1) {

		ret = cu_set_mem_blocking(dev_ptr, val, size_bytes);
		if (ret){
			fprintf(stderr, "Error: was unable to set memory for device ptr of %p of %lu bytes...\n", dev_ptr, size_bytes);
			return -1;
		}

		return 0;
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;
	CUstream stream = cu_streams[stream_id];

	ret = cu_set_mem(stream, dev_ptr, val, size_bytes);
	if (ret){
		fprintf(stderr, "Error: was unable to set memory for device ptr of %p of %lu bytes...\n", dev_ptr, size_bytes);
		return -1;
	}
	return 0;

}


// PAGE-LOCKS THE HOST MEMORY...use sparingly...
// Makes this context know that host memory is pinned...
int cuda_enable_access_to_host_mem(Dataflow_Handle * dataflow_handle, void * host_ptr, uint64_t size_bytes, unsigned int flags){

	int ret;

	int device_id = dataflow_handle -> device_id;

	ret = cu_register_host_mem(host_ptr, size_bytes, flags);
	if (ret){
		fprintf(stderr, "Error: failed to enable device id %d to access host memory with ptr %p of size %lu and flags %u...\n", device_id, host_ptr, size_bytes, flags);
		return -1;
	}

	return 0;
}

// Releases the page-locked memory that 
int cuda_disable_access_to_host_mem(Dataflow_Handle * dataflow_handle, void * host_ptr){

	int ret;

	int device_id = dataflow_handle -> device_id;

	ret = cu_unregister_host_mem(host_ptr);
	if (ret){
		fprintf(stderr, "Error: failed to disable host memory access with ptr %p from device id %d...\n", host_ptr, device_id);
		return -1;
	}

	return 0;
}


int cuda_enable_access_to_peer_mem(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle){

	int ret;	

	if (peer_dataflow_handle -> compute_type != dataflow_handle -> compute_type){
		fprintf(stderr, "Error: cannot enable peer access, must be same compute type...\n");
		return -1;
	}

	int this_device_id = dataflow_handle -> device_id;
	int other_device_id = dataflow_handle -> device_id;

	CUcontext peer_context = *((CUcontext *) peer_dataflow_handle -> ctx);

	ret = cu_enable_peer_ctx(peer_context);
	if (ret){
		fprintf(stderr, "Error: was unable to allow this device (id %d) to access memory on peer device (id %d)\n", this_device_id, other_device_id);
		return -1;
	}

	return 0;
}


int cuda_disable_access_to_peer_mem(Dataflow_Handle * dataflow_handle, Dataflow_Handle * peer_dataflow_handle){

	int ret;	

	if (peer_dataflow_handle -> compute_type != dataflow_handle -> compute_type){
		fprintf(stderr, "Error: cannot enable peer access, must be same compute type...\n");
		return -1;
	}

	int this_device_id = dataflow_handle -> device_id;
	int other_device_id = dataflow_handle -> device_id;

	CUcontext peer_context = *((CUcontext *) peer_dataflow_handle -> ctx);

	ret = cu_disable_peer_ctx(peer_context);
	if (ret){
		fprintf(stderr, "Error: was unable to disable this device (id %d) from accessing memory on peer device (id %d)\n", this_device_id, other_device_id);
		return -1;
	}

	return 0;
}

/* 4. TRANSFER FUNCTIONALITY */


int cuda_submit_inbound_transfer(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * host_src, uint64_t size_bytes){

	if (stream_id == -1){
		return cu_transfer_host_to_dev_blocking(dev_dest, host_src, size_bytes);
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	return cu_transfer_host_to_dev_async(stream, dev_dest, host_src, size_bytes);
}

int cuda_submit_outbound_transfer(Dataflow_Handle * dataflow_handle, int stream_id, void * host_dest, void * dev_src, uint64_t size_bytes){

	if (stream_id == -1){
		return cu_transfer_dev_to_host_blocking(host_dest, dev_src, size_bytes);
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	return cu_transfer_dev_to_host_async(stream, host_dest, dev_src, size_bytes);
	
}

int cuda_submit_peer_transfer(Dataflow_Handle * dataflow_handle, int stream_id, void * dev_dest, void * dev_src, uint64_t size_bytes){

	if (stream_id == -1){
		return cu_transfer_dev_to_dev_blocking(dev_dest, dev_src, size_bytes);
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;

	CUstream stream = cu_streams[stream_id];

	return cu_transfer_dev_to_dev_async(stream, dev_dest, dev_src, size_bytes);
	
}



int cuda_set_device_info(Cuda_Device_Info * device_info, CUdevice dev){

	int ret;

	ret = cu_get_dev_total_mem(&(device_info -> total_mem), dev);
	if (ret){
		fprintf(stderr, "Error: failed to get total mem when setting device info...\n");
		return -1;
	}

	int major_arch_num;
	int minor_arch_num;

	ret = cu_get_dev_attribute(&major_arch_num, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
	if (ret){
		fprintf(stderr, "Error: failed to get major arch num for cuda device...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&minor_arch_num, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
	if (ret){
		fprintf(stderr, "Error: failed to get major arch num for cuda device...\n");
		return -1;
	}

	device_info -> arch_num = 10 * major_arch_num + minor_arch_num;

	
	ret = cu_get_dev_attribute(&(device_info -> sm_count), dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
	if (ret){
		fprintf(stderr, "Error: failed to get total sm count when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> max_threads_per_block), dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
	if (ret){
		fprintf(stderr, "Error: failed to get max threads per sm when setting device info...\n");
		return -1;
	}

	if (device_info -> max_threads_per_block > CUDA_DEV_UPPER_BOUND_MAX_THREADS_ALL_FUNC){
		device_info -> max_threads_per_block = CUDA_DEV_UPPER_BOUND_MAX_THREADS_ALL_FUNC;
	}


	ret = cu_get_dev_attribute(&(device_info -> max_smem_per_block), dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
	if (ret){
		fprintf(stderr, "Error: failed to get max smem per block when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> optin_max_smem_per_block), dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
	if (ret){
		fprintf(stderr, "Error: failed to get optin max smem per block when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_attribute(&(device_info -> host_numa_id), dev, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID);
	if (ret){
		fprintf(stderr, "Error: failed to get host numa id when setting device info...\n");
		return -1;
	}

	ret = cu_get_dev_name(device_info -> device_name, 256, dev);
	if (ret){
		fprintf(stderr, "Error: failed to get device name when setting device info...\n");
		return -1;
	}

	return 0;
}

int cuda_set_pcie_info(Dataflow_Handle * dataflow_handle){

	int ret;

	ret = cuda_nvml_init();
	if (ret){
		fprintf(stderr, "Error: failed to initialize nvml...\n");
		return -1;
	}

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	ret = cuda_nvml_get_pcie_info(dataflow_handle -> device_id, &(device_info -> pcie_link_width), &(device_info -> pcie_link_gen));
	if (ret){
		fprintf(stderr, "Error: failed to get pcie info...\n");
		return -1;
	}

	ret = cuda_nvml_shutdown();
	if (ret){
		fprintf(stderr, "Error: failed to shutdown nvml...\n");
		return -1;
	}

	return 0;
}



int dataflow_init_handle(Dataflow_Handle * dataflow_handle, ComputeType compute_type, int device_id, 
								int ctx_id, unsigned int ctx_flags, 
								int num_streams, int * opt_stream_prios, char ** opt_stream_names) {

	int ret;

	dataflow_handle -> compute_type = compute_type;
	
	// 1.) Ensure cuda has been initialized
	ret = cu_initialize_drv();
	if (ret){
		fprintf(stderr, "Error: failed to initialize cuda driver...\n");
		return -1;
	}

	// 2.) Get device handle
	dataflow_handle -> device_handle = malloc(sizeof(CUdevice));
	if (!dataflow_handle -> device_handle){
		fprintf(stderr, "Error: malloc failed to alloc space for device handle...\n");
		return -1;
	}

	dataflow_handle -> device_id = device_id;

	ret = cu_get_device(dataflow_handle -> device_handle, device_id);
	if (ret){
		fprintf(stderr, "Error: failed to get device handle for dev id: %d...\n", device_id);
		return -1;
	}


	dataflow_handle -> device_info = malloc(sizeof(Cuda_Device_Info));
	if (!dataflow_handle -> device_info){
		fprintf(stderr, "Error: malloc failed to alloc space for compute handle device info...\n");
		return -1;
	}

	ret = cuda_set_device_info(dataflow_handle -> device_info, *((CUdevice *) dataflow_handle -> device_handle));
	if (ret){
		fprintf(stderr, "Error: was unable to set cuda device info for dev id: %d...\n", device_id);
		return -1;
	}



	// 3.) Init context

	dataflow_handle -> ctx_id = ctx_id;

	dataflow_handle -> ctx = malloc(sizeof(CUcontext));
	if (!dataflow_handle -> ctx){
		fprintf(stderr, "Error: malloc failed to allocate space for cucontex container...\n");
		return -1;
	}

	ret = cu_initialize_ctx(dataflow_handle -> ctx, *((CUdevice *) dataflow_handle -> device_handle), ctx_flags);
	if (ret){
		fprintf(stderr, "Error: unable to initialize cuda context for device %d...\n", device_id);
		return -1;
	}



	// 4.) Init streams
	
	dataflow_handle -> num_streams = num_streams;

	// Handle optional arguments...

	// a.) stream priorities
	if (opt_stream_prios){
		memcpy(dataflow_handle -> stream_prios, opt_stream_prios, num_streams * sizeof(int));
	}
	else{
		for (int i = 0; i < num_streams; i++){
			(dataflow_handle -> stream_prios)[i] = CUDA_DEFAULT_STREAM_PRIO;
		}
	}

	for (int i = num_streams; i < MAX_STREAMS; i++){
		(dataflow_handle -> stream_prios)[i] = -1;
	}


	// b.) stream names (for niceness in profiling...)
	size_t stream_name_len;
	if (opt_stream_names){
		for (int i = 0; i < num_streams; i++){
			stream_name_len = strlen(opt_stream_names[i]);
			(dataflow_handle -> stream_names)[i] = malloc(stream_name_len + 1);
			if (!(dataflow_handle -> stream_names)[i]){
				fprintf(stderr, "Error: failed to alloc space to hold stream name...\n");
				return -1;
			}
			strcpy((dataflow_handle -> stream_names)[i], opt_stream_names[i]);
		}
	}
	else{
		for (int i = 0; i < num_streams; i++){
			(dataflow_handle -> stream_names)[i] = NULL;
		}
	}

	for (int i = num_streams; i < MAX_STREAMS; i++){
		(dataflow_handle -> stream_names)[i] = NULL;
	}

	// Actually create the streams + events

	dataflow_handle -> streams = malloc(num_streams * sizeof(CUstream));
	if (!dataflow_handle -> streams){
		fprintf(stderr, "Error: malloc failed to allocate space for custream container...\n");
		return -1;
	}

	dataflow_handle -> stream_states = malloc(num_streams * sizeof(CUevent));
	if (!dataflow_handle -> stream_states){
		fprintf(stderr, "Error: malloc failed to allocate space for cuevent container...\n");
		return -1;
	}

	CUstream * cu_streams = (CUstream *) dataflow_handle -> streams;
	CUevent * cu_events = (CUevent *) dataflow_handle -> stream_states;

	for (int i = 0; i < num_streams; i++){
		ret = cu_initialize_stream(&(cu_streams[i]), (dataflow_handle -> stream_prios)[i]);
		if (ret){
			fprintf(stderr, "Error: unable to initialize stream #%d on device id %d...\n", i, device_id);
			return -1;
		}

		ret = cu_initialize_event(&(cu_events[i]));
		if (ret){
			fprintf(stderr, "Error: unable to initialize event for stream #%d on device id %d...\n", i, device_id);
			return -1;
		}
	}


	// 5.) Iniitalize 0 native function libs...

	dataflow_handle -> num_native_function_libs = 0;
	
	// 6.) Init op table

	Hash_Func hash_func = &cuda_op_table_hash_func;
	uint64_t key_size_bytes = DATAFLOW_OP_IDENTIFIER_FINGERPRINT_NUM_BYTES;
	uint64_t value_size_bytes = sizeof(Cuda_Function);

	uint64_t min_table_size = OP_TABLE_MIN_SIZE;
	uint64_t max_table_size = OP_TABLE_MAX_SIZE;

	float load_factor = 1.0f;
	float shrink_factor = 1.0f;

	ret = dataflow_init_table(&(dataflow_handle -> op_table), hash_func, key_size_bytes, value_size_bytes, min_table_size, max_table_size, load_factor, shrink_factor);
	if (ret){
		fprintf(stderr, "Error: failed to init op table...\n");
		return -1;
	}

	// Set PCIe info...
	ret = cuda_set_pcie_info(dataflow_handle);
	if (ret){
		fprintf(stderr, "Error: failed to set pcie info...\n");
		return -1;
	}
	dataflow_handle -> pcie_link_width = dataflow_handle -> device_info -> pcie_link_width;
	dataflow_handle -> pcie_link_gen = dataflow_handle -> device_info -> pcie_link_gen;

	// SET FUNCTION POINTERS SO COMPUTE HANDLE CAN BE USEFUL...!

	// Accessible Device Info
	// Arch ID
	dataflow_handle -> hardware_arch_type = cuda_get_hardware_arch_type(dataflow_handle);

	// Ops Functionality
	dataflow_handle -> register_native_code = cuda_register_native_code;
	dataflow_handle -> register_external_code = cuda_register_external_code;

	// Compute Functionality
	dataflow_handle -> submit_op = cuda_submit_op;
	dataflow_handle -> submit_host_op = cuda_submit_host_op;

	// Dependency Functionality
	dataflow_handle -> get_stream_state = cuda_get_stream_state;
	dataflow_handle -> submit_dependency = cuda_submit_dependency;
	//dataflow_handle -> submit_stream_post_sem_callback = &cuda_submit_stream_post_sem_callback;
	dataflow_handle -> sync_stream = cuda_sync_stream;
	dataflow_handle -> sync_handle = cuda_sync_handle;

	// Memory Functionality
	dataflow_handle -> alloc_mem = cuda_alloc_mem;
	dataflow_handle -> free_mem = cuda_free_mem;
	dataflow_handle -> set_mem = cuda_set_mem;
	dataflow_handle -> enable_access_to_host_mem = cuda_enable_access_to_host_mem;
	dataflow_handle -> disable_access_to_host_mem = cuda_disable_access_to_host_mem;
	dataflow_handle -> enable_access_to_peer_mem = cuda_enable_access_to_peer_mem;
	dataflow_handle -> disable_access_to_peer_mem = cuda_disable_access_to_peer_mem;

	// Transfer Functionality
	dataflow_handle -> submit_inbound_transfer = cuda_submit_inbound_transfer;
	dataflow_handle -> submit_outbound_transfer = cuda_submit_outbound_transfer;
	dataflow_handle -> submit_peer_transfer = cuda_submit_outbound_transfer;


	// Initilaize profiler functionality...
	(dataflow_handle -> profiler).start = cuda_profiler_start;
	(dataflow_handle -> profiler).stop = cuda_profiler_stop;
	(dataflow_handle -> profiler).name_device = cuda_profiler_name_device;
	(dataflow_handle -> profiler).name_handle = cuda_profiler_name_handle;
	(dataflow_handle -> profiler).name_stream = cuda_profiler_name_stream;
	(dataflow_handle -> profiler).name_thread = cuda_profiler_name_thread;
	(dataflow_handle -> profiler).range_push = cuda_profiler_range_push;
	(dataflow_handle -> profiler).range_pop = cuda_profiler_range_pop;

	// Now name all of the streams...!
	if (opt_stream_names){
		for (int i = 0; i < num_streams; i++){
			if (opt_stream_names[i]){
				cuda_profiler_name_stream(dataflow_handle, i, opt_stream_names[i]);
			}
		}
	}

	return 0;

}
