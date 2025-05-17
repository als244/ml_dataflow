#include "host_ops.h"
#include "adam_step.h"

#define TO_PRINT 1


// Just for testing...
#define TO_SAVE_RESULTS 0
#define TOTAL_LAYERS 16
#define DATA_SAVE_DIR "test_transformer_data"

static int save_file(int layer_id, char * filename, void * ptr, uint64_t num_els, DataflowDatatype dt) {

	int ret;

	size_t el_size = dataflow_sizeof_element(dt);

	char full_filename[1024];

	if (layer_id == TOTAL_LAYERS){
		sprintf(full_filename, "%s/optimizer_states/head/%s.dat", DATA_SAVE_DIR, filename);
	}
	else if (layer_id == -1){
		sprintf(full_filename, "%s/optimizer_states/embedding/%s.dat", DATA_SAVE_DIR, filename);
	}
	else if ((layer_id < -1) || (layer_id > TOTAL_LAYERS)){
		fprintf(stderr, "Error: invalid layer id: %d\n", layer_id);
		return -1;
	}
	else{
		sprintf(full_filename, "%s/optimizer_states/layers/%d/%s.dat", DATA_SAVE_DIR, layer_id, filename);
	}


	FILE * fp = fopen(full_filename, "wb");
	if (!fp){
		fprintf(stderr, "Error: failed to save %s, because couldn't open file: %s...\n", filename, full_filename);
		return -1;
	}

	if (TO_PRINT){
		printf("\n[Saving] %s (%lu bytes)\n", filename, num_els * el_size);
	}

	size_t num_written = fwrite(ptr, el_size, num_els, fp);
	if (num_written != num_els){
		fprintf(stderr, "Error: failed to write to file %s, wrote %zu elements instead of %zu\n", filename, num_written, num_els);
		return -1;
	}

	fclose(fp);

	return 0;
}


// only takes one argument as parameter as 
// this is typically called by (-> submit_host_op)
// dataflow api function
int adam_step_host(void * _adam_host_op_args){

	int ret;

	Adam_Host_Op_Args * adam_host_op_args = (Adam_Host_Op_Args *) _adam_host_op_args;

	DataflowDatatype param_dt = adam_host_op_args -> param_dt;
	DataflowDatatype grad_dt = adam_host_op_args -> grad_dt;
	DataflowDatatype mean_dt = adam_host_op_args -> mean_dt;
	DataflowDatatype var_dt = adam_host_op_args -> var_dt;

	// unlike other ops, these are not references but rather op values themselves...

	int num_threads = adam_host_op_args -> num_threads;
	uint64_t num_els = adam_host_op_args -> num_els;
	int layer_id = adam_host_op_args -> layer_id;
	float lr = adam_host_op_args -> lr;
	float beta1 = adam_host_op_args -> beta1;
	float beta2 = adam_host_op_args -> beta2;
	float weight_decay = adam_host_op_args -> weight_decay;
	float epsilon = adam_host_op_args -> epsilon;

	void * param = adam_host_op_args -> param;
	void * grad = adam_host_op_args -> grad;
	void * mean = adam_host_op_args -> mean;
	void * var = adam_host_op_args -> var;

	if (TO_PRINT){
		printf("[Adam Dispatcher] Optimizing Layer ID: %d...\n\n", layer_id);
	}

	if (TO_SAVE_RESULTS){

		if (TO_PRINT){
			printf("[Adam Dispatcher] Saving Results for Layer ID: %d...\n\n", layer_id);
		}

		save_file(layer_id, "param_pre_step", param, num_els, param_dt);
		save_file(layer_id, "grad_pre_step", mean, num_els, mean_dt);
		save_file(layer_id, "mean_pre_step", mean, num_els, mean_dt);
		save_file(layer_id, "var_pre_step", var, num_els, var_dt);
	}

	 if (__builtin_cpu_supports("avx512f")){
        ret = do_adam_step_host_avx512(param_dt, grad_dt, mean_dt, var_dt, num_threads, num_els, lr, beta1, beta2, weight_decay, epsilon, param, grad, mean, var);
    }
    else{
        ret = do_adam_step_host(param_dt, grad_dt, mean_dt, var_dt, num_threads, num_els, lr, beta1, beta2, weight_decay, epsilon, param, grad, mean, var);
    }

	if (ret){
		fprintf(stderr, "Error: failed to do adam step...\n");
		return -1;
	}

	if (TO_SAVE_RESULTS){

		if (TO_PRINT){
			printf("[Adam Dispatcher] Saving Results for Layer ID: %d...\n\n", layer_id);
		}

		save_file(layer_id, "param_post_step", param, num_els, param_dt);
		save_file(layer_id, "mean_post_step", mean, num_els, mean_dt);
		save_file(layer_id, "var_post_step", var, num_els, var_dt);
	}

	return 0;
}