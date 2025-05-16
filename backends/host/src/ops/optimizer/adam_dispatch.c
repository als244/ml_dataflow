#include "host_ops.h"
#include "adam_step.h"

// only takes one argument as parameter as 
// this is typically called by (-> submit_host_op)
// dataflow api function
int adam_step_host(void * _adam_host_op_args){

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

	printf("[Adam Dispatcher] Optimizing Layer ID: %d...\n\n", layer_id);

	 if (__builtin_cpu_supports("avx512f")){
        return do_adam_step_host_avx512(param_dt, grad_dt, mean_dt, var_dt, num_threads, num_els, lr, beta1, beta2, weight_decay, epsilon, param, grad, mean, var);
    }
    else{
        return do_adam_step_host(param_dt, grad_dt, mean_dt, var_dt, num_threads, num_els, lr, beta1, beta2, weight_decay, epsilon, param, grad, mean, var);
    }
}