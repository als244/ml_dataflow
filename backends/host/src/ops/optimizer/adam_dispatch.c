#include "host_ops.h"
#include "adam_step.h"

// only takes one argument as parameter as 
// this is typically called by (-> submit_host_op)
// dataflow api function
int adam_step_host(void * _op){

	Op * op = (Op *) _op;

	Op_Skeleton * skeleton = &(op -> op_skeleton);
	
	DataflowDatatype * arg_dtypes = (skeleton -> header).arg_dtypes;

	DataflowDatatype param_dt = arg_dtypes[7];
	DataflowDatatype grad_dt = arg_dtypes[8];
	DataflowDatatype mean_dt = arg_dtypes[9];
	DataflowDatatype var_dt = arg_dtypes[10];

	void ** op_args = op -> op_args;

	int num_threads = *((int *) op_args[0]);
	uint64_t num_els = *((uint64_t *) op_args[1]);
	float lr = *((float *) op_args[2]);
	float beta1 = *((float *) op_args[3]);
	float beta2 = *((float *) op_args[4]);
	float weight_decay = *((float *) op_args[5]);
	float epsilon = *((float *) op_args[6]);

	void * param = *((void **) op_args[7]);
	void * grad = *((void **) op_args[8]);
	void * mean = *((void **) op_args[9]);
	void * var = *((void **) op_args[10]);

	 if (__builtin_cpu_supports("avx512f")){
        return do_adam_step_host_avx512(param_dt, grad_dt, mean_dt, var_dt, num_threads, num_els, lr, beta1, beta2, weight_decay, epsilon, param, grad, mean, var);
    }
    else{
        return do_adam_step_host(param_dt, grad_dt, mean_dt, var_dt, num_threads, num_els, lr, beta1, beta2, weight_decay, epsilon, param, grad, mean, var);
    }
}