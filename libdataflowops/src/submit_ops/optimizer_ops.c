#include "dataflow_ops.h"


int dataflow_submit_default_adamw_step(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype param_dt, DataflowDatatype grad_dt, 
						DataflowDatatype mean_dt, DataflowDatatype var_dt,
						uint64_t num_els, int step_num,
						float lr, float beta1, float beta2, float weight_decay, float epsilon,
						void * param, void * grad, void * mean, void * var){

    int ret;

	Op adam_step_op;

	dataflow_set_default_adamw_step_skeleton(&adam_step_op.op_skeleton, param_dt, grad_dt, mean_dt, var_dt);

	void ** op_args = adam_step_op.op_args;

	op_args[0] = &num_els;
    op_args[1] = &step_num;
	op_args[2] = &lr;
	op_args[3] = &beta1;
	op_args[4] = &beta2;
	op_args[5] = &weight_decay;
	op_args[6] = &epsilon;
	op_args[7] = &param;
	op_args[8] = &grad;
	op_args[9] = &mean;
	op_args[10] = &var;


	ret = (handle -> submit_op)(handle, &adam_step_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit adam step...\n");
		return -1;
	}

	return 0;
}


