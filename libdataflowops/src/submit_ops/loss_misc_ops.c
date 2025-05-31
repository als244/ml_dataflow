#include "dataflow_ops.h"


int dataflow_submit_default_softmax(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int n_rows, int n_cols,
						void * X, void * out){

	int ret;

	Op softmax_op;

	dataflow_set_default_softmax_skeleton(&softmax_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = softmax_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &X;
	op_args[3] = &out;

	ret = (handle -> submit_op)(handle, &softmax_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit softmax op...\n");
		return -1;
	}

	return 0;
}



