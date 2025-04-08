#include "dataflow_ops.h"

int dataflow_submit_default_swiglu(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int num_rows, int num_cols, 
						void * x_w1, void * x_w3, void * out) {

	int ret;

	Op swiglu_op;

	dataflow_set_default_swiglu_skeleton(&swiglu_op.op_skeleton, fwd_dt);

	void ** op_args = swiglu_op.op_args;

	op_args[0] = &num_rows;
	op_args[1] = &num_cols;
	op_args[2] = &x_w1;
	op_args[3] = &x_w3;
	op_args[4] = &out;


	ret = (handle -> submit_op)(handle, &swiglu_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu op...\n");
		return -1;
	}

	return 0;
}

int dataflow_submit_default_swiglu_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int num_rows, int num_cols, 
						void * x_w1, void * x_w3, 
						void * upstream_dX, void * dX_w1, void * dX_w3) {

	int ret;
	Op swiglu_bwd_x_op;

	dataflow_set_default_swiglu_bwd_x_skeleton(&swiglu_bwd_x_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = swiglu_bwd_x_op.op_args;

	op_args[0] = &num_rows;
	op_args[1] = &num_cols;
	op_args[2] = &x_w1;
	op_args[3] = &x_w3;
	op_args[4] = &upstream_dX;
	op_args[5] = &dX_w1;
	op_args[6] = &dX_w3;


	ret = (handle -> submit_op)(handle, &swiglu_bwd_x_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu bwd x op...\n");
		return -1;
	}

	return 0;
}