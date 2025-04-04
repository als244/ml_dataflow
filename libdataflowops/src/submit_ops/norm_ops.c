#include "dataflow_ops.h"


int dataflow_submit_rms_norm(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * rms_weight, void * X, void * out, float * weighted_sums, float * rms_vals){
	
	int ret;

	Op rms_norm_op;

	dataflow_set_rms_norm_skeleton(&rms_norm_op.op_skeleton, fwd_dt);

	void ** op_args = rms_norm_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &rms_weight;
	op_args[4] = &X;
	op_args[5] = &out;
	op_args[6] = &weighted_sums;
	op_args[7] = &rms_vals;


	ret = (handle -> submit_op)(handle, &rms_norm_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to rms norm...\n");
		return -1;
	}

	return 0;
}


int dataflow_submit_rms_norm_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_weighted_sums, float * fwd_rms_vals,
								 void * rms_weight, void * X_inp, void * upstream_dX, void * dX){
	
	int ret;

	Op rms_norm_bwd_x_op;

	dataflow_set_rms_norm_bwd_x_skeleton(&rms_norm_bwd_x_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = rms_norm_bwd_x_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &fwd_weighted_sums;
	op_args[4] = &fwd_rms_vals;
	op_args[5] = &rms_weight;
	op_args[6] = &X_inp;
	op_args[7] = &upstream_dX;
	op_args[8] = &dX;


	ret = (handle -> submit_op)(handle, &rms_norm_bwd_x_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd x op...\n");
		return -1;
	}

	return 0;
}


int dataflow_submit_rms_norm_bwd_w(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals, void * X_inp, void * upstream_dX, void * dW) {
	int ret;
	Op rms_norm_bwd_w_op;

	dataflow_set_rms_norm_bwd_w_skeleton(&rms_norm_bwd_w_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = rms_norm_bwd_w_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &fwd_rms_vals;
	op_args[4] = &X_inp;
	op_args[5] = &upstream_dX;
	op_args[6] = &dW;


	ret = (handle -> submit_op)(handle, &rms_norm_bwd_w_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd w op...\n");
		return -1;
	}
}
