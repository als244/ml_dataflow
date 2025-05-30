#include "dataflow_ops.h"


int dataflow_submit_default_rms_norm(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * rms_weight, void * X, void * out, float * rms_vals){
	
	int ret;

	Op rms_norm_op;

	dataflow_set_default_rms_norm_skeleton(&rms_norm_op.op_skeleton, fwd_dt);

	void ** op_args = rms_norm_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &rms_weight;
	op_args[4] = &X;
	op_args[5] = &out;
	op_args[6] = &rms_vals;


	ret = (handle -> submit_op)(handle, &rms_norm_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to rms norm...\n");
		return -1;
	}

	return 0;
}

int dataflow_submit_default_rms_norm_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals,
								void * rms_weight, void * X_inp, void * upstream_dX, void * dX, void * X_out){
	
	int ret;

	Op rms_norm_bwd_x_op;

	dataflow_set_default_rms_norm_bwd_x_skeleton(&rms_norm_bwd_x_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = rms_norm_bwd_x_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &fwd_rms_vals;
	op_args[4] = &rms_weight;
	op_args[5] = &X_inp;
	op_args[6] = &upstream_dX;
	op_args[7] = &dX;
	op_args[8] = &X_out;

	ret = (handle -> submit_op)(handle, &rms_norm_bwd_x_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd x op...\n");
		return -1;
	}

	return 0;
}


int dataflow_submit_default_rms_norm_bwd_w(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals, void * X_inp, void * upstream_dX, void * dW, 
								uint64_t workspaceBytes, void * workspace) {
	int ret;
	Op rms_norm_bwd_w_op;

	int * ret_num_blocks_launched = (int *)workspace;
	void * dW_workspace = workspace + 256;

	dataflow_set_default_rms_norm_bwd_w_skeleton(&rms_norm_bwd_w_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = rms_norm_bwd_w_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &fwd_rms_vals;
	op_args[4] = &X_inp;
	op_args[5] = &upstream_dX;
	op_args[6] = &dW_workspace;
	op_args[7] = &ret_num_blocks_launched;
	// this is not used within the kernel, but it gets passed to the set config function so we can bounds check...
	op_args[8] = &workspaceBytes;

	ret = (handle -> submit_op)(handle, &rms_norm_bwd_w_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd w op...\n");
		return -1;
	}

	Op rms_norm_bwd_w_combine_op;

	dataflow_set_default_rms_norm_bwd_w_combine_skeleton(&rms_norm_bwd_w_combine_op.op_skeleton, bwd_dt);

	op_args = rms_norm_bwd_w_combine_op.op_args;
	
	op_args[0] = &ret_num_blocks_launched;
	op_args[1] = &n_cols;
	op_args[2] = &dW_workspace;
	op_args[3] = &dW;

	ret = (handle -> submit_op)(handle, &rms_norm_bwd_w_combine_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm bwd w combine op...\n");
		return -1;
	}

	return 0;
}


int dataflow_submit_default_rms_norm_noscale(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * X, void * out, float * rms_vals){

	int ret;

	Op rms_norm_noscale_op;

	dataflow_set_default_rms_norm_noscale_skeleton(&rms_norm_noscale_op.op_skeleton, fwd_dt);

	void ** op_args = rms_norm_noscale_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &X;
	op_args[4] = &out;
	op_args[5] = &rms_vals;


	ret = (handle -> submit_op)(handle, &rms_norm_noscale_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to rms norm noscale...\n");
		return -1;
	}

	return 0;


}

int dataflow_submit_default_rms_norm_noscale_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals,
								void * X_inp, void * upstream_dX, void * dX, void * X_out) {
	
	int ret;

	Op rms_norm_noscale_bwd_x_op;

	dataflow_set_default_rms_norm_noscale_bwd_x_skeleton(&rms_norm_noscale_bwd_x_op.op_skeleton, fwd_dt, bwd_dt);

	void ** op_args = rms_norm_noscale_bwd_x_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &eps;
	op_args[3] = &fwd_rms_vals;
	op_args[4] = &X_inp;
	op_args[5] = &upstream_dX;
	op_args[6] = &dX;
	op_args[7] = &X_out;


	ret = (handle -> submit_op)(handle, &rms_norm_noscale_bwd_x_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rms norm noscale bwd x op...\n");
		return -1;
	}

	return 0;
}