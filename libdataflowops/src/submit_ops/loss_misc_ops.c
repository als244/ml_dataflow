#include "dataflow_ops.h"


int dataflow_submit_softmax(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int n_rows, int n_cols,
						void * X, void * out){

	int ret;

	Op softmax_op;

	dataflow_set_softmax_skeleton(&softmax_op.op_skeleton, fwd_dt, bwd_dt);

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


int dataflow_submit_cross_entropy_loss(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype bwd_dt,
								int n_rows, int n_cols,
								void * pred_logits, uint32_t * labels, float * loss_vec, float * loss_sum) {

	int ret;

	Op cross_entropy_loss_op;

	dataflow_set_cross_entropy_loss_skeleton(&cross_entropy_loss_op.op_skeleton, bwd_dt);

	void ** op_args = cross_entropy_loss_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &pred_logits;
	op_args[3] = &labels;
	op_args[4] = &loss_vec;
	op_args[5] = &loss_sum;

	ret = (handle -> submit_op)(handle, &cross_entropy_loss_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit cross entropy loss op...\n");
		return -1;
	}

	return 0;



}
