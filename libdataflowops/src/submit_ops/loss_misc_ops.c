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


int dataflow_submit_default_cross_entropy_loss(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype bwd_dt,
								int n_rows, int n_cols,
								void * pred_logits, uint32_t * labels, float * loss_vec) {

	int ret;

	Op cross_entropy_loss_op;

	dataflow_set_default_cross_entropy_loss_skeleton(&cross_entropy_loss_op.op_skeleton, bwd_dt);

	void ** op_args = cross_entropy_loss_op.op_args;

	op_args[0] = &n_rows;
	op_args[1] = &n_cols;
	op_args[2] = &pred_logits;
	op_args[3] = &labels;
	op_args[4] = &loss_vec;

	ret = (handle -> submit_op)(handle, &cross_entropy_loss_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit cross entropy loss op...\n");
		return -1;
	}

	return 0;



}

// Print Loss Host Op

int dataflow_submit_print_chunk_loss_host(Dataflow_Handle * handle, int stream_id,
									void * print_chunk_loss_host_func, Print_Chunk_Loss_Host_Op_Args * op_buffer,
									int step_num, int round_num, int seq_id, int chunk_id, int num_tokens, float * avg_loss_ref){

	int ret;
    
	op_buffer -> step_num = step_num;
	op_buffer -> round_num = round_num;
    op_buffer -> seq_id = seq_id;
    op_buffer -> chunk_id = chunk_id;
    op_buffer -> num_tokens = num_tokens;
    op_buffer -> avg_loss_ref = avg_loss_ref;


    ret = (handle -> submit_host_op)(handle, print_chunk_loss_host_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit print chunk loss host op...\n");
        return -1;
    }

    return 0;
}


int dataflow_submit_print_step_loss_host(Dataflow_Handle * handle, int stream_id,
									void * print_step_loss_host_func, Print_Step_Loss_Host_Op_Args * op_buffer,
									int step_num, int round_num, int num_chunks, int total_tokens, float * per_chunk_avg_loss){

	int ret;

	op_buffer -> step_num = step_num;
	op_buffer -> round_num = round_num;
	op_buffer -> num_chunks = num_chunks;
	op_buffer -> total_tokens = total_tokens;
	op_buffer -> per_chunk_avg_loss = per_chunk_avg_loss;

	ret = (handle -> submit_host_op)(handle, print_step_loss_host_func, op_buffer, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit print step loss host op...\n");
		return -1;
	}

	return 0;
}


