#include "dataflow_ops.h"

int dataflow_submit_rope(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * X_q, void * X_k){

	int ret;

	Op rope_op;

	dataflow_set_rope_skeleton(&rope_op.op_skeleton, fwd_dt);

	void ** op_args = rope_op.op_args;

	op_args[0] = &N;
	op_args[1] = &model_dim;
	op_args[2] = &head_dim;
	op_args[3] = &num_kv_heads;
	op_args[4] = &theta;
	op_args[5] = &seq_positions;
	op_args[6] = &X_q;
	op_args[7] = &X_k;


	ret = (handle -> submit_op)(handle, &rope_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rope op...\n");
		return -1;
	}

	return 0;
}

int dataflow_submit_rope_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype bwd_dt, 
						uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * dX_q, void * dX_k){

	int ret;

	Op rope_bwd_op;

	dataflow_set_rope_bwd_x_skeleton(&rope_bwd_op.op_skeleton, bwd_dt);

	void ** op_args = rope_bwd_op.op_args;

	op_args[0] = &N;
	op_args[1] = &model_dim;
	op_args[2] = &head_dim;
	op_args[3] = &num_kv_heads;
	op_args[4] = &theta;
	op_args[5] = &seq_positions;
	op_args[6] = &dX_q;
	op_args[7] = &dX_k;


	ret = (handle -> submit_op)(handle, &rope_bwd_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit rope op...\n");
		return -1;
	}

	return 0;
}



int dataflow_submit_copy_to_seq_context(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int total_tokens, int kv_dim, 
						void * X_k, void * X_v, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes) {

	int ret;

	Op copy_to_seq_contrxt_op;

	dataflow_set_copy_to_seq_context_skeleton(&copy_to_seq_contrxt_op.op_skeleton, fwd_dt);

	void ** op_args = copy_to_seq_contrxt_op.op_args;

	op_args[0] = &N;
	op_args[1] = &total_tokens;
	op_args[2] = &kv_dim;
	op_args[3] = &X_k;
	op_args[4] = &X_v;
	op_args[5] = &seq_positions;
	op_args[6] = &seq_context_ptrs;
	op_args[7] = &seq_context_sizes;


	ret = (handle -> submit_op)(handle, &copy_to_seq_contrxt_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit copy to seq context op...\n");
		return -1;
	}

	return 0;


}


