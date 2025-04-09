#include "dataflow_ops.h"

int dataflow_submit_default_embedding_table(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int num_unique_tokens, int embed_dim, 
						uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, 
						void * embedding_table, void * output) {

    
    int ret;

	Op embedding_table_op;

	dataflow_set_default_embedding_table_skeleton(&embedding_table_op.op_skeleton, fwd_dt);

	void ** op_args = embedding_table_op.op_args;

	op_args[0] = &num_unique_tokens;
	op_args[1] = &embed_dim;
	op_args[2] = &sorted_token_ids;
	op_args[3] = &sorted_token_mapping;
	op_args[4] = &unique_token_sorted_inds_start;
	op_args[5] = &embedding_table;
	op_args[6] = &output;

	ret = (handle -> submit_op)(handle, &embedding_table_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit embedding table op...\n");
		return -1;
	}

	return 0;
}

int dataflow_submit_default_embedding_table_bwd_w(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype bwd_dt, 
						int num_unique_tokens, int embed_dim, 
						uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, 
						void * grad_stream, void * grad_embedding_table) {


	int ret;

	Op embedding_table_bwd_w_op;

	dataflow_set_default_embedding_table_bwd_w_skeleton(&embedding_table_bwd_w_op.op_skeleton, bwd_dt);	

	void ** op_args = embedding_table_bwd_w_op.op_args;

	op_args[0] = &num_unique_tokens;
	op_args[1] = &embed_dim;
	op_args[2] = &sorted_token_ids;
	op_args[3] = &sorted_token_mapping;
	op_args[4] = &unique_token_sorted_inds_start;	
	op_args[5] = &grad_stream;
	op_args[6] = &grad_embedding_table;

	ret = (handle -> submit_op)(handle, &embedding_table_bwd_w_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit embedding table bwd w op...\n");
		return -1;	
	}

	return 0;	
}