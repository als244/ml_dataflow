#include "dataflow_ops.h"

int dataflow_submit_default_embedding_table(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int num_tokens, int embed_dim, 
						uint32_t * token_ids, void * embedding_table, void * output){

    
    int ret;

	Op embedding_table_op;

	dataflow_set_default_embedding_table_skeleton(&embedding_table_op.op_skeleton, fwd_dt);

	void ** op_args = embedding_table_op.op_args;

	op_args[0] = &num_tokens;
	op_args[1] = &embed_dim;
	op_args[2] = &token_ids;
	op_args[3] = &embedding_table;
	op_args[4] = &output;

	ret = (handle -> submit_op)(handle, &embedding_table_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit embedding table op...\n");
		return -1;
	}

	return 0;
                        

}