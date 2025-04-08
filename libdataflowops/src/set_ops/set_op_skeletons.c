#include "set_op_skeletons.h"

// This function should just be called during the registration of the op
// Not performance senesitive so the strcmps are fine...

// switches into respective function based on op_base_name
// return 0 on success, -1 on failure
// depending on function might either use fwd_dt or bwd_dt or both
int dataflow_set_op_skeleton(Op_Skeleton * skeleton, char * op_name, DataflowDatatype fwd_dt, DataflowDatatype bwd_dt) {

	if (strcmp(op_name, "default_embedding_table") == 0) {
        dataflow_set_default_embedding_table_skeleton(skeleton, fwd_dt);
    } 
	else if (strcmp(op_name, "default_rms_norm") == 0) {
        dataflow_set_default_rms_norm_skeleton(skeleton, fwd_dt);
    }
	else if (strcmp(op_name, "default_rms_norm_bwd_x") == 0) {
		dataflow_set_default_rms_norm_bwd_x_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "default_rms_norm_bwd_w") == 0) {
		dataflow_set_default_rms_norm_bwd_w_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "default_rms_norm_noscale") == 0) {
		dataflow_set_default_rms_norm_noscale_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "default_rms_norm_noscale_bwd_x") == 0) {
		dataflow_set_default_rms_norm_noscale_bwd_x_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "default_rope") == 0) {
		dataflow_set_default_rope_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "default_rope_bwd_x") == 0) {
		dataflow_set_default_rope_bwd_x_skeleton(skeleton, bwd_dt);
	}
	else if (strcmp(op_name, "default_copy_to_seq_context") == 0) {
		dataflow_set_default_copy_to_seq_context_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "default_select_experts") == 0) {
		dataflow_set_default_select_experts_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "default_swiglu") == 0) {
		dataflow_set_default_swiglu_skeleton(skeleton, fwd_dt);
	}
	else if (strcmp(op_name, "default_swiglu_bwd_x") == 0) {
		dataflow_set_default_swiglu_bwd_x_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "default_softmax") == 0) {
		dataflow_set_default_softmax_skeleton(skeleton, fwd_dt, bwd_dt);
	}
	else if (strcmp(op_name, "default_cross_entropy_loss") == 0) {
		dataflow_set_default_cross_entropy_loss_skeleton(skeleton, bwd_dt);
	}
	else {
		// External ops (for current cuda ops implementation)
		if (strcmp(op_name, "matmul") == 0) {
			dataflow_set_matmul_skeleton(skeleton);
		}
		else if (strcmp(op_name, "flash3_attention_fwd") == 0) {
			dataflow_set_flash3_attention_fwd_skeleton(skeleton);
		}
		else if (strcmp(op_name, "flash3_attention_bwd") == 0) {
			dataflow_set_flash3_attention_bwd_skeleton(skeleton);
		}
		else{
			printf("Cannot set skeleton, unknown op: %s, with fwd_dt: %s, bwd_dt: %s\n", op_name, dataflow_datatype_as_string(fwd_dt), dataflow_datatype_as_string(bwd_dt));
			return -1;
		}
	}
	return 0;
}

void dataflow_set_matmul_skeleton(Op_Skeleton * skeleton) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s", "default_matmul");

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 19;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// num_sms
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;

	// A DataflowDatatype
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// B DataflowDatatype
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	// C DataflowDatatype
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	// D DataflowDatatype
	arg_dtypes[4] = DATAFLOW_INT_SCALAR;
	// Compute Type as DataflowDatatype (FP32, FP16, or BF16)
	arg_dtypes[5] = DATAFLOW_INT_SCALAR;

	// to_trans_a
	arg_dtypes[6] = DATAFLOW_INT_SCALAR;
	// to_trans_b
	arg_dtypes[7] = DATAFLOW_INT_SCALAR;

	// M
	arg_dtypes[8] = DATAFLOW_INT_SCALAR;
	// K
	arg_dtypes[9] = DATAFLOW_INT_SCALAR;
	// N
	arg_dtypes[10] = DATAFLOW_INT_SCALAR;
	// alpha
	arg_dtypes[11] = DATAFLOW_FP32_SCALAR;
	// beta
	arg_dtypes[12] = DATAFLOW_FP32_SCALAR;
	
	// A
	arg_dtypes[13] = DATAFLOW_VOID;
	// B
	arg_dtypes[14] = DATAFLOW_VOID;
	// C
	arg_dtypes[15] = DATAFLOW_VOID;
	// D
	arg_dtypes[16] = DATAFLOW_VOID;

	// workspace bytes
	arg_dtypes[17] = DATAFLOW_UINT64;
	// workspace
	arg_dtypes[18] = DATAFLOW_VOID;
	

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);

}

void dataflow_set_flash3_attention_fwd_skeleton(Op_Skeleton * skeleton) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s", "flash3_attention_fwd");

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	

	int num_args = 20;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// flash_dtype_as_int
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	
	// num_seqs
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// total_q
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	// total_k
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	// q_seq_offsets
	arg_dtypes[4] = DATAFLOW_INT;
	// q_seq_lens
	arg_dtypes[5] = DATAFLOW_INT;
	// max_seqlen_q
	arg_dtypes[6] = DATAFLOW_INT_SCALAR;
	// k_seq_offsets
	arg_dtypes[7] = DATAFLOW_INT;
	// k_seq_lens
	arg_dtypes[8] = DATAFLOW_INT;
	// max_seqlen_k
	arg_dtypes[9] = DATAFLOW_INT_SCALAR;

	// num_q_heads
	arg_dtypes[10] = DATAFLOW_INT_SCALAR;
	// num_kv_heads
	arg_dtypes[11] = DATAFLOW_INT_SCALAR;
	// head_dim
	arg_dtypes[12] = DATAFLOW_INT_SCALAR;
	
	// x_q
	arg_dtypes[13] = DATAFLOW_VOID;
	// x_k
	arg_dtypes[14] = DATAFLOW_VOID;
	// x_v
	arg_dtypes[15] = DATAFLOW_VOID;

	
	// x_attn_out
	arg_dtypes[16] = DATAFLOW_VOID;
	// softmax_lse
	arg_dtypes[17] = DATAFLOW_VOID;

	// workspaceBytes 
	arg_dtypes[18] = DATAFLOW_UINT64_SCALAR;

	// attn_workspace
	arg_dtypes[19] = DATAFLOW_VOID;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_flash3_attention_bwd_skeleton(Op_Skeleton * skeleton) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s", "flash3_attention_bwd");

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	

	int num_args = 24;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// flash_dtype_as_int
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	
	// num_seqs
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// total_q
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	// total_k
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	// q_seq_offsets
	arg_dtypes[4] = DATAFLOW_INT;
	// q_seq_lens
	arg_dtypes[5] = DATAFLOW_INT;
	// max_seqlen_q
	arg_dtypes[6] = DATAFLOW_INT_SCALAR;
	// k_seq_offsets
	arg_dtypes[7] = DATAFLOW_INT;
	// k_seq_lens
	arg_dtypes[8] = DATAFLOW_INT;
	// max_seqlen_k
	arg_dtypes[9] = DATAFLOW_INT_SCALAR;

	// num_q_heads
	arg_dtypes[10] = DATAFLOW_INT_SCALAR;
	// num_kv_heads
	arg_dtypes[11] = DATAFLOW_INT_SCALAR;
	// head_dim
	arg_dtypes[12] = DATAFLOW_INT_SCALAR;
	
	// x_q
	arg_dtypes[13] = DATAFLOW_VOID;
	// x_k
	arg_dtypes[14] = DATAFLOW_VOID;
	// x_v
	arg_dtypes[15] = DATAFLOW_VOID;

	
	// x_attn_out
	arg_dtypes[16] = DATAFLOW_VOID;
	// softmax_lse
	arg_dtypes[17] = DATAFLOW_VOID;

	// dx_out (upstream gradient)
	arg_dtypes[18] = DATAFLOW_VOID;

	// dx_q
	arg_dtypes[19] = DATAFLOW_VOID;
	// dx_k
	arg_dtypes[20] = DATAFLOW_VOID;
	// dx_v
	arg_dtypes[21] = DATAFLOW_VOID;

	// workspaceBytes
	arg_dtypes[22] = DATAFLOW_UINT64_SCALAR;

	// attn_bwd_workspace
	arg_dtypes[23] = DATAFLOW_VOID;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}


void dataflow_set_default_embedding_table_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_embedding_table", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 5;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	// num tokens
	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	// token ids
	arg_dtypes[2] = DATAFLOW_UINT32;
	// embedding table	
	arg_dtypes[3] = fwd_datatype;
	// output
	arg_dtypes[4] = fwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
	
}

void dataflow_set_default_rms_norm_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_rms_norm", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = fwd_datatype;
	arg_dtypes[6] = DATAFLOW_FP32;
	arg_dtypes[7] = DATAFLOW_FP32;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);

}

void dataflow_set_default_rms_norm_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "default_rms_norm_bwd_x", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 9;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = DATAFLOW_FP32;
	arg_dtypes[4] = DATAFLOW_FP32;
	arg_dtypes[5] = fwd_datatype;
	arg_dtypes[6] = fwd_datatype;
	arg_dtypes[7] = bwd_datatype;
	arg_dtypes[8] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_rms_norm_bwd_w_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "default_rms_norm_bwd_w", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 7;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = DATAFLOW_FP32;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = bwd_datatype;
	arg_dtypes[6] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_rms_norm_noscale_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_rms_norm_noscale", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 7;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = DATAFLOW_FP32;
	arg_dtypes[6] = DATAFLOW_FP32;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);

}

void dataflow_set_default_rms_norm_noscale_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "default_rms_norm_noscale_bwd_x", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_FP32_SCALAR;
	arg_dtypes[3] = DATAFLOW_FP32;
	arg_dtypes[4] = DATAFLOW_FP32;
	arg_dtypes[5] = fwd_datatype;
	arg_dtypes[6] = bwd_datatype;
	arg_dtypes[7] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_rope_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_rope", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_UINT64_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	arg_dtypes[4] = DATAFLOW_INT_SCALAR;
	arg_dtypes[5] = DATAFLOW_INT;
	arg_dtypes[6] = fwd_datatype;
	arg_dtypes[7] = fwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_rope_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_rope_bwd_x", dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_UINT64_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = DATAFLOW_INT_SCALAR;
	arg_dtypes[4] = DATAFLOW_INT_SCALAR;
	arg_dtypes[5] = DATAFLOW_INT;
	arg_dtypes[6] = bwd_datatype;
	arg_dtypes[7] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_copy_to_seq_context_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_copy_to_seq_context", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 
	
	int num_args = 8;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_UINT64_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = DATAFLOW_INT;
	arg_dtypes[6] = DATAFLOW_UINT64;
	arg_dtypes[7] = DATAFLOW_INT;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);

}

void dataflow_set_default_select_experts_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_select_experts", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 

	int num_args = 9;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = DATAFLOW_INT_SCALAR;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;
	arg_dtypes[5] = DATAFLOW_UINT16;
	arg_dtypes[6] = DATAFLOW_INT;
	arg_dtypes[7] = DATAFLOW_INT;
	arg_dtypes[8] = DATAFLOW_INT;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);

}

void dataflow_set_default_swiglu_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s", "default_swiglu", dataflow_datatype_as_string(fwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0'; 

	int num_args = 5;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = fwd_datatype;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = fwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);

}

void dataflow_set_default_swiglu_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype) {

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "default_swiglu_bwd_x", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0';

	int num_args = 7;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = fwd_datatype;
	arg_dtypes[3] = fwd_datatype;
	arg_dtypes[4] = bwd_datatype;
	arg_dtypes[5] = bwd_datatype;
	arg_dtypes[6] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_softmax_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];

	sprintf(op_nickname, "%s_%s_%s", "default_softmax", dataflow_datatype_as_string(fwd_datatype), dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0';
	
	int num_args = 4;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = fwd_datatype;
	arg_dtypes[3] = bwd_datatype;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}

void dataflow_set_default_cross_entropy_loss_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype){

	Op_Skeleton_Header * skeleton_header = &(skeleton -> header);

	char op_nickname[MAX_OP_NICKNAME_SIZE];
	
	sprintf(op_nickname, "%s_%s", "default_cross_entropy_loss", dataflow_datatype_as_string(bwd_datatype));

	// MAX nicknmae size is set to 255 with 256 allocated space...
	strncpy(skeleton_header -> op_nickname, op_nickname, MAX_OP_NICKNAME_SIZE);
	// last character must be null no matter what, if nickname is less than null bytes were added prior
	(skeleton_header -> op_nickname)[MAX_OP_NICKNAME_SIZE] = '\0';
	
	int num_args = 5;

	skeleton_header -> num_args = num_args;

	DataflowDatatype * arg_dtypes = skeleton_header -> arg_dtypes;

	arg_dtypes[0] = DATAFLOW_INT_SCALAR;
	arg_dtypes[1] = DATAFLOW_INT_SCALAR;
	arg_dtypes[2] = bwd_datatype;
	arg_dtypes[3] = DATAFLOW_UINT32;
	arg_dtypes[4] = DATAFLOW_FP32;

	for (int i = num_args; i < MAX_OP_ARGS; i++){
		arg_dtypes[i] = DATAFLOW_NONE;
	}

	dataflow_do_fingerprinting(skeleton_header, sizeof(Op_Skeleton_Header), (skeleton -> identifier).fingerprint);
}
