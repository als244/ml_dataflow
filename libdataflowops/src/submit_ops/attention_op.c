#include "dataflow_ops.h"


// FOR FLASH3 ATTENTION:

// Only support for FP16, BF16, and FP8
// if TYPE FP8, output must be BF16
// Softmax LSE is of type FP32 and has length total_q * num_q_heads

// To compute required size of attn_workspace:

// attn_workspace_size = 0

// Occum and LSE accum:
// If num_splits > 1:
//      attn_workspace_size += num_splits * sizeof(float) * num_q_heads * total_q * (1 + head_dim)

// Tile count sem: 
// If arch >= 90 || num_splits > 1:
//      attn_workspace_size += sizeof(int)

// Dynamic split ptr for each seq:
// If num_seqs <= 992:
//      attn_workspace_size += num_seqs * sizeof(int)


// ASSUME CAUSAL

// - cum_q_seqlens should be of length num_seqs + 1, starting with 0
//		- cumsum of # of queries in each sequence
// - cum_k_seqlens should be of length num_seqs + 1, starting with 0
//		- cumsum of total # (prior context + current) of keys in sequence (should be >= # of queries) 
//			- (assumes that if sequence has Q queries and K keys, the starting position of Q_0
//				occurs at position K - Q)

int dataflow_submit_attention(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype fwd_dt,
						int num_seqs, int total_q, int total_k, 
						int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
						int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, 
						void * x_attn_out, float * softmax_lse, 
						int is_causal,
						uint64_t workspaceBytes, void * workspace) {


	int ret;

	Op attention_op;

	dataflow_set_flash3_attention_fwd_skeleton(&attention_op.op_skeleton);

	void ** op_args = attention_op.op_args;

	op_args[0] = &fwd_dt;
	op_args[1] = &num_seqs;
	op_args[2] = &total_q;
	op_args[3] = &total_k;
	op_args[4] = &q_seq_offsets;
	op_args[5] = &q_seq_lens;
	op_args[6] = &max_seqlen_q;
	op_args[7] = &k_seq_offsets;
	op_args[8] = &k_seq_lens;
	op_args[9] = &max_seqlen_k;
	op_args[10] = &num_q_heads;
	op_args[11] = &num_kv_heads;
	op_args[12] = &head_dim;
	op_args[13] = &x_q;
	op_args[14] = &x_k;
	op_args[15] = &x_v;
	op_args[16] = &x_attn_out;
	op_args[17] = &softmax_lse;
	op_args[18] = &is_causal;
	op_args[19] = &workspaceBytes;
	op_args[20] = &workspace;

	ret = (handle -> submit_op)(handle, &attention_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention_op...\n");
		return -1;
	}

	return 0;
}


int dataflow_submit_attention_bwd(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype bwd_dt,
						int num_seqs, int total_q, int total_k, 
						int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
						int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, 
						void * x_attn_out, float * softmax_lse,
						void * dx_out,
						void * dx_q, void * dx_k, void * dx_v, 
						int is_causal,
						uint64_t workspaceBytes, void * workspace) {


	int ret;

	Op attention_bwd_op;

	dataflow_set_flash3_attention_bwd_skeleton(&attention_bwd_op.op_skeleton);

	void ** op_args = attention_bwd_op.op_args;

	op_args[0] = &bwd_dt;
	op_args[1] = &num_seqs;
	op_args[2] = &total_q;
	op_args[3] = &total_k;
	op_args[4] = &q_seq_offsets;
	op_args[5] = &q_seq_lens;
	op_args[6] = &max_seqlen_q;
	op_args[7] = &k_seq_offsets;
	op_args[8] = &k_seq_lens;
	op_args[9] = &max_seqlen_k;
	op_args[10] = &num_q_heads;
	op_args[11] = &num_kv_heads;
	op_args[12] = &head_dim;
	op_args[13] = &x_q;
	op_args[14] = &x_k;
	op_args[15] = &x_v;
	op_args[16] = &x_attn_out;
	op_args[17] = &softmax_lse;
	op_args[18] = &dx_out;
	op_args[19] = &dx_q;
	op_args[20] = &dx_k;
	op_args[21] = &dx_v;
	op_args[22] = &is_causal;
	op_args[23] = &workspaceBytes;
	op_args[24] = &workspace;

	ret = (handle -> submit_op)(handle, &attention_bwd_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention_bwd_op...\n");
		return -1;
	}

	return 0;
}