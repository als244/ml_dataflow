#ifndef DATAFLOW_OPS_H
#define DATAFLOW_OPS_H

#include "dataflow.h"
#include "set_op_skeletons.h"


// CORE COMPUTE FUNCTIONS!
//	- these end up calling external library implementations...

// From matmul.c


// Operates under assumption of all col-major formatting, 
// so can be clever about obtaining result in row major format with transposes


// Note: FP8 Tensor cores can only be used with TN formatting

// Forward pass: 

// I.e. Normal Fwd Case is to compute:
//			- X (m, k), Y (m, n) in row major
//			- W (k, n) but stored in col major
// Y = X @ W
// => Y^T = W^T @ X^T

// If we store W in col-major then from blas perspective
// it needs to transpose

// However if we store X in row-major then blas interprets
// the non-transposed X as already transposed

// We interpret the output (in col-major) Y^T as non-transposed Y in row-major


// to do this can pass in W => A, X => B, Y => D
// and set to_trans_a = 1, to_trans_b = 0
// and also M = n, K = k, N = m

// Now result is Y which is n x m in col-major, 
// or equivalently, m x n in row-major as hoped for


// During backprop, if we want dX to be in row-major, but are 
// working with W stored in col-major format this is not possible
// under "TN" constraints

// I.e. Normal Bwd Case:
//			- dY (m, n), dX (m, k) in row-major
//			- W (k, n), but stored in col major
// dX = dY @ W^T

// => dX^T = W @ dY^T

// if we store W in col-major then no transpose
// if we store dY in row-major then no tranpose because it already interprets as transposed 

// We interpret the output (in col major) of dX^T as non-transposed dX in row-major

// However if we are not using FP8 tensor cores then we can achieve
// the correct results (without physical transposes) by doing:

// W => A, dY => B, dX => D
// and set to_trans_a = 0, to_trans_b = 0
int dataflow_submit_matmul(Dataflow_Handle * handle, int stream_id, 
					DataflowDatatype a_dt, DataflowDatatype b_dt, DataflowDatatype c_dt, DataflowDatatype d_dt,
					DataflowDatatype compute_dt,
					int to_trans_a, int to_trans_b,
					int M, int K, int N,
					float alpha, float beta,
					void * A, void * B, void * C, void * D,
					uint64_t workspaceBytes, void * workspace);



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
						void * x_attn_out, void * softmax_lse, 
						uint64_t workspaceBytes, void * workspace);



// FOR FLASH3 ATTENTION:

// Bwd Dt must be either FP16 or BF16!

// if fwd_dt is FP8, then x_q, x_k, and x_v must be converted to 
// the bwd_dt before submission of this function
int dataflow_submit_attention_bwd(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype bwd_dt,
						int num_seqs, int total_q, int total_k, 
						int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
						int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, 
						void * x_attn_out, void * softmax_lse,
						void * dx_out,
						void * dx_q, void * dx_k, void * dx_v, 
						uint64_t workspaceBytes, void * workspace);


// NATIVE OPS

// From preprocess_ops.c
int dataflow_submit_default_embedding_table(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int num_unique_tokens, int embed_dim, 
						uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, 
						void * embedding_table, void * output);

int dataflow_submit_default_embedding_table_bwd_w(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype bwd_dt, 
						int num_unique_tokens, int embed_dim, 
						uint32_t * sorted_token_ids, uint32_t * sorted_token_mapping, uint32_t * unique_token_sorted_inds_start, 
						void * grad_stream, void * grad_embedding_table);

// From norm_ops.c

// computes correct output along with saving rms_vals
int dataflow_submit_default_rms_norm(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * rms_weight, void * X, void * out, float * rms_vals);


// if X_out is not NULL, then it gets populated the the recomputed value from fwd pass
// accumulates result into dX
int dataflow_submit_default_rms_norm_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals,
								void * rms_weight, void * X_inp, void * upstream_dX, void * dX, void * X_out);


int dataflow_submit_default_rms_norm_bwd_w(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals, void * X_inp, void * upstream_dX, void * dW);



int dataflow_submit_default_rms_norm_noscale(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols, float eps, 
						void * X, void * out, float * rms_vals);


// if X_out is not NULL, then it gets populated the the recomputed value from fwd pass
int dataflow_submit_default_rms_norm_noscale_bwd_x(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype fwd_dt, DataflowDatatype bwd_dt, 
								int n_rows, int n_cols, float eps, 
								float * fwd_rms_vals,
								void * X_inp, void * upstream_dX, void * dX, void * X_out);

// From attn_misc_ops.c

int dataflow_submit_default_rope(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * X_q, void * X_k);

int dataflow_submit_default_rope_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype bwd_dt, 
						uint64_t N, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * dX_q, void * dX_k);


int dataflow_submit_default_copy_to_seq_context(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int total_tokens, int kv_dim, 
						void * X_k, void * X_v, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);


// From moe_ops.c

int dataflow_submit_default_select_experts(Dataflow_Handle * handle, int stream_id, 
                                DataflowDatatype fwd_dt,
                                int total_tokens, int n_experts, int top_k_experts,  
                                void * X_routed, void * token_expert_weights, 
                                uint16_t * chosen_experts, int * expert_counts, 
                                int * expert_counts_cumsum, int * num_routed_by_expert_workspace);
						

// From mlp_misc_ops.c

int dataflow_submit_default_swiglu(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int num_rows, int num_cols, 
						void * x_w1, void * x_w3, void * out);


int dataflow_submit_default_swiglu_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int num_rows, int num_cols, 
						void * x_w1, void * x_w3, 
						void * upstream_dX, void * dX_w1, void * dX_w3);

// From loss_misc_ops.c

int dataflow_submit_default_softmax(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, DataflowDatatype bwd_dt,
						int n_rows, int n_cols,
						void * X, void * out);

int dataflow_submit_default_cross_entropy_loss(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype bwd_dt,
								int n_rows, int n_cols,
								void * pred_logits, uint32_t * labels, float * loss_vec);




#endif
