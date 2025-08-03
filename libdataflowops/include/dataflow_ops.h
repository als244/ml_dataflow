#ifndef DATAFLOW_OPS_H
#define DATAFLOW_OPS_H

#include "dataflow.h"
#include "set_op_skeletons.h"

int dataflow_submit_cast(Dataflow_Handle *handle, int stream_id, DataflowDatatype src_dt, DataflowDatatype dst_dt, uint64_t num_elements, void *src, void *dst);

// C = alpha * A + beta * B
int dataflow_submit_cast_and_add(Dataflow_Handle * handle, int stream_id,
		DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
		uint64_t num_els,
		float alpha, void * A, float beta, void * B, void * C);

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

// returns the required workspace size for the attention op
// exported by libattention_helper.so

int dataflow_get_attention_workspace_size(Dataflow_Handle * handle, DataflowDatatype attn_dt, int is_training, 
											int num_q_heads, int num_kv_heads, int head_dim, 
											int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
											int is_causal,
											uint64_t * ret_workspace_size);


int dataflow_submit_attention(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype fwd_dt,
						int num_seqs, int total_q, int total_k, 
						int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
						int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, 
						void * x_attn_out, float * softmax_lse, 
						int is_causal,
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
						void * x_attn_out, float * softmax_lse,
						void * dx_out,
						void * dx_q, void * dx_k, void * dx_v, 
						int is_causal,
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

// computes correct output along with saving rms_vals
int dataflow_submit_default_rms_norm_recompute(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						int n_rows, int n_cols,
						void * rms_weight, float * rms_vals, void * X, void * out);


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
								float * fwd_rms_vals, void * X_inp, void * upstream_dX, void * dW, 
								uint64_t workspaceBytes, void * workspace);



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
						int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * X_q, void * X_k);

int dataflow_submit_default_rope_bwd_x(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype bwd_dt, 
						int num_tokens, int model_dim, int head_dim, int num_kv_heads, int theta,
						int * seq_positions, void * dX_q, void * dX_k);


/*
int dataflow_submit_default_copy_to_seq_context(Dataflow_Handle * handle, int stream_id, 
						DataflowDatatype fwd_dt, 
						uint64_t N, int total_tokens, int kv_dim, 
						void * X_k, void * X_v, int * seq_positions, uint64_t * seq_context_ptrs, int * seq_context_sizes);
*/


// From moe_ops.c

int dataflow_submit_default_select_experts(Dataflow_Handle * handle, int stream_id, 
                                DataflowDatatype fwd_dt,
                                int total_tokens, int n_experts, int top_k_experts,  
                                void * X_routed, float * token_expert_weights, 
                                uint16_t * chosen_experts, int * expert_counts, 
                                int * expert_counts_cumsum, 
								int * host_expert_counts);


int dataflow_submit_default_build_expert_mapping(Dataflow_Handle * handle, int stream_id, 
                                int total_tokens, int num_routed_experts, int num_selected_experts, 
                                uint16_t * chosen_experts, int * expert_counts_cumsum,
                                int * expert_mapping);

int dataflow_submit_default_prepare_expert_zone(Dataflow_Handle * handle, int stream_id, 
                                DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
                                int model_dim, void * X, 
								int expert_id, int * expert_counts, int * expert_counts_cumsum,
                                int * expert_mapping, 
								void * expert_zone);

int dataflow_submit_default_merge_expert_result(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
								int num_tokens, int model_dim, int top_k_experts, 
								void * expert_zone, int expert_id, 
								int * expert_counts_cumsum, 
								int * expert_mapping,
								float * token_expert_weights,
								uint16_t * chosen_experts,
								void * X_combined);


int	dataflow_submit_router_bwd_x(Dataflow_Handle * handle, int stream_id,
								DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
								int num_tokens, int model_dim, int num_routed_experts, int top_k_active,
								int expert_id,
								int * expert_counts_cumsum,
								int * expert_mapping,
								uint16_t * chosen_experts,
								float * token_expert_weights,
								void * expert_out, void * upstream_dX,
								void * dX_routed, // populating column [expert_id] of router derivs with dot product of expert output and loss gradient corresponding to tokens selected by this expert
								void * dX_expert_out); // repopulating with the rows from inp_grad_stream -> X * weight assoicated with this expert (for each token)...


int dataflow_submit_router_gate_bwd_x(Dataflow_Handle * handle, int stream_id,
								DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
								int num_tokens, int num_routed_experts, int top_k_active,
								uint16_t * chosen_experts,
								float * token_expert_weights,
								void * dX_routed);

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

int dataflow_submit_default_set_average_loss(Dataflow_Handle * handle, int stream_id, int n_tokens, float * loss_vec);

int dataflow_submit_default_adamw_step(Dataflow_Handle * handle, int stream_id,
						DataflowDatatype param_dt, DataflowDatatype grad_dt, 
						DataflowDatatype mean_dt, DataflowDatatype var_dt,
						uint64_t num_els, int step_num,
						float lr, float beta1, float beta2, float weight_decay, float epsilon,
						void * param, void * grad, void * mean, void * var);



/* HOST OPS */

// Require user to pass in host function pointer the backend implementation of the op
// The arguments passed in need to populated in memory at the the execution time of the op
// but this is uncertain, so it is user responssibility to pass in buffer that will be populated
// by these submission functions and shoudn't be overwritten until after the op has completed execution...


// GENERAL / OPTIMIZER OPS
typedef struct set_mem_host_op_args{
    void * ptr;
    size_t size_bytes;
    int value;
} Set_Mem_Host_Op_Args;

int dataflow_submit_set_mem_host(Dataflow_Handle * handle, int stream_id, 
                        void * set_mem_host_func, Set_Mem_Host_Op_Args * op_buffer,
                        void * ptr, int value, size_t size_bytes);


// C = alpha * A + beta * B
typedef struct add_host_op_args{
    DataflowDatatype A_dt;
    DataflowDatatype B_dt;
    DataflowDatatype C_dt;
    void * A;
    void * B;
    void * C;
    float alpha;
    float beta;
    int num_threads;
    int layer_id;
    size_t num_els;
} Add_Host_Op_Args;

// these are within optimizer_ops.c
int dataflow_submit_add_host(Dataflow_Handle * handle, int stream_id, 
                        void * add_host_func, Add_Host_Op_Args * op_buffer,
                        DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                        int num_threads, int layer_id, size_t num_els, void * A, void * B, void * C,
                        float alpha, float beta);


typedef struct Adam_Host_Op_Args{
    DataflowDatatype param_dt;
    DataflowDatatype grad_dt;
    DataflowDatatype mean_dt;
    DataflowDatatype var_dt;
    int num_threads;
    int step_num;
    int layer_id;
    uint64_t num_els;
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float epsilon;
    void * param;
    void * grad;
    void * mean;
    void * var;
} Adam_Host_Op_Args;



int dataflow_submit_adam_step_host(Dataflow_Handle * handle, int stream_id, 
                        void * adam_host_func, Adam_Host_Op_Args * op_buffer,
						DataflowDatatype param_dt, DataflowDatatype grad_dt, 
                        DataflowDatatype mean_dt, DataflowDatatype var_dt,
                        int num_threads, int step_num, int layer_id, uint64_t num_els, 
                        float lr, float beta1, float beta2, float weight_decay, float epsilon,
                        void * param, void * grad, void * mean, void * var);


// LOSS STUFF

typedef struct Print_Chunk_Loss_Host_Op_Args{
    int step_num;
    int round_num;
    int seq_id;
    int chunk_id;
    int num_tokens;
    float * avg_loss_ref;
} Print_Chunk_Loss_Host_Op_Args;


int dataflow_submit_print_chunk_loss_host(Dataflow_Handle * handle, int stream_id,
									void * print_chunk_loss_host_func, Print_Chunk_Loss_Host_Op_Args * op_buffer,
									int step_num, int round_num, int seq_id, int chunk_id, int num_tokens, float * avg_loss_ref);


typedef struct Print_Round_Loss_Host_Op_Args{
	int step_num;
	int round_num;
	int num_seqs;
	int num_chunks;
	int total_tokens;
	float * per_chunk_avg_loss;
} Print_Round_Loss_Host_Op_Args;



int dataflow_submit_print_round_loss_host(Dataflow_Handle * handle, int stream_id,
									void * print_round_loss_host_func, Print_Round_Loss_Host_Op_Args * op_buffer,
									int step_num, int round_num, int num_seqs, int num_chunks, int total_tokens, float * per_chunk_avg_loss);



// METRICS

#define MAX_SEQS_PER_STEP 65536
#define MAX_INP_ONLY_CHUNKS 8192

typedef struct Step_Throughput_Host_Op_Args{
	// populated at beginning of training for all steps
	int model_dim;
	int kv_dim;
	int num_shared_experts;
	int num_total_routed_experts;
	int num_active_routed_experts;
	int expert_dim;
	int vocab_size;
	int num_layers;
	float peak_hardware_flop_rate;
	bool to_print_metrics;
	bool to_print_verbose;

	// these are populated as input to start_step_metrics()
	int step_num;
	int num_seqs;
	int seqlens[MAX_SEQS_PER_STEP];
	int is_causal;
	// to determine recomputation flops...
	int chunk_size;
	int num_inp_attn_saved;
	int num_inp_only_saved;
	int inp_only_seq_lens[MAX_INP_ONLY_CHUNKS];
	int num_seqs_per_round;
	int num_rounds_per_step;
	// populated during start_step_metrics()
	struct timespec start_time;
	int total_tokens;
	float total_fwd_flops;
	float total_head_flops;
	float total_bwd_x_flops;
	float total_bwd_w_flops;
	// sum of above 4
	float total_computation_flops;
	float total_recompute_flops;
	// sum of total_computation_flops and total_recompute_flops
	float total_flops;
	float total_attn_flops;
	float total_matmul_flops;
	// these get populated during end_step_metrics()
	struct timespec end_time;
	uint64_t duration_ns;
	float duration_s;
	// duration_s / total_computation_flops
	float achieved_flop_rate;
	// duration_s / (total_computation_flops / peak_hardware_flop_rate)
	float mfu;

	// having the denomicator include recompute flops...
	float achieved_hardware_flop_rate;
	float hfu;
	// total_tokens / duration_s
	float tokens_per_second;
} Step_Throughput_Host_Op_Args;

// assumes that the op_buffer is already allocated and populated with the static model info...
int dataflow_submit_start_step_metrics_host(Dataflow_Handle * handle, int stream_id, 
                        void * start_step_metrics_func, Step_Throughput_Host_Op_Args * op_buffer,
						int step_num, int num_seqs, int * seqlens);

// assumes the same op_buffer that was used in the start_step_metrics_host op
int dataflow_submit_end_step_metrics_host(Dataflow_Handle * handle, int stream_id, 
                        void * end_step_metrics_func, Step_Throughput_Host_Op_Args * op_buffer);









#endif
