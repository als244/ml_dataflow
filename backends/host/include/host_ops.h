#ifndef HOST_OPS_H
#define HOST_OPS_H

#include "dataflow.h"
#include "dataflow_ops.h"

#include <immintrin.h>  // For AVX512 intrinsics

#define USE_FP16_ARTIHMETIC_FOR_ADD 1



int print_chunk_loss_host(void * _print_chunk_loss_host_op_args);
int print_round_loss_host(void * _print_round_loss_host_op_args);

float get_seq_flops(int seq_len, int vocab_size, int model_dim, int kv_dim, int is_causal, int num_shared_experts, int num_total_routed_experts, int num_active_routed_experts, int expert_dim, int num_layers, 
							float * ret_seq_flops_fwd, float * ret_seq_flops_head, float * ret_seq_flops_bwd_x, float * ret_seq_flops_bwd_w, float * ret_seq_attn_flops, float * ret_seq_matmul_flops);

float get_chunk_block_flops(int chunk_size, int prior_seq_len, int max_seq_len, int model_dim, int kv_dim, int is_causal, 
								int num_shared_experts, int num_total_routed_experts, int num_active_routed_experts, int expert_dim);

int start_step_metrics(void * _step_throughput_op_args);
int end_step_metrics(void * _step_throughput_op_args);

int set_mem_host(void * _set_mem_host_op_args);

int add_host(void * _add_host_op_args);

int adam_step_host(void * _adam_host_op_args);



#endif
