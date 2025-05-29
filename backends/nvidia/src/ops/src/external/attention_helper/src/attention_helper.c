#include "attention_helper.h"

int flash_attention_fwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra) {

	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int arch = device_info -> arch_num;
	int sm_count = device_info -> sm_count;

	CUstream * streams = (CUstream *) (dataflow_handle -> streams);

	CUstream stream = streams[stream_id];

	void ** op_args = op -> op_args;

	// Follow same paradigm as submitting to command queue
	// pass in pointers to each of the arguments...

	// so now we can deference them...

	int flash_dtype_as_int = *((int *)(op_args[0]));

	int num_seqs = *((int *)(op_args[1]));
	int total_q = *((int *)(op_args[2]));
	int total_k = *((int *)(op_args[3]));
	int * q_seq_offsets = *((int **) op_args[4]);
	int * q_seq_lens = *((int **) op_args[5]);
	int max_seqlen_q = *((int *)(op_args[6]));

	int * k_seq_offsets = *((int **) op_args[7]);
	int * k_seq_lens = *((int **) op_args[8]);
	int max_seqlen_k = *((int *)(op_args[9]));
	
	int num_q_heads = *((int *)(op_args[10]));
	int num_kv_heads = *((int *)(op_args[11]));
	int head_dim = *((int *)(op_args[12]));

	void * x_q = *((void **) op_args[13]);
	void * x_k = *((void **) op_args[14]);
	void * x_v = *((void **) op_args[15]);

	void * x_attn_out = *((void **) op_args[16]);
	float * softmax_lse = *((float **) op_args[17]);

	int is_causal = *((int *)(op_args[18]));

	uint64_t workspaceBytes = *((uint64_t *) op_args[19]);
	void * workspace = *((void **) op_args[20]);
	
	// FLASH3 only supports SM80, SM86, SM89, SM90
	if ((arch == 90 && USE_FLASH3_HOPPER) || ((arch == 80 || arch == 86 || arch == 89) && USE_FLASH3_AMPERE)) {
		return flash3_fwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									is_causal,
									workspaceBytes, workspace);
	}

	return flash2_fwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									is_causal,
									workspaceBytes, workspace);
}


// inputs: same as fwd + dx_out (upstream gradient) and possibly different sized workspace

// purpose is to compute dx_q, dx_k, dx_v
int flash_attention_bwd(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra) {

	/*
	int flash3_bwd_wrapper(CUstream stream, int arch, int num_sm,
                            int flash_dtype_as_int, 
                            int num_seqs, int total_q, int total_k, 
                            int * cum_q_seqlens, int max_seqlen_q,
                            int * cum_k_seqlens, int max_seqlen_k,
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, void * softmax_lse, 
                            void * dx_out, 
                            void * dx_q, void * dx_k, void * dx_v,
                            void * attn_bwd_workspace);

	*/


	Cuda_Device_Info * device_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

	int arch = device_info -> arch_num;
	int sm_count = device_info -> sm_count;

	CUstream * streams = (CUstream *) (dataflow_handle -> streams);

	CUstream stream = streams[stream_id];

	void ** op_args = op -> op_args;

	// Follow same paradigm as submitting to command queue
	// pass in pointers to each of the arguments...

	// so now we can deference them...

	int flash_dtype_as_int = *((int *)(op_args[0]));

	int num_seqs = *((int *)(op_args[1]));
	int total_q = *((int *)(op_args[2]));
	int total_k = *((int *)(op_args[3]));
	
	int * q_seq_offsets = *((int **) op_args[4]);
	int * q_seq_lens = *((int **) op_args[5]);
	int max_seqlen_q = *((int *)(op_args[6]));

	int * k_seq_offsets = *((int **) op_args[7]);
	int * k_seq_lens = *((int **) op_args[8]);
	int max_seqlen_k = *((int *)(op_args[9]));
	
	int num_q_heads = *((int *)(op_args[10]));
	int num_kv_heads = *((int *)(op_args[11]));
	int head_dim = *((int *)(op_args[12]));

	void * x_q = *((void **) op_args[13]);
	void * x_k = *((void **) op_args[14]);
	void * x_v = *((void **) op_args[15]);

	void * x_attn_out = *((void **) op_args[16]);
	float * softmax_lse = *((float **) op_args[17]);

	// upstream gradient
	void * dx_out = *((void **) op_args[18]);

	// Gradients we want to compute
	void * dx_q = *((void **) op_args[19]);
	void * dx_k = *((void **) op_args[20]);
	void * dx_v = *((void **) op_args[21]);

	int is_causal = *((int *)(op_args[22]));

	uint64_t workspaceBytes = *((uint64_t *) op_args[23]);

	void * workspace = *((void **) op_args[24]);

	// FLASH3 only supports SM80, SM86, SM89, SM90
	if ((arch == 90 && USE_FLASH3_HOPPER) || ((arch == 80 || arch == 86 || arch == 89) && USE_FLASH3_AMPERE)) {
		return flash3_bwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									dx_out,
									dx_q, dx_k, dx_v,
									is_causal,
									workspaceBytes, workspace);
	} 

	return flash2_bwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									dx_out,
									dx_q, dx_k, dx_v,
									is_causal,
									workspaceBytes, workspace);
}