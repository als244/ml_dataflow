#define __GNU_SOURCE
#include <stdio.h>

#include "dataflow.h"
#include "dataflow_utils.h"
#include "cuda_dataflow_handle.h"
#include "dataflow_ops.h"
#include "register_ops.h"

int main(int argc, char * argv[]){

	// Seed the random number generator with a constant value
    srand(42);
	
	int ret;

	Dataflow_Handle dataflow_handle;
	
	ComputeType compute_type = COMPUTE_CUDA;
	int device_id = 0;

	// In case we want to create multiple contexts per device, 
	// higher level can create multiple instances of dataflow handles...
	int ctx_id = 0;
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;

	int num_streams = 8;
	int opt_stream_prios[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	char * opt_stream_names[8] = {"Inbound (a)", "Compute (a)", "Outbound (a)", "Peer (a)", "Inbound (b)", "Compute (b)", "Outbound (b)", "Peer (b)"};


	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = dataflow_init_handle(&dataflow_handle, compute_type, device_id, 
			ctx_id, ctx_flags, 
			num_streams, opt_stream_prios, opt_stream_names); 
	
	if (ret){
		fprintf(stderr, "Error: failed to init cuda dataflow handle...\n");
		return -1;
	}

	// from backend/nvidia/src/ops/src/register_ops/register_ops.c	
	// handles registering external and native ops within cuda_dataflow_ops...
	int added_funcs = dataflow_register_default_ops(&dataflow_handle);
	printf("Registered %d default ops...\n", added_funcs);
	
	int alignment = 4096;

	void * host_mem;

	// 16 GB...
	size_t host_size_bytes = 1UL << 34;

	printf("Allocating host memory of size: %lu...\n", host_size_bytes);

	ret = posix_memalign(&host_mem, alignment, host_size_bytes);
	if (ret){
		fprintf(stderr, "Error: posix memalign failed...\n");
		return -1;
	}
	memset(host_mem, 0, host_size_bytes);


	printf("Registering host memory...\n");

	ret = dataflow_handle.enable_access_to_host_mem(&dataflow_handle, host_mem, host_size_bytes, 0);
	if (ret){
		fprintf(stderr, "Registration of host memory failed...\n");
		return -1;
	}

	// Llama 70B dimensions, taking example input after embedding stage and weights are loaded 
	// as first layer

	int num_q_heads = 64;
	int num_kv_heads = 8;
	int head_dim = 128;

	uint64_t ffn_dim = 28672;

	uint64_t model_dim = num_q_heads * head_dim;
	uint64_t kv_dim = num_kv_heads * head_dim;

	int theta = 500000;
	float eps = 1e-5;


	DataflowDatatype fwd_dt = DATAFLOW_FP16;
	DataflowDatatype compute_dt = DATAFLOW_FP16;


	uint64_t el_size = dataflow_sizeof_element(fwd_dt);




	printf("Loading Weight Matrices into host mem...\n");

	char * attn_norm_f = "data/weights/attn_norm.dat";
	char * wq_f = "data/weights/wq.dat";
	char * wk_f = "data/weights/wk.dat";
	char * wv_f = "data/weights/wv.dat";
	char * wo_f = "data/weights/wo.dat";
	char * ffn_norm_f = "data/weights/ffn_norm.dat";
	char * w1_f = "data/weights/w1.dat";
	char * w2_f = "data/weights/w2.dat";
	char * w3_f = "data/weights/w3.dat";

	uint64_t attn_norm_size = model_dim * el_size;
	uint64_t wq_size = model_dim * model_dim * el_size;
	uint64_t wk_size = model_dim * kv_dim * el_size;
	uint64_t wv_size = model_dim * kv_dim * el_size;
	uint64_t wo_size = model_dim * model_dim * el_size;
	uint64_t ffn_norm_size = model_dim * el_size;
	uint64_t w1_size = model_dim * ffn_dim * el_size;
	uint64_t w2_size = ffn_dim * model_dim * el_size;
	uint64_t w3_size = model_dim * ffn_dim * el_size;


	void * attn_norm = host_mem;
	void * wq = attn_norm + attn_norm_size;
	void * wk = wq + wq_size;
	void * wv = wk + wk_size;
	void * wo = wv + wv_size;
	void * ffn_norm = wo + wo_size;
	void * w1 = ffn_norm + ffn_norm_size;
	void * w2 = w1 + w1_size;
	void * w3 = w2 + w2_size;


	void * res;

	// NOTE: all of the weights are stored in column-major format 
	//			- (however this doesn't impact loading in as same total size, we just interpret differently)

	res = load_host_matrix_from_file(attn_norm_f, 1, model_dim, fwd_dt, fwd_dt, attn_norm);
	if (!res){
		fprintf(stderr, "Error: could not load in attn norm weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(wq_f, model_dim, model_dim, fwd_dt, fwd_dt, wq);
	if (!res){
		fprintf(stderr, "Error: could not load in attn query proj weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(wk_f, model_dim, kv_dim, fwd_dt, fwd_dt, wk);
	if (!res){
		fprintf(stderr, "Error: could not load in attn key proj weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(wv_f, model_dim, kv_dim, fwd_dt, fwd_dt, wv);
	if (!res){
		fprintf(stderr, "Error: could not load in attn value proj weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(wo_f, model_dim, model_dim, fwd_dt, fwd_dt, wo);
	if (!res){
		fprintf(stderr, "Error: could not load in attn out proj weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(ffn_norm_f, 1, model_dim, fwd_dt, fwd_dt, ffn_norm);
	if (!res){
		fprintf(stderr, "Error: could not load in ffn norm weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(w1_f, model_dim, ffn_dim, fwd_dt, fwd_dt, w1);
	if (!res){
		fprintf(stderr, "Error: could not load in ffn w1 proj weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(w2_f, ffn_dim, model_dim, fwd_dt, fwd_dt, w2);
	if (!res){
		fprintf(stderr, "Error: could not load in ffn w2 proj weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(w3_f, model_dim, ffn_dim, fwd_dt, fwd_dt, w3);
	if (!res){
		fprintf(stderr, "Error: could not load in ffn w3 proj weight file...\n");
		return -1;
	}


	// DEAL WITH EXAMPLE TRAINING SEQ BATCH...

	int num_seqs = 1;

	size_t offsets_size = (num_seqs + 1) * sizeof(int);
	size_t lens_size = num_seqs * sizeof(int);

	void * q_seq_offsets = w3 + w3_size;
	void * q_seq_lens = q_seq_offsets + offsets_size;

	void * k_seq_offsets = q_seq_lens + lens_size;
	void * k_seq_lens = k_seq_offsets + offsets_size;

	// Harcoding for now
	int q_seqlens[] = {4096};
	int total_q = 0;
	int max_seqlen_q = 0;

	
	int kv_seqlens[] = {4096};
	int total_kv = 0;
	int max_seqlen_kv = 0;

	int * q_seq_offsets_casted = (int *) q_seq_offsets;
	int * q_seq_lens_casted = (int *) q_seq_lens;
	int * k_seq_offsets_casted = (int *) k_seq_offsets;
	int * k_seq_lens_casted = (int *) k_seq_lens;

	// hardcoding to 
	q_seq_offsets_casted[0] = 0;

	int cur_len = 0;


	// Ensuring to set values properly within pinned buffer
	// to avoid implicit sync during data transfer
	for (int i = 0; i < num_seqs; i++){
		q_seq_offsets_casted[i + 1] = cur_len + q_seqlens[i];
		q_seq_lens_casted[i] = q_seqlens[i];
		if (q_seqlens[i] > max_seqlen_q){
			max_seqlen_q = q_seqlens[i];
		}


		total_q += q_seqlens[i];
		cur_len += q_seqlens[i];
	}

	cur_len = 0;

	k_seq_offsets_casted[0] = 0;

	for (int i = 0; i < num_seqs; i++){
		k_seq_offsets_casted[i + 1] = cur_len + kv_seqlens[i];
		k_seq_lens_casted[i] = kv_seqlens[i];
		if (kv_seqlens[i] > max_seqlen_kv){
			max_seqlen_kv = kv_seqlens[i];
		}
		total_kv += kv_seqlens[i];
	}

	size_t seq_positions_size = total_q * sizeof(int);

	void * seq_positions = k_seq_lens + lens_size;

	int * seq_positions_casted = (int *) seq_positions;

	int cur_q = 0;
	int cur_seqlen;
	for (int i = 0; i < num_seqs; i++){
		cur_seqlen = q_seqlens[i];
		for (int k = 0; k < cur_seqlen; k++){
			seq_positions_casted[cur_q + k] = k;
		}
		cur_q += cur_seqlen;
	}


	uint64_t N = total_q * model_dim;



	// Set metadata for norms and attn that is used for backpass...

	uint64_t weighted_sums_size = total_q * sizeof(float);
	uint64_t rms_vals_size = total_q * sizeof(float);

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

	// just get enough...

	size_t max_num_splits = 1;


	size_t attn_workspace_size = 0;

	// cover oaccum and lse accum
	attn_workspace_size += max_num_splits * sizeof(float) * (uint64_t) num_q_heads * (uint64_t) total_q * (uint64_t) (1 + head_dim);
	
	// cover potential tile count sem
	attn_workspace_size += sizeof(int);

	// covert potential dynamic split
	attn_workspace_size += num_seqs * sizeof(int);


	uint64_t softmax_lse_size = total_q * num_q_heads * sizeof(float);

	void * attn_norm_weighted_sums = seq_positions + seq_positions_size;
	void * attn_norm_rms_vals = attn_norm_weighted_sums + weighted_sums_size;
	void * attn_softmax_lse = attn_norm_rms_vals + attn_workspace_size;

	void * ffn_norm_weighted_sums = attn_softmax_lse + softmax_lse_size;
	void * ffn_norm_rms_vals = ffn_norm_weighted_sums + weighted_sums_size;

	uint64_t x_size = (uint64_t) total_q * model_dim * el_size;
	uint64_t x_kv_size = (uint64_t) total_q * kv_dim * el_size;
	uint64_t x_ffn_size = (uint64_t) total_q * ffn_dim * el_size;

	char * orig_x_f = "data/orig_x.dat";
	void * orig_x = ffn_norm_rms_vals + rms_vals_size;

	res = load_host_matrix_from_file(orig_x_f, total_q, model_dim, fwd_dt, fwd_dt, orig_x);
	if (!res){
		fprintf(stderr, "Error: could not load in orig x file...\n");
		return -1;
	}


	void * attn_norm_out = orig_x + x_size;

	void * wq_out = attn_norm_out + x_size;

	void * wk_out = wq_out + x_size;

	void * wv_out = wk_out + x_kv_size;

	void * attn_out = wv_out + x_kv_size;

	void * wo_out = attn_out + x_size;

	void * ffn_norm_out = wo_out + x_size;

	void * w1_out = ffn_norm_out + x_size;

	void * w3_out = w1_out + x_ffn_size;

	void * swiglu_out = w3_out + x_ffn_size;

	void * w2_out = swiglu_out + x_ffn_size;

	


	// 16 GB...	
	size_t dev_size_bytes = 1UL << 34;

	printf("Allocating device memory of size: %lu...\n", dev_size_bytes);


	void * dev_mem = dataflow_handle.alloc_mem(&dataflow_handle, dev_size_bytes);
	if (!dev_mem){
		fprintf(stderr, "Error: device memory allocation failed...\n");
		return -1;
	}


	// Transfer all required weights/inputs/metadata/workspace to device before submitting ops...

	void * d_attn_norm = dev_mem;
	void * d_wq = d_attn_norm + attn_norm_size;
	void * d_wk = d_wq + wq_size;
	void * d_wv = d_wk + wk_size;
	void * d_wo = d_wv + wv_size;
	void * d_ffn_norm = d_wo + wo_size;
	void * d_w1 = d_ffn_norm + ffn_norm_size;
	void * d_w2 = d_w1 + w1_size;
	void * d_w3 = d_w2 + w2_size;

	void * d_q_seq_offsets = d_w3 + w3_size;
	void * d_q_seq_lens = d_q_seq_offsets + offsets_size;
	void * d_k_seq_offsets = d_q_seq_lens + lens_size;
	void * d_k_seq_lens = d_k_seq_offsets + offsets_size;

	void * d_seq_positions = d_k_seq_lens + lens_size;

	void * d_attn_norm_weighted_sums = d_seq_positions + seq_positions_size;
	void * d_attn_norm_rms_vals = d_attn_norm_weighted_sums + weighted_sums_size;
	void * d_attn_workspace = d_attn_norm_rms_vals + rms_vals_size;
	void * d_attn_softmax_lse = d_attn_workspace + attn_workspace_size;

	void * d_ffn_norm_weighted_sums = d_attn_softmax_lse + softmax_lse_size;
	void * d_ffn_norm_rms_vals = d_ffn_norm_weighted_sums + weighted_sums_size;

	// In order to use tensor cores all matrices must have 256-byte alignment...
	int dev_alignment = 256;

	int align_spacer = dev_alignment - (((uint64_t) (d_ffn_norm_rms_vals + rms_vals_size)) % dev_alignment);

	void * d_orig_x = d_ffn_norm_rms_vals + rms_vals_size + align_spacer;

	void * d_attn_norm_out = d_orig_x + x_size;

	void * d_wq_out = d_attn_norm_out + x_size;

	void * d_wk_out = d_wq_out + x_size;

	void * d_wv_out = d_wk_out + x_kv_size;

	void * d_attn_out = d_wv_out + x_kv_size;

	void * d_wo_out = d_attn_out + x_size;

	void * d_ffn_norm_out = d_wo_out + x_size;

	void * d_w1_out = d_ffn_norm_out + x_size;

	void * d_w3_out = d_w1_out + x_ffn_size;

	void * d_swiglu_out = d_w3_out + x_ffn_size;

	void * d_w2_out = d_swiglu_out + x_ffn_size;

	uint64_t workspaceBytes = 1UL << 24;
	void * d_matmul_workspace = d_w2_out + x_size;


	printf("Transferring weights to device...\n");


	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_attn_norm, attn_norm, attn_norm_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for attn norm weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_wq, wq, wq_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for attn query proj weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_wk, wk, wk_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for attn key proj weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_wv, wv, wv_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for attn value proj weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_wo, wo, wo_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for attn output proj weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_ffn_norm, ffn_norm, ffn_norm_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for ffn norm weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_w1, w1, w1_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for ffn w1 proj weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_w2, w2, w2_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for ffn w2 proj weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_w3, w3, w3_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for ffn w3 proj weights...\n");
		return -1;
	}


	printf("Transferring seq batch metadata to device...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_seq_positions, seq_positions, seq_positions_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for seq_positions...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_q_seq_offsets, q_seq_offsets, offsets_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for q_seq_offsets...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_q_seq_lens, q_seq_lens, lens_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for q_seq_lens...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_k_seq_offsets, k_seq_offsets, offsets_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for k_seq_offsets...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_k_seq_lens, k_seq_lens, lens_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for k_seq_lens...\n");
		return -1;
	}

	printf("Transferring seq batch to device...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_orig_x, orig_x, x_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for orig_x...\n");
		return -1;
	}


	printf("Syncing with device after transfer...\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer...\n");
		return -1;
	}


	// Assume weights are in col-major format.

	// But we want to process activations in row-major

	// Note that matmul interface assumes col-major storage format

	// Also note that FP8 tensor cores only available in TN format

	// During FWD pass we normally want:


	// Thus to compute Y = X @ W, 
	// we can do Y^T = W^T @ X^T
	// where from matmul perspective ^T means we interpret as row-major
	// However we store W as col-major so we need to transpose it.

	// Also for M, K, N (assuming X: (m, k), W (k, n))
	// we set M = n, K = k, N = m

	// The BWD pass is different because if we want dX's to be in row major we need:

	// dX = dY @ W^T
	// => dX^T = W @ dY^T

	// so if we store W in col-major format we shouldn't transpose it..

	// Now for bwd we set
	// M = k, K = n, N = m
	// where m, k, n are from fwd values of X, W, and Y.

	int to_trans_a = 1;
	int to_trans_b = 0;


	printf("Submitting Attention RMS Norm...!\n");

	ret = dataflow_submit_rms_norm(&dataflow_handle, compute_stream_id_a, 
						fwd_dt, 
						total_q, model_dim, eps, 
						d_attn_norm, d_orig_x, d_attn_norm_out, 
						d_attn_norm_weighted_sums, d_attn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit attention norm...\n");
		return -1;
	}	



	printf("Submitting Q, K, V matmuls...!\n");

	// Q Proj
	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					model_dim, model_dim, total_q,
					1.0, 0.0,
					d_wq, d_attn_norm_out, NULL, d_wq_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit q matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					kv_dim, model_dim, total_q,
					1.0, 0.0,
					d_wk, d_attn_norm_out, NULL, d_wk_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit k matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					kv_dim, model_dim, total_q,
					1.0, 0.0,
					d_wv, d_attn_norm_out, NULL, d_wv_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit v matmul proj...\n");
		return -1;
	}


	printf("Submitting RoPE...!\n");

	ret = dataflow_submit_rope(&dataflow_handle, compute_stream_id_a, 
						fwd_dt, 
						N, model_dim, head_dim, num_kv_heads, theta,
						d_seq_positions, d_wq_out, d_wk_out);
	if (ret){
		fprintf(stderr, "Error: failed to submit rope...\n");
		return -1;
	}

	ret = ((&dataflow_handle) -> set_mem)(&dataflow_handle, compute_stream_id_a, d_attn_workspace, 0, attn_workspace_size);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}

	printf("Submitting Attention...!\n");

	ret = dataflow_submit_attention(&dataflow_handle, compute_stream_id_a,
						 fwd_dt, 
						 num_seqs, total_q, total_kv,
						 d_q_seq_offsets, d_q_seq_lens, max_seqlen_q,
						 d_k_seq_offsets, d_k_seq_lens, max_seqlen_kv,
						 num_q_heads, num_kv_heads, head_dim,
						 d_wq_out, d_wk_out, d_wv_out,
						 d_attn_out, d_attn_softmax_lse, 
						 attn_workspace_size, d_attn_workspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention...\n");
		return -1;
	}


	printf("Submitting Attention Output Matmul...!\n");


	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					model_dim, model_dim, total_q, 
					1.0, 1.0,
					d_wo, d_attn_out, d_orig_x, d_wo_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit o matmul proj...\n");
		return -1;
	}


	printf("Submitting FFN RMS Norm...!\n");

	ret = dataflow_submit_rms_norm(&dataflow_handle, compute_stream_id_a, 
						fwd_dt, 
						total_q, model_dim, eps, 
						d_ffn_norm, d_wo_out, d_ffn_norm_out, 
						d_ffn_norm_weighted_sums, d_ffn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit ffn norm...\n");
		return -1;
	}


	printf("Submitting FFN w1 and w3 matmuls...!\n");

	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					(int) ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					d_w1, d_ffn_norm_out, NULL, d_w1_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w1 matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					(int) ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					d_w3, d_ffn_norm_out, NULL, d_w3_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w3 matmul proj...\n");
		return -1;
	}


	printf("Submitting SwiGLU Activation...!\n");


	ret = dataflow_submit_swiglu(&dataflow_handle, compute_stream_id_a, 
						fwd_dt, 
						total_q, (int) ffn_dim, 
						d_w1_out, d_w3_out, d_swiglu_out);

	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu activation...\n");
		return -1;
	}


	printf("Submitting FFN w2 matmul...!\n");

	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					model_dim, (int) ffn_dim, total_q, 
					1.0, 1.0,
					d_w2, d_swiglu_out, d_wo_out, d_w2_out,
					workspaceBytes, d_matmul_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w2 matmul proj...\n");
		return -1;
	}

	printf("Finished submitting all work, now submitting dependency for outbound transfer...\n");

	void * compute_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id_a);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id_a, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	printf("Submitting outbound transfers for all results...\n");


	// ATTENTION BLOCK!

	// Attn Norm...
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, attn_norm_out, d_attn_norm_out, x_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for attn_norm_out...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, attn_norm_weighted_sums, d_attn_norm_weighted_sums, weighted_sums_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for attn_norm_weighted_sums...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, attn_norm_rms_vals, d_attn_norm_rms_vals, rms_vals_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for attn_norm_rms_vals...\n");
		return -1;
	}

	// Q, K, V proj
	// Resuse same buffer for rope, so d_wq_out and d_wk_out are values after rope is applied...
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, wq_out, d_wq_out, x_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for wq_out (post rope)...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, wk_out, d_wk_out, x_kv_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for wk_out (post rope)...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, wv_out, d_wv_out, x_kv_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for wv_out...\n");
		return -1;
	}

	// Attention
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, attn_out, d_attn_out, x_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for attn_out...\n");
		return -1;
	}


	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, attn_softmax_lse, d_attn_softmax_lse, softmax_lse_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for attn_softmax_lse...\n");
		return -1;
	}


	// Attention output projection
	// Note: Already includes adding residual stream
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, wo_out, d_wo_out, x_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for wo_out...\n");
		return -1;
	}



	// FFN BLOCK!


	// FFN Norm...
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, ffn_norm_out, d_ffn_norm_out, x_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for ffn_norm_out...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, ffn_norm_weighted_sums, d_ffn_norm_weighted_sums, weighted_sums_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for ffn_norm_weighted_sums...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, ffn_norm_rms_vals, d_ffn_norm_rms_vals, rms_vals_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for ffn_norm_rms_vals...\n");
		return -1;
	}


	// W1 and W3 Projections...
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, w1_out, d_w1_out, x_ffn_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for w1_out...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, w3_out, d_w3_out, x_ffn_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for w3_out...\n");
		return -1;
	}


	// SwiGLU Activation...
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, swiglu_out, d_swiglu_out, x_ffn_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for swiglu_out...\n");
		return -1;
	}

	// W2 Projection
	// (Layer output...)
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, w2_out, d_w2_out, x_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for w2_out...\n");
		return -1;
	}

	printf("Syncing with outbound transfer...\n");


	ret = dataflow_handle.sync_stream(&dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}


	printf("Saving results in files...\n");


	// attn_norm_out
	// attn_norm_weighted_sums
	// attn_norm_rms_vals

	// wq_out
	// wk_out
	// wv_out

	// attn_out
	// attn_softmax_lse

	// wo_out

	// ffn_norm_out
	// ffn_norm_weighted_sums
	// ffn_norm_rms_vals

	// w1_out
	// w3_out

	// swiglu_out

	// w2_out


	char * attn_norm_out_f = "test_layer/attn_norm_out.dat";
	char * attn_norm_weighted_sums_f = "test_layer/attn_norm_weighted_sums.dat";
	char * attn_norm_rms_vals_f = "test_layer/attn_norm_rms_vals.dat";

	ret = save_host_matrix(attn_norm_out_f, attn_norm_out, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save attn_norm_out matrix...\n");
		return -1;
	}

	ret = save_host_matrix(attn_norm_weighted_sums_f, attn_norm_weighted_sums, 1, total_q, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save attn_norm_weighted_sums matrix...\n");
		return -1;
	}

	ret = save_host_matrix(attn_norm_rms_vals_f, attn_norm_rms_vals, 1, total_q, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save attn_norm_rms_vals matrix...\n");
		return -1;
	}


	char * wq_out_f = "test_layer/x_q_rope.dat";
	char * wk_out_f = "test_layer/x_k_rope.dat";
	char * wv_out_f = "test_layer/x_v.dat";

	// Note: already applied rope to these out projections...

	ret = save_host_matrix(wq_out_f, wq_out, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_q_rope matrix...\n");
		return -1;
	}

	ret = save_host_matrix(wk_out_f, wk_out, total_q, kv_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_k_rope matrix...\n");
		return -1;
	}

	ret = save_host_matrix(wv_out_f, wv_out, total_q, kv_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_v matrix...\n");
		return -1;
	}


	char * attn_out_f = "test_layer/x_attn.dat";
	char * attn_softmax_lse_f = "test_layer/x_attn_softmax_lse.dat";

	ret = save_host_matrix(attn_out_f, attn_out, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_attn matrix...\n");
		return -1;
	}

	ret = save_host_matrix(attn_softmax_lse_f, attn_softmax_lse, total_q, num_q_heads, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save x_attn_softmax_lse matrix...\n");
		return -1;
	}


	// Note: already includes adding orig x

	char * wo_out_f = "test_layer/x_attn_out.dat";
	ret = save_host_matrix(wo_out_f, wo_out, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_attn_out matrix...\n");
		return -1;
	}



	char * ffn_norm_out_f = "test_layer/ffn_norm_out.dat";
	char * ffn_norm_weighted_sums_f = "test_layer/ffn_norm_weighted_sums.dat";
	char * ffn_norm_rms_vals_f = "test_layer/ffn_norm_rms_vals.dat";

	ret = save_host_matrix(ffn_norm_out_f, ffn_norm_out, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save ffn_norm_out matrix...\n");
		return -1;
	}

	ret = save_host_matrix(ffn_norm_weighted_sums_f, ffn_norm_weighted_sums, 1, total_q, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save ffn_norm_weighted_sums matrix...\n");
		return -1;
	}

	ret = save_host_matrix(ffn_norm_rms_vals_f, ffn_norm_rms_vals, 1, total_q, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save ffn_norm_rms_vals matrix...\n");
		return -1;
	}


	char * w1_out_f = "test_layer/x_w1.dat";
	char * w3_out_f = "test_layer/x_w3.dat";

	ret = save_host_matrix(w1_out_f, w1_out, total_q, (int) ffn_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_w1 matrix...\n");
		return -1;
	}

	ret = save_host_matrix(w3_out_f, w1_out, total_q, (int) ffn_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_w3 matrix...\n");
		return -1;
	}


	char * swiglu_out_f = "test_layer/x_swiglu.dat";

	ret = save_host_matrix(swiglu_out_f, swiglu_out, total_q, (int) ffn_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_swiglu matrix...\n");
		return -1;
	}


	char * w2_out_f = "test_layer/x_layer_out.dat";

	ret = save_host_matrix(w2_out_f, w2_out, total_q, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_layer_out matrix...\n");
		return -1;
	}


	printf("\n\n\nSuccessfully Performed Op...!!!\n");

	return 0;
}
