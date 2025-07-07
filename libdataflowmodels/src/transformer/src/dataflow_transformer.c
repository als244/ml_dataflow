
#include "dataflow_transformer.h"

// toggle to print out before submitting any ops...
// verbose printing of each op
#define TO_PRINT 0

// meta-toggle required to be set to 1 to save any data
// when set to 0, nothing will be saved
#define TO_SAVE_DATA 0


// DETERMINES WHAT DATA TO SAVE...
#define TO_SAVE_EMBEDDING 1
#define TO_SAVE_EMBEDDING_BWD 1

// if 0, then no layers will be saved
#define TO_SAVE_LAYER 1
// if save layer is 1, then layer id specifies which layer to save
// if -1, then all layers will be saved
#define LAYER_ID_TO_SAVE -1

// toggle required to be set to 1 to save any head data
#define TO_SAVE_HEAD 1
#define TO_SAVE_HEAD_BWD 1

#define TO_SAVE_BWD_LAYER 1
#define BWD_LAYER_ID_TO_SAVE -1


// during test the adam step (on host) also saves the model grads within optimizer_states/<layer_id>/grad_pre_step.dat

#define TO_SAVE_MODEL_GRADS 1

// only save the model grads when chunk id 0 finishes (then layer grad is complete)
#define TO_SAVE_MODEL_GRAD_CHUNK_ID 0



// directory to save data to
// NOTE: this should be set to path relative to directory in which executable is run from
// i.e. if executable is run from <proj_root>/test/dataflow_models, then this should be "test_transformer_data/layers"
// to save into <proj_root>/test/dataflow_models/test_transformer_data/layers

#define DATA_SAVE_DIR "test_transformer_data"

static int save_file(Dataflow_Handle * dataflow_handle, int stream_id, int layer_id, int seq_id, int chunk_id, bool is_bwd, char * filename, void * dev_ptr, int M, int N, DataflowDatatype dt){

	int ret;

	size_t num_els = (size_t) M * (size_t) N;

	size_t el_size = dataflow_sizeof_element(dt);

	void * host_ptr = malloc(num_els * el_size);

	ret = (dataflow_handle -> submit_outbound_transfer)(dataflow_handle, stream_id, host_ptr, dev_ptr, num_els * el_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit outbound transfer...\n");
		return -1;
	}

	ret = (dataflow_handle -> sync_stream)(dataflow_handle, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream...\n");
		return -1;
	}

	char full_filename[1024];

	// model weights!
	if (seq_id == -1){
		if (layer_id >= 0){
			if (is_bwd){
				sprintf(full_filename, "%s/model_grads/layers/%d/%s.dat", DATA_SAVE_DIR, layer_id, filename);
			}
			else{
				sprintf(full_filename, "%s/model/layers/%d/%s.dat", DATA_SAVE_DIR, layer_id, filename);
			}
		}
		else{
			// layer id == -2 => embedding
			if (layer_id == -2){
				if (is_bwd){
					sprintf(full_filename, "%s/model_grads/embedding/%s.dat", DATA_SAVE_DIR, filename);
				}
				else{
					sprintf(full_filename, "%s/model/embedding/%s.dat", DATA_SAVE_DIR, filename);
				}
			}
			// layer id == -1 => head
			else if (layer_id == -1){
				if (is_bwd){
					sprintf(full_filename, "%s/model_grads/head/%s.dat", DATA_SAVE_DIR, filename);
				}
				else{
					sprintf(full_filename, "%s/model/head/%s.dat", DATA_SAVE_DIR, filename);
				}
			}
			else{
				fprintf(stderr, "Error: invalid layer id (should be -2 (embedding), -1 (head), or layer id >= 0)): %d\n", layer_id);
				return -1;
			}
		}
	}
	else{
		if (layer_id >= 0){
			if (is_bwd){
				sprintf(full_filename, "%s/layers_bwd/%d/seq_%d_chunk_%d_%s.dat", DATA_SAVE_DIR, layer_id, seq_id, chunk_id, filename);
			} else {
				sprintf(full_filename, "%s/layers_fwd/%d/seq_%d_chunk_%d_%s.dat", DATA_SAVE_DIR, layer_id, seq_id, chunk_id, filename);
			}
		}
		// embed or head 
		else {
			if (layer_id == -2){
				if (is_bwd){
					sprintf(full_filename, "%s/embedding_bwd/seq_%d_chunk_%d_%s.dat", DATA_SAVE_DIR, seq_id, chunk_id, filename);
				} else {
					sprintf(full_filename, "%s/embedding_fwd/seq_%d_chunk_%d_%s.dat", DATA_SAVE_DIR, seq_id, chunk_id, filename);
				}
			}
			else if (layer_id == -1){
				if (is_bwd){
					sprintf(full_filename, "%s/head_bwd/seq_%d_chunk_%d_%s.dat", DATA_SAVE_DIR, seq_id, chunk_id, filename);
				} else {
					sprintf(full_filename, "%s/head_fwd/seq_%d_chunk_%d_%s.dat", DATA_SAVE_DIR, seq_id, chunk_id, filename);
				}
			}
			else{
				fprintf(stderr, "Error: invalid layer id (should be -2 (embedding), -1 (head), or layer id >= 0)): %d\n", layer_id);
				return -1;
			}
		}
	}

	FILE * fp = fopen(full_filename, "wb");
	if (!fp){
		free(host_ptr);
		fprintf(stderr, "Error: failed to save %s, because couldn't open file: %s...\n", filename, full_filename);
		return -1;
	}

	if (TO_PRINT){
		printf("\n[Saving] %s (%lu bytes)\n", filename, num_els * el_size);
	}

	size_t num_written = fwrite(host_ptr, el_size, num_els, fp);
	if (num_written != num_els){
		free(host_ptr);
		fprintf(stderr, "Error: failed to write to file %s, wrote %zu elements instead of %zu\n", filename, num_written, num_els);
		return -1;
	}

	fclose(fp);

	free(host_ptr);

	return 0;
}


int dataflow_submit_transformer_embedding(Dataflow_Handle * dataflow_handle, int compute_stream_id,
											Transformer_Model_Input * model_input,
											Transformer_Embedding_Table * embedding_table,
											Transformer_Block_Transition * embedding_output) {


		int ret;

		Seq_Batch * seq_batch = model_input -> seq_batch;

		int seq_id = seq_batch -> seq_id;
		int chunk_id = seq_batch -> chunk_id;

		int total_tokens = seq_batch -> total_tokens;

		Embedding_Config * embedding_table_config = embedding_table -> config;

		int embedding_dim = embedding_table_config -> embedding_size;

		Seq_Batch_Embedding_Config * batch_embedding_config = &(seq_batch -> embedding_config);

		DataflowDatatype embed_dt = embedding_table_config -> embed_dt;

		int num_unique_tokens = batch_embedding_config -> num_unique_tokens;
		uint32_t * sorted_token_ids = batch_embedding_config -> sorted_token_ids;
		uint32_t * sorted_token_mapping = batch_embedding_config -> sorted_token_mapping;
		uint32_t * unique_token_sorted_inds_start = batch_embedding_config -> unique_token_sorted_inds_start;

		ret = dataflow_submit_default_embedding_table(dataflow_handle, compute_stream_id,
														embed_dt, num_unique_tokens, embedding_dim, 
														sorted_token_ids, sorted_token_mapping, unique_token_sorted_inds_start,
														embedding_table -> embedding_table, embedding_output -> X);

		if (ret){
			fprintf(stderr, "Error: failed to submit embedding table...\n");
			return -1;
		}


		if (TO_SAVE_DATA && TO_SAVE_EMBEDDING) {
			ret = save_file(dataflow_handle, compute_stream_id, -2, seq_id, chunk_id, false, "x_embedding", embedding_output -> X, total_tokens, embedding_dim, embed_dt);
			if (ret){
				fprintf(stderr, "Error: failed to save x_embedding file...\n");
				return -1;
			}
		}

		return 0;
}

// ALL BAKED INTO 1 Large Function for now,
// but really should have subfunctions to do norms, attn, and mlp based on transformer block config...!

int dataflow_submit_transformer_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
								Transformer_Block_Transition * block_input, 
								Transformer_Block * transformer_block, 
								Transformer_Block_Activations * activations, 
								Transformer_Block_Transition * block_output) {

    int ret;

	int layer_id = transformer_block -> layer_id;

	Transformer_Block_Config * block_config = &(transformer_block -> config);
    DataflowDatatype fwd_dt = block_config -> block_dt;
	DataflowDatatype compute_dt = block_config -> compute_dt;

	int model_dim = block_config -> model_dim;
    int kv_dim = block_config -> kv_dim;
    int ffn_dim = block_config -> ffn_dim;

	size_t x_el_size = dataflow_sizeof_element(fwd_dt);

	   
	Seq_Batch * seq_batch = block_input -> seq_batch;

	int seq_id = seq_batch -> seq_id;
	int chunk_id = seq_batch -> chunk_id;

	Seq_Batch_Attention_Config * batch_attention_config = &(seq_batch -> attention_config);
    int num_seqs = batch_attention_config -> num_seqs;
    int total_q = batch_attention_config -> total_q;
    int total_k = batch_attention_config -> total_k;
 

	Seq_Batch_Saved_Activations * working_activations = activations -> working_activations;
	Seq_Batch_Activation_Workspace * activation_workspace = activations -> activation_workspace;

    uint64_t kernelWorkspaceBytes = activation_workspace -> kernelWorkspaceBytes;
    void * kernelWorkspace = activation_workspace -> kernelWorkspace;


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

	int to_transa = 1;
	int to_transb = 0;


	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_act_stream", block_input -> X, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_act_stream file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Saving blocking input into activations buffer...\n");
	}

	ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, working_activations -> x_inp, block_input -> X, (uint64_t) total_q * (uint64_t) model_dim * (uint64_t) x_el_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit peer transfer for x_inp...\n");
		return -1;
	}


	if (TO_PRINT){
		printf("Submitting Attention RMS Norm...!\n");
	}



	ret = dataflow_submit_default_rms_norm(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, (transformer_block -> config).eps, 
						transformer_block -> w_attn_norm, block_input -> X, activation_workspace -> x_temp, 
						working_activations -> attn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit attention norm...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_attn_norm", activation_workspace -> x_temp, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save attention nor file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Q, K, V matmuls...!\n");
	}

	// Q Proj
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_q, activation_workspace -> x_temp, NULL, working_activations -> x_q,
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit Q matmul proj...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_q", working_activations -> x_q, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q file...\n");
			return -1;
		}
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					kv_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_k, activation_workspace -> x_temp, NULL, working_activations -> x_k_local,
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit K matmul proj...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_k", working_activations -> x_k_local, total_q, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_k file...\n");
			return -1;
		}
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					kv_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_v, activation_workspace -> x_temp, NULL, working_activations -> x_v_local,
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit V matmul proj...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_v_local", working_activations -> x_v_local, total_q, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_v file...\n");
			return -1;
		}
	}


	if (TO_PRINT){
		printf("Submitting RoPE...!\n");
	}

	int num_q_heads = (transformer_block -> config).num_q_heads;
	int num_kv_heads = (transformer_block -> config).num_kv_heads;
	int head_dim = (transformer_block -> config).head_dim;


	ret = dataflow_submit_default_rope(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, head_dim, num_kv_heads, (transformer_block -> config).theta,
						batch_attention_config -> seq_positions, working_activations -> x_q, working_activations -> x_k_local);
	if (ret){
		fprintf(stderr, "Error: failed to submit rope...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_q_rope", working_activations -> x_q, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q_rope file...\n");
			return -1;
		}
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_k_rope_local", working_activations -> x_k_local, total_q, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_k_rope file...\n");
			return -1;
		}
	}


	if (TO_PRINT){
		printf("Submitting Attention...!\n");
	}

	// ensure workspace is zerod out beforehand....
	// doing this within attention kernel itself now because only some parts of workspace need to be zeroed out
	/*
	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, kernelWorkspace, 0, kernelWorkspaceBytes);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}
	*/

	void * q_seq_offsets = batch_attention_config -> q_seq_offsets;
	void * q_seq_lens = batch_attention_config -> q_seq_lens;
	int max_seqlen_q = batch_attention_config -> max_seqlen_q;

	void * k_seq_offsets = batch_attention_config -> k_seq_offsets;
	void * k_seq_lens = batch_attention_config -> k_seq_lens;
	int max_seqlen_k = batch_attention_config -> max_seqlen_k;

	Seq_Batch_Context * context = seq_batch -> context;

	void * x_k_global = working_activations -> x_k_local;
	void * x_v_global = working_activations -> x_v_local;

	if (context){
		x_k_global = context -> x_k;
		x_v_global = context -> x_v;

		void * x_k_global_dest = x_k_global + ((uint64_t) context -> cur_tokens_populated * (uint64_t) kv_dim * (uint64_t) x_el_size);
		void * x_v_global_dest = x_v_global + ((uint64_t) context -> cur_tokens_populated * (uint64_t) kv_dim * (uint64_t) x_el_size);

		// need to copy x_k_local and x_v_local into global context...

		ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, x_k_global_dest, working_activations -> x_k_local, (uint64_t) total_q * (uint64_t) kv_dim * (uint64_t) x_el_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit peer transfer for x_k_global...\n");
			return -1;
		}

		ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, x_v_global_dest, working_activations -> x_v_local, (uint64_t) total_q * (uint64_t) kv_dim * (uint64_t) x_el_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit peer transfer for x_v_global...\n");
			return -1;
		}

		context -> cur_tokens_populated += total_q;

		// assert (context -> cur_tokens_populated == total_q);

	}


	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))) {
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_k_global_attn_inp", x_k_global, context -> cur_tokens_populated, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_k_global_attn_inp file...\n");
			return -1;
		}
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_v_global_attn_inp", x_v_global, context -> cur_tokens_populated, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_k_global_attn_inp file...\n");
			return -1;
		}
	}


	int is_causal = 1;

	ret = dataflow_submit_attention(dataflow_handle, compute_stream_id,
						 fwd_dt, 
						 num_seqs, total_q, total_k,
						 q_seq_offsets, q_seq_lens, max_seqlen_q,
						 k_seq_offsets, k_seq_lens, max_seqlen_k,
						 num_q_heads, num_kv_heads, head_dim,
						 working_activations -> x_q, x_k_global, x_v_global,
						 working_activations -> x_attn_out, working_activations -> softmax_lse, 
						 is_causal,
						 kernelWorkspaceBytes, kernelWorkspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention...\n");
		return -1;
	}


	// reset for beginning of next layer so it can populate
	if (context -> cur_tokens_populated == context -> total_context_tokens){
		context -> cur_tokens_populated = 0;
	}
	

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_attn", working_activations -> x_attn_out, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn file...\n");
			return -1;
		}
	}


	if (TO_PRINT){
		printf("Submitting Attention Output Matmul...!\n");
	}


	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 1.0,
					transformer_block -> w_o, working_activations -> x_attn_out, block_input -> X, working_activations -> x_o,
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit o matmul proj...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_attn_final_out", working_activations -> x_o, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_final_out file...\n");
			return -1;
		}
	}


	if (TO_PRINT){
		printf("Submitting FFN RMS Norm...!\n");
	}

	ret = dataflow_submit_default_rms_norm(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, (transformer_block -> config).eps, 
						transformer_block -> w_ffn_norm, working_activations -> x_o, activation_workspace -> x_temp, 
						working_activations -> ffn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit ffn norm...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_ffn_norm_out", activation_workspace -> x_temp, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_final_out file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting FFN w1 and w3 matmuls...!\n");
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					(transformer_block -> w_1)[0], activation_workspace -> x_temp, NULL, (working_activations -> x_1)[0],
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w1 matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					(transformer_block -> w_3)[0], activation_workspace -> x_temp, NULL, (working_activations -> x_3)[0],
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w3 matmul proj...\n");
		return -1;
	}


	if (TO_PRINT){
		printf("Submitting SwiGLU Activation...!\n");
	}


	ret = dataflow_submit_default_swiglu(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, ffn_dim, 
						(working_activations -> x_1)[0], (working_activations -> x_3)[0], activation_workspace -> x_temp_mlp);

	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu activation...\n");
		return -1;
	}


	if (TO_PRINT){
		printf("Submitting FFN w2 matmul...!\n");
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q, 
					1.0, 1.0,
					(transformer_block -> w_2)[0], activation_workspace -> x_temp_mlp, working_activations -> x_o, block_output -> X,
					kernelWorkspaceBytes, kernelWorkspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w2 matmul proj...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, false, "x_act_stream_out", block_output -> X, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_act_stream file...\n");
			return -1;
		}
	}

	// copy to output
	return 0;

}


int dataflow_submit_transformer_head(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                        Transformer_Block_Transition * block_input, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations, 
                        Transformer_Model_Output * model_output,
						// during interference these would be NULL
						Transformer_Head * grad_transformer_head,
						Transformer_Block_Transition * grad_stream) {

    int ret;

	Seq_Batch * seq_batch = block_input -> seq_batch;

	int seq_id = seq_batch -> seq_id;
	int chunk_id = seq_batch -> chunk_id;

    // Get dimensions from embedding config
    int vocab_size = (transformer_head -> embedding_config) -> vocab_size;
    int embedding_size = (transformer_head -> embedding_config) -> embedding_size;

	int total_tokens = head_activations -> num_tokens;

	DataflowDatatype fwd_dt = transformer_head -> fwd_dt;
	DataflowDatatype bwd_dt = transformer_head -> bwd_dt;


	if (TO_SAVE_DATA && TO_SAVE_HEAD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, false, "x_inp", block_input -> X, total_tokens, embedding_size, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_inp file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Head RMS Norm...\n");
	}

    // RMS Normalization
    ret = dataflow_submit_default_rms_norm(dataflow_handle, compute_stream_id,
                         transformer_head -> fwd_dt,
                         head_activations -> num_tokens,
                         embedding_size, transformer_head -> eps,
                         transformer_head -> w_head_norm,
                         block_input -> X,
                         head_activations -> head_norm_out,
                         head_activations -> head_norm_rms_vals);
    if (ret) {
        fprintf(stderr, "Error: Failed to submit RMS normalization in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_HEAD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, false, "x_norm", head_activations -> head_norm_out, total_tokens, embedding_size, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_norm file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Head Output Projection...\n");
	}

    // Output projection
    // Input x_temp is in row-major format
    // Weights w_out are in col-major format
    // Want output in row-major format: head_out[num_tokens, vocab_size] = x_temp[num_tokens, embedding_size] @ w_out[embedding_size, vocab_size]
    // Use transa=1 for row-major input, transb=0 for col-major weights
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                       transformer_head -> fwd_dt,
                       transformer_head -> fwd_dt,
                       DATAFLOW_NONE,
                       transformer_head -> fwd_dt,
                       transformer_head -> compute_dt,
                       1, 0,  // transa=1 for row-major input, transb=0 for col-major weights
                       vocab_size,        // n = vocab_size
					   embedding_size,    // m = embedding_size (due to transa=1)
                       head_activations -> num_tokens,  // k = num_tokens (due to transa=1)
                       1.0, 0.0,
                       transformer_head -> w_head,       // Changed from w_out to w_head
                       head_activations -> head_norm_out,      // B[num_tokens, embedding_size] in row-major
                       NULL,
                       head_activations -> head_out,    // C[num_tokens, vocab_size] in row-major
                       head_activations -> kernelWorkspaceBytes, head_activations -> kernelWorkspace);  // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output projection in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_HEAD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, false, "x_head_out", head_activations -> head_out, total_tokens, vocab_size, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_head_out file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Head Softmax to Determine Final Logits...\n");
	}

    // Apply softmax over vocabulary dimension
    // Each row (corresponding to a token) should sum to 1

	// using an extra N_tokens x vocab_size buffer for the softmax, 
	// because makes it easier if fwd_dtype != bwd_dtype...

	// if necessary can update head_out in place if same dtype...
    ret = dataflow_submit_default_softmax(dataflow_handle, compute_stream_id,
                        transformer_head -> fwd_dt,
                        transformer_head -> bwd_dt,  // Use bwd_dt for backward pass
                        head_activations -> num_tokens,  // Number of rows to process
                        vocab_size,                      // Size of softmax dimension
                        head_activations -> head_out,   // Input logits
                        model_output -> logits);        // Output probabilities to provided buffer
    if (ret) {
        fprintf(stderr, "Error: Failed to submit softmax in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_HEAD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, false, "x_logits", model_output -> logits, total_tokens, vocab_size, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_logits file...\n");
			return -1;
		}
	}

	// IF ONLY FOR FORWARD PASS, THEN RETURN HERE...
	if (!grad_transformer_head){
		return 0;
	}

	// DOING BACKWARDS PASS OF HEAD HERE!!!
	// Getting the grad stream to pass back to the last block...



	

	if (seq_batch -> loss_config.num_tokens_to_predict == 0){
		return 0;
	}

	if (TO_PRINT){	
		printf("Submitting Head Cross Entropy Loss...\n");
	}

	// STARTING BACKPROP HERE!

	// compute cross entropy loss
	// updates logits in-place

    // First compute cross entropy loss gradients
    // This will subtract 1.0 from the correct class logits in-place
    ret = dataflow_submit_default_cross_entropy_loss(dataflow_handle, compute_stream_id,
                                  transformer_head -> bwd_dt,
                                  head_activations -> num_tokens,  // Number of rows (tokens)
                                  vocab_size,                      // Number of columns (vocab size)
                                  model_output -> logits,         // Predicted logits
                                  (seq_batch -> loss_config).labels,  // Ground truth labels
								  (seq_batch -> loss_config).loss_vec);       // per token loss + avg loss as last el
       
	if (ret){
	    fprintf(stderr, "Error: Failed to submit cross entropy loss in transformer head backward...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_HEAD_BWD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, true, "x_logits_loss", model_output -> logits, total_tokens, vocab_size, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_logits_loss file...\n");
			return -1;
		}
	}

	// Now backpropagate through the output projection
    // Input logits is in row-major format after cross entropy
    // Weights w_head are in col-major format
    // Want output in row-major format: dx_temp = dlogits @ w_head^T
    // For backward pass with transa=0, transb=0:
    // M = fwd_K = embedding_size
    // K = fwd_N = vocab_size
    // N = fwd_M = num_tokens

	// Now backpropagate through the output projection

	if (TO_PRINT){
		printf("Submitting Output Projection Weight Gradients...\n");
	}

	float grad_avg_scale = 1.0f / ((float)head_activations -> total_pred_tokens_in_step);

	 // 2. Output projection weight gradients
    
	// transa = 0, transb = 1

	// if both are row-major and we want a col-major output, then "NT" transpose is correct

    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> compute_dt,
                       0, 1,
                       embedding_size, head_activations -> num_tokens, vocab_size,
                       grad_avg_scale, 1.0,  // Accumulate gradients,
					   head_activations -> head_norm_out,    // Input activations [num_tokens, embedding_size]     
					   model_output -> logits, // Gradient of output [num_tokens, vocab_size]
                       grad_transformer_head -> w_head,      // Previous gradient
                       grad_transformer_head -> w_head,      // Output gradient
                       head_activations -> kernelWorkspaceBytes, head_activations -> kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output bwd weight gradient computation in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, -1, -1, -1, true, "w_head", grad_transformer_head -> w_head, embedding_size, vocab_size, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_head file...\n");
			return -1;
		}
	}

	// Now repurpose head_activation -> head_norm_out as the derivative of the norm

	if (TO_PRINT){
		printf("Submitting Bwd X Head Matmul...\n");
	}

    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                       transformer_head -> bwd_dt,
                       transformer_head -> bwd_dt,
                       DATAFLOW_NONE,
                       transformer_head -> bwd_dt,
                       transformer_head -> compute_dt,
                       0, 0,
                       embedding_size,
					   vocab_size, 
					   head_activations -> num_tokens,
                       grad_avg_scale, 0.0,
					   transformer_head -> w_head,     // w_head[embedding_size, vocab_size] in col-major
                       model_output -> logits,         // dlogits[num_tokens, vocab_size] in row-major
                       NULL,     
                       head_activations -> head_norm_out, // dx_temp[num_tokens, embedding_size] in row-major
                       head_activations -> kernelWorkspaceBytes, head_activations -> kernelWorkspace);  // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit bwd x head matmul in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_HEAD_BWD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, true, "x_head_proj_inp", head_activations -> head_norm_out, total_tokens, embedding_size, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_head_proj_inp file...\n");
			return -1;
		}

		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, true, "w_head_proj", transformer_head -> w_head, embedding_size, vocab_size, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_head_proj file...\n");
			return -1;
		}
	}

	
	if (TO_PRINT){
		printf("Submitting Bwd X for RMS Norm to get Gradient Stream...\n");
	}

	// Finally backpropagate through RMS normalization

	// don't need to save downt the recomputed norm value because already have it...
    ret = dataflow_submit_default_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
                               transformer_head -> bwd_dt,
                               transformer_head -> bwd_dt,
                               head_activations -> num_tokens,
                               embedding_size,
                               transformer_head -> eps,
                               head_activations -> head_norm_rms_vals,
                               transformer_head -> w_head_norm,
                               block_input -> X,         // Original input
                               head_activations -> head_norm_out,      // Upstream gradient
                               grad_stream -> X, 
							   NULL);
	if (ret) {
        fprintf(stderr, "Error: Failed to submit bwd x rms norm in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_HEAD_BWD){
		ret = save_file(dataflow_handle, compute_stream_id, -1, seq_id, chunk_id, true, "x_head_norm_inp", grad_stream -> X, total_tokens, embedding_size, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head x_head_norm_inp file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Bwd W for RMS Norm to update Head Norm...\n");
	}

	ret = dataflow_submit_default_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
                               grad_transformer_head -> fwd_dt,
                               grad_transformer_head -> bwd_dt,
                               head_activations -> num_tokens,
                               embedding_size,
                               grad_transformer_head -> eps,
                               head_activations -> head_norm_rms_vals,  // RMS values from forward pass
                               block_input -> X,           				// Original input
                               head_activations -> head_norm_out,         // Upstream gradient
                               grad_transformer_head -> w_head_norm,
							   head_activations -> kernelWorkspaceBytes, head_activations -> kernelWorkspace);   // Output gradient
	if (ret) {
        fprintf(stderr, "Error: Failed to submit bwd w rms norm in transformer head...\n");
        return ret;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, -1, -1, -1, true, "w_head_norm", grad_transformer_head -> w_head_norm, 1, embedding_size, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_head_norm file...\n");
			return -1;
		}
	}

	// FINAL OUTPUT GRADIENT IS IN grad_stream -> X!!!

    return 0;
}




// Assumse that working_activations has been already populated in device memory with the appropriate
// saved activation level amount of data alreay ready to go (prefix of the this chunk). Assumes that
// there is enough memory to save the forward activations for the entire block (repeating the forward pass)
// and that the bwd_x function will be called next!
int dataflow_submit_transformer_block_recompute(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
												Transformer_Block * transformer_block,
												Seq_Batch * seq_batch,
												SavedActivationLevel saved_activation_level,
												Seq_Batch_Saved_Activations * working_activations, Seq_Batch_Context * fwd_context,
												Seq_Batch_Activation_Workspace * activation_workspace) {
	
	int ret;

	if (saved_activation_level == SAVED_ACTIVATION_LEVEL_FULL){
		return 0;
	}

	if (saved_activation_level == SAVED_ACTIVATION_LEVEL_NONE){
		fprintf(stderr, "Error: cannot recompute with no block input...\n");
	}

	Transformer_Block_Config * fwd_block_config = &(transformer_block -> config);
    DataflowDatatype fwd_dt = fwd_block_config -> block_dt;
	DataflowDatatype compute_dt = fwd_block_config -> compute_dt;

	int model_dim = fwd_block_config -> model_dim;
	int ffn_dim = fwd_block_config -> ffn_dim;
	int theta = fwd_block_config -> theta;


    uint64_t kernelWorkspaceBytes = activation_workspace -> kernelWorkspaceBytes;
    void * kernelWorkspace = activation_workspace -> kernelWorkspace;

	int num_q_heads = (transformer_block -> config).num_q_heads;
	int num_kv_heads = (transformer_block -> config).num_kv_heads;
	int head_dim = (transformer_block -> config).head_dim;

	Seq_Batch_Attention_Config * batch_attention_config = &(seq_batch -> attention_config);
    int num_seqs = batch_attention_config -> num_seqs;
    int total_q = batch_attention_config -> total_q;
    int total_k = batch_attention_config -> total_k;

	int is_causal = 1;

	void * q_seq_offsets = batch_attention_config -> q_seq_offsets;
	void * q_seq_lens = batch_attention_config -> q_seq_lens;
	int max_seqlen_q = batch_attention_config -> max_seqlen_q;

	void * k_seq_offsets = batch_attention_config -> k_seq_offsets;
	void * k_seq_lens = batch_attention_config -> k_seq_lens;
	int max_seqlen_k = batch_attention_config -> max_seqlen_k;

	int total_tokens = total_q;

	// In forward pass mode!
	int to_transa = 1;
	int to_transb = 0;

	// recompute the queries regardless
	if ((saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ONLY) || (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY)){
		// need to recompute queries

		ret = dataflow_submit_default_rms_norm_recompute(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim,
						transformer_block -> w_attn_norm, 
						working_activations -> attn_norm_rms_vals,
						working_activations -> x_inp,
						activation_workspace -> x_temp);

		if (ret){
			fprintf(stderr, "Error: failed to submit rms norm recompute for input of attention norm...\n");
			return ret;
		}

		ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_q, activation_workspace -> x_temp, NULL, working_activations -> x_q,
					kernelWorkspaceBytes, kernelWorkspace);
		
		if (ret){
			fprintf(stderr, "Error: failed to submit matmul for q proj during recompute...\n");
			return ret;
		}

		// now do rope, (only for x_q)
		ret = dataflow_submit_default_rope(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, head_dim, num_kv_heads, theta,
						batch_attention_config -> seq_positions, working_activations -> x_q, NULL);
		if (ret){
			fprintf(stderr, "Error: failed to submit rope for q during recompute...\n");
			return ret;
		}
	}

	// recompoute attention if needed
	if (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ONLY){
		// need to recompute attention

		// forward context already has the correct x_k_rope and x_v for this chunk already populated...
		ret = dataflow_submit_attention(dataflow_handle, compute_stream_id,
						 fwd_dt, 
						 num_seqs, total_q, total_k,
						 q_seq_offsets, q_seq_lens, max_seqlen_q,
						 k_seq_offsets, k_seq_lens, max_seqlen_k,
						 num_q_heads, num_kv_heads, head_dim,
						 working_activations -> x_q, fwd_context -> x_k, fwd_context -> x_v,
						 working_activations -> x_attn_out, working_activations -> softmax_lse, 
						 is_causal,
						 kernelWorkspaceBytes, kernelWorkspace);
		if (ret){
			fprintf(stderr, "Error: failed to submit attention during recompute...\n");
			return -1;
		}
	}

	// now do the output projection and ffn no matter what...
	if ((saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ONLY) || (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY)){
	
		// add result to the block input..
		ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 1.0,
					transformer_block -> w_o, working_activations -> x_attn_out, working_activations -> x_inp, working_activations -> x_o,
					kernelWorkspaceBytes, kernelWorkspace);

		if (ret){
			fprintf(stderr, "Error: failed to submit o matmul proj during recompute...\n");
			return -1;
		}

		ret = dataflow_submit_default_rms_norm_recompute(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim,
						transformer_block -> w_ffn_norm, 
						working_activations -> ffn_norm_rms_vals,
						working_activations -> x_o,
						activation_workspace -> x_temp);

		if (ret){
			fprintf(stderr, "Error: failed to submit rms norm recompute for input of ffn norm...\n");
			return ret;
		}

		ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					(transformer_block -> w_1)[0], activation_workspace -> x_temp, NULL, (working_activations -> x_1)[0],
					kernelWorkspaceBytes, kernelWorkspace);

		if (ret){
			fprintf(stderr, "Error: failed to submit w1 matmul proj during recompute...\n");
			return -1;
		}

		ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
						fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
						compute_dt,
						to_transa, to_transb,
						ffn_dim, model_dim, total_q, 
						1.0, 0.0,
						(transformer_block -> w_3)[0], activation_workspace -> x_temp, NULL, (working_activations -> x_3)[0],
						kernelWorkspaceBytes, kernelWorkspace);

		if (ret){
			fprintf(stderr, "Error: failed to submit w3 matmul proj during recompute...\n");
			return -1;
		}

		// only fill up to x_1 and x_3, don't need to recompute swiglu or 2nd matmul...
	}

	return 0;


}















// dX_out is upstream gradient
// it is in col-major orientation

// now to do dX = matmul(dX_out, W^T)
// we remember that W is being stored in col-major as well
// thus we can correctly get dX in col major as:
// we can do matmul(W, dx_Out^T)
// This is because we will get a row major output of 
// dX^T = dX in col major

// To see how we get col major of dX recall:

// thus if we pass A = W in col-major = W^T effectively in row major
// and B = dX_Out in col-major,
// If we pass in M = w_in, K = w_out, N = x_in, which yields
// a (w_in, x_in) matrix which we interpret as col major dX
int dataflow_submit_transformer_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id,
								Transformer_Block * transformer_block, 
								Transformer_Block_Transition * inp_grad_stream, 
								Seq_Batch_Saved_Activations * fwd_activations, Seq_Batch_Context * fwd_context,
								Transformer_Block_Activations * grad_activations,
								Transformer_Block * grad_weights,
								Transformer_Block_Transition * next_grad_stream) {

	int ret;

	int layer_id = transformer_block -> layer_id;

	Transformer_Block_Config * fwd_block_config = &(transformer_block -> config);
    DataflowDatatype fwd_dt = fwd_block_config -> block_dt;

	Transformer_Block_Config * bwd_block_config = &(grad_weights -> config);
	DataflowDatatype bwd_dt = bwd_block_config -> block_dt;
	DataflowDatatype compute_dt = bwd_block_config -> compute_dt;

	int model_dim = bwd_block_config -> model_dim;
    int kv_dim = bwd_block_config -> kv_dim;
    int ffn_dim = bwd_block_config -> ffn_dim;

	size_t x_el_size = dataflow_sizeof_element(fwd_dt);
	size_t x_el_bwd_size = dataflow_sizeof_element(bwd_dt);

	   
	Seq_Batch * seq_batch = inp_grad_stream -> seq_batch;
	Seq_Batch_Attention_Config * batch_attention_config = &(seq_batch -> attention_config);
    int num_seqs = batch_attention_config -> num_seqs;
    int total_q = batch_attention_config -> total_q;
    int total_k = batch_attention_config -> total_k;
 

	Seq_Batch_Saved_Activations * working_activations = grad_activations -> working_activations;
	Seq_Batch_Activation_Workspace * activation_workspace = grad_activations -> activation_workspace;

	uint64_t kernelWorkspaceBytes = activation_workspace -> kernelWorkspaceBytes;
    void * kernelWorkspace = activation_workspace -> kernelWorkspace;

	// context gradients being accumulated
	Seq_Batch_Context * bwd_context = seq_batch -> context;

	int seq_id = seq_batch -> seq_id;
	int chunk_id = seq_batch -> chunk_id;
	

	int to_transa = 0;
	int to_transb = 0;


	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_upstream_grad", inp_grad_stream -> X, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_upstream_grad file...\n");
			return -1;
		}
	}

	// Will be modifying the input gradient stream (adding to it, but need to save it for the bwd_w pass, so make a copy and put it in the next gradient stream)

	ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, next_grad_stream -> X, inp_grad_stream -> X, total_q * model_dim * x_el_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit memcpy to update the the working gradient stream...\n");
		return -1;
	}



	// 1. Backprop through FFN w2 matmul

	if (TO_PRINT){
		printf("Submitting Matmul to get dX for input of W2 FFN...\n");
	}

	// Forward: [num_tokens, ffn_dim] @ [ffn_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = ffn_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q,  // M = ffn_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					(transformer_block -> w_2)[0], inp_grad_stream -> X, NULL, (activation_workspace -> x_temp_mlp),
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w2 backward matmul...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_2_inp", activation_workspace -> x_temp_mlp, total_q, ffn_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_2_inp file...\n");
			return -1;
		}
	}



	// 2. Backprop through SwiGLU

	if (TO_PRINT){
		printf("Submitting SwiGLU Bwd X to get dX for input of W1 and W3 FFNs...\n");
	}

	ret = dataflow_submit_default_swiglu_bwd_x(dataflow_handle, compute_stream_id,
						bwd_dt, bwd_dt,
						total_q, ffn_dim,
						(fwd_activations -> x_1)[0], (fwd_activations -> x_3)[0],
						activation_workspace -> x_temp_mlp,
						(working_activations -> x_1)[0], (working_activations -> x_3)[0]);
	if (ret) {
		fprintf(stderr, "Error: failed to submit swiglu backward...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_1_swiglu_inp", (working_activations -> x_1)[0], total_q, ffn_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_1_swiglu_inp file...\n");
			return -1;
		}

		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_3_swiglu_inp", (working_activations -> x_3)[0], total_q, ffn_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_3_swiglu_inp file...\n");
			return -1;
		}
	}

	// 3. Backprop through w1 and w3 matmuls
	// Forward: [num_tokens, model_dim] @ [model_dim, ffn_dim] -> [num_tokens, ffn_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = ffn_dim
	// N = batch dim = num_tokens

	if (TO_PRINT){
		printf("Submitting Matmuls (W1, W3) to get dX for output of FFN norm...\n");
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q,  // M = model_dim, K = ffn_dim, N = num_tokens
					1.0, 0.0,
					(transformer_block -> w_1)[0], (working_activations -> x_1)[0], NULL, activation_workspace -> x_temp,
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w1 backward matmul...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_1_inp", activation_workspace -> x_temp, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_1_inp file...\n");
			return -1;
		}
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q,  // M = model_dim, K = ffn_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					(transformer_block -> w_3)[0], (working_activations -> x_3)[0], activation_workspace -> x_temp, activation_workspace -> x_temp,
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w3 backward matmul...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_1_plus_x_3_inp", activation_workspace -> x_temp, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_1_plus_x_3_inp file...\n");
			return -1;
		}
	}

	// 4. Backprop through FFN RMS Norm

	if (TO_PRINT){
		printf("Submitting RMS Norm Bwd X for FFN norm...\n");
	}

	// save the recomputed norm value because will need it for bwd_w pass...
	void * ffn_norm_recomputed = working_activations -> recomputed_activations -> recomputed_ffn_norm;

	ret = dataflow_submit_default_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (transformer_block -> config).eps,
							fwd_activations -> ffn_norm_rms_vals,
							transformer_block -> w_ffn_norm,
							fwd_activations -> x_o,  // Input to norm
							activation_workspace -> x_temp,  // Upstream gradient
							next_grad_stream -> X,
							ffn_norm_recomputed); // Saving the result into the working activations -> x_o);                    
	if (ret) {
		fprintf(stderr, "Error: failed to submit ffn norm backward...\n");
		return -1;
	}

	// ensure to save ths gradient flowing out of the ffn norm (and into the attn out proj...)
	ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, working_activations -> x_o, next_grad_stream -> X, (uint64_t) total_q * (uint64_t) model_dim * (uint64_t) x_el_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit memcpy to update the the working gradient stream...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_ffn_norm_inp_plus_upstream_grad", next_grad_stream -> X, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_ffn_norm_inp_plus_upstream_grad file...\n");
			return -1;
		}

		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_dloss_dattn_out", working_activations -> x_o, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_ffn_norm_inp_plus_upstream_grad file...\n");
			return -1;
		}
	}

	// 5. Now that we have the correct upstream gradient also do bwd_w for ffn norm


	if (TO_PRINT){
		printf("Submitting RMS Norm Bwd W for FFN norm...\n");
	}

	ret = dataflow_submit_default_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
								bwd_dt, bwd_dt,
								total_q, model_dim, (transformer_block -> config).eps,
								fwd_activations -> ffn_norm_rms_vals,
								fwd_activations -> x_o,
								activation_workspace -> x_temp,
								grad_weights -> w_ffn_norm,
								kernelWorkspaceBytes, kernelWorkspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit ffn norm weight gradient computation during bwd_x...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_ffn_norm", grad_weights -> w_ffn_norm, 1, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_ffn_norm file...\n");
			return -1;
		}
	}

	// 6. Backprop through attention output projection
	// Forward: [num_tokens, model_dim] @ [model_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens

	if (TO_PRINT){
		printf("Submitting Matmul to get dX for input of W_o...\n");
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,  // M = model_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_o, working_activations -> x_o, NULL, activation_workspace -> x_temp,
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention output backward matmul...\n");
		return -1;
	}


	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_attn_out_proj_inp", activation_workspace -> x_temp, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_out_proj_inp file...\n");
			return -1;
		}
	}


	// ensure workspace is zerod out beforehand....
	// doing this within attention kernel itself now because only some parts of workspace need to be zeroed out
	/*
	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, kernelWorkspace, 0, kernelWorkspaceBytes);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}
	*/

	if (TO_PRINT){
		printf("Submitting attention bwd to get dX for output of W_q (+ rope), W_k (+ rope), and W_v...\n");
	}

	/*
	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, working_activations -> x_q, 0, (uint64_t) total_q * (uint64_t) model_dim * (uint64_t) x_el_bwd_size);
	if (ret){
		fprintf(stderr, "Error: unable to set x_q workspace mem to 0 before submitting...\n");
		return -1;
	}
	*/

	int is_causal = 1;

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "fwd_x_q_rope", fwd_activations -> x_q, total_q, model_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q_rope file...\n");
			return -1;
		}
	}


	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "fwd_x_k_global_attn_inp", fwd_context -> x_k, total_k, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_k_global_inp file...\n");
			return -1;
		}
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "fwd_x_v_global_attn_inp", fwd_context -> x_v, total_k, kv_dim, fwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_v_global_inp file...\n");
			return -1;
		}
	}

	// 7. Backprop through attention
	ret = dataflow_submit_attention_bwd(dataflow_handle, compute_stream_id,
							bwd_dt,
							num_seqs, total_q, total_k,
							batch_attention_config -> q_seq_offsets,
							batch_attention_config -> q_seq_lens,
							batch_attention_config -> max_seqlen_q,
							batch_attention_config -> k_seq_offsets,
							batch_attention_config -> k_seq_lens,
							batch_attention_config -> max_seqlen_k,
							(transformer_block -> config).num_q_heads,
							(transformer_block -> config).num_kv_heads,
							(transformer_block -> config).head_dim,
							fwd_activations -> x_q,       // Q input
							fwd_context -> x_k,  // K input (full context input keys)
							fwd_context -> x_v,  // V input (full context input values)
							fwd_activations -> x_attn_out,     // Attention output
							fwd_activations -> softmax_lse,// Softmax scaling factors
							activation_workspace -> x_temp,// Upstream gradient
							working_activations -> x_q,   // dQ output
							bwd_context -> x_k,  // dK output (full context key grads)
							bwd_context -> x_v, // dV output (full context grads)
							is_causal,
							kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention backward...\n");
		return -1;
	}


	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_attn_q_inp", working_activations -> x_q, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_q_inp file...\n");
			return -1;
		}
	}


	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_attn_k_global_inp", bwd_context -> x_k, total_k, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_k_global_inp file...\n");
			return -1;
		}
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_attn_v_global_inp", bwd_context -> x_v, total_k, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_v_global_inp file...\n");
			return -1;
		}
	}




	// Now need to copy the correct parts of bwd_context -> x_k and bwd_context -> x_v to the correct locations in working_activations -> x_k_local and working_activations -> x_v_local!...

	uint64_t start_local_token_ind = (bwd_context -> total_context_tokens - bwd_context -> cur_tokens_populated) - total_q;
	void * x_k_global_src = bwd_context -> x_k + ((uint64_t) start_local_token_ind * (uint64_t) kv_dim * (uint64_t) x_el_bwd_size);
	void * x_v_global_src = bwd_context -> x_v + ((uint64_t) start_local_token_ind * (uint64_t) kv_dim * (uint64_t) x_el_bwd_size);
	
	// need to copy gradients from global ctx into local...

	ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, working_activations -> x_k_local, x_k_global_src, (uint64_t) total_q * (uint64_t) kv_dim * (uint64_t) x_el_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit peer transfer for x_k_global...\n");
		return -1;
	}

	ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, compute_stream_id, working_activations -> x_v_local, x_v_global_src, (uint64_t) total_q * (uint64_t) kv_dim * (uint64_t) x_el_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit peer transfer for x_v_global...\n");
		return -1;
	}


	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_attn_k_local_inp", working_activations -> x_k_local, total_q, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_k_local_inp file...\n");
			return -1;
		}
	}

	if (TO_SAVE_DATA && TO_SAVE_LAYER && ((LAYER_ID_TO_SAVE == -1) || (layer_id == LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_attn_v_local_inp", working_activations -> x_v_local, total_q, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_attn_v_local_inp file...\n");
			return -1;
		}
	}

	// ensure that we zero out the graidnet context before the next layer...
	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, x_k_global_src, 0, (uint64_t) total_q * (uint64_t) kv_dim * (uint64_t) x_el_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to zero out k grad context for layer #%d...\n", layer_id);
		return -1;
	}

	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, x_v_global_src, 0, (uint64_t) total_q * (uint64_t) kv_dim * (uint64_t) x_el_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to zero out v grad context for layer #%d...\n", layer_id);
		return -1;
	}

	// Update the cur_tokens_populated...
	bwd_context -> cur_tokens_populated += total_q;

	if (bwd_context -> cur_tokens_populated == bwd_context -> total_context_tokens){
		bwd_context -> cur_tokens_populated = 0;
	}



	// Now workng_activations -> x_k_locat should be pointer within bwd_context -> x_k
	// with the correct accumulated gradients...

	// 8. Backprop through RoPE

	if (TO_PRINT){
		printf("Submitting RoPE Bwd X to get dX for output of W_q and W_k...\n");
	}

	ret = dataflow_submit_default_rope_bwd_x(dataflow_handle, compute_stream_id,
						bwd_dt,
						total_q,
						model_dim,
						(transformer_block -> config).head_dim,
						(transformer_block -> config).num_kv_heads,
						(transformer_block -> config).theta,
						batch_attention_config -> seq_positions,
						working_activations -> x_q,
						working_activations -> x_k_local);
	if (ret) {
		fprintf(stderr, "Error: failed to submit rope backward...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_q_rope_inp", working_activations -> x_q, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q_rope_inp file...\n");
			return -1;
		}
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_k_local_rope_inp", working_activations -> x_k_local, total_q, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_k_local_rope_inp file...\n");
			return -1;
		}
	}

	// 9. Backprop through Q, K, V projections
	// Q Forward: [num_tokens, model_dim] @ [model_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// For Q:
	// M = output rows of dX = model_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens

	if (TO_PRINT){
		printf("Submitting Matmuls (W_q, W_k, W_v) to get dX for output of Attenttion Norm...\n");
	}


	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,  // M = model_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_q, working_activations -> x_q, NULL, activation_workspace -> x_temp,
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit Q projection backward matmul...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_q_inp", activation_workspace -> x_temp, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q_inp file...\n");
			return -1;
		}
	}

	// For K/V:
	// Forward: [num_tokens, model_dim] @ [model_dim, kv_dim] -> [num_tokens, kv_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = kv_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q,  // M = model_dim, K = kv_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_k, working_activations -> x_k_local, activation_workspace -> x_temp, activation_workspace -> x_temp,
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit K projection backward matmul...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_q_inp_plus_k_local_inp", activation_workspace -> x_temp, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q_inp_plus_k_local_inp file...\n");
			return -1;
		}
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q,  // M = model_dim, K = kv_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_v, working_activations -> x_v_local, activation_workspace -> x_temp, activation_workspace -> x_temp,
					kernelWorkspaceBytes, kernelWorkspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit V projection backward matmul...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_q_inp_plus_k_local_inp_plus_v_local_inp", activation_workspace -> x_temp, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_q_inp_plus_k_local_inp_plus_v_local_inp file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting RMS Norm Bwd X for Attenttion Norm to get final updated gradient stream...\n");
	}


	// 10. Finally backprop through attention RMS norm

	// save the recomputed norm value because will need it for bwd_w pass...
	void * attn_norm_recomputed = working_activations -> recomputed_activations -> recomputed_attn_norm;


	ret = dataflow_submit_default_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (transformer_block -> config).eps,
							fwd_activations -> attn_norm_rms_vals,
							transformer_block -> w_attn_norm,
							fwd_activations -> x_inp,  // Input to norm
							activation_workspace -> x_temp,  // Upstream gradient
							next_grad_stream -> X,
							attn_norm_recomputed);                    // Final output gradient
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention norm backward...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_BWD_LAYER && ((BWD_LAYER_ID_TO_SAVE == -1) || (layer_id == BWD_LAYER_ID_TO_SAVE))){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, seq_id, chunk_id, true, "x_next_grad_stream", next_grad_stream -> X, total_q, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save x_next_grad_stream file...\n");
			return -1;
		}
	}

	// While we have the correct upstream gradient, also do bwd_w for attn norm
	if (TO_PRINT){
		printf("Submitting RMS Norm Bwd W for Attenttion Norm...\n");
	}

	ret = dataflow_submit_default_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
								bwd_dt, bwd_dt,
								total_q, model_dim, (transformer_block -> config).eps,
								fwd_activations -> attn_norm_rms_vals,
								fwd_activations -> x_inp,
								activation_workspace -> x_temp,
								grad_weights -> w_attn_norm,
								kernelWorkspaceBytes, kernelWorkspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention norm weight gradient computation during bwd_w...\n");
		return -1;
	}

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_attn_norm", grad_weights -> w_attn_norm, 1, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_attn_norm file...\n");
			return -1;
		}
	}

	// now all gradients should be computed and saved within grad_activations -> working_activations...!!!
	return 0;
}

int dataflow_submit_transformer_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                                Transformer_Block_Transition * grad_stream,
                                Seq_Batch_Saved_Activations * fwd_activations, 
                                Transformer_Block_Activations * grad_activations, 
                                Transformer_Block * grad_weights) {
    
    int ret;

	int layer_id = grad_weights -> layer_id;

    DataflowDatatype bwd_dt = (grad_weights -> config).block_dt;
	// just assume same for now...
    DataflowDatatype fwd_dt = bwd_dt;

    DataflowDatatype compute_dt = (grad_weights -> config).compute_dt;

    Seq_Batch * seq_batch = grad_stream -> seq_batch;
	int seq_id = seq_batch -> seq_id;
	int chunk_id = seq_batch -> chunk_id;

	Seq_Batch_Attention_Config * batch_attention_config = &(seq_batch -> attention_config);
    int num_seqs = seq_batch -> attention_config.num_seqs;
    int total_q = seq_batch -> attention_config.total_q;
    int total_k = seq_batch -> attention_config.total_k;
    
    int model_dim = (grad_weights -> config).model_dim;
    int kv_dim = (grad_weights -> config).kv_dim;
    int ffn_dim = (grad_weights -> config).ffn_dim;
    
    Seq_Batch_Saved_Activations * bwd_activations = grad_activations -> working_activations;
	Seq_Batch_Activation_Workspace * activation_workspace = grad_activations -> activation_workspace;
    
    uint64_t kernelWorkspaceBytes = activation_workspace -> kernelWorkspaceBytes;
    void * kernelWorkspace = activation_workspace -> kernelWorkspace;

	// Need 0,1 for transa, transb if both input matrices are row major
	// and we want a col-major output...
    int to_transa = 0;
    int to_transb = 1;

	

	// 1.) Recompute-Swiglu in order to compute w2 grad
	if (TO_PRINT){
		printf("Recomputing SwiGLU to get input of W2 FFN...\n");
		
	}

	ret = dataflow_submit_default_swiglu(dataflow_handle, compute_stream_id,
						fwd_dt,
						total_q, ffn_dim,
						(fwd_activations -> x_1)[0], (fwd_activations -> x_3)[0],
						activation_workspace -> x_temp_mlp);
	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu when recomputing for w2 grad....\n");
		return -1;
	}

    // 2. FFN w2 weight gradients
    // dW2^T = x_temp_mlp @ dX_out^T
	
	if (TO_PRINT){
		printf("Submitting Matmul to get dW_2...\n");
	}

    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    fwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    ffn_dim, total_q, model_dim, 
                    1.0, 1.0,  // Accumulate gradients
                    activation_workspace -> x_temp_mlp, grad_stream -> X, (grad_weights -> w_2)[0], (grad_weights -> w_2)[0],
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit w2 weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_2", (grad_weights -> w_2)[0], ffn_dim, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_2 file...\n");
			return -1;
		}
	}

    // 2. FFN w1 and w3 weight gradients after SwiGLU

	if (TO_PRINT){
		printf("Submitting Matmul to get dW_1...\n");
	}
	
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, total_q, ffn_dim, 
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> recomputed_activations -> recomputed_ffn_norm, (bwd_activations -> x_1)[0], (grad_weights -> w_1)[0], (grad_weights -> w_1)[0],
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit w1 weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_1", (grad_weights -> w_1)[0], model_dim, ffn_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_1 file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Matmul to get dW_3...\n");
	}

    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, total_q, ffn_dim,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> recomputed_activations -> recomputed_ffn_norm, (bwd_activations -> x_3)[0], (grad_weights -> w_3)[0], (grad_weights -> w_3)[0],
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit w3 weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_3", (grad_weights -> w_3)[0], model_dim, ffn_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_3 file...\n");
			return -1;
		}
	}

    // 3. FFN RMS Norm weight gradients -- already done in bwd_x while we hda correct streaming grad

	if (TO_PRINT){
		printf("Submitting Matmul to get dW_o..\n");
	}


    // 4. Attention output projection weight gradients
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, total_q, model_dim, 
                    1.0, 1.0,  // Accumulate gradients
                    fwd_activations -> x_attn_out, bwd_activations -> x_o, grad_weights -> w_o, grad_weights -> w_o,
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit attention output weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_o", grad_weights -> w_o, model_dim, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_o file...\n");
			return -1;
		}
	}

    // 5. Q, K, V projection weight gradients

	if (TO_PRINT){
		printf("Submitting Matmul to get dW_v..\n");
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, total_q, kv_dim,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> recomputed_activations -> recomputed_attn_norm, bwd_activations -> x_v_local, grad_weights -> w_v, grad_weights -> w_v,
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit V projection weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_v", grad_weights -> w_v, model_dim, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_v file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Matmul to get dW_k..\n");
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, total_q, kv_dim,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> recomputed_activations -> recomputed_attn_norm, bwd_activations -> x_k_local, grad_weights -> w_k, grad_weights -> w_k,
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit K projection weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_k", grad_weights -> w_k, model_dim, kv_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_k file...\n");
			return -1;
		}
	}

	if (TO_PRINT){
		printf("Submitting Matmul to get dW_q..\n");
	}


    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, total_q, model_dim,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> recomputed_activations -> recomputed_attn_norm, bwd_activations -> x_q, grad_weights -> w_q, grad_weights -> w_q,
                    kernelWorkspaceBytes, kernelWorkspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit Q projection weight gradient computation...\n");
        return -1;
    }

	if (TO_SAVE_DATA && TO_SAVE_MODEL_GRADS && TO_SAVE_MODEL_GRAD_CHUNK_ID == chunk_id){
		ret = save_file(dataflow_handle, compute_stream_id, layer_id, -1, -1, true, "w_q", grad_weights -> w_q, model_dim, model_dim, bwd_dt);
		if (ret){
			fprintf(stderr, "Error: failed to save head w_q file...\n");
			return -1;
		}
	}

    // 6. Attention RMS Norm weight gradients
    // -- already done in bwd_x while we had the correct streaming grad

    return 0;
}


int dataflow_submit_transformer_embedding_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
											Transformer_Block_Transition * grad_stream,
											Transformer_Embedding_Table * grad_embedding_table) {

		int ret;

		Seq_Batch * seq_batch = grad_stream -> seq_batch;

		Embedding_Config * embedding_table_config = grad_embedding_table -> config;

		int vocab_size = embedding_table_config -> vocab_size;
		int embedding_dim = embedding_table_config -> embedding_size;

		Seq_Batch_Embedding_Config * batch_embedding_config = &(seq_batch -> embedding_config);

		DataflowDatatype embed_dt = embedding_table_config -> embed_dt;

		int num_unique_tokens = batch_embedding_config -> num_unique_tokens;
		uint32_t * sorted_token_ids = batch_embedding_config -> sorted_token_ids;
		uint32_t * sorted_token_mapping = batch_embedding_config -> sorted_token_mapping;
		uint32_t * unique_token_sorted_inds_start = batch_embedding_config -> unique_token_sorted_inds_start;

		if (TO_PRINT){
			printf("Submitting Embedding Bwd W...\n");
		}

		ret = dataflow_submit_default_embedding_table_bwd_w(dataflow_handle, compute_stream_id,
														embed_dt, num_unique_tokens, embedding_dim, 
														sorted_token_ids, sorted_token_mapping, unique_token_sorted_inds_start,
														grad_stream -> X, grad_embedding_table -> embedding_table);

		if (ret){
			fprintf(stderr, "Error: failed to submit embedding table bwd_w...\n");
			return -1;
		}

		if (TO_SAVE_DATA && TO_SAVE_EMBEDDING_BWD){
			ret = save_file(dataflow_handle, compute_stream_id, -2, -1, -1, true, "tok_embeddings", grad_embedding_table -> embedding_table, vocab_size, embedding_dim, embed_dt);
			if (ret){
				fprintf(stderr, "Error: failed to save embedding table bwd_w file...\n");
				return -1;
			}
		}

		return 0;
}
