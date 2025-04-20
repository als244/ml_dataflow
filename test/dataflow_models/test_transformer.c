#include "dataflow_transformer.h"
#include "dataflow_seq_batch.h"
#include "cuda_dataflow_handle.h"
#include "register_ops.h"

int main(int argc, char * argv[]){

	int ret;


	// Initialize dataflow handle...

	Dataflow_Handle dataflow_handle;
	
	ComputeType compute_type = COMPUTE_CUDA;
	int device_id = 0;

	// In case we want to create multiple contexts per device, 
	// higher level can create multiple instances of dataflow handles...
	int ctx_id = 0;
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;

	int num_streams = 4;
	int opt_stream_prios[4] = {0, 0, 0, 0};
	char * opt_stream_names[4] = {"Inbound", "Compute", "Outbound", "Peer"};


	int inbound_stream_id = 0;
	int compute_stream_id = 1;
	int outbound_stream_id = 2;
	int peer_stream_id = 3;

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
	printf("Registered %d default ops...\n\n", added_funcs);



	// 64 GB...
	void * host_mem;

	int host_alignment = 4096;
	size_t host_size_bytes = 1UL << 36;

	printf("Allocating host memory of size: %lu...\n", host_size_bytes);

	ret = posix_memalign(&host_mem, host_alignment, host_size_bytes);
	if (ret){
		fprintf(stderr, "Error: posix memalign failed...\n");
		return -1;
	}
	memset(host_mem, 0, host_size_bytes);


	printf("Registering host memory...\n\n");

	ret = dataflow_handle.enable_access_to_host_mem(&dataflow_handle, host_mem, host_size_bytes, 0);
	if (ret){
		fprintf(stderr, "Registration of host memory failed...\n");
		return -1;
	}

	// 16 GB...	
	size_t dev_size_bytes = 1UL << 34;

	int dev_alignment = 256;

	printf("Allocating device memory of size: %lu...\n\n", dev_size_bytes);


	void * dev_mem = dataflow_handle.alloc_mem(&dataflow_handle, dev_size_bytes);
	if (!dev_mem){
		fprintf(stderr, "Error: device memory allocation failed...\n");
		return -1;
	}

	void * cur_host_mem = host_mem;
	void * cur_dev_mem = dev_mem;


	// Preparing model...

	DataflowDatatype block_dt = DATAFLOW_BF16;

	size_t block_dt_size = dataflow_sizeof_element(block_dt);

	// for matmul accumulations...
	// on Geforce using FP16 gets double perf,
	// on datacenter cards should use DATAFLOW_FP32

	// however for BF dataftype, requires FP32 compute...
	DataflowDatatype compute_dt = DATAFLOW_FP32;


	DataflowNormalizationType norm_type = DATAFLOW_RMSNORM;

	DataflowPositionEmbeddingType pos_emb_type = DATAFLOW_ROPE;

	DataflowAttentionType attn_type = DATAFLOW_EXACT_ATTENTION;

	DataflowMLPType mlp_type = DATAFLOW_GATED_MLP;

	DataflowActivationType activ_type = DATAFLOW_SWIGLU;

	float eps = 1e-5;
	int theta = 500000;

	// llama3 70B config
	/*
	int num_q_heads = 64;
	int num_kv_heads = 8;
	int head_dim = 128;
	int ffn_dim = 28672;
	*/

	// llama3 8b config
	int num_q_heads = 32;
	int num_kv_heads = 8;
	int head_dim = 128;
	int ffn_dim = 14336;
	int model_dim = num_q_heads * head_dim;
	int kv_dim = num_kv_heads * head_dim;

	int n_layers = 32;

	int vocab_size = 128256;

	MoE_Config * moe_config = NULL;


	// setting to host page size.
	// really needs to be 256 in order to use tensor cores
	// depending on filesystem in order to use O_RDONLY | O_DIRECT, alignment may be different...
	
	
	// for now using 0 alignment to directly read from combined file...
	int pointer_alignment = 256;


	printf("Loading embedding table...\n");

	Embedding_Config * embedding_config = malloc(sizeof(Embedding_Config));
	if (!embedding_config){
		fprintf(stderr, "Error: failed to allocate embedding_config...\n");
		return -1;
	}

	embedding_config -> vocab_size = vocab_size;
	embedding_config -> embedding_size = model_dim;
	embedding_config -> embed_dt = block_dt;

	Transformer_Embedding_Table * sys_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!sys_embedding_table){
		fprintf(stderr, "Error: failed to allocate embedding_table...\n");
		return -1;
	}

	sys_embedding_table -> config = embedding_config;
	
	uint64_t embedding_table_els = (uint64_t) vocab_size * (uint64_t) model_dim;
	sys_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_dt_size;
	sys_embedding_table -> embedding_table = cur_host_mem;

	cur_host_mem += sys_embedding_table -> embedding_table_size;


	FILE * fp = fopen("../data/8B/embed/tok_embeddings.weight", "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open data/embed/tok_embedding.weight...\n");
		return -1;
	}

	size_t read_els = fread(sys_embedding_table -> embedding_table, block_dt_size, embedding_table_els, fp);
	if (read_els != embedding_table_els){
		fprintf(stderr, "Error: failed to read tok_embedding.weight, read_els: %zu, expected: %lu\n", read_els, embedding_table_els);
		return -1;
	}

	fclose(fp);

	
	Transformer_Embedding_Table * embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!embedding_table){
		fprintf(stderr, "Error: failed to allocate embedding_table...\n");
		return -1;
	}

	embedding_table -> config = embedding_config;
	embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_dt_size;
	embedding_table -> embedding_table = cur_dev_mem;

	cur_dev_mem += embedding_table -> embedding_table_size;

	printf("Copying embedding table to device...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, embedding_table -> embedding_table, sys_embedding_table -> embedding_table, embedding_table -> embedding_table_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit inbound transfer for embedding table...\n");
		return -1;
	}

	printf("Preparing all sys transformer blocks...\n");

	Transformer_Block ** sys_blocks = malloc(n_layers * sizeof(Transformer_Block *));
	if (!sys_blocks){
		fprintf(stderr, "Error: failed to allocate sys_blocks...\n");
		return -1;
	}
	
	for (int i = 0; i < n_layers; i++){
		sys_blocks[i] = init_transformer_block(block_dt, compute_dt,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);

		if (!sys_blocks[i]){
			fprintf(stderr, "Error: failed to init transformer block...\n");
			return -1;
		}
	}

	uint64_t raw_size = get_transformer_block_raw_size(sys_blocks[0]);
	uint64_t aligned_size = get_transformer_block_aligned_size(sys_blocks[0]);

	printf("\nTransformer Block Sizes (bytes):\n\tRaw: %lu\n\tAligned (%d): %lu\n\n", raw_size, pointer_alignment, aligned_size);

	printf("Binding all sys transformer blocks...\n");

	char * layer_base_path = "../data/8B/layers";

	char layer_path[PATH_MAX];
	for (int i = 0; i < n_layers; i++){
		printf("Binding sys transformer block #%d...\n", i);
		ret = bind_transformer_block(cur_host_mem, sys_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind transformer block #%d...\n", i);
		return -1;
		}

		sprintf(layer_path, "%s/%d/combined_layer.weight", layer_base_path, i);

		printf("Loading transformer block from: %s...\n", layer_path);
		ret = load_transformer_block(layer_path, sys_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to load transformer block #%d from: %s...\n", i, layer_path);
			return -1;
		}

		cur_host_mem += aligned_size;
	}

	int num_dev_blocks = 2;

	Transformer_Block ** blocks = malloc(num_dev_blocks * sizeof(Transformer_Block *));
	if (!blocks){
		fprintf(stderr, "Error: failed to allocate blocks...\n");
		return -1;
	}
	
	for (int i = 0; i < num_dev_blocks; i++){
		blocks[i] = init_transformer_block(block_dt, compute_dt,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);
		if (!blocks[i]){
			fprintf(stderr, "Error: failed to init transformer block #%d...\n", i);
			return -1;
		}

		// bind dev block
		ret = bind_transformer_block(cur_dev_mem, blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind transformer block #%d...\n", i);
			return -1;
		}

		cur_dev_mem += aligned_size;

		// copy sys block to dev block

		printf("Submitting inbound transfer for dev transformer block #%d...\n", i);

		ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[i] -> buffer, sys_blocks[i] -> buffer, aligned_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit inbound transfer for transformer block #%d...\n", i);
			return -1;
		}
	}

	// Loading head...

	printf("Loading head...\n");

	Transformer_Head * sys_head = malloc(sizeof(Transformer_Head));
	if (!sys_head){
		fprintf(stderr, "Error: failed to allocate sys_head...\n");
		return -1;
	}
	
	sys_head -> fwd_dt = block_dt;
	sys_head -> bwd_dt = block_dt;
	sys_head -> compute_dt = compute_dt;
	sys_head -> eps = eps;
	sys_head -> embedding_config = embedding_config;

	sys_head -> buffer = cur_host_mem;
	sys_head -> w_head_norm = sys_head -> buffer;
	sys_head -> w_head = sys_head -> w_head_norm + model_dim * block_dt_size;

	
	uint64_t combined_head_size = (uint64_t) model_dim * block_dt_size + (uint64_t) model_dim * (uint64_t) vocab_size * block_dt_size;

	cur_host_mem += combined_head_size;

	fp = fopen("../data/8B/head/combined_head.weight", "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open data/head/combined_head.weight...\n");
		return -1;
	}

	read_els = fread(sys_head -> buffer, block_dt_size, (uint64_t) model_dim * (uint64_t) vocab_size, fp);
	if (read_els != (uint64_t) model_dim * (uint64_t) vocab_size){
		fprintf(stderr, "Error: failed to read combined_head.weight, read_els: %zu, expected: %lu\n", read_els, (uint64_t) model_dim * (uint64_t) vocab_size);
		return -1;
	}

	fclose(fp);


	Transformer_Head * head = malloc(sizeof(Transformer_Head));
	if (!head){
		fprintf(stderr, "Error: failed to allocate head...\n");
		return -1;
	}

	head -> fwd_dt = block_dt;
	head -> bwd_dt = block_dt;
	head -> compute_dt = compute_dt;
	head -> eps = eps;
	head -> embedding_config = embedding_config;
	head -> buffer = cur_dev_mem;
	head -> w_head_norm = head -> buffer;
	head -> w_head = head -> w_head_norm + model_dim * block_dt_size;

	cur_dev_mem += combined_head_size;

	printf("Submitting inbound transfer for dev head...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, head -> buffer, sys_head -> buffer, combined_head_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit inbound transfer for dev head...\n");
		return -1;
	}


	/*
	printf("Waiting for inbound transfers to complete then preparing seq batch...\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync inbound stream...\n");
		return -1;
	}
	*/






	// Now we can prepare seq batch...

	printf("Preparing seq batch...\n");
	


	int total_tokens = 2048;
	int num_seqs = 1;

	uint64_t metadata_buffer_size = get_seq_batch_metadata_buffer_size(num_seqs, total_tokens);

	printf("Batch Config:\n\tTotal Tokens: %d\n\tNum Seqs: %d\n\n\tSeq Batch Metadata Buffer Size: %lu\n\n\n", total_tokens, num_seqs, metadata_buffer_size);

	Seq_Batch * seq_batch = malloc(sizeof(Seq_Batch));
	if (!seq_batch){
		fprintf(stderr, "Error: failed to allocate seq_batch...\n");
		return -1;
	}

	int max_total_local_expert_tokens = total_tokens;
	ret = init_seq_batch_offsets(seq_batch, total_tokens, num_seqs, &(sys_blocks[0] -> config), max_total_local_expert_tokens);
	if (ret){
		fprintf(stderr, "Error: failed to init seq_batch offsets...\n");
		return -1;
	}


	// 
	ret = bind_seq_batch_metadata_buffer(seq_batch, cur_dev_mem, metadata_buffer_size);
	if (ret){
		fprintf(stderr, "Error: failed to bind seq_batch metadata buffer...\n");
		return -1;
	}

	cur_dev_mem += metadata_buffer_size;

	if ((dev_alignment > 0) && ((uint64_t) cur_dev_mem % dev_alignment != 0)) {
		cur_dev_mem += dev_alignment - ((uint64_t) cur_dev_mem % dev_alignment);
	}


	// Populating the metadata buffer (on dev mem, via registered host mem)...

	uint32_t * sys_token_ids = malloc(total_tokens * sizeof(uint32_t));

	fp = fopen("../data/token_ids_uint32.dat", "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open data/token_ids_uint32.dat...\n");
		return -1;
	}

	read_els = fread(sys_token_ids, sizeof(uint32_t), total_tokens, fp);
	if (read_els != total_tokens){
		fprintf(stderr, "Error: failed to read token_id_uint32.dat, read_els: %zu, expected: %d\n", read_els, total_tokens);
		return -1;
	}
	fclose(fp);

	uint32_t * sys_labels = malloc(total_tokens * sizeof(uint32_t));

	fp = fopen("../data/labels_uint32.dat", "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open data/labels_uint32.dat...\n");
		return -1;
	}

	read_els = fread(sys_labels, sizeof(uint32_t), total_tokens, fp);
	if (read_els != total_tokens){
		fprintf(stderr, "Error: failed to read labels_uint32.dat, read_els: %zu, expected: %d\n", read_els, total_tokens);
		return -1;
	}
	fclose(fp);

	int * sys_seq_positions = malloc(total_tokens * sizeof(int));

	for (int i = 0; i < total_tokens; i++){
		sys_seq_positions[i] = i;
	}

	int sys_q_seq_offsets[] = {0, 2048};
	int sys_q_seq_lens[] = {2048};


	int sys_k_seq_offsets[] = {0, 2048};
	int sys_k_seq_lens[] = {2048};
	
	printf("Populating Seq Batch's device metadata buffer with contents passed from host...\n");

	ret = populate_seq_batch_metadata_buffer(&dataflow_handle, inbound_stream_id, 
                                        seq_batch,
                                        cur_host_mem, metadata_buffer_size,
                                        total_tokens, num_seqs,
                                        sys_token_ids, sys_labels,
                                        sys_seq_positions, 
                                        sys_q_seq_offsets, sys_q_seq_lens,
                                        sys_k_seq_offsets, sys_k_seq_lens);

	if (ret){
		fprintf(stderr, "Error: failed to populate seq_batch metadata buffer...\n");
		return -1;
	}


	// IF we are following up with embedding, ensure to wait for inbound stream....


	printf("Waiting for data transfer of metadata buffer to complete before submitting onto compute stream...\n");

	void * inbound_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, inbound_stream_id);
	if (!inbound_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = dataflow_handle.submit_dependency(&dataflow_handle, compute_stream_id, inbound_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	// For now set seq_batch context to NULL to indicate no other context...

	seq_batch -> context = NULL;
	/*

	// For now just calling sync_stream to inspect outputs (with cuda-gdb...)
	printf("Waiting for data transfer of metadata buffer to complete...\n\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync inbound stream...\n");
		return -1;
	}
	*/

	printf("\n\nReady for embedding...\n");


	Transformer_Model_Input * model_input = malloc(sizeof(Transformer_Model_Input));
	if (!model_input){
		fprintf(stderr, "Error: failed to allocate model_input...\n");
		return -1;
	}

	model_input -> seq_batch = seq_batch;
	
	uint64_t block_transition_size = (uint64_t) total_tokens * (uint64_t) model_dim * (uint64_t) block_dt_size;

	int num_block_transitions = 2;

	Transformer_Block_Transition * sys_block_transitions = malloc(num_block_transitions * sizeof(Transformer_Block_Transition));
	if (!sys_block_transitions){
		fprintf(stderr, "Error: failed to allocate sys_block_transitions...\n");
		return -1;
	}
	
	Transformer_Block_Transition * block_transitions = malloc(num_block_transitions * sizeof(Transformer_Block_Transition));
	if (!block_transitions){
		fprintf(stderr, "Error: failed to allocate block_transitions...\n");
		return -1;
	}

	for (int i = 0; i < num_block_transitions; i++){
		sys_block_transitions[i].seq_batch = seq_batch;
		sys_block_transitions[i].X = cur_host_mem;
		cur_host_mem += block_transition_size;

		block_transitions[i].seq_batch = seq_batch;
		block_transitions[i].X = cur_dev_mem;
		cur_dev_mem += block_transition_size;
	}

	ret = dataflow_submit_transformer_embedding(&dataflow_handle, compute_stream_id,
											model_input,
											embedding_table,
											&(block_transitions[0]));
	if (ret){
		fprintf(stderr, "Error: failed to submit transformer embedding...\n");
		return -1;
	}

	// now save the block transition, first ensure sync with compute stream...

	ret = dataflow_handle.sync_stream(&dataflow_handle, compute_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync compute stream...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_block_transitions[0].X, block_transitions[0].X, block_transition_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit outbound transfer for block transition...\n");
		return -1;
	}

	// save the block transitions...

	ret = save_host_matrix("test_transformer_data/example_embed_output.dat", sys_block_transitions[0].X, total_tokens, model_dim, block_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save example embed output...\n");
		return -1;
	}

	printf("Successfully saved example embed output...!!!\n\n");



	// TODO:

	// need to save and bind activations...

	int num_saved_activation_buffers = 2;
	Seq_Batch_Saved_Activations * saved_activations = malloc(num_saved_activation_buffers * sizeof(Seq_Batch_Saved_Activations));
	if (!saved_activations){
		fprintf(stderr, "Error: failed to allocate saved_activations...\n");
		return -1;
	}

	uint64_t saved_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batch);
	
	for (int i = 0; i < num_saved_activation_buffers; i++){
		ret = bind_seq_batch_saved_activations_buffer(seq_batch, &(saved_activations[i]), cur_dev_mem, saved_activations_buffer_size, i);
		if (ret){
			fprintf(stderr, "Error: failed to bind seq_batch saved_activations buffer...\n");
			return -1;
		}

		cur_dev_mem += saved_activations_buffer_size;
	}
	

	uint64_t activation_workspace_size = get_seq_batch_activation_workspace_buffer_size(seq_batch, &(blocks[0] -> config));
	Seq_Batch_Activation_Workspace * activation_workspace = malloc(sizeof(Seq_Batch_Activation_Workspace));
	if (!activation_workspace){
		fprintf(stderr, "Error: failed to allocate activation_workspace...\n");
		return -1;
	}

	activation_workspace -> activationWorkspaceBuffer = cur_dev_mem;
	activation_workspace -> activationWorkspaceBytes = activation_workspace_size;

	activation_workspace -> x_temp = cur_dev_mem;
	activation_workspace -> x_temp_mlp = activation_workspace -> x_temp + ((uint64_t) total_tokens * (uint64_t) model_dim * (uint64_t) block_dt_size);
	
	cur_dev_mem += activation_workspace_size;


	uint64_t kernelWorkspaceBytes = 1UL << 24;
	void * kernelWorkspace = cur_dev_mem;
	cur_dev_mem += kernelWorkspaceBytes;

	Transformer_Block_Activations * activations = malloc(sizeof(Transformer_Block_Activations));
	if (!activations){
		fprintf(stderr, "Error: failed to allocate activations...\n");
		return -1;
	}


	activations -> working_activations = &(saved_activations[0]);
	activation_workspace -> kernelWorkspace = kernelWorkspace;
	activation_workspace -> kernelWorkspaceBytes = kernelWorkspaceBytes;
	activations -> activation_workspace = activation_workspace;
	
	printf("Submitting transformer block...!\n\n");
	

	ret = dataflow_submit_transformer_block(&dataflow_handle, compute_stream_id, 
								&(block_transitions[0]), 
								blocks[0], 
								activations, 
								&(block_transitions[1]));

	if (ret){
		fprintf(stderr, "Error: failed to submit transformer block...\n");
		return -1;
	}

	void * compute_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get compute stream state...\n");
		return -1;
	}

	ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_block_transitions[1].X, block_transitions[1].X, block_transition_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit outbound transfer for block transition...\n");
		return -1;
	}


	printf("Ensuring block transition arrives before saving layer output...\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, outbound_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync outbound stream...\n");
		return -1;
	}

	// save the block transitions...

	printf("Saving layer output...\n");

	ret = save_host_matrix("test_transformer_data/example_layer_output.dat", sys_block_transitions[1].X, total_tokens, model_dim, block_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save example layer output...\n");
		return -1;
	}

	printf("Successfully saved example layer output...!!!\n\n");

	return 0;
}