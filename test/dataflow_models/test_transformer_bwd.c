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

	// 22 GB...	
	size_t dev_size_bytes = 21 * (1UL << 30);

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

	// for GeForce cards FP16 is double performance of BF16 because can use FP16 compute...
	//DataflowDatatype block_dt = DATAFLOW_FP16;
	//DataflowDatatype block_dt_bwd = DATAFLOW_FP16;

	DataflowDatatype block_dt = DATAFLOW_BF16;
	DataflowDatatype block_dt_bwd = DATAFLOW_BF16;

	size_t block_dt_size = dataflow_sizeof_element(block_dt);
	size_t block_dt_bwd_size = dataflow_sizeof_element(block_dt_bwd);

	// for matmul accumulations...
	// on Geforce using FP16 gets double perf,
	// on datacenter cards should use DATAFLOW_FP32
	//DataflowDatatype compute_dt = DATAFLOW_FP16;

	// however for BF dataftype, requires FP32 compute...
	DataflowDatatype compute_dt = DATAFLOW_FP32;
	DataflowDatatype compute_dt_bwd = DATAFLOW_FP32;


	DataflowNormalizationType norm_type = DATAFLOW_RMSNORM;

	DataflowPositionEmbeddingType pos_emb_type = DATAFLOW_ROPE;

	DataflowAttentionType attn_type = DATAFLOW_EXACT_ATTENTION;

	DataflowMLPType mlp_type = DATAFLOW_GATED_MLP;

	DataflowActivationType activ_type = DATAFLOW_SWIGLU;

	float eps = 1e-5;
	int theta = 500000;

	// llama3 70B config
	/*
	int n_layers = 80;
	int num_q_heads = 64;
	int num_kv_heads = 8;
	int head_dim = 128;
	int ffn_dim = 28672;
	*/

	// llama3 8b config
	/*
	int n_layers = 32;
	int num_q_heads = 32;
	int num_kv_heads = 8;
	int head_dim = 128;
	int ffn_dim = 14336;
	int model_dim = num_q_heads * head_dim;
	int kv_dim = num_kv_heads * head_dim;
	*/


	// llama3 1B config
	int n_layers = 16;
	int num_q_heads = 32;
	int num_kv_heads = 8;
	int head_dim = 64;
	int ffn_dim = 8192;
	int model_dim = num_q_heads * head_dim;
	int kv_dim = num_kv_heads * head_dim;

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


	FILE * fp = fopen("../data/1B/embed/tok_embeddings.weight", "rb");
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
		sys_blocks[i] = init_transformer_block(i, block_dt, compute_dt,
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

	char * layer_base_path = "../data/1B/layers";

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

	int num_dev_blocks = n_layers;

	Transformer_Block ** blocks = malloc(num_dev_blocks * sizeof(Transformer_Block *));
	if (!blocks){
		fprintf(stderr, "Error: failed to allocate blocks...\n");
		return -1;
	}
	
	for (int i = 0; i < num_dev_blocks; i++){
		blocks[i] = init_transformer_block(i, block_dt, compute_dt,
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
	sys_head -> bwd_dt = block_dt_bwd;
	sys_head -> compute_dt = compute_dt;
	sys_head -> eps = eps;
	sys_head -> embedding_config = embedding_config;

	sys_head -> buffer = cur_host_mem;
	sys_head -> w_head_norm = sys_head -> buffer;
	sys_head -> w_head = sys_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) block_dt_size;

	// model dim els for norm + model dim * vocab size els for head projection
	uint64_t combined_head_els = (uint64_t) model_dim * ((uint64_t) 1 + (uint64_t) vocab_size);
	uint64_t combined_head_size = combined_head_els * block_dt_size;

	cur_host_mem += combined_head_size;

	fp = fopen("../data/1B/head/combined_head.weight", "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open data/head/combined_head.weight...\n");
		return -1;
	}

	read_els = fread(sys_head -> buffer, block_dt_size, combined_head_els, fp);
	if (read_els != combined_head_els) {
		fprintf(stderr, "Error: failed to read combined_head.weight, read_els: %zu, expected: %lu\n", read_els, combined_head_els);
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
	head -> w_head = head -> w_head_norm + (uint64_t) model_dim * (uint64_t) block_dt_size;

	cur_dev_mem += combined_head_size;

	printf("Submitting inbound transfer for dev head...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, head -> buffer, sys_head -> buffer, combined_head_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit inbound transfer for dev head...\n");
		return -1;
	}




	// GRADIENTS!

	// JUST FOR NOW (while testing for correctness) keeping all block grads on device...
	int num_dev_block_grads = n_layers;

	Transformer_Block ** grad_blocks = malloc(num_dev_block_grads * sizeof(Transformer_Block *));
	if (!grad_blocks){
		fprintf(stderr, "Error: failed to allocate grad_blocks...\n");
		return -1;
	}

	for (int i = 0; i < num_dev_block_grads; i++){
		grad_blocks[i] = init_transformer_block(i, block_dt_bwd, compute_dt_bwd,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);
		if (!grad_blocks[i]){
			fprintf(stderr, "Error: failed to init transformer grad block #%d...\n", i);
			return -1;
		}

		// bind dev block
		ret = bind_transformer_block(cur_dev_mem, grad_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind transformer grad block #%d...\n", i);
			return -1;
		}

		ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, grad_blocks[i] -> buffer, 0, aligned_size);
		if (ret){
			fprintf(stderr, "Error: failed to set mem to 0 for grad block #%d...\n", i);
			return -1;
		}

		cur_dev_mem += aligned_size;
	}

	// Embedding Table Gradients
	Transformer_Embedding_Table * grad_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!grad_embedding_table){
		fprintf(stderr, "Error: failed to allocate grad_embedding_table...\n");
		return -1;
	}

	grad_embedding_table -> config = embedding_config;
	grad_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_dt_bwd_size;
	grad_embedding_table -> embedding_table = cur_dev_mem;

	ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, grad_embedding_table -> embedding_table, 0, grad_embedding_table -> embedding_table_size);
	if (ret){
		fprintf(stderr, "Error: failed to set mem for grad_embedding_table...\n");
		return -1;
	}

	cur_dev_mem += grad_embedding_table -> embedding_table_size;
	
	

	// Head Gradients
	Transformer_Head * grad_head = malloc(sizeof(Transformer_Head));
	if (!grad_head){
		fprintf(stderr, "Error: failed to allocate grad_head...\n");
		return -1;
	}

	grad_head -> fwd_dt = block_dt;
	grad_head -> bwd_dt = block_dt_bwd;
	grad_head -> compute_dt = compute_dt;
	grad_head -> eps = eps;
	grad_head -> embedding_config = embedding_config;
	grad_head -> buffer = cur_dev_mem;
	grad_head -> w_head_norm = grad_head -> buffer;
	grad_head -> w_head = grad_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) block_dt_bwd_size;

	

	ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, grad_head -> buffer, 0, combined_head_size);
	if (ret){
		fprintf(stderr, "Error: failed to set mem for grad_head...\n");
		return -1;
	}

	uint64_t combined_head_bwd_size = combined_head_els * block_dt_bwd_size;

	cur_dev_mem += combined_head_bwd_size;

	
	
	









	// CONTEXT AND GRAD CONTEXTS!

	int total_tokens = 2048;
	int num_seqs = 1;


	int max_tokens_per_chunk = 2048;
	int num_chunks = MY_CEIL(total_tokens, max_tokens_per_chunk);
	
	


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

	FILE * f_token_ids = fopen("test_transformer_data/token_ids_uint32.dat", "wb");
	if (!f_token_ids){
		fprintf(stderr, "Error: failed to open token_ids_uint32.dat...\n");
		return -1;
	}
	
	size_t wrote_els = fwrite(sys_token_ids, sizeof(uint32_t), total_tokens, f_token_ids);
	if (wrote_els != total_tokens){
		fprintf(stderr, "Error: failed to write token_ids_uint32.dat, sys_wrote_els: %zu, expected: %d\n", wrote_els, total_tokens);
		return -1;
	}
	fclose(f_token_ids);
	
	

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

	FILE * f_labels = fopen("test_transformer_data/labels_uint32.dat", "wb");
	if (!f_labels){
		fprintf(stderr, "Error: failed to open labels_uint32.dat...\n");
		return -1;
	}
	
	wrote_els = fwrite(sys_labels, sizeof(uint32_t), total_tokens, f_labels);
	if (wrote_els != total_tokens){
		fprintf(stderr, "Error: failed to write labels_uint32.dat, sys_wrote_els: %zu, expected: %d\n", wrote_els, total_tokens);
		return -1;
	}
	fclose(f_labels);

	// Now we can prepare seq batch...

	printf("Preparing seq batch...\n");
	

	

	uint64_t metadata_buffer_size = get_seq_batch_metadata_buffer_size(num_seqs, max_tokens_per_chunk);

	printf("Batch Config:\n\tTotal Tokens: %d\n\tNum Seqs: %d\n\n\tSeq Batch Metadata Buffer Size: %lu\n\n\n", max_tokens_per_chunk, num_seqs, metadata_buffer_size);

	Seq_Batch ** seq_batches = malloc(num_chunks * sizeof(Seq_Batch *));
	if (!seq_batches){
		fprintf(stderr, "Error: failed to allocate seq_batches...\n");
		return -1;
	}

	int max_total_local_expert_tokens = max_tokens_per_chunk;

	int remain_tokens = total_tokens;
	int chunk_tokens;
	int cur_token = 0;
	int * sys_seq_positions = malloc(total_tokens * sizeof(int));


	// these are offsets relative to local q
	// for all chunks except last should be [0, max_tokens_per_chunk
	// for last chunk should be [0, remain_tokens]
	int * cur_sys_q_seq_offsets = malloc(2 * sizeof(int));
	// this should be [max_tokens_per_chunk]
	// for all chunks except last
	// for last chunk should be [remain_tokens]
	int * cur_sys_q_seq_lens = malloc(sizeof(int));

	// these are offsets relative to the global k context
	// thus for chunk 0 should be [0, max_tokens_per_chunk]
	// for chunk 1 should be [0, 2 * max_tokens_per_chunk]
	// etc.
	// until last chunk should be [0, total_tokens]
	int * cur_sys_k_seq_offsets = malloc(2 * sizeof(int));
	// this is size of keys to look over
	// thus for chunk 0 should be [max_tokens_per_chunk]
	// for chunk 1 should be [2 * max_tokens_per_chunk]
	// etc.
	// until last chunk should be [total_tokens]
	int * cur_sys_k_seq_lens = malloc(sizeof(int));

	uint32_t * cur_sys_token_ids = sys_token_ids;
	uint32_t * cur_sys_labels = sys_labels;


	
	int seq_id = 0;

	int * cur_sys_seq_positions = malloc(max_tokens_per_chunk * sizeof(int));

	for (int i = 0; i < num_chunks; i++){
		chunk_tokens = MY_MIN(remain_tokens, max_tokens_per_chunk);
		remain_tokens -= chunk_tokens;

		seq_batches[i] = malloc(sizeof(Seq_Batch));
		if (!seq_batches[i]){
			fprintf(stderr, "Error: failed to allocate seq_batch...\n");
			return -1;
		}

		ret = init_seq_batch_offsets(seq_batches[i], chunk_tokens, num_seqs, &(sys_blocks[0] -> config), max_total_local_expert_tokens);
		if (ret){
			fprintf(stderr, "Error: failed to init seq_batch offsets...\n");
			return -1;
		}

		ret = bind_seq_batch_metadata_buffer(seq_batches[i], cur_dev_mem, metadata_buffer_size);
		if (ret){
			fprintf(stderr, "Error: failed to bind seq_batch metadata buffer...\n");
			return -1;
		}

		cur_dev_mem += metadata_buffer_size;

		if ((dev_alignment > 0) && ((uint64_t) cur_dev_mem % dev_alignment != 0)) {
			cur_dev_mem += dev_alignment - ((uint64_t) cur_dev_mem % dev_alignment);
		}

		

		for (int j = 0; j < chunk_tokens; j++){
			cur_sys_seq_positions[j] = cur_token;
			cur_token++;
		}

		cur_sys_q_seq_offsets[0] = 0;
		cur_sys_q_seq_offsets[1] = chunk_tokens;
		cur_sys_q_seq_lens[0] = chunk_tokens;

		cur_sys_k_seq_offsets[0] = 0;
		cur_sys_k_seq_offsets[1] = cur_token;
		cur_sys_k_seq_lens[0] = cur_token;

		ret = populate_seq_batch_metadata_buffer(&dataflow_handle, inbound_stream_id, 
										seq_batches[i],
										cur_host_mem, metadata_buffer_size,
										seq_id, i, chunk_tokens, num_seqs,
										cur_sys_token_ids, cur_sys_labels,
										cur_sys_seq_positions, 
										cur_sys_q_seq_offsets, cur_sys_q_seq_lens,
										cur_sys_k_seq_offsets, cur_sys_k_seq_lens);

		if (ret){
			fprintf(stderr, "Error: failed to populate seq_batch metadata buffer for chunk #%d...\n", i);
			return -1;
		}

		ret = (dataflow_handle.sync_stream)(&dataflow_handle, inbound_stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to sync inbound stream after populating seq_batch metadata buffer for chunk #%d...\n", i);
			return -1;
		}

		// ADVANCE TO NEXT CHUNK

		cur_sys_token_ids += chunk_tokens;
		cur_sys_labels += chunk_tokens;
		
		
	}

	free(cur_sys_seq_positions);
	free(cur_sys_q_seq_offsets);
	free(cur_sys_q_seq_lens);
	free(cur_sys_k_seq_offsets);
	free(cur_sys_k_seq_lens);

	assert(remain_tokens == 0);



	// CREATE DEVICE CONTEXT THAT ALL CHUNKS WILL REFERENCE...
	// FOR NOW KEEPING N_LAYERS OF CONTEXT BECAUSE NOT TRANSFERRING BACK TO HOST...
	// AND NEEDED FOR BACKPROP THROUGH TIME...

	// in reality only need 1 fwd context and 1 bwd context...
	int num_fwd_contexts = n_layers;


	Seq_Batch_Context * fwd_contexts = malloc(num_fwd_contexts * sizeof(Seq_Batch_Context));
	if (!fwd_contexts){
		fprintf(stderr, "Error: failed to allocate fwd_contexts...\n");
		return -1;
	}

	Seq_Batch_Context * bwd_context = malloc(sizeof(Seq_Batch_Context));
	if (!bwd_context){
		fprintf(stderr, "Error: failed to allocate bwd_context...\n");
		return -1;
	}


	uint64_t context_buffer_size = 2 * (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;


	for (int i = 0; i < num_fwd_contexts; i++){
		(&(fwd_contexts[i])) -> contextBuffer = cur_dev_mem;
		(&(fwd_contexts[i])) -> contextBufferBytes = context_buffer_size;
	
		(&(fwd_contexts[i])) -> cur_tokens_populated = 0;
		(&(fwd_contexts[i])) -> total_context_tokens = total_tokens;

		(&(fwd_contexts[i])) -> x_k = cur_dev_mem;
		(&(fwd_contexts[i])) -> x_v = (&(fwd_contexts[i])) -> x_k + (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;

		cur_dev_mem += context_buffer_size;
	} 


	uint64_t context_buffer_bwd_size = 2 * (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_bwd_size;
	
	(bwd_context) -> contextBuffer = cur_dev_mem;
	(bwd_context) -> contextBufferBytes = context_buffer_bwd_size;
	
	(bwd_context) -> cur_tokens_populated = 0;
	(bwd_context) -> total_context_tokens = total_tokens;

	(bwd_context) -> x_k = cur_dev_mem;
	(bwd_context) -> x_v = (bwd_context) -> x_k + (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_bwd_size;

	cur_dev_mem += context_buffer_bwd_size;

	
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

	// SKELETONS FOR OUTER STRUCTURES...

	Transformer_Model_Input * model_input = malloc(sizeof(Transformer_Model_Input));
	if (!model_input){
		fprintf(stderr, "Error: failed to allocate model_input...\n");
		return -1;
	}

	// EACh TIME CHUNK GOES THROUGH EMBEDDING NEED TO FILL IN :
	
	//model_input -> seq_batch = seq_batches[i];


	int num_block_transitions = 2 * num_chunks;

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
	
	uint64_t block_transition_size = (uint64_t) max_tokens_per_chunk * (uint64_t) model_dim * (uint64_t) block_dt_size;


	
	for (int i = 0; i < num_block_transitions; i++){;
		sys_block_transitions[i].X = cur_host_mem;
		sys_block_transitions[i].seq_batch = seq_batches[i / 2];
		cur_host_mem += block_transition_size;

		block_transitions[i].X = cur_dev_mem;
		block_transitions[i].seq_batch = seq_batches[i / 2];
		cur_dev_mem += block_transition_size;
	}

	// SAME KERNEL WORKSPACE ACROSS ALL COMPUTATIONS!

	uint64_t kernelWorkspaceBytes = 1UL << 28;
	void * kernelWorkspace = cur_dev_mem;
	cur_dev_mem += kernelWorkspaceBytes;

	// each block transition needs to fill in:


	// TODO:
	// need to save and bind activations...
	// IGNORING SENDING BACK SAVED ACTIVATIONS FOR NOW...


	// Need 1 activation workspace per device...
	Seq_Batch_Activation_Workspace * activation_workspace = malloc(sizeof(Seq_Batch_Activation_Workspace));
	if (!activation_workspace){
		fprintf(stderr, "Error: failed to allocate activation_workspace...\n");
		return -1;
	}

	uint64_t activation_workspace_size = get_seq_batch_activation_workspace_buffer_size(seq_batches[0], &(blocks[0] -> config));

	activation_workspace -> activationWorkspaceBuffer = cur_dev_mem;
	activation_workspace -> activationWorkspaceBytes = activation_workspace_size;

	activation_workspace -> x_temp = cur_dev_mem;
	activation_workspace -> x_temp_mlp = activation_workspace -> x_temp + ((uint64_t) max_tokens_per_chunk * (uint64_t) model_dim * (uint64_t) block_dt_size);
	

	activation_workspace -> kernelWorkspace = kernelWorkspace;
	activation_workspace -> kernelWorkspaceBytes = kernelWorkspaceBytes;

	cur_dev_mem += activation_workspace_size;



	// Saved Actviations will live on device and might be transferred back to host and retrieved prior to bwd pass...


	// JUST FOR NOW (while testing for correctness): NOT DEALING WITH DATA TRANSFERS AND SAVING ALL ACTIVATIONS ON DEVICE...

	int num_saved_activation_buffers = n_layers * num_chunks;

	Seq_Batch_Saved_Activations * saved_activations = malloc(num_saved_activation_buffers * sizeof(Seq_Batch_Saved_Activations));
	if (!saved_activations){
		fprintf(stderr, "Error: failed to allocate saved_activations...\n");
		return -1;
	}

	uint64_t saved_activations_buffer_size;

	
	for (int i = 0; i < num_saved_activation_buffers; i++){

		saved_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[(i % num_chunks)]);
		ret = bind_seq_batch_saved_activations_buffer(seq_batches[(i % num_chunks)], &(saved_activations[i]), cur_dev_mem, saved_activations_buffer_size, i);
		if (ret){
			fprintf(stderr, "Error: failed to bind seq_batch saved_activations buffer...\n");
			return -1;
		}

		cur_dev_mem += saved_activations_buffer_size;

		// only the grad activations will have a recomputed activations buffer...
		saved_activations[i].recomputed_activations = NULL;
	}
	
	
	
	
	// Now we will have a certain amount of activations on device that will be paired with a saved activations buffer in order
	// to actually have output space....


	// JUST FOR NOW (while testing for correctness): NOT DEALING WITH DATA TRANSFERS AND SAVING ALL ACTIVATIONS ON DEVICE...

	int num_dev_activations = n_layers * num_chunks;

	Transformer_Block_Activations ** activations = malloc(num_dev_activations * sizeof(Transformer_Block_Activations *));
	if (!activations){
		fprintf(stderr, "Error: failed to allocate top level activations container...\n");
		return -1;
	}

	for (int i = 0; i < num_dev_activations; i++){
		activations[i] = malloc(sizeof(Transformer_Block_Activations));
		if (!activations[i]){
			fprintf(stderr, "Error: failed to allocate activations container for chunk #%d...\n", i);
			return -1;
		}

        // Just for now we know that that the saved activations buffer will be the same size as the activations buffer...
		// (all chunks + layers)
		
		// In reality, the working activations will be tied to a saved activations buffer popped
		// from a queue holding "free" saved activations buffers...
		(activations[i]) -> working_activations = &(saved_activations[i]);

		// can all resuse the same workspace...
		(activations[i]) -> activation_workspace = activation_workspace;
	}

	// ONLY NEED ONE GRAD ACTIVATIONS STRUCT WHICH WILL BE RE-USED FOR EACH CHUNK AND EACH LAYER...
	// (the workspace during bwd x, and then passed in for bwd w)

	Seq_Batch_Saved_Activations * grad_saved_activations = malloc(sizeof(Seq_Batch_Saved_Activations));
	if (!grad_saved_activations){
		fprintf(stderr, "Error: failed to allocate grad_saved_activations...\n");
		return -1;
	}

	// seq batch 0 is the largest, so won't need more space than this...
	uint64_t grad_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[0]);

	// using seq batch 0 offsets is safe because all seq batches are either the same or smaller (in terms of total tokens, thus saved activations offsets...)
	ret = bind_seq_batch_saved_activations_buffer(seq_batches[0], grad_saved_activations, cur_dev_mem, grad_activations_buffer_size, 0);
	if (ret){
		fprintf(stderr, "Error: failed to bind seq_batch grad_saved_activations buffer...\n");
		return -1;
	}

	cur_dev_mem += grad_activations_buffer_size;



	// RECOMPUTED ACTIVATIONS BUFFER SPACE FOR RE-CALCULATING NORM VALUES...
	// This attaches to grad activations...

	Seq_Batch_Recomputed_Activations * recomputed_activations = malloc(sizeof(Seq_Batch_Recomputed_Activations));
	if (!recomputed_activations){
		fprintf(stderr, "Error: failed to allocate recomputed_activations cotnainer...\n");
		return -1;
	}

	uint64_t recomputed_activations_buffer_size = get_seq_batch_recomputed_activations_buffer_size(seq_batches[0]);



	void * recomputed_activations_buffer = cur_dev_mem;

	ret = bind_seq_batch_recomputed_activations_buffer(&(seq_batches[0] -> recomputed_activations_offsets), recomputed_activations, recomputed_activations_buffer, recomputed_activations_buffer_size);
	if (ret){
		fprintf(stderr, "Error: failed to bind seq_batch recomputed_activations buffer...\n");
		return -1;
	}

	cur_dev_mem += recomputed_activations_buffer_size;

	// set the recomputed activations buffer in grad activations...
	grad_saved_activations -> recomputed_activations = recomputed_activations;


	Transformer_Block_Activations * grad_activations = malloc(sizeof(Transformer_Block_Activations));
	if (!grad_activations){
		fprintf(stderr, "Error: failed to allocate grad_activations...\n");
		return -1;
	}

	grad_activations -> working_activations = grad_saved_activations;
	(grad_activations) -> activation_workspace = activation_workspace;



	// PREPARING SPECIAL HEAD ACTIVATIONS STRUCT...

	Transformer_Head_Activations * head_activations = malloc(sizeof(Transformer_Head_Activations));
	if (!head_activations){
		fprintf(stderr, "Error: failed to allocate head_activations...\n");
		return -1;
	}
	
	head_activations -> buffer = cur_dev_mem;
	head_activations -> head_norm_out = head_activations -> buffer;
	uint64_t head_norm_out_size = (uint64_t) max_tokens_per_chunk * (uint64_t) model_dim * (uint64_t) block_dt_size;
	cur_dev_mem += head_norm_out_size;
	head_activations -> head_norm_weighted_sums = cur_dev_mem;
	uint64_t head_norm_weighted_sums_size = (uint64_t) max_tokens_per_chunk * (uint64_t) sizeof(float);
	cur_dev_mem += head_norm_weighted_sums_size;
	head_activations -> head_norm_rms_vals = cur_dev_mem;
	uint64_t head_norm_rms_vals_size = (uint64_t) max_tokens_per_chunk * (uint64_t) sizeof(float);
	cur_dev_mem += head_norm_rms_vals_size;
	head_activations -> head_out = cur_dev_mem;
	uint64_t head_out_size = (uint64_t) max_tokens_per_chunk * (uint64_t) vocab_size * (uint64_t) block_dt_size;
	cur_dev_mem += head_out_size;
	head_activations -> kernelWorkspace = kernelWorkspace;
	head_activations -> kernelWorkspaceBytes = kernelWorkspaceBytes;


	// EACH HEAD ACTIVATIONS STRUCT NEEDS TO BE FILLED IN WITH:
	// head_activations -> num_tokens = total_tokens;
	

	// PREPARING SPECIAL MODEL OUTPUT STRUCT...


	uint64_t logits_size = (uint64_t) max_tokens_per_chunk * (uint64_t) vocab_size * block_dt_bwd_size;
	void * sys_logits = cur_host_mem;
	cur_host_mem += logits_size;

	Transformer_Model_Output * model_output = malloc(sizeof(Transformer_Model_Output));
	if (!model_output){
		fprintf(stderr, "Error: failed to allocate model_output...\n");
		return -1;
	}

	model_output -> logits = cur_dev_mem;
	cur_dev_mem += logits_size;


	// EACH MODEL OUTPUT STRUCT NEEDS TO BE FILLED IN WITH:
	// model_output -> seq_batch = seq_batches[i];


	Transformer_Block_Activations * cur_activations;
	Seq_Batch_Saved_Activations * cur_fwd_activations;

	// 1.) DOING EMBEDDDING....

	// for layer 0 we include the embedding table...

	for (int i = 0; i < num_chunks; i++){
		printf("\n\nSubmitting embedding for chunk #%d...\n\n", i);
		
		model_input -> seq_batch = seq_batches[i];

		ret = dataflow_submit_transformer_embedding(&dataflow_handle, compute_stream_id,
											model_input,
											embedding_table,
											&(block_transitions[2 * i]));
		if (ret){
			fprintf(stderr, "Error: failed to submit transformer embedding...\n");
			return -1;
		}

		printf("\n\nSubmitting layer #0 for chunk #%d...\n\n", i);

		cur_activations = activations[i];

		// set the context
		seq_batches[i] -> context = &(fwd_contexts[0]);

		ret = dataflow_submit_transformer_block(&dataflow_handle, compute_stream_id, 
								&(block_transitions[2 * i]), 
								blocks[0], 
								cur_activations, 
								&(block_transitions[2 * i + 1]));

		if (ret){
			fprintf(stderr, "Error: failed to submit transformer block...\n");
			return -1;
		}


	}

	// 2.) DOING CORE BLOCKS...
	for (int k = 1; k < n_layers; k++){
		for (int i = 0; i < num_chunks; i++){
			
			printf("\n\nSubmitting transformer for chunk #%d, block #%d...!\n\n", i, k);

			cur_activations = activations[k * num_chunks + i];

			// set the context
			seq_batches[i] -> context = &(fwd_contexts[k]);
			

			ret = dataflow_submit_transformer_block(&dataflow_handle, compute_stream_id, 
									&(block_transitions[2 * i + (k % 2)]), 
									blocks[k], 
									cur_activations, 
									&(block_transitions[2 * i + ((k + 1) % 2)])) ;

			if (ret){
				fprintf(stderr, "Error: failed to submit transformer block for chunk #%d, block #%d...\n", i, k);
				return -1;
			}
		}
	}

	// 3.) NOW DOING HEAD, FOR NOW IN REVERSE ORDER...

	Transformer_Block_Transition * final_block_output_transition;

	Transformer_Block_Transition * grad_stream_from_head;

	for (int i = num_chunks - 1; i >= 0; i--){
		printf("\n\nSubmitting head for chunk #%d...\n\n", i);

		final_block_output_transition = &(block_transitions[2 * i + (n_layers % 2)]);

		grad_stream_from_head = &(block_transitions[2 * i + ((n_layers - 1) % 2)]);

		// ensure grad stream is zeroed out...
		ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, grad_stream_from_head -> X, 0, block_transition_size);
		if (ret){
			fprintf(stderr, "Error: failed to zero out grad stream for chunk #%d before head...\n", i);
			return -1;
		}

		model_output -> seq_batch = seq_batches[i];
		head_activations -> num_tokens = seq_batches[i] -> total_tokens;
	
		ret = dataflow_submit_transformer_head(&dataflow_handle, compute_stream_id,
												final_block_output_transition, head,
												head_activations, 
												model_output,
												// during interference these would be NULL
												grad_head,
												grad_stream_from_head);


		if (ret){
			fprintf(stderr, "Error: failed to submit transformer head...\n");
			return -1;
		}

		// Ensure that this seq batch's context will refere the bwd context,
		// in order to correctly backprop through the block...
		seq_batches[i] -> context = bwd_context;
	}

	for (int k = n_layers - 1; k >= 0; k--){

		// need to ensure that grad context is zeroed out before starting each layer...
		ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, bwd_context -> contextBuffer, 0, bwd_context -> contextBufferBytes);
		if (ret){
			fprintf(stderr, "Error: failed to zero out grad context for layer #%d...\n", k);
			return -1;
		}

		for (int i = num_chunks - 1; i >= 0; i--){


			cur_fwd_activations = &(saved_activations[k * num_chunks + i]);

			
			printf("\n\nSubmitting bwd_x for chunk #%d, block #%d...\n\n", i, k);

			ret = dataflow_submit_transformer_block_bwd_x(&dataflow_handle, compute_stream_id,
								blocks[k], 
								&(block_transitions[2 * i + (k % 2)]), 
								cur_fwd_activations, &(fwd_contexts[k]),
								grad_activations,
								grad_blocks[k],
								&(block_transitions[2 * i + ((k + 1) % 2)]));

			if (ret){
				fprintf(stderr, "Error: failed to submit transformer block bwd_x for chunk #%d, block #%d...\n", i, k);
				return -1;
			}

			printf("\n\nSubmitting bwd_w for chunk #%d, block #%d...\n\n", i, k);

			// utilizing the newly populated grad_activations struct
			// to update the grad_weights...

			ret = dataflow_submit_transformer_block_bwd_w(&dataflow_handle, compute_stream_id,
                                &(block_transitions[2 * i + (k % 2)]),
                                cur_fwd_activations, 
                                grad_activations, 
                                grad_blocks[k]);

			if (ret){
				fprintf(stderr, "Error: failed to submit transformer block bwd_w for chunk #%d, block #%d...\n", i, k);
				return -1;
			}
		}
	}


	for (int i = num_chunks - 1; i >= 0; i--){
		printf("\n\nSubmitting embedding bwd_w for chunk #%d...\n\n", i);

		// layer 0'ths output grad stream will be at index 1 (for given chunk)
		ret = dataflow_submit_transformer_embedding_bwd_w(&dataflow_handle, compute_stream_id,
											&(block_transitions[2 * i + 1]),
											grad_embedding_table);

		if (ret){
			fprintf(stderr, "Error: failed to submit transformer embedding bwd_w for chunk #%d...\n", i);
			return -1;
		}
	}
	

	printf("Finished enqueueing all dataflow operations! Waiting to sync...\n\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, compute_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync compute stream...\n");
		return -1;
	}

	printf("All operations complete! Exiting...\n\n");

	return 0;
}