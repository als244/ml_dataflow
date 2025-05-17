#include "dataflow_transformer.h"
#include "dataflow_seq_batch.h"
#include "cuda_dataflow_handle.h"
#include "register_ops.h"
#include "host_ops.h"

// these shoudl be auto-cofigured, testing manually for now...
// could also take in as command line argument...
#define NUM_DEV_BLOCKS 4
#define NUM_DEV_ACTIVATION_SLOTS 16

#define NUM_DEV_BLOCK_GRADS 2
#define NUM_SYS_GRAD_RESULTS 2

#define NUM_ADD_THREADS 16	
#define NUM_ADAM_THREADS 16	

#define HOST_MEM_GB 100
#define DEV_MEM_GB 20

#define MODEL_CONFIG_SIZE_B 1
#define MODEL_PATH "../data/1B"


#define NUM_TOKENS 2048
#define MAX_TOKENS_PER_CHUNK 2048

#define TOKEN_IDS_PATH "../data/2048_token_ids_uint32.dat"
#define TOKEN_LABELS_PATH "../data/2048_labels_uint32.dat"


// number of seqs to process before calling optimizer step...
// (i.e. how many times gradients are accumulated before optimizer step)
#define NUM_SEQS_PER_STEP 1

// right now only testing with 1 sequence, but can have this sequence do fwd/bwd repeats
// just for perf testing...
#define NUM_TOTAL_SEQ_LAPS 1

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

	int num_streams = 6;
	int opt_stream_prios[6] = {0, 0, 0, 0, 0, 0};
	char * opt_stream_names[6] = {"Inbound", "Compute", "Outbound", "Peer", "Host Ops", "Inbound Fwd Context"};


	int inbound_stream_id = 0;
	int compute_stream_id = 1;
	int outbound_stream_id = 2;
	int peer_stream_id = 3;
	int host_ops_stream_id = 4;
	int inbound_fwd_ctx_stream_id = 5;
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
	size_t host_size_bytes = HOST_MEM_GB * (1UL << 30);

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

		
	size_t dev_size_bytes = DEV_MEM_GB * (1UL << 30);

	int dev_alignment = 256;

	printf("Allocating device memory of size: %lu...\n\n", dev_size_bytes);


	void * dev_mem = dataflow_handle.alloc_mem(&dataflow_handle, dev_size_bytes);
	if (!dev_mem){
		fprintf(stderr, "Error: device memory allocation failed...\n");
		return -1;
	}

	void * cur_host_mem = host_mem;
	void * cur_dev_mem = dev_mem;

	size_t used_host_mem = 0;
	size_t used_dev_mem = 0;


	// Preparing model...

	// for GeForce cards FP16 is double performance of BF16 because can use FP16 compute...
	//DataflowDatatype block_dt = DATAFLOW_FP16;

	DataflowDatatype block_dt = DATAFLOW_BF16;
	
	// for now summing bwd same as fwd...
	// conversions not set up yet...
	DataflowDatatype block_bwd_dt = block_dt;

	size_t block_dt_size = dataflow_sizeof_element(block_dt);
	size_t block_bwd_dt_size = dataflow_sizeof_element(block_bwd_dt);

	// for matmul accumulations...
	// on Geforce using FP16 gets double perf,
	// on datacenter cards should use DATAFLOW_FP32
	//DataflowDatatype compute_dt = DATAFLOW_FP16;

	// however for BF dataftype, requires FP32 compute
	DataflowDatatype compute_dt = DATAFLOW_FP32;
	
	// for now summing bwd same as fwd...
	// conversions not set up yet...
	DataflowDatatype compute_bwd_dt = compute_dt;


	
	
	//DataflowDatatype compute_bwd_dt = compute_dt;


	// optimizer data types...
	DataflowDatatype opt_mean_dt = block_dt;
	DataflowDatatype opt_var_dt = block_dt;

	size_t opt_mean_dt_size = dataflow_sizeof_element(opt_mean_dt);
	size_t opt_var_dt_size = dataflow_sizeof_element(opt_var_dt);


	DataflowNormalizationType norm_type = DATAFLOW_RMSNORM;

	DataflowPositionEmbeddingType pos_emb_type = DATAFLOW_ROPE;

	DataflowAttentionType attn_type = DATAFLOW_EXACT_ATTENTION;

	DataflowMLPType mlp_type = DATAFLOW_GATED_MLP;

	DataflowActivationType activ_type = DATAFLOW_SWIGLU;

	float eps = 1e-5;
	int theta = 500000;

	int n_layers;
	int num_q_heads;
	int num_kv_heads;
	int head_dim;
	int ffn_dim;
	int model_dim;
	int kv_dim;
	int vocab_size;

	if (MODEL_CONFIG_SIZE_B == 70){
	// llama3 70B config
		n_layers = 80;
		num_q_heads = 64;
		num_kv_heads = 8;
		head_dim = 128;
		ffn_dim = 28672;
		model_dim = num_q_heads * head_dim;
		kv_dim = num_kv_heads * head_dim;
		vocab_size = 128256;
	}
	else if (MODEL_CONFIG_SIZE_B == 8){
		// llama3 8b config
		n_layers = 32;
		num_q_heads = 32;
		num_kv_heads = 8;
		head_dim = 128;
		ffn_dim = 14336;
		model_dim = num_q_heads * head_dim;
		kv_dim = num_kv_heads * head_dim;
		vocab_size = 128256;
	}
	else if (MODEL_CONFIG_SIZE_B == 1){
		// llama3 1b config
		n_layers = 16;
		num_q_heads = 32;
		num_kv_heads = 8;
		head_dim = 64;
		ffn_dim = 8192;
		model_dim = num_q_heads * head_dim;
		kv_dim = num_kv_heads * head_dim;
		vocab_size = 128256;
	}
	else{
		fprintf(stderr, "Error: invalid model config size (B): %d\n", MODEL_CONFIG_SIZE_B);
		return -1;
	}
	

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
	used_host_mem += sys_embedding_table -> embedding_table_size;



	
	

	printf("Preparing all sys transformer blocks...\n");

	

	Transformer_Block ** sys_blocks = malloc(n_layers * sizeof(Transformer_Block *));
	if (!sys_blocks){
		fprintf(stderr, "Error: failed to allocate sys_blocks...\n");
		return -1;
	}


	uint64_t raw_block_size;
	uint64_t aligned_block_size;
	
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

		if (i == 0){
			raw_block_size = get_transformer_block_raw_size(sys_blocks[i]);
			aligned_block_size = get_transformer_block_aligned_size(sys_blocks[i]);
		}

		printf("Binding sys transformer block #%d...\n", i);
		ret = bind_transformer_block(cur_host_mem, sys_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind transformer block #%d...\n", i);
			return -1;
		}

		cur_host_mem += aligned_block_size;
		used_host_mem += aligned_block_size;
	}


	Transformer_Head * sys_head = malloc(sizeof(Transformer_Head));
	if (!sys_head){
		fprintf(stderr, "Error: failed to allocate sys_head...\n");
		return -1;
	}
	
	sys_head -> fwd_dt = block_dt;
	sys_head -> bwd_dt = block_bwd_dt;
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
	used_host_mem += combined_head_size;


	uint64_t all_blocks_size = aligned_block_size * n_layers;
	uint64_t all_model_size = sys_embedding_table -> embedding_table_size + all_blocks_size + combined_head_size;

	printf("\nTransformer Block Size (bytes):\n\tRaw: %lu\n\tSize With Matrix Alignment (%d): %lu\n\n", raw_block_size, pointer_alignment, aligned_block_size);

	printf("\n\n\nModel Sizing (bytes):\n\tEmbedding: %lu\n\tBlock: %lu\n\t\tTotal: %lu\n\tHead: %lu\nTOTAL MODEL SIZE: %lu\n\n\n", sys_embedding_table -> embedding_table_size, aligned_block_size, all_blocks_size, combined_head_size, all_model_size);


	

	// number of elements to pass into optimizer...

	uint64_t embedding_num_els = (uint64_t) vocab_size * (uint64_t) model_dim;
	uint64_t block_num_els = raw_block_size / block_dt_size;
	uint64_t block_aligned_num_els = aligned_block_size / block_dt_size;
	uint64_t all_blocks_num_els = block_num_els * n_layers;
	uint64_t head_num_els = (uint64_t) model_dim * ((uint64_t) 1 + (uint64_t) vocab_size);

	uint64_t all_model_num_els = embedding_num_els + all_blocks_num_els + head_num_els;

	printf("\n\n\nModel Parameter Counts:\n\tEmbedding: %lu\n\tBlock: %lu\n\t\tTotal: %lu\n\tHead: %lu\nTOTAL MODEL PARAMETERS: %lu\n\n\n", embedding_num_els, block_num_els, all_blocks_num_els, head_num_els, all_model_num_els);


	// Loading in from checkpoint...

	printf("\n\nLOADING MODEL FROM CHECKPOINT: %s...\n", MODEL_PATH);

	char layer_path[PATH_MAX];


	printf("Loading embedding table...\n");

	sprintf(layer_path, "%s/embed/tok_embeddings.weight", MODEL_PATH);
	FILE * fp = fopen(layer_path, "rb");
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



	

	printf("Loading all sys transformer blocks...\n");

	
	for (int i = 0; i < n_layers; i++){
		

		sprintf(layer_path, "%s/layers/%d/combined_layer.weight", MODEL_PATH, i);

		printf("Loading transformer block from: %s...\n", layer_path);
		ret = load_transformer_block(layer_path, sys_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to load transformer block #%d from: %s...\n", i, layer_path);
			return -1;
		}
	}



	printf("Loading head...\n");

	sprintf(layer_path, "%s/head/combined_head.weight", MODEL_PATH);

	fp = fopen(layer_path, "rb");
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


	// HANDLING OPTIMIZER (assuming same dtype as block for now)...


	// Embedding opt state...

	Transformer_Embedding_Table * sys_opt_mean_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!sys_opt_mean_embedding_table){
		fprintf(stderr, "Error: failed to allocate sys_opt_mean_embedding_table...\n");
		return -1;
	}

	sys_opt_mean_embedding_table -> config = embedding_config;
	sys_opt_mean_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * opt_mean_dt_size;
	sys_opt_mean_embedding_table -> embedding_table = cur_host_mem;

	cur_host_mem += sys_opt_mean_embedding_table -> embedding_table_size;
	used_host_mem += sys_opt_mean_embedding_table -> embedding_table_size;
	
	
	Transformer_Embedding_Table * sys_opt_var_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!sys_opt_var_embedding_table){
		fprintf(stderr, "Error: failed to allocate sys_opt_var_embedding_table...\n");
		return -1;
	}

	sys_opt_var_embedding_table -> config = embedding_config;
	sys_opt_var_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * opt_var_dt_size;
	sys_opt_var_embedding_table -> embedding_table = cur_host_mem;

	cur_host_mem += sys_opt_var_embedding_table -> embedding_table_size;
	used_host_mem += sys_opt_var_embedding_table -> embedding_table_size;
	// Blocks opt state

	Transformer_Block ** sys_opt_mean_blocks = malloc(n_layers * sizeof(Transformer_Block *));
	if (!sys_opt_mean_blocks){
		fprintf(stderr, "Error: failed to allocate sys_opt_mean_blocks...\n");
		return -1;
	}

	Transformer_Block ** sys_opt_var_blocks = malloc(n_layers * sizeof(Transformer_Block *));
	if (!sys_opt_var_blocks){
		fprintf(stderr, "Error: failed to allocate sys_opt_var_blocks...\n");
		return -1;
	}

	for (int i = 0; i < n_layers; i++){
		sys_opt_mean_blocks[i] = init_transformer_block(i, opt_mean_dt, compute_dt,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);
		if (!sys_opt_mean_blocks[i]){
			fprintf(stderr, "Error: failed to init sys_opt_mean_block #%d...\n", i);
			return -1;
		}

		ret = bind_transformer_block(cur_host_mem, sys_opt_mean_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind sys_opt_mean_block #%d...\n", i);
			return -1;
		}

		cur_host_mem += aligned_block_size;
		used_host_mem += aligned_block_size;

		memset(sys_opt_mean_blocks[i] -> buffer, 0, aligned_block_size);
		

		sys_opt_var_blocks[i] = init_transformer_block(i, opt_var_dt, compute_dt,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);
		if (!sys_opt_var_blocks[i]){
			fprintf(stderr, "Error: failed to init sys_opt_var_block #%d...\n", i);
			return -1;
		}

		ret = bind_transformer_block(cur_host_mem, sys_opt_var_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind sys_opt_var_block #%d...\n", i);
			return -1;
		}

		memset(sys_opt_var_blocks[i] -> buffer, 0, aligned_block_size);

		cur_host_mem += aligned_block_size;
		used_host_mem += aligned_block_size;
	}


	// Head opt state

	Transformer_Head * sys_opt_mean_head = malloc(sizeof(Transformer_Head));
	if (!sys_opt_mean_head){
		fprintf(stderr, "Error: failed to allocate sys_opt_mean_head...\n");
		return -1;
	}

	sys_opt_mean_head -> fwd_dt = block_dt;
	sys_opt_mean_head -> bwd_dt = block_bwd_dt;
	sys_opt_mean_head -> compute_dt = compute_dt;
	sys_opt_mean_head -> eps = eps;
	sys_opt_mean_head -> embedding_config = embedding_config;
	sys_opt_mean_head -> buffer = cur_host_mem;
	sys_opt_mean_head -> w_head_norm = sys_opt_mean_head -> buffer;
	sys_opt_mean_head -> w_head = sys_opt_mean_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) opt_mean_dt_size;

	cur_host_mem += combined_head_size;
	used_host_mem += combined_head_size;

	Transformer_Head * sys_opt_var_head = malloc(sizeof(Transformer_Head));
	if (!sys_opt_var_head){
		fprintf(stderr, "Error: failed to allocate sys_opt_var_head...\n");
		return -1;
	}

	sys_opt_var_head -> fwd_dt = block_dt;
	sys_opt_var_head -> bwd_dt = block_bwd_dt;
	sys_opt_var_head -> compute_dt = compute_dt;
	sys_opt_var_head -> eps = eps;
	sys_opt_var_head -> embedding_config = embedding_config;
	sys_opt_var_head -> buffer = cur_host_mem;
	sys_opt_var_head -> w_head_norm = sys_opt_var_head -> buffer;
	sys_opt_var_head -> w_head = sys_opt_var_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) opt_var_dt_size;

	cur_host_mem += combined_head_size;
	used_host_mem += combined_head_size;
	
	




	// PARTIONING DEVICE MEMORY...!!!

	Transformer_Embedding_Table * embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!embedding_table){
		fprintf(stderr, "Error: failed to allocate embedding_table...\n");
		return -1;
	}

	embedding_table -> config = embedding_config;
	embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_dt_size;
	embedding_table -> embedding_table = cur_dev_mem;

	cur_dev_mem += embedding_table -> embedding_table_size;
	used_dev_mem += embedding_table -> embedding_table_size;
	// ensure alignment for matmuls..

	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

	printf("Copying embedding table to device...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, embedding_table -> embedding_table, sys_embedding_table -> embedding_table, embedding_table -> embedding_table_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit inbound transfer for embedding table...\n");
		return -1;
	}



	int num_dev_blocks = NUM_DEV_BLOCKS;

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

		cur_dev_mem += aligned_block_size;
		used_dev_mem += aligned_block_size;

		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

		// copy sys block to dev block

		printf("Submitting inbound transfer for dev transformer block #%d...\n", i);

		ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[i] -> buffer, sys_blocks[i] -> buffer, aligned_block_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit inbound transfer for transformer block #%d...\n", i);
			return -1;
		}
	}

	sem_t * is_block_ready = malloc(n_layers * sizeof(sem_t));
	if (!is_block_ready){
		fprintf(stderr, "Error: failed to allocate is_block_ready...\n");
		return -1;
	}

	for (int i = 0; i < n_layers; i++){
		sem_init(&(is_block_ready[i]), 0, 0);
	}

	for (int i = 0; i < num_dev_blocks; i++){
		sem_post(&(is_block_ready[i]));
	}
	
	

	// Loading head...



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
	used_dev_mem += combined_head_size;

	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

	printf("Submitting inbound transfer for dev head...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, head -> buffer, sys_head -> buffer, combined_head_size);
	if (ret){
		fprintf(stderr, "Error: failed to submit inbound transfer for dev head...\n");
		return -1;
	}




	// GRADIENTS!

	uint64_t aligned_block_bwd_size;

	Transformer_Block ** sys_grad_blocks = malloc(n_layers * sizeof(Transformer_Block *));
	if (!sys_grad_blocks){
		fprintf(stderr, "Error: failed to allocate sys_grad_blocks...\n");
		return -1;
	}

	for (int i = 0; i < n_layers; i++){
		sys_grad_blocks[i] = init_transformer_block(i, block_bwd_dt, compute_bwd_dt,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);

		if (!sys_grad_blocks[i]){
			fprintf(stderr, "Error: failed to init transformer block grad #%d...\n", i);
			return -1;
		}

		if (i == 0){
			aligned_block_bwd_size = get_transformer_block_aligned_size(sys_grad_blocks[i]);
		}

		ret = bind_transformer_block(cur_host_mem, sys_grad_blocks[i]);
		if (ret){
			fprintf(stderr, "Error: failed to bind transformer block grad #%d...\n", i);
			return -1;
		}

		memset(sys_grad_blocks[i] -> buffer, 0, aligned_block_bwd_size);

		cur_host_mem += aligned_block_bwd_size;
		used_host_mem += aligned_block_bwd_size;
	}
	


	// JUST FOR NOW (while testing for correctness) keeping all block grads on device...
	int num_dev_block_grads = NUM_DEV_BLOCK_GRADS;

	Transformer_Block ** grad_blocks = malloc(num_dev_block_grads * sizeof(Transformer_Block *));
	if (!grad_blocks){
		fprintf(stderr, "Error: failed to allocate grad_blocks...\n");
		return -1;
	}

	for (int i = 0; i < num_dev_block_grads; i++){
		grad_blocks[i] = init_transformer_block(i, block_bwd_dt, compute_bwd_dt,
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

		ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, grad_blocks[i] -> buffer, 0, aligned_block_bwd_size);
		if (ret){
			fprintf(stderr, "Error: failed to set mem to 0 for grad block #%d...\n", i);
			return -1;
		}

		cur_dev_mem += aligned_block_bwd_size;
		used_dev_mem += aligned_block_bwd_size;

		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);
	}

	sem_t * is_grad_block_ready = malloc(num_dev_block_grads * sizeof(sem_t));
	if (!is_grad_block_ready){
		fprintf(stderr, "Error: failed to allocate is_grad_block_ready...\n");
		return -1;
	}

	for (int i = num_dev_block_grads - 1; i >= 0; i--){
		sem_init(&(is_grad_block_ready[i]), 0, 0);
		sem_post(&(is_grad_block_ready[i]));
	}
	
	

	// Embedding Table Gradients

	Transformer_Embedding_Table * sys_grad_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!sys_grad_embedding_table){
		fprintf(stderr, "Error: failed to allocate sys_grad_embedding_table...\n");
		return -1;
	}

	sys_grad_embedding_table -> config = embedding_config;
	sys_grad_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_bwd_dt_size;
	sys_grad_embedding_table -> embedding_table = cur_host_mem;
	
	
	ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, sys_grad_embedding_table -> embedding_table, 0, sys_grad_embedding_table -> embedding_table_size);
	if (ret){
		fprintf(stderr, "Error: failed to set mem for sys_grad_embedding_table...\n");
		return -1;
	}

	cur_host_mem += sys_grad_embedding_table -> embedding_table_size;
	used_host_mem += sys_grad_embedding_table -> embedding_table_size;
	
	


	Transformer_Embedding_Table * grad_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
	if (!grad_embedding_table){
		fprintf(stderr, "Error: failed to allocate grad_embedding_table...\n");
		return -1;
	}

	grad_embedding_table -> config = embedding_config;
	grad_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_bwd_dt_size;
	grad_embedding_table -> embedding_table = cur_dev_mem;

	ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, grad_embedding_table -> embedding_table, 0, grad_embedding_table -> embedding_table_size);
	if (ret){
		fprintf(stderr, "Error: failed to set mem for grad_embedding_table...\n");
		return -1;
	}

	cur_dev_mem += grad_embedding_table -> embedding_table_size;
	used_dev_mem += grad_embedding_table -> embedding_table_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);
	
	

	// Head Gradients

	uint64_t combined_head_bwd_size = combined_head_els * block_bwd_dt_size;

	Transformer_Head * sys_grad_head = malloc(sizeof(Transformer_Head));
	if (!sys_grad_head){
		fprintf(stderr, "Error: failed to allocate sys_grad_head...\n");
		return -1;
	}

	sys_grad_head -> fwd_dt = block_dt;
	sys_grad_head -> bwd_dt = block_bwd_dt;
	sys_grad_head -> compute_dt = compute_dt;
	sys_grad_head -> eps = eps;
	sys_grad_head -> embedding_config = embedding_config;
	sys_grad_head -> buffer = cur_host_mem;
	sys_grad_head -> w_head_norm = sys_grad_head -> buffer;
	sys_grad_head -> w_head = sys_grad_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) block_bwd_dt_size;


	ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, sys_grad_head -> buffer, 0, combined_head_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to set mem for sys_grad_head...\n");
		return -1;
	}


	cur_host_mem += combined_head_bwd_size;
	used_host_mem += combined_head_bwd_size;
	
	


	Transformer_Head * grad_head = malloc(sizeof(Transformer_Head));
	if (!grad_head){
		fprintf(stderr, "Error: failed to allocate grad_head...\n");
		return -1;
	}

	grad_head -> fwd_dt = block_dt;
	grad_head -> bwd_dt = block_bwd_dt;
	grad_head -> compute_dt = compute_dt;
	grad_head -> eps = eps;
	grad_head -> embedding_config = embedding_config;
	grad_head -> buffer = cur_dev_mem;
	grad_head -> w_head_norm = grad_head -> buffer;
	grad_head -> w_head = grad_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) block_bwd_dt_size;

	

	ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, grad_head -> buffer, 0, combined_head_bwd_size);
	if (ret){
		fprintf(stderr, "Error: failed to set mem for grad_head...\n");
		return -1;
	}



	cur_dev_mem += combined_head_bwd_size;
	used_dev_mem += combined_head_bwd_size;

	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);




	int num_sys_grad_results = NUM_SYS_GRAD_RESULTS;
	void ** sys_grad_results = malloc(num_sys_grad_results * sizeof(void *));
	if (!sys_grad_results){
		fprintf(stderr, "Error: failed to allocate sys_grad_results...\n");
		return -1;
	}

	uint64_t block_bwd_size = aligned_block_bwd_size;
	uint64_t embedding_bwd_size = sys_grad_embedding_table -> embedding_table_size;
	uint64_t head_bwd_size = combined_head_bwd_size;

	uint64_t sys_grad_result_size = MY_MAX(block_bwd_size, MY_MAX(embedding_bwd_size, head_bwd_size));



	for (int i = 0; i < num_sys_grad_results; i++){
		sys_grad_results[i] = cur_host_mem;
		cur_host_mem += sys_grad_result_size;
		used_host_mem += sys_grad_result_size;
	}

	sem_t * is_sys_grad_result_ready = malloc(num_sys_grad_results * sizeof(sem_t));
	if (!is_sys_grad_result_ready){
		fprintf(stderr, "Error: failed to allocate is_sys_grad_result_ready...\n");
		return -1;
	}

	for (int i = 0; i < num_sys_grad_results; i++){
		sem_init(&(is_sys_grad_result_ready[i]), 0, 0);
		sem_post(&(is_sys_grad_result_ready[i]));
	}
	

	
	
	









	// CONTEXT AND GRAD CONTEXTS!

	int total_tokens = NUM_TOKENS;


	// for now just testing the 1 sequence case for debugging...
	int num_seqs = 1;


	int max_tokens_per_chunk = MAX_TOKENS_PER_CHUNK;
	int num_chunks = MY_CEIL(total_tokens, max_tokens_per_chunk);
	
	
	char inp_file_path[PATH_MAX];
	sprintf(inp_file_path, "%s", TOKEN_IDS_PATH);

	uint32_t * sys_token_ids = malloc(total_tokens * sizeof(uint32_t));

	fp = fopen(inp_file_path, "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open %s...\n", inp_file_path);
		return -1;
	}

	read_els = fread(sys_token_ids, sizeof(uint32_t), total_tokens, fp);
	if (read_els != total_tokens){
		fprintf(stderr, "Error: failed to read %s, read_els: %zu, expected: %d\n", inp_file_path, read_els, total_tokens);
		return -1;
	}
	fclose(fp);

	// writing back in same dir as other data for clarity...

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

	sprintf(inp_file_path, "%s", TOKEN_LABELS_PATH);

	fp = fopen(inp_file_path, "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open %s...\n", inp_file_path);
		return -1;
	}

	read_els = fread(sys_labels, sizeof(uint32_t), total_tokens, fp);
	if (read_els != total_tokens){
		fprintf(stderr, "Error: failed to read %s, read_els: %zu, expected: %d\n", inp_file_path, read_els, total_tokens);
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
		used_dev_mem += metadata_buffer_size;


		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

		

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


	Seq_Batch_Context * fwd_context = malloc(sizeof(Seq_Batch_Context));
	if (!fwd_context){
		fprintf(stderr, "Error: failed to allocate fwd_context...\n");
		return -1;
	}

	Seq_Batch_Context * bwd_context = malloc(sizeof(Seq_Batch_Context));
	if (!bwd_context){
		fprintf(stderr, "Error: failed to allocate bwd_context...\n");
		return -1;
	}


	uint64_t context_buffer_size = 2 * (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;


	fwd_context -> contextBuffer = cur_dev_mem;
	fwd_context -> contextBufferBytes = context_buffer_size;
	
	fwd_context -> cur_tokens_populated = 0;
	fwd_context -> total_context_tokens = total_tokens;

	fwd_context -> x_k = cur_dev_mem;
	fwd_context -> x_v = fwd_context -> x_k + (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;

	cur_dev_mem += context_buffer_size;
	used_dev_mem += context_buffer_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);


	uint64_t context_buffer_bwd_size = 2 * (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_bwd_dt_size;
	
	(bwd_context) -> contextBuffer = cur_dev_mem;
	(bwd_context) -> contextBufferBytes = context_buffer_bwd_size;
	
	(bwd_context) -> cur_tokens_populated = 0;
	(bwd_context) -> total_context_tokens = total_tokens;

	(bwd_context) -> x_k = cur_dev_mem;
	(bwd_context) -> x_v = (bwd_context) -> x_k + (uint64_t) total_tokens * (uint64_t) kv_dim * (uint64_t) block_bwd_dt_size;

	cur_dev_mem += context_buffer_bwd_size;
	used_dev_mem += context_buffer_bwd_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

	
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
		used_host_mem += block_transition_size;
		block_transitions[i].X = cur_dev_mem;
		block_transitions[i].seq_batch = seq_batches[i / 2];
		cur_dev_mem += block_transition_size;
		used_dev_mem += block_transition_size;

		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);
	}

	// SAME KERNEL WORKSPACE ACROSS ALL COMPUTATIONS!

	uint64_t kernelWorkspaceBytes = 1UL << 28;
	void * kernelWorkspace = cur_dev_mem;
	cur_dev_mem += kernelWorkspaceBytes;
	used_dev_mem += kernelWorkspaceBytes;
	// ensure alignment for matmuls..	
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

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
	used_dev_mem += activation_workspace_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);



	// Saved Actviations will live on device and might be transferred back to host and retrieved prior to bwd pass...

	int num_sys_saved_activations = n_layers * num_chunks;

	Seq_Batch_Saved_Activations * sys_saved_activations = malloc(num_sys_saved_activations * sizeof(Seq_Batch_Saved_Activations));
	if (!sys_saved_activations){
		fprintf(stderr, "Error: failed to allocate sys_saved_activations...\n");
		return -1;
	}

	uint64_t saved_activations_buffer_size;

	for (int i = 0; i < num_sys_saved_activations; i++){

		saved_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[(i % num_chunks)]);
		ret = bind_seq_batch_saved_activations_buffer(seq_batches[(i % num_chunks)], &(sys_saved_activations[i]), cur_host_mem, saved_activations_buffer_size, i);
		if (ret){
			fprintf(stderr, "Error: failed to bind seq_batch saved_activations buffer...\n");
			return -1;
		}

		cur_host_mem += saved_activations_buffer_size;
		used_host_mem += saved_activations_buffer_size;
		sys_saved_activations[i].recomputed_activations = NULL;
	}


	int num_saved_activation_buffers = NUM_DEV_ACTIVATION_SLOTS;

	Seq_Batch_Saved_Activations * saved_activations = malloc(num_saved_activation_buffers * sizeof(Seq_Batch_Saved_Activations));
	if (!saved_activations){
		fprintf(stderr, "Error: failed to allocate saved_activations...\n");
		return -1;
	}

	
	
	for (int i = 0; i < num_saved_activation_buffers; i++){

		saved_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[(i % num_chunks)]);
		
		if (i == 0){
			printf("Saved Activations buffer size: %lu\n\n", saved_activations_buffer_size);
		
		}

		ret = bind_seq_batch_saved_activations_buffer(seq_batches[(i % num_chunks)], &(saved_activations[i]), cur_dev_mem, saved_activations_buffer_size, i);
		if (ret){
			fprintf(stderr, "Error: failed to bind seq_batch saved_activations buffer...\n");
			return -1;
		}

		cur_dev_mem += saved_activations_buffer_size;
		used_dev_mem += saved_activations_buffer_size;
		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

		// only the grad activations will have a recomputed activations buffer...
		saved_activations[i].recomputed_activations = NULL;
	}

	sem_t * is_saved_activation_ready = malloc(num_saved_activation_buffers * sizeof(sem_t));
	if (!is_saved_activation_ready){
		fprintf(stderr, "Error: failed to allocate is_saved_activation_ready...\n");
		return -1;
	}

	for (int i = 0; i < num_saved_activation_buffers; i++){
		sem_init(&(is_saved_activation_ready[i]), 0, 0);
	}

	for (int i = 0; i < num_saved_activation_buffers; i++){
		sem_post(&(is_saved_activation_ready[i]));
	}


	sem_t * is_fwd_activations_ready = malloc(num_chunks * n_layers * sizeof(sem_t));
	if (!is_fwd_activations_ready){
		fprintf(stderr, "Error: failed to allocate is_fwd_activations_ready...\n");
		return -1;
	}

	for (int i = 0; i < num_chunks * n_layers; i++){
		sem_init(&(is_fwd_activations_ready[i]), 0, 0);
	}

	sem_t * is_saved_act_home = malloc(num_chunks * n_layers * sizeof(sem_t));
	if (!is_saved_act_home){
		fprintf(stderr, "Error: failed to allocate is_saved_act_home...\n");
		return -1;
	}

	for (int i = 0; i < num_chunks * n_layers; i++){
		sem_init(&(is_saved_act_home[i]), 0, 0);
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
	used_dev_mem += grad_activations_buffer_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);



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
	used_dev_mem += recomputed_activations_buffer_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

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
	used_dev_mem += head_norm_out_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);
	head_activations -> head_norm_rms_vals = cur_dev_mem;
	uint64_t head_norm_rms_vals_size = (uint64_t) max_tokens_per_chunk * (uint64_t) sizeof(float);
	cur_dev_mem += head_norm_rms_vals_size;
	used_dev_mem += head_norm_rms_vals_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);
	head_activations -> head_out = cur_dev_mem;
	uint64_t head_out_size = (uint64_t) max_tokens_per_chunk * (uint64_t) vocab_size * (uint64_t) block_dt_size;
	cur_dev_mem += head_out_size;
	used_dev_mem += head_out_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

	head_activations -> kernelWorkspace = kernelWorkspace;
	head_activations -> kernelWorkspaceBytes = kernelWorkspaceBytes;


	// EACH HEAD ACTIVATIONS STRUCT NEEDS TO BE FILLED IN WITH:
	// head_activations -> num_tokens = total_tokens;
	

	// PREPARING SPECIAL MODEL OUTPUT STRUCT...


	uint64_t logits_size = (uint64_t) max_tokens_per_chunk * (uint64_t) vocab_size * block_bwd_dt_size;
	void * sys_logits = cur_host_mem;
	cur_host_mem += logits_size;
	used_host_mem += logits_size;
	Transformer_Model_Output * model_output = malloc(sizeof(Transformer_Model_Output));
	if (!model_output){
		fprintf(stderr, "Error: failed to allocate model_output...\n");
		return -1;
	}


	// EACH MODEL OUTPUT STRUCT NEEDS TO BE FILLED IN WITH:
	// model_output -> seq_batch = seq_batches[i];

	model_output -> logits = cur_dev_mem;
	cur_dev_mem += logits_size;
	used_dev_mem += logits_size;
	// ensure alignment for matmuls..
	used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
	cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);





	printf("Setup Complete!\n\n");

	printf("\nMEMORY USAGE (GB):\n\tHost: %.3f\n\tDevice: %.3f\n\n\n\n", (float) used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) used_dev_mem / (1024.0 * 1024.0 * 1024.0));

	if ((used_host_mem > host_size_bytes) || (used_dev_mem > dev_size_bytes)) {
		fprintf(stderr, "ERROR. Cannot run with current configuration of %d dev parameter blocks and %d dev activation slots...\n", NUM_DEV_BLOCKS, NUM_DEV_ACTIVATION_SLOTS);
		
		if (used_host_mem > host_size_bytes){
			fprintf(stderr, "\nHost Memory Overflow: Have %.3f GB allocated, but requires %.3f GB with current setting...\n", (float) used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) host_size_bytes / (1024.0 * 1024.0 * 1024.0));
		}

		if (used_dev_mem > dev_size_bytes){
			fprintf(stderr, "\nDevice Memory Overflow: Have %.3f GB allocated, but requires %.3f GB with current setting...\n", (float) used_dev_mem / (1024.0 * 1024.0 * 1024.0), (float) dev_size_bytes / (1024.0 * 1024.0 * 1024.0));
		}
		
		return -1;
	}


	// TRAINING LOOP BELOW....













	Transformer_Block_Activations * cur_activations;
	Seq_Batch_Saved_Activations * cur_fwd_activations;

	int num_add_threads = NUM_ADD_THREADS;
	int num_adam_threads = NUM_ADAM_THREADS;
	
	// Dealing with gradient accumulation, optimization, and resetting memory...

	// Ensuring that arguments to the host functions remain intact...
	// these are populated within the dataflowops helper functions...
	// + 2 for the head and embedding


	Add_Host_Op_Args * add_op_buffers = calloc(n_layers + 2, sizeof(Add_Host_Op_Args));
	if (!add_op_buffers){
		fprintf(stderr, "Error: failed to allocate add_op_buffers...\n");
		return -1;
	}
	
	
	Adam_Host_Op_Args * adam_op_buffers = calloc(n_layers + 2, sizeof(Adam_Host_Op_Args));
	if (!adam_op_buffers){
		fprintf(stderr, "Error: failed to allocate adam_op_buffers...\n");
		return -1;
	}

	Set_Mem_Host_Op_Args * set_mem_op_buffers = calloc(n_layers + 2, sizeof(Set_Mem_Host_Op_Args));
	if (!set_mem_op_buffers){
		fprintf(stderr, "Error: failed to allocate set_mem_op_buffers...\n");
		return -1;
	}



	
	
	
	
	
	
	



	// ADAM OPTIMIZER PARAMS...

	// learning rate should have a set schedule...
	float lr = 1e-4;
	float beta1 = 0.9;
	float beta2 = 0.999;
	float weight_decay = 1e-3;
	float epsilon = 1e-8;


	void * sys_activation_home;
	void * cur_stream_state;

	int cur_act = 0;

	int total_acts = n_layers * num_chunks;


	int working_layer_ind = 0;
	int replacement_layer_ind = 0;
	int next_layer_id = num_dev_blocks;
	int working_act_buffer_ind = 0;

	Transformer_Block * working_block;

	void * working_sys_grad_result;

	int final_saved_act_buffer_ind = -1;


	Transformer_Block_Transition * final_block_output_transition;

	Transformer_Block_Transition * grad_stream_from_head;


	int working_grad_block_ind;
	int working_sys_grad_result_ind = 0;
	int cur_fwd_prefetch_act_ind;


	void * fwd_x_k_global = fwd_context -> x_k;
	void * fwd_x_v_global = fwd_context -> x_v;

	void * global_fwd_ctx_k_dest;
	void * global_fwd_ctx_v_dest;

	Transformer_Block * working_grad_block;

	int cur_tokens;
	int fwd_ctx_tokens_replaced;

	int cur_global_token_replacement_ind;

	Seq_Batch_Saved_Activations * prior_layer_sys_saved_activations;
	int prior_layer_dev_saved_act_ind;
	Seq_Batch_Saved_Activations * prior_layer_dev_saved_activations;

	int is_opt_step;

	// after how many sequences should we do an optimizer step...
	// CRTICIAL TODO: right now we aren't doing any accumulation of gradients, 
	// so if this isn't 1 the math is wrong..
	// but can still simulate more realsitic perf by setting this to a higher value...
	int seqs_per_step = NUM_SEQS_PER_STEP;


	// Simulating training loop...
	// Don't have the dataset loading properly ready yet...
	// just setting here for simulating perf...
	int num_train_seqs = NUM_TOTAL_SEQ_LAPS;


	int seq_in_step = 0;

	int cur_step = 0;


	for (int s = 0; s < num_train_seqs; s++){

		if ((seqs_per_step != 0) && (((seq_in_step + 1) % seqs_per_step == 0) || (s == (num_train_seqs - 1))) ) {
			printf("\n\n[Step %d] Will perform optimizer step after working through current seq (#%d of %d)...\n\n", cur_step, s, num_train_seqs);
			is_opt_step = 1;
			cur_step++;
		}
		else{
			is_opt_step = 0;
		}

		// FWD PASS...

		// 1.) DOING EMBEDDDING....

		// for layer 0 we include the embedding table...

		for (int i = 0; i < num_chunks; i++){
			printf("\n\nSubmitting embedding for seq #%d, chunk #%d...\n\n", s, i);
			
			model_input -> seq_batch = seq_batches[i];

			ret = dataflow_submit_transformer_embedding(&dataflow_handle, compute_stream_id,
												model_input,
												embedding_table,
												&(block_transitions[2 * i]));
			if (ret){
				fprintf(stderr, "Error: failed to submit transformer embedding...\n");
				return -1;
			}

			// set the context
			seq_batches[i] -> context = fwd_context;

		}

		// 2.) DOING CORE BLOCKS...

		// reset what the next layer to fetch going forwards is...
		next_layer_id = num_dev_blocks;

		for (int k = 0; k < n_layers; k++){

			// Ensure layer is ready before submitting ops...

			printf("\n\n[Fwd] Waiting for layer id #%d to be ready...\n\n", k);

			sem_wait(&(is_block_ready[k]));

			printf("\n\n[Fwd] Layer id #%d is ready! Submitting ops...\n\n", k);

			working_block = blocks[working_layer_ind];
			working_block -> layer_id = k;

			for (int i = 0; i < num_chunks; i++){
				
				printf("\n\nSubmitting fwd block %d for seq #%d, chunk #%d...!\n\n", k, s, i);

				sem_wait(&(is_saved_activation_ready[working_act_buffer_ind]));

				cur_activations = activations[k * num_chunks + i];

				cur_activations -> working_activations = &(saved_activations[working_act_buffer_ind]);
				cur_activations -> working_activations -> layer_id = k;
				cur_activations -> working_activations -> seq_batch = seq_batches[i];

				// set the context
				seq_batches[i] -> context = fwd_context;

				ret = dataflow_submit_transformer_block(&dataflow_handle, compute_stream_id, 
										&(block_transitions[2 * i + (k % 2)]), 
										working_block, 
										cur_activations, 
										&(block_transitions[2 * i + ((k + 1) % 2)])) ;

				if (ret){
					fprintf(stderr, "Error: failed to submit transformer block for seq #%d, chunk #%d, block #%d...\n", s, i, k);
					return -1;
				}


				// ENSURE WE SAVE THE STATE AFTER ALL OPS HAVE BEEN QUEUED...

				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for seq #%d, chunk #%d, layer id #%d's activations...\n", s, i, k);
					return -1;
				}

			

				/* TODO WHEN DEALING WITH NON-TRAIN CHUNKS...
				if (k < (n_layers - 1)){
					// send back the context that is needed during bwd pass (all chunks)
					// if last layer than already has correct context set...

					
				}
				*/

				// send back activation buffer needed during bwd pass...

				// otherwise we have the correct activations saved on device and don't need to send back...
				if (cur_act < (total_acts - num_saved_activation_buffers)) {

					// ensure depedency is set...

					printf("Sending seq #%d, chunk #%d, layer id #%d's activations to host (act #%d of %d)...\n\n", s, i, k, cur_act, total_acts);


					ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
					if (ret){
						fprintf(stderr, "Error: failed to submit dependency to send seq #%d, chunk #%d, layer id #%d's activations to host...\n", s, i, k);
						return -1;
					}

					sys_activation_home = sys_saved_activations[k * num_chunks + i].savedActivationsBuffer;

					ret = (dataflow_handle.submit_outbound_transfer)(&dataflow_handle, outbound_stream_id, sys_activation_home, cur_activations -> working_activations -> savedActivationsBuffer, 
																	cur_activations -> working_activations -> savedActivationsBufferBytes);

					if (ret){
						fprintf(stderr, "Error: failed to submit outbound transfer to send seq #%d, chunk #%d, layer id #%d's activations to host...\n", s, i, k);
						return -1;
					}

					// add back 
					ret = (dataflow_handle.submit_host_op)(&dataflow_handle, post_sem_callback, (void *) &(is_saved_activation_ready[working_act_buffer_ind]), outbound_stream_id);
					if (ret){
						fprintf(stderr, "Error: failed to submit host op to enqueue seq #%d, chunk #%d, layer id #%d's activations...\n", s, i, k);
						return -1;
					}

					printf("Submitting host op to enqueue act index %d sys activations as home...\n\n", k * num_chunks + i);
					ret = (dataflow_handle.submit_host_op)(&dataflow_handle, post_sem_callback, (void *) &(is_saved_act_home[k * num_chunks + i]), outbound_stream_id);
					if (ret){
						fprintf(stderr, "Error: failed to submit host op to enqueue seq #%d, chunk #%d, layer id #%d's activations as home...\n", s, i, k);
						return -1;
					}


					final_saved_act_buffer_ind = k * num_chunks + i;
				}
				// otherwise we are close to turning around and will save it on device until we reverse...
				else{
					sem_post(&(is_fwd_activations_ready[k * num_chunks + i]));
					sem_post(&(is_saved_activation_ready[working_act_buffer_ind]));
				}


				cur_act++;

				// ensure we don't increment on the final activation...
				if ((k < (n_layers - 1)) || (i < (num_chunks - 1))){
					working_act_buffer_ind = (working_act_buffer_ind + 1) % num_saved_activation_buffers;
				}
			}


			// prefetch next layer...
			if (next_layer_id < n_layers){

				printf("\n\nPrefetching next layer id #%d (replacing layer at index %d)...\n\n", next_layer_id, replacement_layer_ind);

				ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency to prefetch next layer...\n");
					return -1;
				}

				ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[replacement_layer_ind] -> buffer, sys_blocks[next_layer_id] -> buffer, aligned_block_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit inbound transfer for transformer block #%d...\n", next_layer_id);
					return -1;
				}

				ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[next_layer_id]), inbound_stream_id);
				if (ret){
					fprintf(stderr, "Error: failed to submit host op to enqueue is_block_ready for next layer...\n");
					return -1;
				}
				
				// ensure the last layer will be the first one to be used during bwds, don't increment past it...
				if (next_layer_id < (n_layers - 1)){
					replacement_layer_ind = (replacement_layer_ind + 1) % num_dev_blocks;
					
				}
				next_layer_id++;
			}

			// don't move past the the last block as this will be the first one to be used during bwds...
			if (k < (n_layers - 1)){
				working_layer_ind = (working_layer_ind + 1) % num_dev_blocks;
			}
		}

		// 3.) NOW DOING HEAD, FOR NOW IN REVERSE ORDER...


		for (int i = num_chunks - 1; i >= 0; i--){
			printf("\n\nSubmitting head for seq #%d, chunk #%d...\n\n", s, i);


			final_block_output_transition = &(block_transitions[2 * i + (n_layers % 2)]);

			grad_stream_from_head = &(block_transitions[2 * i + ((n_layers - 1) % 2)]);

			// ensure grad stream is zeroed out...
			ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, grad_stream_from_head -> X, 0, block_transition_size);
			if (ret){
				fprintf(stderr, "Error: failed to zero out grad stream for seq #%d, chunk #%d before head...\n", s, i);
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

		// Send back grad head to host...

		sem_wait(&(is_sys_grad_result_ready[working_sys_grad_result_ind]));

		working_sys_grad_result = sys_grad_results[working_sys_grad_result_ind];


		printf("\n\nSending grad head to host for seq #%d...\n\n", s);

		cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
		if (!cur_stream_state){
			fprintf(stderr, "Error: failed to get stream state for head...\n");
			return -1;
		}
		
		ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
		if (ret){
			fprintf(stderr, "Error: failed to submit dependency to send head to host...\n");
			return -1;
		}

		ret = (dataflow_handle.submit_outbound_transfer)(&dataflow_handle, outbound_stream_id, working_sys_grad_result, grad_head -> buffer, combined_head_bwd_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit outbound transfer to send head to host...\n");
			return -1;
		}

		// IN REALITY THE HEAD WILL NOT HAVE A SEPERATE STRUCTURE AND DEDICATED GRAD BUFFER ON DEVICE...

		// However for correctness for now, need to reset grad workspace buffer for next sequence
		// because accumulating gradients on host side
		// So will we need to reset grad workspace buffer for next use...
		// (along with posting sem when done...)

		// This really can be submitted after setting dependency on host ops stream...
		ret = (dataflow_handle.set_mem)(&dataflow_handle, outbound_stream_id, grad_head -> buffer, 0, combined_head_bwd_size);
		if (ret){
			fprintf(stderr, "Error: failed to set mem for grad head...\n");
			return -1;
		}
		
		
		// Ensure to add results to existing grad buffers on host then make the results available for reuse...

		cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
		if (!cur_stream_state){
			fprintf(stderr, "Error: failed to get stream state for head...\n");
			return -1;
		}

		ret = dataflow_handle.submit_dependency(&dataflow_handle, host_ops_stream_id, cur_stream_state);
		if (ret){
			fprintf(stderr, "Error: failed to submit dependency to submit optimizer op...\n");
			return -1;
		}

		// Add results to existing grad buffers on host...

		ret = dataflow_submit_add_host(&dataflow_handle, host_ops_stream_id, 
											&add_host, &(add_op_buffers[n_layers + 1]),
											block_bwd_dt, block_bwd_dt, block_bwd_dt,
											num_add_threads, n_layers, block_num_els, 
											sys_grad_head -> buffer, working_sys_grad_result, sys_grad_head -> buffer,
											1.0, 1.0);
		if (ret){
			fprintf(stderr, "Error: failed to submit add host for grad head...\n");
			return -1;
		}

		ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_sys_grad_result_ready[working_sys_grad_result_ind]), host_ops_stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to submit host op to enqueue is_sys_grad_result_ready for next grad block...\n");
			return -1;
		}

		working_sys_grad_result_ind = (working_sys_grad_result_ind + 1) % num_sys_grad_results;

		if (is_opt_step) {
			// speicfying adam host functio as adam_step_host and passing this function address 
			// which will be submitted to the host ops stream...
			ret = dataflow_submit_adam_step_host(&dataflow_handle, host_ops_stream_id, 
								&adam_step_host, &(adam_op_buffers[n_layers + 1]),
								block_dt, block_bwd_dt, 
								opt_mean_dt, opt_var_dt,
								num_adam_threads, n_layers, head_num_els, 
								lr, beta1, beta2, weight_decay, epsilon,
								sys_head -> buffer, sys_grad_head -> buffer, sys_opt_mean_head -> buffer, sys_opt_var_head -> buffer);
			if (ret){
				fprintf(stderr, "Error: failed to submit adam step host for head...\n");
				return -1;
			}

			// Reset gradient accumulation buffer to 0...
			ret = dataflow_submit_set_mem_host(&dataflow_handle, host_ops_stream_id, 
                        &set_mem_host, &(set_mem_op_buffers[n_layers + 1]),
                        sys_grad_head -> buffer, combined_head_bwd_size, 0);

			if (ret){
				fprintf(stderr, "Error: failed to submit set mem host for grad head...\n");
				return -1;
			}
		}
		






		// BACKWARDS PASS...
		

		// send back gradient for head and run optimizer..
		
		// RESET NEXT LAYER ID TO THE LAST FORWARD BLOCK WE DON"T HAVE...!
		next_layer_id = n_layers - 1 - num_dev_blocks;
		working_grad_block_ind = num_dev_block_grads - 1;
		cur_fwd_prefetch_act_ind = final_saved_act_buffer_ind;

		// working_layer_ind and replacement_layer_ind should be properly set correctly poniting at last block, now working opposite direction...

		// ensure that we set the correct layers to be ready (that are already sitting from the fwd pass)

		for (int k = n_layers - 1; k > n_layers - 1 - num_dev_blocks; k--){
			sem_post(&(is_block_ready[k]));
		}



		cur_global_token_replacement_ind = 0;



		for (int k = n_layers - 1; k >= 0; k--){

			// Ensure we have forward layer...

			printf("\n\n[Bwd] Waiting for layer id #%d to be ready...\n\n", k);

			sem_wait(&(is_block_ready[k]));

			// if we are less than num dev blocks we will need this going forwards, so set to ready...
			if (k < num_dev_blocks){
				sem_post(&(is_block_ready[k]));
			}

			printf("\n\n[Bwd] Layer id #%d is ready! Submitting ops...\n\n", k);

			working_block = blocks[working_layer_ind];
			working_block -> layer_id = k;

			// Ensure we have a fresh gradient buffer to work over...
			printf("\n\n[Bwd] Waiting for grad block workspace to be ready...\n\n");

			sem_wait(&(is_grad_block_ready[working_grad_block_ind]));

			working_grad_block = grad_blocks[working_grad_block_ind];

			working_grad_block -> layer_id = k;

			fwd_ctx_tokens_replaced = 0;

			printf("\n\n[Bwd] Waiting for sys grad result #%d to be ready...\n\n", working_sys_grad_result_ind);

			sem_wait(&(is_sys_grad_result_ready[working_sys_grad_result_ind]));

			working_sys_grad_result = sys_grad_results[working_sys_grad_result_ind];

			// ensure fwd context is populated correctly before doing backprop
			// for the last block this already is, but then all prior blocks do replacements...
			cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, inbound_fwd_ctx_stream_id);
			if (!cur_stream_state){
				fprintf(stderr, "Error: failed to get stream state for fwd context...\n");
				return -1;
			}

			ret = dataflow_handle.submit_dependency(&dataflow_handle, compute_stream_id, cur_stream_state);
			if (ret){
				fprintf(stderr, "Error: failed to submit dependency to for compute stream to wait on fwd context transfers...\n");
				return -1;
			}
			
			for (int i = num_chunks - 1; i >= 0; i--){

				// ensure that we have the correct activations ready...
				printf("\n\n[Bwd] Waiting for fwd activations of chunk #%d to be ready...\n\n", i);
				
				sem_wait(&(is_fwd_activations_ready[k * num_chunks + i]));

				sem_wait(&(is_saved_activation_ready[working_act_buffer_ind]));
				
				cur_fwd_activations = &(saved_activations[working_act_buffer_ind]);
				cur_fwd_activations -> layer_id = k;
				cur_fwd_activations -> seq_batch = seq_batches[i];

				grad_activations -> working_activations -> layer_id = k;
				grad_activations -> working_activations -> seq_batch = seq_batches[i];

			
				printf("\n\nSubmitting bwd_x for seq #%d, chunk #%d, block #%d...\n\n", s, i, k);

				ret = dataflow_submit_transformer_block_bwd_x(&dataflow_handle, compute_stream_id,
									working_block, 
									&(block_transitions[2 * i + (k % 2)]), 
									cur_fwd_activations, fwd_context,
									grad_activations,
									working_grad_block,
									&(block_transitions[2 * i + ((k + 1) % 2)]));

				if (ret){
					fprintf(stderr, "Error: failed to submit transformer block bwd_x for seq #%d, chunk #%d, block #%d...\n", s, i, k);
					return -1;
				}

				// prefetch fwd context for next layer...
				// the bwds context being reset to 0 occurs in the bwd_x op...

				cur_tokens = seq_batches[i] -> total_tokens;

				if (k > 0){

					cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
					if (!cur_stream_state){
						fprintf(stderr, "Error: failed to get stream state for grad block #%d...\n", k);
						return -1;
					}

					ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_fwd_ctx_stream_id, cur_stream_state);
					if (ret){
						fprintf(stderr, "Error: failed to submit dependency to prefetch fwd context for next layer...\n");
						return -1;
					}


					cur_global_token_replacement_ind = fwd_context -> total_context_tokens - fwd_ctx_tokens_replaced - cur_tokens;

					global_fwd_ctx_k_dest = fwd_x_k_global + ((uint64_t) cur_global_token_replacement_ind * (uint64_t) kv_dim * (uint64_t) block_dt_size);
					global_fwd_ctx_v_dest = fwd_x_v_global + ((uint64_t) cur_global_token_replacement_ind * (uint64_t) kv_dim * (uint64_t) block_dt_size);

					// if we know that the prior layer's saved activations are waiting on the host
					// they might already prefetched or on way, but easier this way...
					if (((k - 1) * num_chunks + i) <= final_saved_act_buffer_ind){

						sem_wait(&(is_saved_act_home[(k - 1) * num_chunks + i]));
						sem_post(&(is_saved_act_home[(k - 1) * num_chunks + i]));

						prior_layer_sys_saved_activations = &(sys_saved_activations[(k - 1) * num_chunks + i]);

						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_k_dest, prior_layer_sys_saved_activations -> x_k_local, cur_tokens * kv_dim * block_dt_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer to prefetch fwd context for next layer...\n");
							return -1;
						}

						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_v_dest, prior_layer_sys_saved_activations -> x_v_local, cur_tokens * kv_dim * block_dt_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer to prefetch fwd context for next layer...\n");
							return -1;
						}
					}
					// otherwise the context is only on the device so we can just copy it...
					else{

						if (num_chunks <= working_act_buffer_ind){
							prior_layer_dev_saved_act_ind = working_act_buffer_ind - num_chunks;
						}
						// need to wrap around...
						else{
							prior_layer_dev_saved_act_ind = num_saved_activation_buffers - (num_chunks - working_act_buffer_ind);
						}

						prior_layer_dev_saved_activations = &(saved_activations[prior_layer_dev_saved_act_ind]);

						ret = dataflow_handle.submit_peer_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_k_dest, prior_layer_dev_saved_activations -> x_k_local, cur_tokens * kv_dim * block_dt_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit peer transfer to prefetch fwd context for next layer...\n");
							return -1;
						}

						ret = dataflow_handle.submit_peer_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_v_dest, prior_layer_dev_saved_activations -> x_v_local, cur_tokens * kv_dim * block_dt_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit peer transfer to prefetch fwd context for next layer...\n");
							return -1;
						}
					}

					fwd_ctx_tokens_replaced += cur_tokens;
				}

				// could prefetch next layer weights here, but doing this after the bwd w is done for cleanliness...

				
				printf("\n\nSubmitting bwd_w for seq #%d, chunk #%d, block #%d...\n\n", s, i, k);

				// utilizing the newly populated grad_activations struct
				// to update the grad_weights...

				ret = dataflow_submit_transformer_block_bwd_w(&dataflow_handle, compute_stream_id,
									&(block_transitions[2 * i + (k % 2)]),
									cur_fwd_activations, 
									grad_activations, 
									working_grad_block);

				if (ret){
					fprintf(stderr, "Error: failed to submit transformer block bwd_w for seq #%d, chunk #%d, block #%d...\n", s, i, k);
					return -1;
				}


				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for grad block #%d...\n", k);
					return -1;
				}

				// start prefetch for next activation buffer...

				if (cur_fwd_prefetch_act_ind >= 0){

					ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
					if (ret){
						fprintf(stderr, "Error: failed to submit dependency to prefetch next activation buffer...\n");
						return -1;
					}

					// ensuring the saved activation made it home before prefetching......
					sem_wait(&(is_saved_act_home[cur_fwd_prefetch_act_ind]));
					sem_post(&(is_saved_act_home[cur_fwd_prefetch_act_ind]));


					// overwriting the contents at working_act_buffer_ind....
					ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, cur_fwd_activations -> savedActivationsBuffer, sys_saved_activations[cur_fwd_prefetch_act_ind].savedActivationsBuffer, sys_saved_activations[cur_fwd_prefetch_act_ind].savedActivationsBufferBytes);
					if (ret){
						fprintf(stderr, "Error: failed to submit inbound transfer to prefetch next activation buffer...\n");
						return -1;
					}

					ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_fwd_activations_ready[cur_fwd_prefetch_act_ind]), inbound_stream_id);
					if (ret){
						fprintf(stderr, "Error: failed to submit host op to enqueue is_fwd_activations_ready for next activation buffer...\n");
						return -1;
					}

					ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_saved_activation_ready[working_act_buffer_ind]), inbound_stream_id);
					if (ret){
						fprintf(stderr, "Error: failed to submit host op to enqueue is_saved_activation_ready for next activation buffer...\n");
						return -1;
					}

					cur_fwd_prefetch_act_ind -= 1;
				}
				// otherwise we can just overwrite this buffer during the next fwd pass...
				else{
					sem_post(&(is_saved_activation_ready[working_act_buffer_ind]));
				}

				if (working_act_buffer_ind > 0){
					working_act_buffer_ind -= 1;
				}
				else{
					working_act_buffer_ind = num_saved_activation_buffers - 1;
				}

			}

			cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
			if (!cur_stream_state){
				fprintf(stderr, "Error: failed to get stream state for grad block #%d...\n", k);
				return -1;
			}

			// PREFETCHING NEXT (= PREVIOUS) FORWARD BLOCK...

			if (next_layer_id >= 0){

				printf("[Bwd] Prefetching block layer id %d into slot %d...\n\n", next_layer_id, replacement_layer_ind);

				ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency to prefetch block #%d...\n", next_layer_id);
					return -1;
				}

				ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[replacement_layer_ind] -> buffer, sys_blocks[next_layer_id] -> buffer, aligned_block_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit inbound transfer for block #%d...\n", next_layer_id);
					return -1;
				}

				ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[next_layer_id]), inbound_stream_id);
				if (ret){
					fprintf(stderr, "Error: failed to submit host op to enqueue is_block_ready for next block...\n");
					return -1;
				}

				if (next_layer_id > 0){
					if (replacement_layer_ind > 0){
						replacement_layer_ind -= 1;
					}
					else{
						replacement_layer_ind = num_dev_blocks - 1;
					}
				}
				next_layer_id--;
			}

			// in case we immediate follow up with another forward pass, don't move past the first layer...
			if (k > 0){
				if (working_layer_ind > 0){
					working_layer_ind -= 1;
				}
				else{
					working_layer_ind = num_dev_blocks - 1;
				}
			}

			// send back gradient to host, and run optimizer...

			// SENDING BACK GRAD BLOCK...

			printf("[Bwd] Sending back grad block for seq #%d, layer id %d at index %d...\n\n", s, k, working_grad_block_ind);

			ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
			if (ret){
				fprintf(stderr, "Error: failed to submit dependency to send grad block #%d to host...\n", k);
				return -1;
			}

	
			ret = (dataflow_handle.submit_outbound_transfer)(&dataflow_handle, outbound_stream_id, working_sys_grad_result, working_grad_block -> buffer, aligned_block_size);
			if (ret){
				fprintf(stderr, "Error: failed to submit outbound transfer to send grad block #%d to host...\n", k);
				return -1;
			}

			// After completing, can reset the grad buffer for next use...
			ret = (dataflow_handle.set_mem)(&dataflow_handle, outbound_stream_id, working_grad_block -> buffer, 0, aligned_block_size);
			if (ret){
				fprintf(stderr, "Error: failed to set mem for grad block #%d...\n", k);
				return -1;
			}

			ret = (dataflow_handle.submit_host_op)(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[working_grad_block_ind]), outbound_stream_id);
			if (ret){
				fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for next grad block...\n");
				return -1;
			}


			// Ensure to add results to existing grad buffers on host then make the results available for reuse...

			cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
			if (!cur_stream_state){
				fprintf(stderr, "Error: failed to get stream state for head...\n");
				return -1;
			}

			ret = dataflow_handle.submit_dependency(&dataflow_handle, host_ops_stream_id, cur_stream_state);
			if (ret){
				fprintf(stderr, "Error: failed to submit dependency to submit optimizer op...\n");
				return -1;
			}

			// Add results to existing grad buffers on host...

			ret = dataflow_submit_add_host(&dataflow_handle, host_ops_stream_id, 
											&add_host, &(add_op_buffers[k + 1]),
											block_bwd_dt, block_bwd_dt, block_bwd_dt,
											num_add_threads, k, block_num_els, 
											sys_grad_blocks[k] -> buffer, working_sys_grad_result, sys_grad_blocks[k] -> buffer,
											1.0, 1.0);
			if (ret){
				fprintf(stderr, "Error: failed to submit add host for grad block #%d...\n", k);
				return -1;
			}

			ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_sys_grad_result_ready[working_sys_grad_result_ind]), host_ops_stream_id);
			if (ret){
				fprintf(stderr, "Error: failed to submit host op to enqueue is_sys_grad_result_ready for next grad block...\n");
				return -1;
			}

			working_sys_grad_result_ind = (working_sys_grad_result_ind + 1) % num_sys_grad_results;


			// Calling optimizer step...
			if (is_opt_step) {
				

				// speicfying adam host functio as adam_step_host and passing this function address 
				// which will be submitted to the host ops stream...
				ret = dataflow_submit_adam_step_host(&dataflow_handle, host_ops_stream_id, 
									&adam_step_host, &(adam_op_buffers[k + 1]),
									block_dt, block_bwd_dt, 
									opt_mean_dt, opt_var_dt,
									num_adam_threads, k, block_aligned_num_els, 
									lr, beta1, beta2, weight_decay, epsilon,
									sys_blocks[k] -> buffer, sys_grad_blocks[k] -> buffer, sys_opt_mean_blocks[k] -> buffer, sys_opt_var_blocks[k] -> buffer);
				if (ret){
					fprintf(stderr, "Error: failed to submit adam step host for block #%d...\n", k);
					return -1;
				}

				// Reset gradient accumulation buffer to 0...
				ret = dataflow_submit_set_mem_host(&dataflow_handle, host_ops_stream_id, 
                        &set_mem_host, &(set_mem_op_buffers[k + 1]),
                        sys_grad_blocks[k] -> buffer, aligned_block_size, 0);

				if (ret){
					fprintf(stderr, "Error: failed to submit set mem host for grad block #%d...\n", k);
					return -1;
				}
			}

			// update the working grad block index...

			if (working_grad_block_ind > 0){
				working_grad_block_ind -= 1;
			}
			else{
				working_grad_block_ind = num_dev_block_grads - 1;
			}


			if (working_sys_grad_result_ind > 0){
				working_sys_grad_result_ind -= 1;
			}
			else{
				working_sys_grad_result_ind = num_sys_grad_results - 1;
			}
		}


		for (int i = num_chunks - 1; i >= 0; i--){
			printf("\n\nSubmitting embedding bwd_w for seq #%d, chunk #%d...\n\n", s, i);

			// layer 0'ths output grad stream will be at index 1 (for given chunk)
			ret = dataflow_submit_transformer_embedding_bwd_w(&dataflow_handle, compute_stream_id,
												&(block_transitions[2 * i + 1]),
												grad_embedding_table);

			if (ret){
				fprintf(stderr, "Error: failed to submit transformer embedding bwd_w for seq #%d, chunk #%d...\n", s, i);
				return -1;
			}
		}


		// FINSIHED FWD + BWD PASS FOR SEQ #%d...

		printf("\n\nFINISHED FWD + BWD PASS FOR SEQ #%d...\n\n", s);



		sem_wait(&(is_sys_grad_result_ready[working_sys_grad_result_ind]));

		working_sys_grad_result = sys_grad_results[working_sys_grad_result_ind];


		cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
		if (!cur_stream_state){
			fprintf(stderr, "Error: failed to get stream state for embedding table...\n");
			return -1;
		}
		
		ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
		if (ret){
			fprintf(stderr, "Error: failed to submit dependency to send embedding table to host...\n");
			return -1;
		}

		ret = (dataflow_handle.submit_outbound_transfer)(&dataflow_handle, outbound_stream_id, working_sys_grad_result, grad_embedding_table -> embedding_table, grad_embedding_table -> embedding_table_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit outbound transfer to send embedding table to host...\n");
			return -1;
		}

		// IN REALITY THE EMBEDDING TABLE WILL NOT HAVE A SEPERATE STRUCTURE AND DEDICATED GRAD BUFFER ON DEVICE...

		// However for correctness for now, need to reset grad workspace buffer for next sequence
		// because accumulating gradients on host side
		// So will we need to reset grad workspace buffer for next use...
		// (along with posting sem when done...)

		// This really can be submitted after setting dependency on host ops stream...
		ret = (dataflow_handle.set_mem)(&dataflow_handle, outbound_stream_id, grad_embedding_table -> embedding_table, 0, grad_embedding_table -> embedding_table_size);
		if (ret){
			fprintf(stderr, "Error: failed to set mem for grad embedding table...\n");
			return -1;
		}
		
		
		// Ensure to add results to existing grad buffers on host then make the results available for reuse...

		cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
		if (!cur_stream_state){
			fprintf(stderr, "Error: failed to get stream state for embedding table...\n");
			return -1;
		}

		ret = dataflow_handle.submit_dependency(&dataflow_handle, host_ops_stream_id, cur_stream_state);
		if (ret){
			fprintf(stderr, "Error: failed to submit dependency to submit optimizer op...\n");
			return -1;
		}

		// Add results to existing grad buffers on host...

		ret = dataflow_submit_add_host(&dataflow_handle, host_ops_stream_id, 
											&add_host, &(add_op_buffers[0]),
											block_bwd_dt, block_bwd_dt, block_bwd_dt,
											num_add_threads, -1, block_num_els, 
											sys_grad_embedding_table -> embedding_table, working_sys_grad_result, sys_grad_embedding_table -> embedding_table,
											1.0, 1.0);
		if (ret){
			fprintf(stderr, "Error: failed to submit add host for grad embedding table...\n");
			return -1;
		}

		ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_sys_grad_result_ready[working_sys_grad_result_ind]), host_ops_stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to submit host op to enqueue is_sys_grad_result_ready for next grad block...\n");
			return -1;
		}

		working_sys_grad_result_ind = (working_sys_grad_result_ind + 1) % num_sys_grad_results;

		if (is_opt_step) {
			// speicfying adam host functio as adam_step_host and passing this function address 
			// which will be submitted to the host ops stream...
			ret = dataflow_submit_adam_step_host(&dataflow_handle, host_ops_stream_id, 
								&adam_step_host, &(adam_op_buffers[0]),
								block_dt, block_bwd_dt, 
								opt_mean_dt, opt_var_dt,
								num_adam_threads, -1, head_num_els, 
								lr, beta1, beta2, weight_decay, epsilon,
								sys_embedding_table -> embedding_table, sys_grad_embedding_table -> embedding_table, sys_opt_mean_embedding_table -> embedding_table, sys_opt_var_embedding_table -> embedding_table);
			if (ret){
				fprintf(stderr, "Error: failed to submit adam step host for embedding table...\n");
				return -1;
			}

			// Reset gradient accumulation buffer to 0...
			ret = dataflow_submit_set_mem_host(&dataflow_handle, host_ops_stream_id, 
                        &set_mem_host, &(set_mem_op_buffers[0]),
                        sys_grad_embedding_table -> embedding_table, sys_grad_embedding_table -> embedding_table_size, 0);

			if (ret){
				fprintf(stderr, "Error: failed to submit set mem host for grad head...\n");
				return -1;
			}

			// if we have more seqs to process, then we need to load updated layers after step...
			if (s < num_seqs - 1){
				printf("Submitting dependency to load updated layers after step...\n\n");

				ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency to load updated blocks...\n");
					return -1;
				}

				// For now assumes we always have embedding table and head on device for simplicity...

				ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, embedding_table -> embedding_table, sys_embedding_table -> embedding_table, embedding_table -> embedding_table_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit inbound transfer to load updated blocks...\n");
					return -1;
				}


				ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, head -> buffer, sys_head -> buffer, combined_head_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit inbound transfer to load updated blocks...\n");
					return -1;
				}

				// make the compute stream wait for these to be ready (really only need embedding table, but easier to just have head be ready as well)...
				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, inbound_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for head...\n");
					return -1;
				}

				ret = dataflow_handle.submit_dependency(&dataflow_handle, compute_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency to load updated blocks...\n");
					return -1;
				}
				
				
				// now the compute stream will wait for the blocks to be ready as they arrive...

				for (int i = 0; i < num_dev_blocks; i++){

					printf("Setting block #%d as non-ready so it will wait for it to be loaded back during next forward pass...\n\n", i);

					sem_wait(&(is_block_ready[i]));

					ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[i] -> buffer, sys_blocks[i] -> buffer, aligned_block_size);
					if (ret){
						fprintf(stderr, "Error: failed to submit inbound transfer to load updated blocks...\n");
						return -1;
					}

					// at this point we should have all of the forward blocks posted as ready, but we need to mark them as not because we are going to retrieve the updated blocks...
					ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[i]), inbound_stream_id);
					if (ret){
						fprintf(stderr, "Error: failed to submit host op to enqueue is_block_ready for next block...\n");
						return -1;
					}
					
				}
			}

			// reset the indices of where working / replace inds are...
			// (might've gotten out of wack if non-divisble number of layers/num_dev_block size...)

			working_layer_ind = 0;
			replacement_layer_ind = 0;

			seq_in_step = 0;
		}
		else{
			seq_in_step += 1;
		}
		
	}
	


	printf("Finished enqueueing all dataflow operations! Waiting to sync...\n\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, host_ops_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync host ops stream at end of transformer...\n");
		return -1;
	}

	printf("All operations complete! Exiting...\n\n");

	return 0;
}
