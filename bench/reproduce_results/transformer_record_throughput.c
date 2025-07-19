	#include "dataflow_transformer.h"
	#include "dataflow_seq_batch.h"
	#include "cuda_dataflow_handle.h"
	#include "register_ops.h"
	#include "host_ops.h"

	#include <math.h>

	// peak flops found in:
	// https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
	#define A100_PEAK_BF16_FLOPS 3.12e14
	#define H100_PEAK_BF16_FLOPS 9.89e14
	#define RTX_3090_PEAK_BF16_FLOPS 7.1e13
	#define RTX_4090_PEAK_BF16_FLOPS 1.65e14
	#define RTX_5090_PEAK_BF16_FLOPS 2.095e14
	
	// this is just for testing,.. in 
	// reality determined dynamically...

	#define NUM_TOKENS_EXAMPLE_SEQ 65536
	#define TOKEN_IDS_PATH "../../data/65536_token_ids_uint32.dat"
	#define TOKEN_LABELS_PATH "../../data/65536_labels_uint32.dat"

	// this (along with num seqs per round) modulates how frequently we will step 
	// the optimizer...

	// THIS IS TARGET FOR 1B MODEL, BUT CODE ALREADY MODIFIES IF LARGER...
	#define TARGET_DURATION_PER_STEP_S 6.0f
	// to help determien how many rounds per step
	#define FLOP_EFFICIENCY_ESTIMATE 0.6f

	#define PCIE_LINK_EFFICIENCY 0.75f

	//#define NUM_STEPS 5

	//#define NUM_STEPS_TO_SKIP_FOR_RECORDING 2

	// num_chunks = num_chunks_per_seq * num_seq_groups_per_round
	// num_chunks_per_seq = seqlen / chunk_size
	// for now should ensure that:
	// if seqlen > chunk_size:
	// 		seqlen % chunk_size == 0
	// if seqlen < chunk_size:
	// 		chunk_size % seqlen == 0

	// up to num_chunks (per round for now, because just repeating) to save...
	#define NUM_RAW_CHUNK_IDS_LABELS_TO_SAVE 0







	// config for what to print...

	#define TO_PRINT_SETUP_CONFIG_SUMMARY 0
	#define TO_PRINT_MEMORY_PARTITION_CONFIG 0
	#define TO_PRINT_MEMORY_BREAKDOWN_VERBOSE 0
	#define TO_PRINT_MODEL_SIZING 0
	#define TO_PRINT_BATCH_CONFIG 0
	

	#define TO_PRINT_THROUGHPUT_METRICS 0
	#define TO_PRINT_THROUGHPUT_METRICS_VERBOSE 0

	#define TO_PRINT_ROUND_LOSS 0
	#define TO_PRINT_CHUNK_LOSS 0


	#define TO_PRINT_IS_STEP 0
	#define TO_PRINT_POST_STEP_RELOADING 0

	#define TO_PRINT_SUBMITTING 0

	#define TO_PRINT_FWD_WAITING 0
	#define TO_PRINT_BWD_WAITING 0
	#define TO_PRINT_ACT_WAITING 0
	#define TO_PRINT_OPT_STEP_WAITING 0

	#define TO_PRINT_GRAD_BLOCK_WAITING 0
	#define TO_PRINT_FWD_ACT_WAITING 0

	#define TO_PRINT_FWD_PREFETCHING 0
	#define TO_PRINT_BWD_PREFETCHING 0
	#define TO_PRINT_GRAD_BLOCK_PREFETCHING 0

	#define TO_PRINT_ACT_TRANSFERRING 0
	#define TO_PRINT_CTX_TRANSFERRING 0
	#define TO_PRINT_GRAD_TRANSFERRING 0


	#define TO_SAVE_UPDATED_PARAMS 0
	#define TO_SAVE_UPDATED_PARAMS_PATH "updated_params"

	int save_updated_params(Dataflow_Handle * dataflow_handle, int stream_id, int step_num, int layer_num, bool is_head, bool is_embed, void * updated_layer_host, size_t updated_layer_size){

		int ret;

		ret = dataflow_handle->sync_stream(dataflow_handle, stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to sync stream: %d...\n", stream_id);
			return -1;
		}

		char filename[100];
		if (is_head){
			sprintf(filename, "%s/step_%d_head.dat", TO_SAVE_UPDATED_PARAMS_PATH, step_num);
		}
		else if (is_embed){
			sprintf(filename, "%s/step_%d_embed.dat", TO_SAVE_UPDATED_PARAMS_PATH, step_num);
		}
		else{
			sprintf(filename, "%s/step_%d_layer_%d.dat", TO_SAVE_UPDATED_PARAMS_PATH, step_num, layer_num);
		}

		FILE * f = fopen(filename, "w");
		if (!f){
			fprintf(stderr, "Error: failed to open file: %s...\n", filename);
			return -1;
		}

		fwrite(updated_layer_host, updated_layer_size, 1, f);
		fclose(f);

		return 0;
	}

	#define TO_SAVE_GRAD_BLOCKS_PRE_STEP 0
	#define TO_SAVE_GRAD_BLOCKS_PRE_STEP_PATH "grad_blocks_pre_step"

	int save_grad_blocks_pre_step(Dataflow_Handle * dataflow_handle, int stream_id, int step_num, int layer_num, bool is_head, bool is_embed, void * grad_block_dev, size_t grad_block_size){

		int ret;

		
		char * host_grad_block = malloc(grad_block_size);
		if (!host_grad_block){
			fprintf(stderr, "Error: failed to allocate host grad block...\n");
			return -1;
		}

		ret = (dataflow_handle -> submit_outbound_transfer)(dataflow_handle, stream_id, host_grad_block, grad_block_dev, grad_block_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit outbound transfer...\n");
			return -1;
		}

		ret = dataflow_handle->sync_stream(dataflow_handle, stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to sync stream: %d...\n", stream_id);
			return -1;
		}

		
		

		char filename[100];
		if (is_head){
			sprintf(filename, "%s/step_%d_head.dat", TO_SAVE_GRAD_BLOCKS_PRE_STEP_PATH, step_num);
		}
		else if (is_embed){
			sprintf(filename, "%s/step_%d_embed.dat", TO_SAVE_GRAD_BLOCKS_PRE_STEP_PATH, step_num);
		}
		else{
			sprintf(filename, "%s/step_%d_layer_%d.dat", TO_SAVE_GRAD_BLOCKS_PRE_STEP_PATH, step_num, layer_num);
		}
		FILE * f = fopen(filename, "w");
		if (!f){
			fprintf(stderr, "Error: failed to open file: %s...\n", filename);
			return -1;
		}

		fwrite(host_grad_block, grad_block_size, 1, f);
		fclose(f);

		return 0;
	}
	

	uint64_t get_chunk_activations_size(uint64_t chunk_num_tokens, uint64_t model_dim, uint64_t kv_dim, uint64_t num_active_experts, uint64_t expert_dim, DataflowDatatype fwd_dt){

		uint64_t chunk_act_els = 0;

		// input
		chunk_act_els += chunk_num_tokens * model_dim;

		// q proj
		chunk_act_els += chunk_num_tokens * model_dim;

		// k, v projs
		chunk_act_els += 2 * chunk_num_tokens * kv_dim;

		// attn output
		chunk_act_els += chunk_num_tokens * model_dim;

		// attn proj
		chunk_act_els += chunk_num_tokens * model_dim;

		// saved x1 and x3 ffn activations
		chunk_act_els += num_active_experts * 2 * chunk_num_tokens * expert_dim;

		// multiply by dtype size (for now assuming all same dtype, but in reality the ffn dtype might be 8 bit, and attn 16-bit)

		uint64_t dtype_size = dataflow_sizeof_element(fwd_dt);

		return chunk_act_els * dtype_size; 

	}


	int main(int argc, char * argv[]){

		int ret;

		if (argc != 8){
			fprintf(stderr, "Error. Usage: ./transformerRecordThroughput <host_mem_gb> <dev_mem_gb> <seqlen: [num tokens]> <model size billions: [1 | 8]> <num_steps> <num_steps_to_skip_for_recording> <output_filepath>\n");
			return -1;
		}

		int HOST_MEM_GB = atoi(argv[1]);
		int DEV_MEM_GB = atoi(argv[2]);

		int DEMO_SEQ_LEN = atoi(argv[3]);

		int MAX_SEQLEN = DEMO_SEQ_LEN;

		int MODEL_CONFIG_SIZE_B = atoi(argv[4]);
		if (MODEL_CONFIG_SIZE_B != 1 && MODEL_CONFIG_SIZE_B != 8 && MODEL_CONFIG_SIZE_B != 70){
			fprintf(stderr, "Error. Invalid model config size: %d. Choose from (1 or 8)\n", MODEL_CONFIG_SIZE_B);
			return -1;
		}

		int NUM_STEPS = atoi(argv[5]);
		int NUM_STEPS_TO_SKIP_FOR_RECORDING = atoi(argv[6]);

		if (NUM_STEPS_TO_SKIP_FOR_RECORDING >= NUM_STEPS){
			fprintf(stderr, "Error. num_steps_to_skip_for_recording must be less than num_steps...\n");
			return -1;
		}

		char * output_filepath = argv[7];

		char MODEL_PATH[100];
		sprintf(MODEL_PATH, "../../models/%dB", MODEL_CONFIG_SIZE_B);



		// Initialize dataflow handle...

		Dataflow_Handle dataflow_handle;
		
		ComputeType compute_type = COMPUTE_CUDA;
		int device_id = 0;

		// In case we want to create multiple contexts per device, 
		// higher level can create multiple instances of dataflow handles...
		int ctx_id = 0;
		//unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;
		unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC;

		int num_streams = 7;
		int opt_stream_prios[7] = {0, 0, 0, 0, 0, 0, 0};
		char * opt_stream_names[7] = {"Inbound", "Compute", "Outbound", "Peer", "Host Ops", "Inbound Fwd Context", "Loss Update"};


		int inbound_stream_id = 0;
		int compute_stream_id = 1;
		int outbound_stream_id = 2;
		int peer_stream_id = 3;
		int host_ops_stream_id = 4;
		int inbound_fwd_ctx_stream_id = 5;
		int loss_stream_id = 6;

		ret = dataflow_init_handle(&dataflow_handle, compute_type, device_id, 
				ctx_id, ctx_flags, 
				num_streams, opt_stream_prios, opt_stream_names); 
		
		if (ret){
			fprintf(stderr, "Error: failed to init cuda dataflow handle...\n");
			return -1;
		}

		HardwareArchType hardware_arch_type = dataflow_handle.hardware_arch_type;
		
		int MIN_CHUNK_SIZE = 8192;

		float PEAK_BF16_FLOPS;

		switch (hardware_arch_type){
			case BACKEND_ARCH_A100:
				PEAK_BF16_FLOPS = A100_PEAK_BF16_FLOPS;
				break;
			case BACKEND_ARCH_H100:
				PEAK_BF16_FLOPS = H100_PEAK_BF16_FLOPS;
				// need higher arith intensity to saturate (for 1B model at least)
				MIN_CHUNK_SIZE = 16384;
				break;
			case BACKEND_ARCH_RTX_3090:
				PEAK_BF16_FLOPS = RTX_3090_PEAK_BF16_FLOPS;
				break;
			case BACKEND_ARCH_RTX_4090:
				PEAK_BF16_FLOPS = RTX_4090_PEAK_BF16_FLOPS;
				break;
			case BACKEND_ARCH_RTX_5090:
				PEAK_BF16_FLOPS = RTX_5090_PEAK_BF16_FLOPS;
				break;
			default:
				fprintf(stderr, "Error: unknown hardware architecture, cannot set peak bf16 tflops and record MFU...\n");
				PEAK_BF16_FLOPS = 0;
				break;
		}

		// from backend/nvidia/src/ops/src/register_ops/register_ops.c	
		// handles registering external and native ops within cuda_dataflow_ops...
		int added_funcs = dataflow_register_default_ops(&dataflow_handle);
		//printf("Registered %d default ops...\n\n", added_funcs);



		// 64 GB...
		void * host_mem;

		int host_alignment = 4096;
		size_t host_size_bytes = HOST_MEM_GB * (1UL << 30);

		//printf("Allocating host memory of size: %lu...\n", host_size_bytes);

		ret = posix_memalign(&host_mem, host_alignment, host_size_bytes);
		if (ret){
			fprintf(stderr, "Error: posix memalign failed...\n");
			return -1;
		}
		memset(host_mem, 0, host_size_bytes);


		//printf("Registering host memory...\n\n");

		ret = dataflow_handle.enable_access_to_host_mem(&dataflow_handle, host_mem, host_size_bytes, 0);
		if (ret){
			fprintf(stderr, "Registration of host memory failed...\n");
			return -1;
		}

			
		size_t dev_size_bytes = DEV_MEM_GB * (1UL << 30);

		int dev_alignment = 256;

		//printf("Allocating device memory of size: %lu...\n\n", dev_size_bytes);


		void * dev_mem = dataflow_handle.alloc_mem(&dataflow_handle, dev_size_bytes);
		if (!dev_mem){
			fprintf(stderr, "Error: device memory allocation failed...\n");
			return -1;
		}

		void * cur_host_mem = host_mem;
		void * cur_dev_mem = dev_mem;

		size_t used_host_mem = 0;
		size_t used_dev_mem = 0;

		//printf("\n\nInput Parameters:\n\tHost Mem: %d GB\n\tDevice Mem: %d GB\n\tSeqlen (Tokens): %d\n\tModel Size (B): %d\n\nPREPARING DEMO RUN...\n", HOST_MEM_GB, DEV_MEM_GB, DEMO_SEQ_LEN, MODEL_CONFIG_SIZE_B);

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
		int is_causal = 1;
		

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


		int num_shared_experts = 1;
		int num_total_routed_experts = 0;
		int num_active_routed_experts = 0;
		int num_total_active_experts = num_shared_experts + num_active_routed_experts;
		int expert_dim = ffn_dim;



		MoE_Config * moe_config = NULL;


		// setting to host page size.
		// really needs to be 256 in order to use tensor cores
		// depending on filesystem in order to use O_RDONLY | O_DIRECT, alignment may be different...
		
		
		// for now using 0 alignment to directly read from combined file...
		int pointer_alignment = 256;


		// printf("Loading embedding table...\n");

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



		
		

		// printf("Preparing all sys transformer blocks...\n");

		

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

			// printf("Binding sys transformer block #%d...\n", i);
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
		

		// number of elements to pass into optimizer...

		uint64_t embedding_num_els = (uint64_t) vocab_size * (uint64_t) model_dim;
		uint64_t block_num_els = raw_block_size / block_dt_size;
		uint64_t block_aligned_num_els = aligned_block_size / block_dt_size;
		uint64_t all_blocks_num_els = block_num_els * n_layers;
		uint64_t head_num_els = (uint64_t) model_dim * ((uint64_t) 1 + (uint64_t) vocab_size);

		uint64_t all_model_num_els = embedding_num_els + all_blocks_num_els + head_num_els;


		if (TO_PRINT_MODEL_SIZING){

			printf("\nTransformer Block Size (bytes):\n\tRaw: %lu\n\tSize With Matrix Alignment (%d): %lu\n\n", raw_block_size, pointer_alignment, aligned_block_size);

			printf("\n\n\nModel Sizing (bytes):\n\tEmbedding: %lu\n\tBlock: %lu\n\t\tTotal: %lu\n\tHead: %lu\nTOTAL MODEL SIZE: %lu\n\n\n", sys_embedding_table -> embedding_table_size, aligned_block_size, all_blocks_size, combined_head_size, all_model_size);

			printf("\n\n\nModel Parameter Counts:\n\tEmbedding: %lu\n\tBlock: %lu\n\t\tTotal: %lu\n\tBlock Aligned: %lu\n\tHead: %lu\nTOTAL MODEL PARAMETERS: %lu\n\n\n", embedding_num_els, block_num_els, all_blocks_num_els, block_aligned_num_els, head_num_els, all_model_num_els);
			
		}







		// Loading in from checkpoint...

		//printf("\nConfiguring Dataflow & Loading model from checkpoint: %s\n\n", MODEL_PATH);	

		char layer_path[PATH_MAX];

		


		//printf("Loading embedding table...\n");

		sprintf(layer_path, "%s/embed/tok_embeddings.weight", MODEL_PATH);
		FILE * fp = fopen(layer_path, "rb");
		if (!fp){
			fprintf(stderr, "Error: failed to open %s...\n", layer_path);
			return -1;
		}

		size_t read_els = fread(sys_embedding_table -> embedding_table, block_dt_size, embedding_table_els, fp);
		if (read_els != embedding_table_els){
			fprintf(stderr, "Error: failed to read tok_embedding.weight, read_els: %zu, expected: %lu\n", read_els, embedding_table_els);
			return -1;
		}

		fclose(fp);



		

		//printf("Loading all sys transformer blocks...\n");

		
		for (int i = 0; i < n_layers; i++){
			

			sprintf(layer_path, "%s/layers/%d/combined_layer.weight", MODEL_PATH, i);

			//printf("Loading transformer block from: %s...\n", layer_path);
			ret = load_transformer_block(layer_path, sys_blocks[i]);
			if (ret){
				fprintf(stderr, "Error: failed to load transformer block #%d from: %s...\n", i, layer_path);
				return -1;
			}
		}



		//printf("Loading head...\n");

		sprintf(layer_path, "%s/head/combined_head.weight", MODEL_PATH);

		fp = fopen(layer_path, "rb");
		if (!fp){
			fprintf(stderr, "Error: failed to open %s...\n", layer_path);
			return -1;
		}

		read_els = fread(sys_head -> buffer, block_dt_size, combined_head_els, fp);
		if (read_els != combined_head_els) {
			fprintf(stderr, "Error: failed to read combined_head.weight, read_els: %zu, expected: %lu\n", read_els, combined_head_els);
			return -1;
		}

		fclose(fp);


		// SYS GRADIENTS!

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


		// HANDLING OPTIMIZER (assuming same dtype as block for now)...


		// Embedding opt state (only on device)....

		/*
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
		*/

		
		/*
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
		*/



		
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

			cur_host_mem += block_aligned_num_els * opt_mean_dt_size;
			used_host_mem += block_aligned_num_els * opt_mean_dt_size;

			memset(sys_opt_mean_blocks[i] -> buffer, 0, block_aligned_num_els * opt_mean_dt_size);
			

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

			memset(sys_opt_var_blocks[i] -> buffer, 0, block_aligned_num_els * opt_var_dt_size);

			cur_host_mem += block_aligned_num_els * opt_var_dt_size;
			used_host_mem += block_aligned_num_els * opt_var_dt_size;
		}


		// Head opt state (only on device)

		/*
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

		cur_host_mem += head_num_els * opt_mean_dt_size;
		used_host_mem += head_num_els * opt_mean_dt_size;

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

		cur_host_mem += head_num_els * opt_var_dt_size;
		used_host_mem += head_num_els * opt_var_dt_size;
		*/
		

		// DETERMINE NUM DEV BLOCKS, NUM GRAD BLOCKS, and NUM DEV ACTIVATIONS

		uint64_t fwd_block_size = aligned_block_size;
		uint64_t bwd_block_size = aligned_block_bwd_size;

		// DETERMINING CHUNK SIZE AND KERNEL WORKSPACE SIZE!

		int seq_len = DEMO_SEQ_LEN;
		int max_seqlen = seq_len;
		uint64_t min_chunk_size = MIN_CHUNK_SIZE;

		uint64_t chunk_size;

		// If seq_len is greater than min_chunk_size, find the smallest
		// divisor of seq_len that is at least min_chunk_size.
		if (seq_len > min_chunk_size) {
			// Start checking from min_chunk_size upwards.
			for (int chunk_candidate = min_chunk_size; chunk_candidate <= seq_len; chunk_candidate++) {
				if (seq_len % chunk_candidate == 0) {
					chunk_size = chunk_candidate;
					break; // Found the smallest divisor, so we can exit the loop.
				}
			}
		} 
		// Otherwise, find the smallest multiple of seq_len that is
		// at least min_chunk_size.
		else {
			if (min_chunk_size % seq_len == 0) {
				// min_chunk_size is already a multiple of seq_len.
				chunk_size = min_chunk_size;
			} else {
				// Calculate the next multiple of seq_len greater than min_chunk_size.
				// C's integer division automatically handles the floor operation.
				chunk_size = (min_chunk_size / seq_len + 1) * seq_len;
			}
		}

		int max_tokens_per_chunk = chunk_size;

		int num_chunks_per_seq;
		int num_seqs_per_chunk;

		if (seq_len <= chunk_size){
			num_chunks_per_seq = 1;
			num_seqs_per_chunk = chunk_size / seq_len;
		} else {
			num_chunks_per_seq = MY_CEIL(seq_len, chunk_size);
			num_seqs_per_chunk = 1;
		}

		int max_seqs_in_chunk = num_seqs_per_chunk;



		// SAME KERNEL WORKSPACE ACROSS ALL COMPUTATIONS (serialized by stream)!

        // at least use 200MB (ensure enough for rms bwd w and good matmul perf)
		// this is more than enough, except for attention...
		uint64_t baseKernelWorkspaceBytes = 200 * (1UL << 20);

		// attention kernel bwd needs good amount of workspace...
		 // e.g. flash2 with chunksize 8k and seqlen 256k requires 5GB
		// should be an easy API call -- attn bwd most likely the largest consumer...

		uint64_t required_attn_workspace;

		int is_training = 1;

		ret = dataflow_get_attention_workspace_size(&dataflow_handle, block_dt, is_training,
													num_q_heads, num_kv_heads, head_dim,
													max_tokens_per_chunk, max_seqlen, max_seqs_in_chunk,
													is_causal,
													&required_attn_workspace);
		if (ret){
			fprintf(stderr, "Error: failed to get attention workspace size...\n");
			return -1;
		}

		// really could just take the max here, but leaving a little room in case...
		uint64_t kernelWorkspaceBytes = baseKernelWorkspaceBytes + required_attn_workspace;

    	void * kernelWorkspace = cur_dev_mem;
        cur_dev_mem += kernelWorkspaceBytes;
        used_dev_mem += kernelWorkspaceBytes;
        // ensure alignment for matmuls..       
        used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
        cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);



		
		// DETERMINING DEVICE MEMORY PARTITIONING!
		

		uint64_t chunk_act_size = get_chunk_activations_size(chunk_size, model_dim, kv_dim, num_total_active_experts, expert_dim, block_dt);

		//printf("Chunk Act Size: %lu\n", chunk_act_size);

		// int num_chunks = num_chunks_per_seq * seq_groups_per_round;

		// Backing out how many seq grops per round based on equating 
		// the layer size and the amount of activations sent back (if no recomputation)...

		float num_chunks_equal_data_weights = (float) fwd_block_size / (float) chunk_act_size;

		/*
		if (TO_PRINT_SETUP_CONFIG_SUMMARY){
			printf("Num Chunks (with activations of size %lu MB) to equal size of layer: %f\n\n", chunk_act_size / (1UL << 20), num_chunks_equal_data_weights);
		}
		*/

		int num_seq_groups_per_round = MY_MAX(1, round(num_chunks_equal_data_weights / num_chunks_per_seq));

		// the old #define still laying around even though auto-cofig'ed
		int NUM_SEQ_GROUPS_PER_ROUND = num_seq_groups_per_round;


		int num_chunks = num_chunks_per_seq * num_seq_groups_per_round;


		uint64_t context_tokens = MY_MAX(max_tokens_per_chunk, max_seqlen);

		uint64_t context_buffer_size = 2 * (uint64_t) context_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;

		uint64_t context_buffer_bwd_size = 2 * (uint64_t) context_tokens * (uint64_t) kv_dim * (uint64_t) block_bwd_dt_size;

		uint64_t total_sticky_context_size = context_buffer_size + context_buffer_bwd_size;

		uint64_t sticky_embedding_size = embedding_num_els * block_dt_size;
		uint64_t sticky_embedding_grad_size = embedding_num_els * block_bwd_dt_size;
		uint64_t sticky_embedding_opt_size = embedding_num_els * (opt_mean_dt_size + opt_var_dt_size);
		uint64_t total_sticky_embedding_size = sticky_embedding_size + sticky_embedding_grad_size + sticky_embedding_opt_size;

		uint64_t sticky_head_size = head_num_els * block_dt_size;
		uint64_t sticky_head_grad_size = head_num_els * block_bwd_dt_size;
		uint64_t sticky_head_opt_size = head_num_els * (opt_mean_dt_size + opt_var_dt_size);
		uint64_t total_sticky_head_size = sticky_head_size + sticky_head_grad_size + sticky_head_opt_size;

		uint64_t total_base_dev_mem = kernelWorkspaceBytes + total_sticky_context_size + total_sticky_embedding_size + total_sticky_head_size;

		uint64_t sticky_transitions_size = 2 * (uint64_t) num_chunks * chunk_size * (uint64_t) model_dim * block_dt_size;
		uint64_t sticky_dev_logits_size = chunk_size * (uint64_t) vocab_size * block_bwd_dt_size;
		uint64_t sticky_dev_recomputed_buffer_size = 2 * chunk_size * (uint64_t) model_dim * block_dt_size;
		uint64_t sticky_dev_working_grad_act_size = get_chunk_activations_size(chunk_size, model_dim, kv_dim, num_total_active_experts, expert_dim, block_bwd_dt);
		uint64_t sticky_dev_head_act_size = chunk_size * (((uint64_t) model_dim + (uint64_t) vocab_size) * block_dt_size + sizeof(float)); 
		uint64_t sticky_act_workspace_size = chunk_size * ((uint64_t) model_dim + (uint64_t) ffn_dim) * block_dt_size;
		// now also incoporate the other sicy buffers...
		total_base_dev_mem += sticky_transitions_size + sticky_dev_logits_size + sticky_dev_recomputed_buffer_size + sticky_dev_working_grad_act_size + sticky_dev_head_act_size + sticky_act_workspace_size;
		
		// save 200 MB just in case...
		uint64_t extra_padding = 200 * (1UL << 20);
		total_base_dev_mem += extra_padding;

		// printf("Total Base Dev Mem: %lu\n", total_base_dev_mem);

		if (total_base_dev_mem > dev_size_bytes){
			fprintf(stderr, "Error: not enough memory to hold sticky buffers. Requires %lu bytes, but only have %lu bytes", total_base_dev_mem, dev_size_bytes);
			return -1;
		}
		
		uint64_t remain_dev_mem = dev_size_bytes - total_base_dev_mem;

		if (remain_dev_mem <= 0){
			fprintf(stderr, "Error: not enough memory to hold embeddding (& grad + opt) + head (& grad + op) + context fwd + bwd + kernel workspace size. Requires %lu bytes, but only have %lu bytes", total_base_dev_mem, dev_size_bytes);
			return -1;
		}

		

		uint64_t per_layer_act_size = chunk_act_size * (uint64_t) num_chunks;

		uint64_t required_mem = fwd_block_size + bwd_block_size + 2 * chunk_act_size;

		if (remain_dev_mem < required_mem){
			fprintf(stderr, "Error: not enough device memory to run training. Requires 1 fwd block 1 bwd block and 2 chunk activations = %lu bytes, but after sticky buffers only have: %lu...\n", required_mem, remain_dev_mem);
			return -1;
		}

		// TODO: fix this below, for now just assuming even ratio of fwd and bwd...
		// we want it such that we properly ratio the number of saved activations and layers
		// Probably is be better to have more fwd blocks vs. bwd blocks if the model doens't fully fit..
		// so by default making it a 2:1 ratio unless the model fits...
		
		//uint64_t per_layer_full_size = fwd_block_size + per_layer_act_size + 0.5 * bwd_block_size;

		uint64_t per_layer_full_size = fwd_block_size + per_layer_act_size + bwd_block_size;

		//printf("Per Layer Full Size: %lu\n", per_layer_full_size);

		//printf("Remain Dev Mem: %lu\n", remain_dev_mem);

		int num_full_layers_on_dev = MY_MIN(remain_dev_mem / per_layer_full_size, n_layers);

		//printf("Num Full Layers on Dev: %d\n", num_full_layers_on_dev);

		int NUM_DEV_BLOCKS;
		int NUM_DEV_GRAD_BLOCKS;
		int NUM_DEV_ACTIVATION_SLOTS;

		// we already passed check to ensure enough memory for 1 fwd, 1 bwd, 2 chunks, so this means
		// that we don't have enough memory for all chunks...
		if (num_full_layers_on_dev == 0){
			
		

			NUM_DEV_BLOCKS = 1;
			NUM_DEV_GRAD_BLOCKS = 1;
			NUM_DEV_ACTIVATION_SLOTS = (remain_dev_mem - fwd_block_size - bwd_block_size) / chunk_act_size;
			// should never get here because already checked...
			if (NUM_DEV_ACTIVATION_SLOTS < 2){
				fprintf(stderr, "Error: not enough device memory to run training. Requires 1 fwd block 1 bwd block and 2 chunk activations = %lu bytes, but after sticky buffers only have: %lu...\n", required_mem, remain_dev_mem);
				return -1;
			}
			
		}
		else{
			NUM_DEV_BLOCKS = num_full_layers_on_dev;
			NUM_DEV_ACTIVATION_SLOTS = num_full_layers_on_dev * num_chunks;
			NUM_DEV_GRAD_BLOCKS = num_full_layers_on_dev;

			if (NUM_DEV_GRAD_BLOCKS == 0){
				fprintf(stderr, "Error: not enough dev memory to store at least 1 grad block, cannot run training...\n");
				return -1;
			}

			if (num_full_layers_on_dev < n_layers){
				uint64_t base_fwd_space = (uint64_t) NUM_DEV_BLOCKS * fwd_block_size + (uint64_t) NUM_DEV_ACTIVATION_SLOTS * chunk_act_size;
				uint64_t base_bwd_space = ((uint64_t) NUM_DEV_GRAD_BLOCKS * bwd_block_size);
				uint64_t base_space = base_fwd_space + base_bwd_space;


				uint64_t leftover_space = remain_dev_mem - base_space;
				
				if (leftover_space > fwd_block_size){
					NUM_DEV_BLOCKS++;
					leftover_space -= fwd_block_size;
				}

				uint64_t remaining_slots = n_layers * num_chunks - NUM_DEV_ACTIVATION_SLOTS;

				// for a given a layer there is at most num_chunks corresponding slots...
				uint64_t leftover_act_slots = MY_MIN(remaining_slots, leftover_space / chunk_act_size);

				NUM_DEV_ACTIVATION_SLOTS += leftover_act_slots;

			}
		}

		if (NUM_DEV_BLOCKS == 1){
			fprintf(stderr, "!!! WARNING !!!: not enough memory to store mulitple layers on device only holding 1 block and 1 gradient at a time...; performance may be severely impacted...\n");
		}

		if (TO_PRINT_MEMORY_PARTITION_CONFIG){
			printf("\nMEMORY PARTITION CONFIGURATION:\n\tNum Dev Blocks: %d\n\tNum Dev Grad Blocks: %d\n\tNum Dev Activation Slots: %d\n\n", NUM_DEV_BLOCKS, NUM_DEV_GRAD_BLOCKS, NUM_DEV_ACTIVATION_SLOTS);
		}

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

		// printf("Copying embedding table to device...\n");

		ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, embedding_table -> embedding_table, sys_embedding_table -> embedding_table, embedding_table -> embedding_table_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit inbound transfer for embedding table...\n");
			return -1;
		}



		int num_dev_blocks = MY_MIN(n_layers, NUM_DEV_BLOCKS);

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

			// printf("Submitting inbound transfer for dev transformer block #%d...\n", i);

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

		// printf("Submitting inbound transfer for dev head...\n");

		ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, head -> buffer, sys_head -> buffer, combined_head_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit inbound transfer for dev head...\n");
			return -1;
		}

		ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to sync inbound stream...\n");
			return -1;
		}

		// printf("Finished loading model...\n\n");




		// DEV GRADIENTS!
		


		// JUST FOR NOW (while testing for correctness) keeping all block grads on device...
		int num_dev_grad_blocks = MY_MIN(n_layers, NUM_DEV_GRAD_BLOCKS);

		Transformer_Block ** grad_blocks = malloc(num_dev_grad_blocks * sizeof(Transformer_Block *));
		if (!grad_blocks){
			fprintf(stderr, "Error: failed to allocate grad_blocks...\n");
			return -1;
		}

		for (int i = 0; i < num_dev_grad_blocks; i++){
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

		sem_t * is_grad_block_ready = malloc(n_layers * sizeof(sem_t));
		if (!is_grad_block_ready){
			fprintf(stderr, "Error: failed to allocate is_grad_block_ready...\n");
			return -1;
		}

		for (int i = n_layers - 1; i >= 0; i--){
			sem_init(&(is_grad_block_ready[i]), 0, 0);
		}
		
		

		// Embedding Table Gradients (only on device)

		/*
		Transformer_Embedding_Table * sys_grad_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
		if (!sys_grad_embedding_table){
			fprintf(stderr, "Error: failed to allocate sys_grad_embedding_table...\n");
			return -1;
		}

		sys_grad_embedding_table -> config = embedding_config;
		sys_grad_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * block_bwd_dt_size;
		sys_grad_embedding_table -> embedding_table = cur_host_mem;
		
		memset(sys_grad_embedding_table -> embedding_table, 0, sys_grad_embedding_table -> embedding_table_size);

		cur_host_mem += sys_grad_embedding_table -> embedding_table_size;
		used_host_mem += sys_grad_embedding_table -> embedding_table_size;
		*/
		


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
		
		

		// Head Gradients (only on device)

		uint64_t combined_head_bwd_size = combined_head_els * block_bwd_dt_size;

		/*
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


		memset(sys_grad_head -> buffer, 0, combined_head_bwd_size);
		


		cur_host_mem += combined_head_bwd_size;
		used_host_mem += combined_head_bwd_size;
		*/
		


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



		uint64_t model_used_dev_mem = used_dev_mem;
		uint64_t model_used_host_mem = used_host_mem;


		// KEEP Opt State of Embedding and Head on device for now (simpler...)

		// Embedding opt state

		Transformer_Embedding_Table * opt_mean_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
		if (!opt_mean_embedding_table){
			fprintf(stderr, "Error: failed to allocate opt_mean_embedding_table...\n");
			return -1;
		}

		opt_mean_embedding_table -> config = embedding_config;
		opt_mean_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * opt_mean_dt_size;
		opt_mean_embedding_table -> embedding_table = cur_dev_mem;

		cur_dev_mem += opt_mean_embedding_table -> embedding_table_size;
		used_dev_mem += opt_mean_embedding_table -> embedding_table_size;
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);


		Transformer_Embedding_Table * opt_var_embedding_table = malloc(sizeof(Transformer_Embedding_Table));
		if (!opt_var_embedding_table){
			fprintf(stderr, "Error: failed to allocate opt_var_embedding_table...\n");
			return -1;
		}

		opt_var_embedding_table -> config = embedding_config;
		opt_var_embedding_table -> embedding_table_size = (uint64_t) vocab_size * (uint64_t) model_dim * opt_var_dt_size;
		opt_var_embedding_table -> embedding_table = cur_dev_mem;

		cur_dev_mem += opt_var_embedding_table -> embedding_table_size;
		used_dev_mem += opt_var_embedding_table -> embedding_table_size;
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

		// Head opt state

		Transformer_Head * opt_mean_head = malloc(sizeof(Transformer_Head));
		if (!opt_mean_head){
			fprintf(stderr, "Error: failed to allocate opt_mean_head...\n");
			return -1;
		}

		opt_mean_head -> fwd_dt = block_dt;
		opt_mean_head -> bwd_dt = block_bwd_dt;
		opt_mean_head -> compute_dt = compute_dt;
		opt_mean_head -> eps = eps;
		opt_mean_head -> embedding_config = embedding_config;
		opt_mean_head -> buffer = cur_dev_mem;
		opt_mean_head -> w_head_norm = opt_mean_head -> buffer;
		opt_mean_head -> w_head = opt_mean_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) opt_mean_dt_size;

		cur_dev_mem += head_num_els * opt_mean_dt_size;
		used_dev_mem += head_num_els * opt_mean_dt_size;
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

		Transformer_Head * opt_var_head = malloc(sizeof(Transformer_Head));
		if (!opt_var_head){
			fprintf(stderr, "Error: failed to allocate opt_var_head...\n");
			return -1;
		}

		opt_var_head -> fwd_dt = block_dt;
		opt_var_head -> bwd_dt = block_bwd_dt;
		opt_var_head -> compute_dt = compute_dt;
		opt_var_head -> eps = eps;
		opt_var_head -> embedding_config = embedding_config;
		opt_var_head -> buffer = cur_dev_mem;
		opt_var_head -> w_head_norm = opt_var_head -> buffer;
		opt_var_head -> w_head = opt_var_head -> w_head_norm + (uint64_t) model_dim * (uint64_t) opt_var_dt_size;

		cur_dev_mem += head_num_els * opt_var_dt_size;
		used_dev_mem += head_num_els * opt_var_dt_size;
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);


		uint64_t dev_embed_head_opt_state_size = used_dev_mem - model_used_dev_mem;

		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Dev Embed Head Opt State Size: %.3f GB\n", (float) dev_embed_head_opt_state_size / (1024.0 * 1024.0 * 1024.0));
		}




		
		
		
		
		
		
		

		
		
		









	
		
		
		char inp_file_path[PATH_MAX];
		sprintf(inp_file_path, "%s", TOKEN_IDS_PATH);


		int num_tokens_example_seq = NUM_TOKENS_EXAMPLE_SEQ;

		uint32_t * sys_token_ids = malloc(num_tokens_example_seq * sizeof(uint32_t));

		fp = fopen(inp_file_path, "rb");
		if (!fp){
			fprintf(stderr, "Error: failed to open %s...\n", inp_file_path);
			return -1;
		}

		read_els = fread(sys_token_ids, sizeof(uint32_t), num_tokens_example_seq, fp);
		if (read_els != num_tokens_example_seq){
			fprintf(stderr, "Error: failed to read %s, read_els: %zu, expected: %d\n", inp_file_path, read_els, num_tokens_example_seq);
			return -1;
		}
		fclose(fp);
		
		

		uint32_t * sys_labels = malloc(num_tokens_example_seq * sizeof(uint32_t));

		sprintf(inp_file_path, "%s", TOKEN_LABELS_PATH);

		fp = fopen(inp_file_path, "rb");
		if (!fp){
			fprintf(stderr, "Error: failed to open %s...\n", inp_file_path);
			return -1;
		}

		read_els = fread(sys_labels, sizeof(uint32_t), num_tokens_example_seq, fp);
		if (read_els != num_tokens_example_seq){
			fprintf(stderr, "Error: failed to read %s, read_els: %zu, expected: %d\n", inp_file_path, read_els, num_tokens_example_seq);
			return -1;
		}
		fclose(fp);

		if (num_tokens_example_seq < seq_len){
			sys_token_ids = realloc(sys_token_ids, seq_len * sizeof(uint32_t));
			if (!sys_token_ids){
				fprintf(stderr, "Error: failed to realloc sys_token_ids...\n");
				return -1;
			}

			sys_labels = realloc(sys_labels, seq_len * sizeof(uint32_t));
			if (!sys_labels){
				fprintf(stderr, "Error: failed to realloc sys_labels...\n");
				return -1;
			}

			int remain_tokens = seq_len;

			uint32_t * cur_loc_sys_token_ids = sys_token_ids;
			uint32_t * cur_loc_sys_labels = sys_labels;

			int new_tokens;

			while (remain_tokens > 0){

				if (remain_tokens < num_tokens_example_seq){
					new_tokens = remain_tokens;
					
				}
				else{
					new_tokens = num_tokens_example_seq;
				}

				memcpy(cur_loc_sys_token_ids, sys_token_ids, new_tokens * sizeof(uint32_t));
				memcpy(cur_loc_sys_labels, sys_labels, new_tokens * sizeof(uint32_t));

				cur_loc_sys_token_ids += new_tokens;
				cur_loc_sys_labels += new_tokens;

				remain_tokens -= new_tokens;
			}
			
		}

		// ensuring we can populate the seq batch/chunk with correct amount of tokens...
		if (num_seqs_per_chunk > 1){
			sys_token_ids = realloc(sys_token_ids, num_tokens_example_seq * num_seqs_per_chunk * sizeof(uint32_t));
			if (!sys_token_ids){
				fprintf(stderr, "Error: failed to realloc sys_token_ids...\n");
				return -1;
			}

			sys_labels = realloc(sys_labels, num_tokens_example_seq * num_seqs_per_chunk * sizeof(uint32_t));
			if (!sys_labels){
				fprintf(stderr, "Error: failed to realloc sys_labels...\n");
				return -1;
			}

			for (int i = 1; i < num_seqs_per_chunk; i++){
				memcpy(sys_token_ids + i * num_tokens_example_seq, sys_token_ids, num_tokens_example_seq * sizeof(uint32_t));
				memcpy(sys_labels + i * num_tokens_example_seq, sys_labels, num_tokens_example_seq * sizeof(uint32_t));
			}
		}



		// Now we can prepare seq batch...

		//printf("\n\n\nPreparing seq batch...\n");

		

		uint64_t metadata_buffer_size = get_seq_batch_metadata_buffer_size(num_seqs_per_chunk, max_tokens_per_chunk);


		if (TO_PRINT_BATCH_CONFIG){
			printf("Batch Config:\n\tTotal Tokens: %d\n\tNum Seqs Per Chunk: %d\n\n\n", max_tokens_per_chunk, num_seqs_per_chunk);
		}

		Seq_Batch ** seq_batches = malloc(num_chunks * sizeof(Seq_Batch *));
		if (!seq_batches){
			fprintf(stderr, "Error: failed to allocate seq_batches...\n");
			return -1;
		}

		int max_total_local_expert_tokens = max_tokens_per_chunk;


		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("MEMORY BREAKDOWN...\n\n");


			printf("Model:\n\tUsed Host Mem: %.3f GB\n\tUsed Dev Mem: %.3f GB\n", (float) model_used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) model_used_dev_mem / (1024.0 * 1024.0 * 1024.0));
		}


		uint32_t ** chunk_sys_token_ids = malloc(num_chunks * sizeof(uint32_t *));
		uint32_t ** chunk_sys_labels = malloc(num_chunks * sizeof(uint32_t *));
		int ** chunk_sys_seq_positions = malloc(num_chunks * sizeof(int * ));
		
		for (int i = 0; i < num_chunks; i++){
			chunk_sys_token_ids[i] = cur_host_mem;
			cur_host_mem += max_tokens_per_chunk * sizeof(uint32_t);
			used_host_mem += max_tokens_per_chunk * sizeof(uint32_t);
			chunk_sys_labels[i] = cur_host_mem;
			cur_host_mem += max_tokens_per_chunk * sizeof(uint32_t);
			used_host_mem += max_tokens_per_chunk * sizeof(uint32_t);
			chunk_sys_seq_positions[i] = cur_host_mem;
			cur_host_mem += max_tokens_per_chunk * sizeof(int);
			used_host_mem += max_tokens_per_chunk * sizeof(int);
		}





		// these are offsets relative to local q
		// for all chunks except last should be [0, max_tokens_per_chunk
		// for last chunk should be [0, remain_tokens]
		int * cur_sys_q_seq_offsets = malloc((num_seqs_per_chunk + 1) * sizeof(int));
		// this should be [max_tokens_per_chunk]
		// for all chunks except last
		// for last chunk should be [remain_tokens]
		int * cur_sys_q_seq_lens = malloc(num_seqs_per_chunk * sizeof(int));

		// these are offsets relative to the global k context
		// thus for chunk 0 should be [0, max_tokens_per_chunk]
		// for chunk 1 should be [0, 2 * max_tokens_per_chunk]
		// etc.
		// until last chunk should be [0, num_tokens_example_seq]
		int * cur_sys_k_seq_offsets = malloc((num_seqs_per_chunk + 1) * sizeof(int));
		// this is size of keys to look over
		// thus for chunk 0 should be [max_tokens_per_chunk]
		// for chunk 1 should be [2 * max_tokens_per_chunk]
		// etc.
		// until last chunk should be [num_tokens_example_seq]
		int * cur_sys_k_seq_lens = malloc(num_seqs_per_chunk * sizeof(int));

		// the location where to copy token ids from
		uint32_t * cur_ref_sys_token_ids = sys_token_ids;
		uint32_t * cur_ref_sys_labels = sys_labels;
		
		// will contain pointers into the pinned slab
		uint32_t * cur_sys_token_ids;
		uint32_t * cur_sys_labels;
		
		int * cur_sys_seq_positions;


		
		int chunk_id;

		int chunk_tokens;
		int cur_token;


		for (int seq_group = 0; seq_group < num_seq_groups_per_round; seq_group++){

			// for now make every sequence the same tokens...
			cur_ref_sys_token_ids = sys_token_ids;
			cur_ref_sys_labels = sys_labels;

			cur_token = 0;

			for (int c = 0; c < num_chunks_per_seq; c++){

				chunk_id = seq_group * num_chunks_per_seq + c;

				
				cur_sys_token_ids = chunk_sys_token_ids[chunk_id];
				cur_sys_labels = chunk_sys_labels[chunk_id];

				// just assume for now new the number of chunks per seq evenly divides the seq len
				// we already ensure that if multiple seqs fit in chunk the original example input sequence was extended....

				memcpy(cur_sys_token_ids, cur_ref_sys_token_ids, chunk_size * sizeof(uint32_t));
				memcpy(cur_sys_labels, cur_ref_sys_labels, chunk_size * sizeof(uint32_t));


				// for now make every chuk full and the same size
				chunk_tokens = chunk_size;

				seq_batches[chunk_id] = malloc(sizeof(Seq_Batch));
				if (!seq_batches[chunk_id]){
					fprintf(stderr, "Error: failed to allocate seq_batch...\n");
					return -1;
				}

				ret = init_seq_batch_offsets(seq_batches[chunk_id], chunk_tokens, num_seqs_per_chunk, &(sys_blocks[0] -> config), max_total_local_expert_tokens);
				if (ret){
					fprintf(stderr, "Error: failed to init seq_batch offsets...\n");
					return -1;
				}

				ret = bind_seq_batch_metadata_buffer(seq_batches[chunk_id], cur_dev_mem, metadata_buffer_size);
				if (ret){
					fprintf(stderr, "Error: failed to bind seq_batch metadata buffer...\n");
					return -1;
				}

				cur_dev_mem += metadata_buffer_size;
				used_dev_mem += metadata_buffer_size;


				// ensure alignment for matmuls..
				used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
				cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);


				// populating the seq positions slab that will be saved in seq batch struct
				cur_sys_seq_positions = chunk_sys_seq_positions[chunk_id];

				cur_sys_q_seq_offsets[0] = 0;
				cur_sys_k_seq_offsets[0] = 0;

				if (num_seqs_per_chunk > 1){
					for (int s_in_chunk = 0; s_in_chunk < num_seqs_per_chunk; s_in_chunk++){

						for (int j = 0; j < seq_len; j++){
							cur_sys_seq_positions[s_in_chunk * seq_len + j] = j;
						}

						cur_sys_q_seq_offsets[s_in_chunk + 1] = seq_len * (s_in_chunk + 1);
						cur_sys_k_seq_offsets[s_in_chunk + 1] = seq_len * (s_in_chunk + 1);
						cur_sys_q_seq_lens[s_in_chunk] = seq_len;
						cur_sys_k_seq_lens[s_in_chunk] = seq_len;
					}
				}
				else{

					for (int j = 0; j < chunk_tokens; j++){
						cur_sys_seq_positions[j] = cur_token;
						cur_token++;
					}

					cur_sys_q_seq_offsets[1] = chunk_tokens;
					cur_sys_q_seq_lens[0] = chunk_tokens;
					cur_sys_k_seq_offsets[1] = cur_token;
					cur_sys_k_seq_lens[0] = cur_token;
				}

				ret = populate_seq_batch_metadata_buffer(&dataflow_handle, inbound_stream_id, 
												seq_batches[chunk_id],
												cur_host_mem, metadata_buffer_size,
												seq_group, chunk_id, chunk_tokens, num_seqs_per_chunk,
												cur_sys_token_ids, cur_sys_labels,
												cur_sys_seq_positions, 
												cur_sys_q_seq_offsets, cur_sys_q_seq_lens,
												cur_sys_k_seq_offsets, cur_sys_k_seq_lens);

				if (ret){
					fprintf(stderr, "Error: failed to populate seq_batch metadata buffer for chunk #%d...\n", chunk_id);
					return -1;
				}

				ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id);
				if (ret){
					fprintf(stderr, "Error: failed to sync inbound stream after populating seq_batch metadata buffer for chunk #%d...\n", chunk_id);
					return -1;
				}

				// ADVANCE TO NEXT CHUNK

				cur_ref_sys_token_ids += chunk_tokens;
				cur_ref_sys_labels += chunk_tokens;
			}
		}

		free(cur_sys_q_seq_offsets);
		free(cur_sys_q_seq_lens);
		free(cur_sys_k_seq_offsets);
		free(cur_sys_k_seq_lens);

		// now already copied the inputs into the seq batch structs...
		// so are done with the raw token ids and labels...
		free(sys_token_ids);
		free(sys_labels);

		FILE * f_token_ids;
		size_t wrote_els;
		FILE * f_labels;


		uint64_t metadata_used_dev_mem = used_dev_mem - model_used_dev_mem;
		uint64_t metadata_used_host_mem = used_host_mem - model_used_host_mem;


		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Metadata:\n\tUsed Host Mem: %.3f GB\n\tUsed Dev Mem: %.3f GB\n", (float) metadata_used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) metadata_used_dev_mem / (1024.0 * 1024.0 * 1024.0));
		}

		
		char file_path[PATH_MAX];
		for (int i = 0; i < NUM_RAW_CHUNK_IDS_LABELS_TO_SAVE; i++){

			sprintf(file_path, "test_transformer_data/chunk_%d_token_ids_uint32.dat", i);


			// writing back in same dir as other data for clarity...
			f_token_ids = fopen(file_path, "wb");
			if (!f_token_ids){
				fprintf(stderr, "Error: failed to open %s...\n", file_path);
				return -1;
			}
		
			size_t wrote_els = fwrite(seq_batches[i] -> sys_token_ids, sizeof(uint32_t), seq_batches[i] -> total_tokens, f_token_ids);
			if (wrote_els != seq_batches[i] -> total_tokens){
				fprintf(stderr, "Error: failed to write %s, sys_wrote_els: %zu, expected: %d\n", file_path, wrote_els, seq_batches[i] -> total_tokens);
				return -1;
			}
			fclose(f_token_ids);


			sprintf(file_path, "test_transformer_data/chunk_%d_labels_uint32.dat", i);

			f_labels = fopen(file_path, "wb");
			if (!f_labels){
				fprintf(stderr, "Error: failed to open %s...\n", file_path);
				return -1;
			}
		
			wrote_els = fwrite(seq_batches[i] -> sys_labels, sizeof(uint32_t), seq_batches[i] -> total_tokens, f_labels);
			if (wrote_els != seq_batches[i] -> total_tokens){
				fprintf(stderr, "Error: failed to write %s, sys_wrote_els: %zu, expected: %d\n", file_path, wrote_els, seq_batches[i] -> total_tokens);
				return -1;
			}
			fclose(f_labels);

		}

		







		


		// CONTEXT AND GRAD CONTEXTS!


		// CREATE DEVICE CONTEXT THAT ALL CHUNKS WILL REFERENCE...



		// FOR SINGLE DEVICE CASE WE ONLY NEED TWO, 
		// BUT FOR MULTI-DEVICE WE MAY HAVE CONCURRENT FORWARD AND BACKWARD CHUNKS
		// REFERRING TO DIFFERENT SEQUENCES, THUS THE BACKWARDS SEQUENCE NEEDS BOTH FWDS AND BWDS
		// AND THE FORWARDS CHUNKS NEEDS A SEPERATE WORKSPACE TO ADAVNCE....


		Seq_Batch_Context * fwd_context = malloc(sizeof(Seq_Batch_Context));
		if (!fwd_context){
			fprintf(stderr, "Error: failed to allocate fwd_context...\n");
			return -1;
		}


		/* FOR OVERLAPPING FWD/BWD SEQS WE NEED AN ADDITONAL BUFFER FOR FWD CONTEXT TO ADVANCE... */
		// not implementeing this for now...
		/*
		Seq_Batch_Context * working_fwd_context = malloc(sizeof(Seq_Batch_Context));
		if (!working_fwd_context){
			fprintf(stderr, "Error: failed to allocate working_fwd_context...\n");
			return -1;
		}
		*/

		Seq_Batch_Context * bwd_context = malloc(sizeof(Seq_Batch_Context));
		if (!bwd_context){
			fprintf(stderr, "Error: failed to allocate bwd_context...\n");
			return -1;
		}


		


		fwd_context -> contextBuffer = cur_dev_mem;
		fwd_context -> contextBufferBytes = context_buffer_size;
		
		fwd_context -> cur_tokens_populated = 0;
		fwd_context -> total_context_tokens = context_tokens;

		fwd_context -> x_k = cur_dev_mem;
		fwd_context -> x_v = fwd_context -> x_k + (uint64_t) context_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;

		cur_dev_mem += context_buffer_size;
		used_dev_mem += context_buffer_size;
		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);


		/*
		working_fwd_context -> contextBuffer = cur_dev_mem;
		working_fwd_context -> contextBufferBytes = context_buffer_size;
		
		working_fwd_context -> cur_tokens_populated = 0;
		working_fwd_context -> total_context_tokens = context_tokens;

		working_fwd_context -> x_k = cur_dev_mem;
		working_fwd_context -> x_v = working_fwd_context -> x_k + (uint64_t) context_tokens * (uint64_t) kv_dim * (uint64_t) block_dt_size;

		cur_dev_mem += context_buffer_size;
		used_dev_mem += context_buffer_size;
		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);
		*/

		
		(bwd_context) -> contextBuffer = cur_dev_mem;
		(bwd_context) -> contextBufferBytes = context_buffer_bwd_size;
		
		(bwd_context) -> cur_tokens_populated = 0;
		(bwd_context) -> total_context_tokens = context_tokens;

		(bwd_context) -> x_k = cur_dev_mem;
		(bwd_context) -> x_v = (bwd_context) -> x_k + (uint64_t) context_tokens * (uint64_t) kv_dim * (uint64_t) block_bwd_dt_size;

		cur_dev_mem += context_buffer_bwd_size;
		used_dev_mem += context_buffer_bwd_size;
		// ensure alignment for matmuls..
		used_dev_mem += 256 - ((uint64_t) cur_dev_mem % 256);
		cur_dev_mem = (void *) ((uint64_t)(cur_dev_mem + 255) & ~255UL);

		ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, fwd_context -> contextBuffer, 0, fwd_context -> contextBufferBytes);
		if (ret){
			fprintf(stderr, "Error: failed to set mem for fwd_context...\n");
			return -1;
		}

		ret = dataflow_handle.set_mem(&dataflow_handle, inbound_stream_id, bwd_context -> contextBuffer, 0, bwd_context -> contextBufferBytes);
		if (ret){
			fprintf(stderr, "Error: failed to set mem for bwd_context...\n");
			return -1;
		}
		
		uint64_t context_used_dev_mem = used_dev_mem - model_used_dev_mem - metadata_used_dev_mem;
		uint64_t context_used_host_mem = used_host_mem - model_used_host_mem - metadata_used_host_mem;

		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Context:\n\tUsed Host Mem: %.3f GB\n\tUsed Dev Mem: %.3f GB\n", (float) context_used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) context_used_dev_mem / (1024.0 * 1024.0 * 1024.0));
		}
		

		
		/*

		// For now just calling sync_stream to inspect outputs (with cuda-gdb...)
		printf("Waiting for data transfer of metadata buffer to complete...\n\n");

		ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to sync inbound stream...\n");
			return -1;
		}
		*/


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

		uint64_t block_transition_used_dev_mem = used_dev_mem - model_used_dev_mem - metadata_used_dev_mem - context_used_dev_mem;
		uint64_t block_transition_used_host_mem = used_host_mem - model_used_host_mem - metadata_used_host_mem - context_used_host_mem;
		
		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Block Transition:\n\tUsed Host Mem: %.3f GB\n\tUsed Dev Mem: %.3f GB\n", (float) block_transition_used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) block_transition_used_dev_mem / (1024.0 * 1024.0 * 1024.0));
		}
		
		
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


		int total_acts = n_layers * num_chunks;

		int num_saved_activation_buffers = MY_MIN(total_acts, NUM_DEV_ACTIVATION_SLOTS);

		int total_home_acts = total_acts - num_saved_activation_buffers;
		int total_dev_acts = num_saved_activation_buffers;


		// DETERMINE THE savedActivationLevel's based on host mem!



		if (used_host_mem > host_size_bytes){
			fprintf(stderr, "Error: not enough host memory to fit any activations. Already used %.3f GB of %.3f GB...\n", (float) used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) host_size_bytes / (1024.0 * 1024.0 * 1024.0));
			return -1;
		}

		// Shouldn't need this but calculation is going wrong somewhere off slightly...
		uint64_t extra_host_mem = 200 * (1UL << 20);
		uint64_t remaining_host_mem = host_size_bytes - used_host_mem - extra_host_mem;

		uint64_t full_saved_size = get_seq_batch_saved_activations_buffer_size(seq_batches[0], SAVED_ACTIVATION_LEVEL_FULL);
		uint64_t inp_attn_saved_size = get_seq_batch_saved_activations_buffer_size(seq_batches[0], SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY);
		uint64_t inp_only_saved_size = get_seq_batch_saved_activations_buffer_size(seq_batches[0], SAVED_ACTIVATION_LEVEL_INP_ONLY);

		// First ensure that we can fit all using inp only...
		uint64_t all_inp_only_size = inp_only_saved_size * total_home_acts;
		if (all_inp_only_size > remaining_host_mem){
			fprintf(stderr, "Error: not enough host memory to fit all activations using inp only. Needed %.3f GB, remaining: %.3f GB...\n", (float) all_inp_only_size / (1024.0 * 1024.0 * 1024.0), (float) remaining_host_mem / (1024.0 * 1024.0 * 1024.0));
			return -1;
		}

		// Given "total_home_acts" = #, "remaining_host_mem" = #
		// Given inp_only_saved_size < inp_attn_saved_size < full_saved_size
		// Objective:
		// 1.) ensure that sum of num_inp_only_saved, num_inp_attn_saved, num_full_saved == total_home_acts
		// 2.) Ensure that all total_home_acts can at least fit if they are inp_only
		// 3.) Before using full_saved ensure that all total_home_acts can use inp_attn_saved
		// 4.) If not all can use inp_attn_saved then use as many as can
		// 5.) If all can use inp_attn_saved but not enough for all to use full_saved, then use as many full_saved as can...
		
		
		// First, see how many activations can be upgraded from inp_only to inp_attn_saved
        uint64_t mem_for_attn_upgrades = remaining_host_mem - all_inp_only_size;
        uint64_t inp_attn_upgrade_cost = inp_attn_saved_size - inp_only_saved_size;
        uint64_t num_upgraded_to_attn = 0;
        if (inp_attn_upgrade_cost > 0) {
            num_upgraded_to_attn = mem_for_attn_upgrades / inp_attn_upgrade_cost;
        }

        // We can't upgrade more than the total number of activations
        if (num_upgraded_to_attn > total_home_acts) {
            num_upgraded_to_attn = total_home_acts;
        }

        int num_inp_attn_saved = num_upgraded_to_attn;
        int num_inp_only_saved = total_home_acts - num_inp_attn_saved;

        // Now, from the ones upgraded to inp_attn, see how many can be further upgraded to full_saved
        uint64_t current_mem_usage = (num_inp_attn_saved * inp_attn_saved_size) + (num_inp_only_saved * inp_only_saved_size);
        uint64_t mem_for_full_upgrades = remaining_host_mem - current_mem_usage;
        uint64_t full_upgrade_cost = full_saved_size - inp_attn_saved_size;
        uint64_t num_upgraded_to_full = 0;
        if (full_upgrade_cost > 0) {
            num_upgraded_to_full = mem_for_full_upgrades / full_upgrade_cost;
        }
        
        // We can't upgrade more than what we've already allocated for inp_attn
        if (num_upgraded_to_full > num_inp_attn_saved) {
            num_upgraded_to_full = num_inp_attn_saved;
        }

        int num_full_saved = num_upgraded_to_full;

		// Assign the saved activation levels based on remaining memory...

		// Do a round robin assignment to balance the I/O load...

		// Given: num_inp_only_saved > 0 => num_full_saved = 0; if num_full_saved > 0 => num_inp_only_saved = 0;
		// num_inp_only_saved + num_inp_attn_saved + num_full_saved = total_home_acts

		SavedActivationLevel * saved_activation_levels = malloc(total_acts * sizeof(SavedActivationLevel));
		if (!saved_activation_levels){
			fprintf(stderr, "Error: failed to allocate saved_activation_level...\n");
			return -1;
		}

		for (int i = 0; i < total_acts; i++){
			saved_activation_levels[i] = SAVED_ACTIVATION_LEVEL_NONE;
		}

		// Now bound by the total number of home acts...

		int remain_home_acts = total_home_acts;
		num_full_saved  = MY_MIN(num_full_saved, remain_home_acts);
		remain_home_acts -= num_full_saved;
		num_inp_attn_saved = MY_MIN(num_inp_attn_saved, remain_home_acts);
		remain_home_acts -= num_inp_attn_saved;
		assert(num_inp_only_saved >= remain_home_acts);
		num_inp_only_saved = remain_home_acts;

		int full_to_assign = num_full_saved;
		int attn_to_assign = num_inp_attn_saved;
		int only_to_assign = num_inp_only_saved;

		int inp_only_seq_lens[MAX_INP_ONLY_CHUNKS];
		for (int i = 0; i < MAX_INP_ONLY_CHUNKS; i++){
			inp_only_seq_lens[i] = 0;
		}
		
		int cur_inp_only_seq_len = 0;

		int cur_inp_only_assigned = 0;

		float min_window_flops;

		int prior_seq_len = 0;
		int cur_seq_len = 0;
		for (int i = 0; i < total_dev_acts; i++){
			min_window_flops += get_chunk_block_flops(chunk_size, prior_seq_len, DEMO_SEQ_LEN, model_dim, kv_dim, is_causal, num_shared_experts, num_total_routed_experts, num_active_routed_experts, expert_dim);
			if (chunk_size + prior_seq_len < DEMO_SEQ_LEN){
				prior_seq_len += chunk_size;
			}
			else{
				prior_seq_len = 0;
			}
		}

		

		float runtime_dev_window_sec = min_window_flops / (FLOP_EFFICIENCY_ESTIMATE * PEAK_BF16_FLOPS);

		float theo_link_speed_bytes_per_sec = get_home_link_speed_bytes_per_sec(dataflow_handle.pcie_link_width, dataflow_handle.pcie_link_gen);

		float link_speed_bytes_per_sec = PCIE_LINK_EFFICIENCY * theo_link_speed_bytes_per_sec;

		float max_bytes_per_window_saved = runtime_dev_window_sec * link_speed_bytes_per_sec;

		int full_windows_saved = total_home_acts / total_dev_acts;
		int non_window_home_acts = total_home_acts % total_dev_acts;

		// CORRECT FOR LINK CONGESTION! 
		// potentially downgrade full saved activations to inp+attn if too much data is being transferred

		int full_per_window;
		int attn_per_window;
		if (full_windows_saved > 0){
			full_per_window = full_to_assign / full_windows_saved;
			attn_per_window = attn_to_assign / full_windows_saved;
		}
		else{
			full_per_window = 0;
			attn_per_window = 0;
		}

		if (only_to_assign == 0 && (DEMO_SEQ_LEN <= chunk_size) && full_windows_saved > 0){
			
			// might need to downgrade full to assign based on max_bytes_per_window_saved

			uint64_t full_window_attn_inp_only_size = inp_attn_saved_size * total_dev_acts;
			if (full_window_attn_inp_only_size > max_bytes_per_window_saved){
				// downgrade all full windows to attn
				attn_to_assign += full_to_assign;
				full_to_assign = 0;
				full_per_window = 0;
				attn_per_window = total_dev_acts;
			}
			else{
				uint64_t room_for_full = max_bytes_per_window_saved - full_window_attn_inp_only_size;
				uint64_t num_space_for_full_per_window = room_for_full / full_upgrade_cost;
				if (num_space_for_full_per_window == 0){
					full_to_assign = MY_MIN(full_to_assign, non_window_home_acts);
					attn_to_assign = total_home_acts - full_to_assign;
					full_per_window = 0;
					attn_per_window = total_dev_acts;
				}
				else{
					
					int target_attn_to_assign = (total_dev_acts - num_space_for_full_per_window) * full_windows_saved;
					int target_full_to_assign = total_home_acts - target_attn_to_assign;
					
					// now need to downgrade in order to prevent congestion (recompute is better than idle)
					if (target_full_to_assign < full_to_assign){
						full_to_assign = target_full_to_assign;
						attn_to_assign = total_home_acts - full_to_assign;
						full_per_window = num_space_for_full_per_window;
						attn_per_window = total_dev_acts - full_per_window;
					}
					else{
						full_per_window = full_to_assign / full_windows_saved;
						attn_per_window = total_dev_acts - full_per_window;
					}
				}
			}

			// Reset how many of each type we are assigning...
			num_full_saved = full_to_assign;
			num_inp_attn_saved = attn_to_assign;
		}

		// for each of the full windows, do the same ordering

		// make sure to recompute attention on the short seq portions...

		if (only_to_assign > 0){

			int only_chunks_per_layer = only_to_assign / n_layers;
			int only_remain = only_to_assign % n_layers;

			for (int k = 0; k < n_layers; k++){

				for (int chunk_id = 0; chunk_id < only_chunks_per_layer; chunk_id++){
					saved_activation_levels[k*num_chunks + chunk_id] = SAVED_ACTIVATION_LEVEL_INP_ONLY;
					if (chunk_id == 0){
						cur_inp_only_seq_len = MY_MIN(chunk_size, DEMO_SEQ_LEN);
					}
					else{
						cur_inp_only_seq_len = (chunk_id + 1) * chunk_size;
					}

					inp_only_seq_lens[cur_inp_only_assigned] = cur_inp_only_seq_len;
					cur_inp_only_assigned++;
				}

				if (only_remain > 0){
					int chunk_id = only_chunks_per_layer;
					for (int k = 0; k < only_remain; k++){
						saved_activation_levels[k*num_chunks + chunk_id] = SAVED_ACTIVATION_LEVEL_INP_ONLY;

					}
					if (chunk_id == 0){
						cur_inp_only_seq_len = MY_MIN(chunk_size, DEMO_SEQ_LEN);
					}
					else{
						cur_inp_only_seq_len = (chunk_id + 1) * chunk_size;
					}

					inp_only_seq_lens[cur_inp_only_assigned] = cur_inp_only_seq_len;
					cur_inp_only_assigned++;
				}
			}

			// assign the rest to inp_attn...
			for (int i = 0; i < total_home_acts; i++){
				if (saved_activation_levels[i] == SAVED_ACTIVATION_LEVEL_NONE){
					saved_activation_levels[i] = SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY;
				}
			}
		}
		// only distributing between full and inp+attn, so it doesn't really matter
		// which chunk saves/recomputes besides for load balancing I/O.. (thus can alternate).
		else{

			for (int i = 0; i < full_windows_saved; i++){

				int cur_window_full_to_assign = full_per_window;
				int cur_window_attn_to_assign = attn_per_window;

				for (int j = i * total_dev_acts; j < (i + 1) * total_dev_acts; j++){

					if ((i % 2 == 1) && (cur_window_full_to_assign > 0)){
						saved_activation_levels[j] = SAVED_ACTIVATION_LEVEL_FULL;
						cur_window_full_to_assign--;
						full_to_assign--;
					}
					else if ((i % 2 == 0) && (cur_window_attn_to_assign > 0)){
						saved_activation_levels[j] = SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY;
						cur_window_attn_to_assign--;
						attn_to_assign--;
					}
					else{
						if (cur_window_full_to_assign > 0){
							saved_activation_levels[j] = SAVED_ACTIVATION_LEVEL_FULL;
							cur_window_full_to_assign--;
							full_to_assign--;
						}
						else{
							saved_activation_levels[j] = SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY;
							cur_window_attn_to_assign--;
							attn_to_assign--;
						}
					}
				}
			}

			for (int j = full_windows_saved * total_dev_acts; j < total_home_acts; j++){
				if (full_to_assign > 0){
					saved_activation_levels[j] = SAVED_ACTIVATION_LEVEL_FULL;
					full_to_assign--;
				}
				else{
					saved_activation_levels[j] = SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY;
					attn_to_assign--;
				}
			}
		}

		// Don't save any of the activations that are at end of orderining
		// as these will stay on device...
		for (int i = total_home_acts; i < total_acts; i++){
			saved_activation_levels[i] = SAVED_ACTIVATION_LEVEL_NONE;
		}
		
		

		// If we have enough remaining host mem to fit the full activations buffer for any of the chunks, then do so...
		


		// Saved Actviations will live on device and might be transferred back to host and retrieved prior to bwd pass...

		int num_sys_saved_activations = total_home_acts;

		Seq_Batch_Saved_Activations * sys_saved_activations = malloc(num_sys_saved_activations * sizeof(Seq_Batch_Saved_Activations));
		if (!sys_saved_activations){
			fprintf(stderr, "Error: failed to allocate sys_saved_activations...\n");
			return -1;
		}

		uint64_t saved_activations_buffer_size;

		SavedActivationLevel cur_saved_activation_level;

		for (int i = 0; i < num_sys_saved_activations; i++){

			cur_saved_activation_level = saved_activation_levels[i];

			saved_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[(i % num_chunks)], cur_saved_activation_level);
			ret = bind_seq_batch_saved_activations_buffer(seq_batches[(i % num_chunks)], &(sys_saved_activations[i]), cur_host_mem, cur_saved_activation_level, saved_activations_buffer_size, i);
			if (ret){
				fprintf(stderr, "Error: failed to bind seq_batch saved_activations buffer...\n");
				return -1;
			}

			cur_host_mem += saved_activations_buffer_size;
			used_host_mem += saved_activations_buffer_size;
			sys_saved_activations[i].recomputed_activations = NULL;
		}




		// BLOCKS OPT STATE...


		// At the end of each round we will load in the opt state from sys, compute, and then send back
		// When we are donig this step there are no other pieces of data needed on the device
		// (all of the contexts/grads/activations) so we can alias that memory to be used for loading in the opt state
		// when the time comes...
		void * cur_opt_state_loc = cur_dev_mem;
		uint64_t dev_opt_state_size = dev_size_bytes - used_dev_mem;

		// determine how many layers worth of opt state we can fit in the available space

		uint64_t block_mean_opt_state_size = block_aligned_num_els * opt_mean_dt_size;
		uint64_t block_var_opt_state_size = block_aligned_num_els * opt_var_dt_size;

		uint64_t block_opt_state_size = block_mean_opt_state_size + block_var_opt_state_size;

		int num_dev_opt_blocks = dev_opt_state_size / block_opt_state_size;

		if (num_dev_opt_blocks == 0){
			fprintf(stderr, "Error: not enough device memory to fit even a single layer of opt state (available: %.3f GB, needed: %.3f GB)...\n", (float) dev_opt_state_size / (1024.0 * 1024.0 * 1024.0), (float) block_opt_state_size / (1024.0 * 1024.0 * 1024.0));
			return -1;
		}

		num_dev_opt_blocks = MY_MIN(n_layers, num_dev_opt_blocks);

		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Num Dev Opt Blocks (Aliasing into Activation Buffers no longer needed at end of step...): %d\n", num_dev_opt_blocks);
		}

		Transformer_Block ** opt_mean_blocks = malloc(num_dev_opt_blocks * sizeof(Transformer_Block *));
		if (!opt_mean_blocks){
			fprintf(stderr, "Error: failed to allocate opt_mean_blocks...\n");
			return -1;
		}

		Transformer_Block ** opt_var_blocks = malloc(num_dev_opt_blocks * sizeof(Transformer_Block *));
		if (!opt_var_blocks){
			fprintf(stderr, "Error: failed to allocate opt_var_blocks...\n");
			return -1;
		}

		uint64_t opt_state_alias_used_size = 0;

		// WE ARE BINDING TO CUR OPT_STATE LOC!
		// ENSURE THAT THIS DOESN'T COUNT TOWARDS THE CUR DEV MEM BECAUSE it is only used rarely when the parts below are not needed...

		for (int i = 0; i < num_dev_opt_blocks; i++){
			opt_mean_blocks[i] = init_transformer_block(i, opt_mean_dt, compute_dt,
															norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
															eps, theta,
															num_q_heads, num_kv_heads, head_dim,
															ffn_dim,
															moe_config,
															pointer_alignment);
			if (!opt_mean_blocks[i]){
				fprintf(stderr, "Error: failed to init opt_mean_block #%d...\n", i);
				return -1;
			}

			ret = bind_transformer_block(cur_opt_state_loc, opt_mean_blocks[i]);
			if (ret){
				fprintf(stderr, "Error: failed to bind opt_mean_block #%d...\n", i);
				return -1;
			}

			cur_opt_state_loc += block_aligned_num_els * opt_mean_dt_size;
			opt_state_alias_used_size += block_aligned_num_els * opt_mean_dt_size;
			opt_state_alias_used_size += 256 - ((uint64_t) cur_opt_state_loc % 256);
			cur_opt_state_loc = (void *) ((uint64_t)(cur_opt_state_loc + 255) & ~255UL);

			opt_var_blocks[i] = init_transformer_block(i, opt_var_dt, compute_dt,
															norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
															eps, theta,
															num_q_heads, num_kv_heads, head_dim,
															ffn_dim,
															moe_config,
															pointer_alignment);
			if (!opt_var_blocks[i]){
				fprintf(stderr, "Error: failed to init opt_var_block #%d...\n", i);
				return -1;
			}

			ret = bind_transformer_block(cur_opt_state_loc, opt_var_blocks[i]);
			if (ret){
				fprintf(stderr, "Error: failed to bind opt_var_block #%d...\n", i);
				return -1;
			}

			cur_opt_state_loc += block_aligned_num_els * opt_var_dt_size;
			opt_state_alias_used_size += block_aligned_num_els * opt_var_dt_size;
			opt_state_alias_used_size += 256 - ((uint64_t) cur_opt_state_loc % 256);
			cur_opt_state_loc = (void *) ((uint64_t)(cur_opt_state_loc + 255) & ~255UL);
		}

		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Opt State Alias Used Size: %.3f GB\n", (float) opt_state_alias_used_size / (1024.0 * 1024.0 * 1024.0));
		}

		sem_t * is_opt_layer_ready = malloc(n_layers * sizeof(sem_t));
		if (!is_opt_layer_ready){
			fprintf(stderr, "Error: failed to allocate is_opt_layer_ready...\n");
			return -1;
		}

		for (int i = 0; i < n_layers; i++){
			sem_init(&is_opt_layer_ready[i], 0, 0);
		}


		// ensure that if we need to refetch the next set of blocks for forward pass
		// that we can do so...
		sem_t * is_block_home = malloc(n_layers * sizeof(sem_t));
		if (!is_block_home){
			fprintf(stderr, "Error: failed to allocate is_block_home...\n");
			return -1;
		}

		for (int i = 0; i < n_layers; i++){
			sem_init(&is_block_home[i], 0, 0);
		}
		
		// during the opt step, if there are overlapping layers that we need to prefetch 
		// but are already on device we need to reshuffle them in order to start every 
		// step with layer id = 0 existing at index 0 in the blocks array...
		Transformer_Block ** temp_blocks = malloc(num_dev_blocks * sizeof(Transformer_Block *));
		if (!temp_blocks){
			fprintf(stderr, "Error: failed to allocate temp_blocks...\n");
			return -1;
		}






		Seq_Batch_Saved_Activations * saved_activations = malloc(num_saved_activation_buffers * sizeof(Seq_Batch_Saved_Activations));
		if (!saved_activations){
			fprintf(stderr, "Error: failed to allocate saved_activations...\n");
			return -1;
		}


		// HAVE THE DEVICE SAVED ACTIVATIONS BUFFERS BE FULL SO THEY CAN BE POPULATED DURING FWD PASS...
		
		for (int i = 0; i < num_saved_activation_buffers; i++){

			saved_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[(i % num_chunks)], SAVED_ACTIVATION_LEVEL_FULL);

			ret = bind_seq_batch_saved_activations_buffer(seq_batches[(i % num_chunks)], &(saved_activations[i]), cur_dev_mem, SAVED_ACTIVATION_LEVEL_FULL, saved_activations_buffer_size, i);
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

		sem_t * is_saved_act_home = malloc(total_home_acts * sizeof(sem_t));
		if (!is_saved_act_home){
			fprintf(stderr, "Error: failed to allocate is_saved_act_home...\n");
			return -1;
		}

		for (int i = 0; i < total_home_acts; i++){
			sem_init(&(is_saved_act_home[i]), 0, 0);
			sem_post(&(is_saved_act_home[i]));
		}
		

		int * dev_act_ind = malloc(num_chunks * n_layers * sizeof(int));
		if (!dev_act_ind){
			fprintf(stderr, "Error: failed to allocate dev_act_ind...\n");
			return -1;
		}

		for (int i = 0; i < num_chunks * n_layers; i++){
			dev_act_ind[i] = -1;
		}
		

		
		
		
		
		
		
		
		
		
		
		// We will maintain contains for each corresponding (chunk_id, layer_id) pair and then 
		// match these with the saved activations buffer when it is ready...

		Transformer_Block_Activations ** activations = malloc(total_acts * sizeof(Transformer_Block_Activations *));
		if (!activations){
			fprintf(stderr, "Error: failed to allocate top level activations container...\n");
			return -1;
		}

		for (int i = 0; i < total_acts; i++){
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
		uint64_t grad_activations_buffer_size = get_seq_batch_saved_activations_buffer_size(seq_batches[0], SAVED_ACTIVATION_LEVEL_FULL);

		// using seq batch 0 offsets is safe because all seq batches are either the same or smaller (in terms of total tokens, thus saved activations offsets...)
		ret = bind_seq_batch_saved_activations_buffer(seq_batches[0], grad_saved_activations, cur_dev_mem, SAVED_ACTIVATION_LEVEL_FULL, grad_activations_buffer_size, 0);
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
		// head_activations -> num_tokens = num_tokens_example_seq;
		

		// PREPARING SPECIAL MODEL OUTPUT STRUCT...


		// in case we want to save the logits...
		uint64_t logits_size = (uint64_t) max_tokens_per_chunk * (uint64_t) vocab_size * block_bwd_dt_size;

		/*
		void * sys_logits = cur_host_mem;
		cur_host_mem += logits_size;
		used_host_mem += logits_size;
		*/

		






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


		uint64_t activations_used_dev_mem = used_dev_mem - model_used_dev_mem - metadata_used_dev_mem - context_used_dev_mem - block_transition_used_dev_mem;
		uint64_t activations_used_host_mem = used_host_mem - model_used_host_mem - metadata_used_host_mem - context_used_host_mem - block_transition_used_host_mem;

		if (TO_PRINT_MEMORY_BREAKDOWN_VERBOSE){
			printf("Activations (+ additional workspace):\n\t(Chunk, Layer) Activation Size: %.3f GB\n\tHost # Acts: %d\n\tDev # Acts: %d\n\tUsed Host Mem: %.3f GB\n\tUsed Dev Mem: %.3f GB\n\n\n", saved_activations_buffer_size / (1024.0 * 1024.0 * 1024.0), total_home_acts, total_dev_acts, (float) activations_used_host_mem / (1024.0 * 1024.0 * 1024.0), (float) activations_used_dev_mem / (1024.0 * 1024.0 * 1024.0));
		}





		float used_host_mem_gb = (float) used_host_mem / (1024.0 * 1024.0 * 1024.0);
		float used_dev_mem_gb = (float) used_dev_mem / (1024.0 * 1024.0 * 1024.0);

		if (TO_PRINT_SETUP_CONFIG_SUMMARY){
			printf("Setup Complete!\n\n");
			
			printf("\nMEMORY USAGE (GB):\n\tHost: %.3f\n\tDevice: %.3f\n\n", used_host_mem_gb, used_dev_mem_gb);
		}

		if ((used_host_mem > host_size_bytes) || (used_dev_mem > dev_size_bytes)) {
			fprintf(stderr, "ERROR. Cannot run with current configuration of %d dev parameter blocks,%d dev activation slots, %d dev block grads, %d min chunk size (=> xhunk size %lu), and %d seq groups per round => %d chunks per round...\n", NUM_DEV_BLOCKS, NUM_DEV_ACTIVATION_SLOTS, NUM_DEV_GRAD_BLOCKS, MIN_CHUNK_SIZE, chunk_size, NUM_SEQ_GROUPS_PER_ROUND, num_chunks);
			
			if (used_host_mem > host_size_bytes){
				fprintf(stderr, "\nHost Memory Overflow: Have %.3f GB allocated, but requires %.3f GB with current setting...\n", (float) host_size_bytes / (1024.0 * 1024.0 * 1024.0), (float) used_host_mem / (1024.0 * 1024.0 * 1024.0));
			}

			if (used_dev_mem > dev_size_bytes){
				fprintf(stderr, "\nDevice Memory Overflow: Have %.3f GB allocated, but requires %.3f GB with current setting...\n", (float) dev_size_bytes / (1024.0 * 1024.0 * 1024.0), (float) used_dev_mem / (1024.0 * 1024.0 * 1024.0));
			}
			
			return -1;
		}


		// TRAINING LOOP BELOW....


		Transformer_Block_Activations * cur_activations;
		Seq_Batch_Saved_Activations * cur_fwd_activations;


		
		
		
		
		
		
		



		// ADAM OPTIMIZER PARAMS...

		// learning rate should have a set schedule...
		float lr = 2e-5;
		float beta1 = 0.9;
		float beta2 = 0.999;
		float weight_decay = 1e-4;
		float epsilon = 1e-8;


		void * sys_activation_home;
		void * cur_stream_state;

		int cur_act = 0;




		int working_layer_ind = 0;
		int replacement_layer_ind = 0;
		int next_layer_id = num_dev_blocks;
		int working_act_buffer_ind = 0;

		Transformer_Block * working_block;

		int final_saved_act_buffer_ind = -1;


		Transformer_Block_Transition * final_block_output_transition;

		Transformer_Block_Transition * grad_stream_from_head;


		int working_grad_block_ind = num_dev_grad_blocks - 1;
		int replacement_grad_layer_ind = num_dev_grad_blocks - 1;
		// every round starts with the last grad block...
		int next_grad_block_id = n_layers - 1;
		int cur_fwd_prefetch_act_ind;


		void * fwd_x_k_global = fwd_context -> x_k;
		void * fwd_x_v_global = fwd_context -> x_v;

		void * global_fwd_ctx_k_dest;
		void * global_fwd_ctx_v_dest;

		Transformer_Block * working_grad_block;

		int cur_tokens;
		int fwd_ctx_tokens_replaced;

		int cur_global_token_replacement_ind;

		Seq_Batch_Saved_Activations * prior_group_sys_saved_activations;
		int prior_group_dev_saved_act_ind;
		Seq_Batch_Saved_Activations * prior_group_dev_saved_activations;

		int next_opt_layer_id = 0;
		int replacement_opt_layer_ind = 0;
		int working_opt_layer_ind = 0;

		Transformer_Block * working_opt_mean_block;
		Transformer_Block * working_opt_var_block;



		// In order to appropriately scale the gradients during the head stage we multiply 
		// loss (chunk size x vocab size) matrix by (1 / total_pred_tokens_in_step)

		// this should be set at the beginning of each step...

		int num_steps = NUM_STEPS;

		// Determine number of rounds per step to be a target duration...

		int seqs_per_round = num_seq_groups_per_round * num_seqs_per_chunk;

		float per_seq_flops = get_seq_flops(MAX_SEQLEN, vocab_size, model_dim, kv_dim, is_causal, num_shared_experts, num_total_routed_experts, num_active_routed_experts, expert_dim, n_layers, 
											NULL, NULL, NULL, NULL, NULL, NULL);

		float flops_per_round = per_seq_flops * seqs_per_round;

		float target_duration_per_step_s = TARGET_DURATION_PER_STEP_S;
		
		if (MODEL_CONFIG_SIZE_B == 8){
			target_duration_per_step_s *= 8;
		}

		float flop_efficiency_estimate = FLOP_EFFICIENCY_ESTIMATE;

		float per_round_duration_s_est = flops_per_round / (flop_efficiency_estimate * PEAK_BF16_FLOPS);

		int num_rounds_per_step = MY_MAX(1, round(target_duration_per_step_s / per_round_duration_s_est));

		//printf("NUM ROUNDS PER STEP: %d\n", num_rounds_per_step);


		uint64_t loss_tracker_size = num_steps * num_rounds_per_step * num_chunks * sizeof(float);
		float * sys_loss_tracker = (float *) cur_host_mem;
		cur_host_mem += loss_tracker_size;
		used_host_mem += loss_tracker_size;

		Print_Round_Loss_Host_Op_Args * print_round_loss_op_buffer = calloc(num_steps * num_rounds_per_step, sizeof(Print_Round_Loss_Host_Op_Args));
		if (!print_round_loss_op_buffer){
			fprintf(stderr, "Error: failed to allocate print_round_loss_op_buffer...\n");
			return -1;
		}


		Print_Chunk_Loss_Host_Op_Args * print_chunk_loss_op_buffer = calloc(num_steps * num_rounds_per_step * num_chunks, sizeof(Print_Chunk_Loss_Host_Op_Args));
		if (!print_chunk_loss_op_buffer){
			fprintf(stderr, "Error: failed to allocate print_chunk_loss_op_buffer...\n");
			return -1;
		}



		float * dev_loss_vec;
		int total_pred_tokens_in_step = num_rounds_per_step * num_seq_groups_per_round * num_chunks_per_seq * chunk_size;
	
		int round_tokens = num_seq_groups_per_round * num_chunks_per_seq * chunk_size;
		
		int total_train_tokens = num_steps * total_pred_tokens_in_step;
		
		

		// Prepare to save down metrics...
		Step_Throughput_Host_Op_Args * step_throughput_op_buffers = calloc(num_steps, sizeof(Step_Throughput_Host_Op_Args));
		if (!step_throughput_op_buffers){
			fprintf(stderr, "Error: failed to allocate step_throughput_op_buffers...\n");
			return -1;
		}

		// Populate with static model info...

		for (int t = 0; t < num_steps; t++){
			step_throughput_op_buffers[t].model_dim = model_dim;
			step_throughput_op_buffers[t].kv_dim = kv_dim;
			step_throughput_op_buffers[t].is_causal = is_causal;
			step_throughput_op_buffers[t].num_shared_experts = num_shared_experts;
			step_throughput_op_buffers[t].num_total_routed_experts = num_total_routed_experts;
			step_throughput_op_buffers[t].num_active_routed_experts = num_active_routed_experts;
			step_throughput_op_buffers[t].expert_dim = expert_dim;
			step_throughput_op_buffers[t].vocab_size = vocab_size;
			step_throughput_op_buffers[t].num_layers = n_layers;
			step_throughput_op_buffers[t].peak_hardware_flop_rate = PEAK_BF16_FLOPS;
			step_throughput_op_buffers[t].to_print_metrics = TO_PRINT_THROUGHPUT_METRICS;
			step_throughput_op_buffers[t].to_print_verbose = TO_PRINT_THROUGHPUT_METRICS_VERBOSE;
			

			// To determine recomputation flops...
			step_throughput_op_buffers[t].num_seqs_per_round = seqs_per_round;
			step_throughput_op_buffers[t].num_rounds_per_step = num_rounds_per_step;
			step_throughput_op_buffers[t].chunk_size = chunk_size;
			step_throughput_op_buffers[t].num_inp_attn_saved = num_inp_attn_saved;
			step_throughput_op_buffers[t].num_inp_only_saved = num_inp_only_saved;
			memcpy(step_throughput_op_buffers[t].inp_only_seq_lens, inp_only_seq_lens, MAX_INP_ONLY_CHUNKS * sizeof(int));
		}

		// JUST FOR DEMO we are using the same sequence distribution for every round and eveyr step...

		// seqs per chunk = 1 if seq uses >= 1 chunks, otherwise packing multiple seqs per chunk...
		
		int seqs_per_step = seqs_per_round * num_rounds_per_step;

		if (TO_PRINT_SETUP_CONFIG_SUMMARY){
			printf("SETUP CONFIG OVERVIEW:\n");
			printf("\tKernel Workspace Bytes: %lu\n", kernelWorkspaceBytes);
			printf("\tChunk size: %lu\n", chunk_size);
			printf("\tChunks per round: %d\n", num_chunks);
			printf("\tRound tokens: %d\n", round_tokens);
			printf("\tNum rounds per step: %d\n", num_rounds_per_step);
			printf("\tTotal tokens per step: %d\n", total_pred_tokens_in_step);
			printf("\tTotal train tokens: %d\n\n", total_train_tokens);

			printf("\tSeqlen: %d\n", MAX_SEQLEN);
			printf("\tSeqs per round: %d\n", seqs_per_round);
			printf("\tSeqs per step: %d\n\n", seqs_per_step);

			printf("\tPCIe Connection:\n");
			printf("\t\tLink Gen: %u\n", dataflow_handle.pcie_link_gen);
			printf("\t\t# Lanes: %u\n", dataflow_handle.pcie_link_width);
			printf("\t\tTheo BW: %d GB/s\n\n", (int) (theo_link_speed_bytes_per_sec / (1024.0 * 1024.0 * 1024.0)));

			printf("\tHost Activations: %d\n", total_home_acts);
			printf("\t\tNum Full Saved Activations: %d\n", num_full_saved);
			printf("\t\tNum Inp + Attn Saved Activations: %d\n", num_inp_attn_saved);
			printf("\t\tNum Inp Only Saved Activations: %d\n", num_inp_only_saved);
			printf("\tDevice Activations: %d\n\n", total_dev_acts);

			printf("# Model Params: %.2fB\n\n", all_model_num_els / 1e9);
		}
		

		int * seqlens = calloc(seqs_per_step, sizeof(int));
		if (!seqlens){
			fprintf(stderr, "Error: failed to allocate seqlens...\n");
			return -1;
		}

		// all seqs use the same length...
		for (int i = 0; i < seqs_per_step; i++){
			seqlens[i] = MAX_SEQLEN;
		}

		
		
		
		

		//printf("------ STARTING TRAINING ------\n\n");

		int cur_round_num_seqs;
		int cur_round_num_chunks;
		int cur_round_num_tokens;

		// start profiling...

		//printf("Starting profiling...\n\n");
		ret = dataflow_handle.profiler.start();
		if (ret){
			fprintf(stderr, "Error: failed to start profiling...\n");
			return -1;
		}

		char profile_msg[256];

		for (int t = 1; t < num_steps + 1; t++){

			sprintf(profile_msg, "Step #%d", t);
			dataflow_handle.profiler.range_push(profile_msg);

			// start the metrics for this step...
			ret = dataflow_submit_start_step_metrics_host(&dataflow_handle, compute_stream_id, 
							start_step_metrics, &(step_throughput_op_buffers[t - 1]),
							t, seqs_per_step, seqlens);
			if (ret){
				fprintf(stderr, "Error: failed to submit start step metrics for step #%d...\n", t);
				return -1;
			}

			// at init, or after each step, reset the layer indices...
			// (if # layers non-divisible by # blocks in dev, then these might get of out whack, 
			// within rounds, but that is ok, because they are properly being ping/ponged 
			// these are referring to indices within the dev blocks on device...
			working_layer_ind = 0;
			replacement_layer_ind = 0;


			for (int r = 0; r < num_rounds_per_step; r++){

				// in reality these wouldn't be the same every round...
				cur_round_num_seqs = seqs_per_round;
				cur_round_num_chunks = num_chunks;
				cur_round_num_tokens = round_tokens;

				sprintf(profile_msg, "Round #%d", r);
				dataflow_handle.profiler.range_push(profile_msg);

				if (r == 0){
					// no need to prefetch grad blocks for first round...
					// because gradients will be 0...
					for (int i = n_layers - 1; i >= (n_layers - num_dev_grad_blocks); i--){
						sem_post(&(is_grad_block_ready[i]));
					}
				}

				// ensure the next grad block is set properly...
				next_grad_block_id = n_layers - 1;
				replacement_grad_layer_ind = num_dev_grad_blocks - 1;
				// this works downwards from the last grad block...
				// and then works updwards doring opt step...
				working_grad_block_ind = num_dev_grad_blocks - 1;

				for (int i = 0; i < num_chunks * n_layers; i++){
					dev_act_ind[i] = -1;
					// reset for each home act...
					if (i < total_home_acts){
						// we will reset this every round
						sem_wait(&(is_saved_act_home[i]));
					}
				}


				// ADVANCE ALL SEQUENCES FORWARD...

				// FWD PASS...

				cur_act = 0;
				working_act_buffer_ind = 0;
				final_saved_act_buffer_ind = -1;

				// EMBEDDING...

				sprintf(profile_msg, "Fwd");
				dataflow_handle.profiler.range_push(profile_msg);


				sprintf(profile_msg, "Embedding");
				dataflow_handle.profiler.range_push(profile_msg);


				// might be mulitiple seqs per chunk => chunks per seq == 1
				for (int seq_group = 0; seq_group < num_seq_groups_per_round; seq_group++){

					sprintf(profile_msg, "Seq Group: %d", seq_group);
					dataflow_handle.profiler.range_push(profile_msg);

					
					for (int c = 0; c < num_chunks_per_seq; c++){

						chunk_id = seq_group * num_chunks_per_seq + c;

						sprintf(profile_msg, "Chunk: %d", chunk_id);
						dataflow_handle.profiler.range_push(profile_msg);

						if (TO_PRINT_SUBMITTING){
							printf("\n\nSubmitting embedding for seq group #%d, chunk #%d...\n\n", seq_group, chunk_id);
						}
							
						model_input -> seq_batch = seq_batches[chunk_id];

						seq_batches[chunk_id] -> seq_id = seq_group;
						seq_batches[chunk_id] -> chunk_id = chunk_id;

						ret = dataflow_submit_transformer_embedding(&dataflow_handle, compute_stream_id,
																model_input,
																embedding_table,
																&(block_transitions[2 * chunk_id]));
						if (ret){
							fprintf(stderr, "Error: failed to submit transformer embedding...\n");
							return -1;
						}

						// set the context

						
						seq_batches[chunk_id] -> context = fwd_context;

						dataflow_handle.profiler.range_pop();


						/*
						// special case for single-device where we know the first seq group will be the first sequence going backwards,
						// so we can have it directly populate the fwd_context buffer (the one that prefetches prior context info, needed for bwd pass)
						// otherwise the last layer's context will be populated with the last sequence instead of first
						// (which is ok for single-device, where we could schedule seqs in reverse-order going backwards, 
						// but not ideal for overlapping fwd/bwd chunks in multi-device ring)
						if (seq_group == 0){
							seq_batches[chunk_id] -> context = fwd_context;
						}
						*/
					}

					dataflow_handle.profiler.range_pop();
				}

				// pop from "Embedding"
				dataflow_handle.profiler.range_pop();

				sprintf(profile_msg, "Layers");
				dataflow_handle.profiler.range_push(profile_msg);
				// 2.) DOING CORE BLOCKS...

				// reset what the next layer to fetch going forwards is...
				next_layer_id = num_dev_blocks;

				for (int k = 0; k < n_layers; k++){

					sprintf(profile_msg, "Layer: %d", k);
					dataflow_handle.profiler.range_push(profile_msg);


					// Ensure layer is ready before submitting ops...

					if (TO_PRINT_FWD_WAITING){
						printf("\n\n[Fwd] Waiting for layer id #%d to be ready (at index %d)...\n\n", k, working_layer_ind);
					}

					sem_wait(&(is_block_ready[k]));

					working_block = blocks[working_layer_ind];
					working_block -> layer_id = k;

					for (int seq_group = 0; seq_group < num_seq_groups_per_round; seq_group++){

						sprintf(profile_msg, "Seq Group: %d", seq_group);
						dataflow_handle.profiler.range_push(profile_msg);

						for (int c = 0; c < num_chunks_per_seq; c++) {

							chunk_id = seq_group * num_chunks_per_seq + c;

							sprintf(profile_msg, "Chunk: %d", chunk_id);
							dataflow_handle.profiler.range_push(profile_msg);

							if (TO_PRINT_ACT_WAITING){
								printf("\n\n[Fwd] Waiting for saved activation buffer at index %d to be ready...\n\n", working_act_buffer_ind);
							}

							sem_wait(&(is_saved_activation_ready[working_act_buffer_ind]));

							if (TO_PRINT_SUBMITTING){
								printf("\n\nSubmitting fwd block %d for seq group #%d, chunk #%d...!\n\n", k, seq_group, chunk_id);
							}

							cur_activations = activations[k * num_chunks + chunk_id];

							cur_activations -> working_activations = &(saved_activations[working_act_buffer_ind]);
							cur_activations -> working_activations -> layer_id = k;
							cur_activations -> working_activations -> seq_batch = seq_batches[chunk_id];

							// set the context
							seq_batches[chunk_id] -> context = fwd_context;

							ret = dataflow_submit_transformer_block(&dataflow_handle, compute_stream_id, 
															&(block_transitions[2 * chunk_id + (k % 2)]), 
															working_block, 
															cur_activations, 
															&(block_transitions[2 * chunk_id + ((k + 1) % 2)])) ;

							if (ret){
								fprintf(stderr, "Error: failed to submit transformer block for seq group #%d, chunk #%d, block #%d...\n", seq_group, chunk_id, k);
								return -1;
							}


							// ENSURE WE SAVE THE STATE AFTER ALL OPS HAVE BEEN QUEUED...

							cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
							if (!cur_stream_state){
								fprintf(stderr, "Error: failed to get stream state for seq group #%d, chunk #%d, layer id #%d's activations...\n", seq_group, chunk_id, k);
								return -1;
							}


							// send back activation buffer needed during bwd pass...

							// otherwise we have the correct activations saved on device and don't need to send back...
							if (cur_act < total_home_acts) {

								// ensure depedency is set...

								if (TO_PRINT_ACT_TRANSFERRING){
									printf("Sending seq group #%d, chunk #%d, layer id #%d's activations to host (act #%d of %d)...\n\n", seq_group, chunk_id, k, cur_act, total_acts);
								}


								ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
								if (ret){
									fprintf(stderr, "Error: failed to submit dependency to send seq group #%d, chunk #%d, layer id #%d's activations to host...\n", seq_group, chunk_id, k);
									return -1;
								}

								sys_activation_home = sys_saved_activations[k * num_chunks + chunk_id].savedActivationsBuffer;

								

								ret = (dataflow_handle.submit_outbound_transfer)(&dataflow_handle, outbound_stream_id, sys_activation_home, cur_activations -> working_activations -> savedActivationsBuffer, 
																					sys_saved_activations[k * num_chunks + chunk_id].savedActivationsBufferBytes);

								if (ret){
									fprintf(stderr, "Error: failed to submit outbound transfer to send seq group #%d, chunk #%d, layer id #%d's activations to host...\n", seq_group, chunk_id, k);
									return -1;
								}

								// add back to ready queue...
								ret = (dataflow_handle.submit_host_op)(&dataflow_handle, post_sem_callback, (void *) &(is_saved_activation_ready[working_act_buffer_ind]), outbound_stream_id);
								if (ret){
									fprintf(stderr, "Error: failed to submit host op to enqueue seq group #%d, chunk #%d, layer id #%d's activations...\n", seq_group, chunk_id, k);
									return -1;
								}

								if (TO_PRINT_ACT_TRANSFERRING){
									printf("Submitting host op to enqueue act index %d sys activations as home...\n\n", k * num_chunks + chunk_id);
								}

								ret = (dataflow_handle.submit_host_op)(&dataflow_handle, post_sem_callback, (void *) &(is_saved_act_home[k * num_chunks + chunk_id]), outbound_stream_id);
								if (ret){
									fprintf(stderr, "Error: failed to submit host op to enqueue seq group #%d, chunk #%d, layer id #%d's activations as home...\n", seq_group, chunk_id, k);
									return -1;
								}


								final_saved_act_buffer_ind = k * num_chunks + chunk_id;
							}
							// otherwise we are close to turning around and will save it on device until we reverse...
							else{
								sem_post(&(is_fwd_activations_ready[k * num_chunks + chunk_id]));

								dev_act_ind[k * num_chunks + chunk_id] = working_act_buffer_ind;

								if (TO_PRINT_ACT_TRANSFERRING){
									printf("Saving seq group #%d, chunk #%d, layer id #%d's at working buffer index %d (act #%d of %d)...\n\n", seq_group, chunk_id, k, working_act_buffer_ind, cur_act, total_acts);
								}

								sem_post(&(is_saved_activation_ready[working_act_buffer_ind]));
							}


							cur_act++;

							// ensure we don't increment on the final activation...
							if ((k < (n_layers - 1)) || (chunk_id < (num_chunks - 1))){
								working_act_buffer_ind = (working_act_buffer_ind + 1) % num_saved_activation_buffers;
							}

							// pop from "Chunk: %d"
							dataflow_handle.profiler.range_pop();
						}

						// pop from "Seq Group: %d"
						dataflow_handle.profiler.range_pop();
					}





					// finsihed processing this layer forwards

					// prefetch next layer...
					if (next_layer_id < n_layers){

						if (TO_PRINT_FWD_PREFETCHING){
							printf("\n\nPrefetching next layer id #%d (replacing layer at index %d)...\n\n", next_layer_id, replacement_layer_ind);
						}

						ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
						if (ret){
							fprintf(stderr, "Error: failed to submit dependency to prefetch next layer...\n");
							return -1;
						}

						sprintf(profile_msg, "Prefetching next layer id #%d...", next_layer_id);
						dataflow_handle.profiler.range_push(profile_msg);

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

						dataflow_handle.profiler.range_pop();
								
						// ensure the last layer will be the first one to be used during bwds, don't increment past it...
						if (next_layer_id < (n_layers - 1)){
							replacement_layer_ind = (replacement_layer_ind + 1) % num_dev_blocks;
									
						}
						next_layer_id++;
					}
					// otherwise see if we need to prefetch a grad block...
					else{
						// post that this block is ready for bwd...
						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[k]), compute_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for grad block #%d...\n", k);
							return -1;
						}

						// the first round all gradients will be 0, so no need to prefetch...
						if ((r > 0) && (next_grad_block_id >= (n_layers - num_dev_grad_blocks))){

							if (TO_PRINT_GRAD_BLOCK_PREFETCHING){
								printf("\n\nPrefetching next grad block id #%d (replacing grad block at index %d)...\n\n", next_grad_block_id, replacement_grad_layer_ind);
							}

							// prefetch next grad block...
							ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
							if (ret){
								fprintf(stderr, "Error: failed to submit dependency to prefetch next grad block...\n");
								return -1;
							}

							sprintf(profile_msg, "Prefetching next grad block id #%d...", next_grad_block_id);
							dataflow_handle.profiler.range_push(profile_msg);
							
							ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, grad_blocks[replacement_grad_layer_ind] -> buffer, sys_grad_blocks[next_grad_block_id] -> buffer, aligned_block_bwd_size);
							if (ret){
								fprintf(stderr, "Error: failed to submit inbound transfer for grad block #%d...\n", next_grad_block_id);
								return -1;
							}

							ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[next_grad_block_id]), inbound_stream_id);
							if (ret){
								fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for next grad block...\n");
								return -1;
							}

							dataflow_handle.profiler.range_pop();

							if (next_grad_block_id > 0){
								if (replacement_grad_layer_ind > 0){
									replacement_grad_layer_ind--;
								}
								else{
									replacement_grad_layer_ind = num_dev_grad_blocks - 1;
								}
							}
							next_grad_block_id--;
						}
					}

					// don't move past the the last block as this will be the first one to be used during bwds...
					if (k < (n_layers - 1)){
						working_layer_ind = (working_layer_ind + 1) % num_dev_blocks;
					}

					// pop from added Layer %d
					dataflow_handle.profiler.range_pop();
				}

				// pop from "Layers"
				dataflow_handle.profiler.range_pop();

				// pop from "Fwd"
				dataflow_handle.profiler.range_pop();

				if (r > 0){
					int num_grad_blocks_to_prefetch = (next_grad_block_id + 1) - (n_layers - num_dev_grad_blocks);
					if (num_grad_blocks_to_prefetch > 0){
						sprintf(profile_msg, "Prefetching remaining %d grad blocks to fill buffer...", num_grad_blocks_to_prefetch);
						dataflow_handle.profiler.range_push(profile_msg);

						while (next_grad_block_id >= (n_layers - num_dev_grad_blocks)){
							ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, grad_blocks[replacement_grad_layer_ind] -> buffer, sys_grad_blocks[next_grad_block_id] -> buffer, aligned_block_bwd_size);
							if (ret){
								fprintf(stderr, "Error: failed to submit inbound transfer for grad block #%d...\n", next_grad_block_id);
								return -1;
							}

							ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[next_grad_block_id]), inbound_stream_id);
							if (ret){
								fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for next grad block...\n");
								return -1;
							}

							if (next_grad_block_id > 0){
								if (replacement_grad_layer_ind > 0){
									replacement_grad_layer_ind--;
								}
								else{
									replacement_grad_layer_ind = num_dev_grad_blocks - 1;
								}
							}
							next_grad_block_id--;
						}

						dataflow_handle.profiler.range_pop();
					}
				}
				else{
					next_grad_block_id = n_layers - 1 - num_dev_grad_blocks;
				}

				sprintf(profile_msg, "Head");
				dataflow_handle.profiler.range_push(profile_msg);

				// 3.) NOW DOING HEAD, FOR NOW IN REVERSE SEQUENCE ORDER AND REVERSE ORDER WITHIN SEQUENCE...
				// for multi-device ring with overlapping fwd/bwd chunks it would be better to process in same sequence order,
				// and reverse within sequence because this gets the bwd flow moving faster...

				// Keeping track of stack ordering and act prefetching is easier if everything is fully reversed...
				// Also the fwd context is populated with the correct value...

				for (int seq_group = num_seq_groups_per_round - 1; seq_group >= 0; seq_group--){

					sprintf(profile_msg, "Seq Group: %d", seq_group);
					dataflow_handle.profiler.range_push(profile_msg);

					for (int c = num_chunks_per_seq -1; c >= 0; c--){

						chunk_id = seq_group * num_chunks_per_seq + c;


						sprintf(profile_msg, "Chunk: %d", chunk_id);
						dataflow_handle.profiler.range_push(profile_msg);

						if (TO_PRINT_SUBMITTING){
							printf("\n\nSubmitting head for seq group #%d, chunk #%d...\n\n", seq_group, chunk_id);
						}


						final_block_output_transition = &(block_transitions[2 * chunk_id + (n_layers % 2)]);

						grad_stream_from_head = &(block_transitions[2 * chunk_id + ((n_layers - 1) % 2)]);

						// ensure grad stream is zeroed out...
						ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, grad_stream_from_head -> X, 0, block_transition_size);
						if (ret){
							fprintf(stderr, "Error: failed to zero out grad stream for seq group #%d, chunk #%d before head...\n", seq_group, chunk_id);
							return -1;
						}

						model_output -> seq_batch = seq_batches[chunk_id];
						head_activations -> num_tokens = seq_batches[chunk_id] -> total_tokens;
						head_activations -> total_pred_tokens_in_step = total_pred_tokens_in_step;

						dev_loss_vec = (model_output -> seq_batch -> loss_config).loss_vec;

						// ensure prior loss is zeroed out...
						ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, dev_loss_vec, 0, (head_activations -> num_tokens + 1) * sizeof(float));
						if (ret){
							fprintf(stderr, "Error: failed to zero out loss vec for seq group #%d, chunk #%d before head...\n", seq_group, chunk_id);
							return -1;
						}

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


						// save the loss tracker...
						
						if ((TO_PRINT_CHUNK_LOSS) || (TO_PRINT_ROUND_LOSS)) {
							cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
							if (!cur_stream_state){
								fprintf(stderr, "Error: failed to get stream state for head...\n");
								return -1;
							}
							
							ret = dataflow_handle.submit_dependency(&dataflow_handle, loss_stream_id, cur_stream_state);
							if (ret){
								fprintf(stderr, "Error: failed to submit dependency to send head to host...\n");
								return -1;
							}

							

							ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, loss_stream_id, &(sys_loss_tracker[(t - 1) * num_rounds_per_step * num_chunks + r * num_chunks + chunk_id]), &(dev_loss_vec[head_activations -> num_tokens]), sizeof(float));

							if (ret){
								fprintf(stderr, "Error: failed to submit outbound transfer for loss tracker...\n");
								return -1;
							}
						}

						if (TO_PRINT_CHUNK_LOSS) {

							ret = dataflow_submit_print_chunk_loss_host(&dataflow_handle, loss_stream_id,
													&print_chunk_loss_host, &(print_chunk_loss_op_buffer[(t - 1) * num_rounds_per_step * num_chunks + r * num_chunks + chunk_id]),
													t, r, seq_group, chunk_id, head_activations -> num_tokens, &(sys_loss_tracker[(t - 1) * num_rounds_per_step * num_chunks + r * num_chunks + chunk_id]));

							if (ret){
								fprintf(stderr, "Error: failed to submit print loss host...\n");
								return -1;
							}
						}

						// Ensure that this seq batch's context will refere the bwd context,
						// in order to correctly backprop through the block...
						seq_batches[chunk_id] -> context = bwd_context;

						dataflow_handle.profiler.range_pop();
					}

					dataflow_handle.profiler.range_pop();
				}

				if (TO_PRINT_ROUND_LOSS) {

					ret = dataflow_submit_print_round_loss_host(&dataflow_handle, loss_stream_id,
												&print_round_loss_host, &(print_round_loss_op_buffer[(t - 1) * num_rounds_per_step + r]),
												t, r, cur_round_num_seqs, cur_round_num_chunks, cur_round_num_tokens, &(sys_loss_tracker[(t - 1) * num_rounds_per_step * num_chunks + r * num_chunks]));

					if (ret){
						fprintf(stderr, "Error: failed to submit print step loss host...\n");
						return -1;
					}
				}

				// pop from "Head"
				dataflow_handle.profiler.range_pop();


				// BACKWARDS PASS...

				// send back gradient for head and run optimizer..
				
				// RESET NEXT LAYER ID TO THE LAST FORWARD BLOCK WE DON"T HAVE...!
				next_layer_id = n_layers - 1 - num_dev_blocks;
				cur_fwd_prefetch_act_ind = final_saved_act_buffer_ind;

				// working_layer_ind and replacement_layer_ind should be properly set correctly poniting at last block, now working opposite direction...

				cur_global_token_replacement_ind = 0;
				int submitted_outbound_grad = 0;

				sprintf(profile_msg, "Bwd");
				dataflow_handle.profiler.range_push(profile_msg);

				for (int k = n_layers - 1; k >= 0; k--){

					sprintf(profile_msg, "Layer: %d", k);
					dataflow_handle.profiler.range_push(profile_msg);

					// Ensure we have forward layer...

					sprintf(profile_msg, "Waiting for layer id #%d to be ready", k);
					dataflow_handle.profiler.range_push(profile_msg);

					if (TO_PRINT_BWD_WAITING){
						printf("\n\n[Bwd] Waiting for layer id #%d to be ready (at index %d)...\n\n", k, working_layer_ind);
					}

					sem_wait(&(is_block_ready[k]));
					
					int block_ready_val;
					sem_getvalue(&(is_block_ready[k]), &block_ready_val);

					assert(block_ready_val == 0);

					dataflow_handle.profiler.range_pop();

					// if we are less than num dev blocks we will need this going forwards either in next round or in the step following if last round...
					if (k < num_dev_blocks){
						sem_post(&(is_block_ready[k]));
					}


					working_block = blocks[working_layer_ind];
					working_block -> layer_id = k;

					// Ensure we have a fresh gradient buffer to work over...
					if (TO_PRINT_GRAD_BLOCK_WAITING){
						printf("\n\n[Bwd] Waiting for grad block %d (at index %d) to be ready...\n\n", k, working_grad_block_ind);
					}

					sprintf(profile_msg, "Waiting for grad block %d (at index %d) to be ready", k, working_grad_block_ind);
					dataflow_handle.profiler.range_push(profile_msg);

					sem_wait(&(is_grad_block_ready[k]));

					int grad_block_ready_val;
					sem_getvalue(&(is_grad_block_ready[k]), &grad_block_ready_val);

					dataflow_handle.profiler.range_pop();

					working_grad_block = grad_blocks[working_grad_block_ind];

					working_grad_block -> layer_id = k;

					if (r == 0){
						// enusre that grad block is set to 0...
						ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, working_grad_block -> buffer, 0, aligned_block_bwd_size);
						if (ret){
							fprintf(stderr, "Error: failed to set mem for grad block #%d...\n", working_grad_block_ind);
							return -1;
						}
					}

					// PROCESS EACH SEQUENCE IN REVERSE AS FORWARDS AND REVERSED WITHIN SEQUENCE...
					// would be better to process in same sequence order, but not implementing this for now...

					// WE WANT TO PREFETCH TOKENS INTO THE "fwd_context" buffer, and we set it so the first
					// sequence group has this buffer ready to go on the last block...
					for (int seq_group = num_seq_groups_per_round - 1; seq_group >= 0; seq_group--){

						sprintf(profile_msg, "Seq Group: %d", seq_group);
						dataflow_handle.profiler.range_push(profile_msg);

						// EVERY SEQUENCE GROUP NEEDS TO HAVE ITS CONTEXT READY BEFORE THE CHUNKS CAN 

						// reset fwd context tokens replaced...
						fwd_ctx_tokens_replaced = 0;

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

						for (int c = num_chunks_per_seq - 1; c >= 0; c--){

							chunk_id = seq_group * num_chunks_per_seq + c;

							sprintf(profile_msg, "Chunk: %d", chunk_id);
							dataflow_handle.profiler.range_push(profile_msg);

							// ensure that we have the correct activations ready...
							if (TO_PRINT_FWD_ACT_WAITING){
								printf("\n\n[Bwd] Waiting for fwd activations of layer #%d, chunk #%d to be ready at working buffer index %d...\n\n", k, c, working_act_buffer_ind);
							}

							sprintf(profile_msg, "Waiting for fwd activations of layer #%d, chunk #%d to be ready at working buffer index %d", k, c, working_act_buffer_ind);
							dataflow_handle.profiler.range_push(profile_msg);
							
							sem_wait(&(is_fwd_activations_ready[k * num_chunks + chunk_id]));

							sem_wait(&(is_saved_activation_ready[working_act_buffer_ind]));

							dataflow_handle.profiler.range_pop();
							
							cur_fwd_activations = &(saved_activations[working_act_buffer_ind]);
							cur_fwd_activations -> layer_id = k;
							cur_fwd_activations -> seq_batch = seq_batches[chunk_id];

							cur_saved_activation_level = saved_activation_levels[num_chunks * k + chunk_id];
							cur_fwd_activations -> saved_activation_level = cur_saved_activation_level;

							grad_activations -> working_activations -> layer_id = k;
							grad_activations -> working_activations -> seq_batch = seq_batches[chunk_id];

							
							if ((cur_saved_activation_level != SAVED_ACTIVATION_LEVEL_NONE) && (cur_saved_activation_level != SAVED_ACTIVATION_LEVEL_FULL)){
								
								if (TO_PRINT_SUBMITTING){
									printf("\n\nSubmitting recompute for seq group #%d, chunk #%d, block #%d...\n\n", seq_group, chunk_id, k);
								}

								sprintf(profile_msg, "Recompute X: seq group #%d, chunk #%d, block #%d", seq_group, chunk_id, k);
								dataflow_handle.profiler.range_push(profile_msg);
								ret = dataflow_submit_transformer_block_recompute(&dataflow_handle, compute_stream_id, 
												working_block,
												seq_batches[chunk_id],
												cur_saved_activation_level,
												cur_fwd_activations, fwd_context,
												activation_workspace);
								if (ret){
									fprintf(stderr, "Error: failed to submit transformer block recompute for seq group #%d, chunk #%d, block #%d...\n", seq_group, chunk_id, k);
									return -1;
								}
								dataflow_handle.profiler.range_pop();
							}

						
							if (TO_PRINT_SUBMITTING){
								printf("\n\nSubmitting bwd_x for seq group #%d, chunk #%d, block #%d...\n\n", seq_group, chunk_id, k);
							}

							sprintf(profile_msg, "Bwd X: seq group #%d, chunk #%d, block #%d", seq_group, chunk_id, k);
							dataflow_handle.profiler.range_push(profile_msg);

							ret = dataflow_submit_transformer_block_bwd_x(&dataflow_handle, compute_stream_id,
												working_block, 
												&(block_transitions[2 * chunk_id + (k % 2)]), 
												cur_fwd_activations, fwd_context,
												grad_activations,
												working_grad_block,
												&(block_transitions[2 * chunk_id + ((k + 1) % 2)]));

							if (ret){
								fprintf(stderr, "Error: failed to submit transformer block bwd_x for seq group #%d, chunk #%d, block #%d...\n", seq_group, chunk_id, k);
								return -1;
							}

							dataflow_handle.profiler.range_pop();

							// prefetch fwd context for the chunk as same position in next seq group....

							cur_tokens = seq_batches[chunk_id] -> total_tokens;

							// if we change to increasing sequence order going backwards, then
							// we should prefetch the in increasing seq order

							// for now doing fully reversed (not ideal for fwd/bwd overlap, but this 
							// doesn't matter for single-device "ring"...)
							if ((k > 0) || (seq_group > 0)){

								int next_chunk_id_context;
								int next_home_act_ind_context;

								// get the next sequence group's chunk id and home act ind...

								// if final sequence group, then we need to prefetch the first sequence group of previous layer...
								if (seq_group == 0){
									next_chunk_id_context = num_chunks_per_seq * (num_seq_groups_per_round - 1) + c;
									next_home_act_ind_context = num_chunks * (k - 1) + next_chunk_id_context;
								}
								// get seq group at same layer but "behind" in bwd ordering...
								else{
									next_chunk_id_context = num_chunks_per_seq * (seq_group - 1) + c;
									next_home_act_ind_context = num_chunks * k + next_chunk_id_context;
								}
								

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

								int has_act;

								sem_getvalue(&(is_fwd_activations_ready[next_home_act_ind_context]), &has_act);

								// most of the time we should have the activations ready, but special case
								// in which acts are running behind we need to grad it from host...
								if (!has_act){

									if (TO_PRINT_FWD_ACT_WAITING){
										printf("\n\n[Bwd] Waiting for prior layer's saved activations to be home...\n\n");
									}

									// wait for this to be home, but then can let it stay there...
									sem_wait(&(is_saved_act_home[next_home_act_ind_context]));
									sem_post(&(is_saved_act_home[next_home_act_ind_context]));

									if (TO_PRINT_CTX_TRANSFERRING){
										printf("\n\n[Bwd] Transferring prior layer's saved context at home index %d from host to device...\n\n", next_home_act_ind_context);
									}

									prior_group_sys_saved_activations = &(sys_saved_activations[next_home_act_ind_context]);

									ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_k_dest, prior_group_sys_saved_activations -> x_k_local, cur_tokens * kv_dim * block_dt_size);
									if (ret){
										fprintf(stderr, "Error: failed to submit inbound transfer to prefetch fwd context for next layer...\n");
										return -1;
									}

									ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_v_dest, prior_group_sys_saved_activations -> x_v_local, cur_tokens * kv_dim * block_dt_size);
									if (ret){
										fprintf(stderr, "Error: failed to submit inbound transfer to prefetch fwd context for next layer...\n");
										return -1;
									}
								}
								// otherwise the context is only on the device so we can just copy it...
								else{

									// we know that it is ready, so lets get the index...

									prior_group_dev_saved_act_ind = dev_act_ind[next_home_act_ind_context];

									prior_group_dev_saved_activations = &(saved_activations[prior_group_dev_saved_act_ind]);

									if (TO_PRINT_CTX_TRANSFERRING){
										printf("\n\n[Bwd] Transferring prior group/layer (act index %d) saved context from self memory at dev index %d...\n\n", next_home_act_ind_context, prior_group_dev_saved_act_ind);
									}

									ret = dataflow_handle.submit_peer_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_k_dest, prior_group_dev_saved_activations -> x_k_local, cur_tokens * kv_dim * block_dt_size);
									if (ret){
										fprintf(stderr, "Error: failed to submit peer transfer to prefetch fwd context for next layer...\n");
										return -1;
									}

									ret = dataflow_handle.submit_peer_transfer(&dataflow_handle, inbound_fwd_ctx_stream_id, global_fwd_ctx_v_dest, prior_group_dev_saved_activations -> x_v_local, cur_tokens * kv_dim * block_dt_size);
									if (ret){
										fprintf(stderr, "Error: failed to submit peer transfer to prefetch fwd context for next layer...\n");
										return -1;
									}
								}

								fwd_ctx_tokens_replaced += cur_tokens;
							}

							// could prefetch next layer weights here, but doing this after the bwd w is done for cleanliness...

							if (TO_PRINT_SUBMITTING){
								printf("\n\nSubmitting bwd_w for seq group #%d, chunk #%d, block #%d...\n\n", seq_group, chunk_id, k);
							}

							// utilizing the newly populated grad_activations struct
							// to update the grad_weights...

							sprintf(profile_msg, "Bwd W: seq group #%d, chunk #%d, block #%d", seq_group, chunk_id, k);
							dataflow_handle.profiler.range_push(profile_msg);

							// uses the same input transition as bwd_x...
							ret = dataflow_submit_transformer_block_bwd_w(&dataflow_handle, compute_stream_id,
												&(block_transitions[2 * chunk_id + (k % 2)]),
												cur_fwd_activations, 
												grad_activations, 
												working_grad_block);

							if (ret){
								fprintf(stderr, "Error: failed to submit transformer block bwd_w for seq group #%d, chunk #%d, block #%d...\n", seq_group, chunk_id, k);
								return -1;
							}

							dataflow_handle.profiler.range_pop();


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

								int fwd_layer = cur_fwd_prefetch_act_ind / num_chunks;
								int fwd_chunk = cur_fwd_prefetch_act_ind % num_chunks;

								if (TO_PRINT_ACT_TRANSFERRING){
									printf("\n\n[Bwd] Waiting for saved activation at index %d (fwd layer %d, chunk %d) to be home and bringing into working buffer index %d...\n\n", cur_fwd_prefetch_act_ind, fwd_layer, fwd_chunk, working_act_buffer_ind);
								}

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

								dev_act_ind[cur_fwd_prefetch_act_ind] = working_act_buffer_ind;

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

							// pop from pushing "Chunk: %d", chunk_id
							dataflow_handle.profiler.range_pop();
						}


						// pop from pushing "Seq Group: %d", seq_group
						dataflow_handle.profiler.range_pop();
					}

					// pop from Layer %d
					dataflow_handle.profiler.range_pop();

					// Finished this layer, moving onto next...

					cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
					if (!cur_stream_state){
						fprintf(stderr, "Error: failed to get stream state for grad block #%d...\n", k);
						return -1;
					}


					// if we are about to do opt step and can hold onto this grad block...
					// then we don't need to send it back to host...
					submitted_outbound_grad = 0;
					if (((r == num_rounds_per_step - 1) && (k >= num_dev_grad_blocks)) || (r < num_rounds_per_step - 1)){
						// send back gradient to host, and run optimizer if needed

						ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
						if (ret){
							fprintf(stderr, "Error: failed to submit dependency to send grad block #%d to host...\n", k);
							return -1;
						}

						sprintf(profile_msg, "Sending grad block #%d to host...", k);
						dataflow_handle.profiler.range_push(profile_msg);
					
						ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_grad_blocks[k] -> buffer, working_grad_block -> buffer, aligned_block_bwd_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit outbound transfer to send grad block #%d to host...\n", k);
							return -1;
						}
						
						submitted_outbound_grad = 1;

						dataflow_handle.profiler.range_pop();

						// this buffer is now
						// because don't need to do inbound transfer (but need to wait for it to be sent back...)
						if (r == 0){
							if (next_grad_block_id >= 0){
								ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[next_grad_block_id]), outbound_stream_id);
								if (ret){
									fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for grad block #%d...\n", k - num_dev_grad_blocks);
									return -1;
								}
								next_grad_block_id--;
							}
						}					
					}
					// otherwise we can leave this grad block on device to be ready for opt step...
					else{
						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[k]), compute_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for grad block #%d...\n", k);
							return -1;
						}
					}
					

					// PREFETCHING NEXT NEEDED (which is) <= PREVIOUS LAYER ID) FORWARD BLOCK and GRADIENT BLOCK...
					// choose the one that is needed next, or both if both are needed...
					if (r == 0){
						if (next_layer_id >= 0){
						// will will definitely need the forward block next...
							if (TO_PRINT_BWD_PREFETCHING){
								printf("[Bwd] Prefetching fwd block layer id %d into slot %d...\n\n", next_layer_id, replacement_layer_ind);
							}

							ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
							if (ret){
								fprintf(stderr, "Error: failed to submit dependency to prefetch block #%d...\n", next_layer_id);
								return -1;
							}

							sprintf(profile_msg, "Prefetching fwd block layer id %d...", next_layer_id);	
							dataflow_handle.profiler.range_push(profile_msg);

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

							dataflow_handle.profiler.range_pop();

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
					}
					else{
						if ((next_layer_id == next_grad_block_id) && (next_layer_id >= 0)) {


							// will will definitely need the forward block next...
							if (TO_PRINT_BWD_PREFETCHING){
								printf("[Bwd] Prefetching fwd block layer id %d into slot %d...\n\n", next_layer_id, replacement_layer_ind);
							}

							ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
							if (ret){
								fprintf(stderr, "Error: failed to submit dependency to prefetch block #%d...\n", next_layer_id);
								return -1;
							}

							sprintf(profile_msg, "Prefetching fwd block layer id %d...", next_layer_id);	
							dataflow_handle.profiler.range_push(profile_msg);

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

							dataflow_handle.profiler.range_pop();

							if (next_layer_id > 0){
								if (replacement_layer_ind > 0){
									replacement_layer_ind -= 1;
								}
								else{
									replacement_layer_ind = num_dev_blocks - 1;
								}
							}
							next_layer_id--;

							// for now just prefetching grad block every time in this case, 
							// but less data then might be to alternate between prefetching and not...
							if (TO_PRINT_GRAD_BLOCK_PREFETCHING){
								printf("\n\nPrefetching next grad block id #%d (replacing grad block at index %d)...\n\n", next_grad_block_id, replacement_grad_layer_ind);
							}


							if (submitted_outbound_grad){
								// here cur stream state is either done with computation (if replacement ind is different than working ind, 
								// or it is waiting for the grad block to have finished making its way home if replacement ind is same as working ind)
								// need to set dependency for the outbound stream to ensure this grad block has finished making its way home..
								cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
								if (!cur_stream_state){
									fprintf(stderr, "Error: failed to get stream state for grad block #%d...\n", k);
									return -1;
								}

								// prefetch next grad block...
								ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
								if (ret){
									fprintf(stderr, "Error: failed to submit dependency to prefetch next grad block...\n");
									return -1;
								}
							}

							sprintf(profile_msg, "Prefetching next grad block id #%d...", next_grad_block_id);
							dataflow_handle.profiler.range_push(profile_msg);
								
							ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, grad_blocks[replacement_grad_layer_ind] -> buffer, sys_grad_blocks[next_grad_block_id] -> buffer, aligned_block_bwd_size);
							if (ret){
								fprintf(stderr, "Error: failed to submit inbound transfer for grad block #%d...\n", next_grad_block_id);
								return -1;
							}

							ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[next_grad_block_id]), inbound_stream_id);
							if (ret){
								fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for next grad block...\n");
								return -1;
							}

							dataflow_handle.profiler.range_pop();

							if (replacement_grad_layer_ind > 0){
								replacement_grad_layer_ind--;
							}
							else{
								replacement_grad_layer_ind = num_dev_grad_blocks - 1;
							}
							next_grad_block_id--;
						}
						// only forward block is needed...
						else if ((next_layer_id > next_grad_block_id) && (next_layer_id >= 0)){
							// will will definitely need the forward block next...
							if (TO_PRINT_BWD_PREFETCHING){
								printf("[Bwd] Prefetching fwd block layer id %d into slot %d...\n\n", next_layer_id, replacement_layer_ind);
							}

							ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
							if (ret){
								fprintf(stderr, "Error: failed to submit dependency to prefetch block #%d...\n", next_layer_id);
								return -1;
							}

							sprintf(profile_msg, "Prefetching fwd block layer id %d...", next_layer_id);	
							dataflow_handle.profiler.range_push(profile_msg);

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

							dataflow_handle.profiler.range_pop();

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
						// only gradient block is needed...
						else if ((next_grad_block_id > next_layer_id) && (next_grad_block_id >= 0)){
							
							if (TO_PRINT_GRAD_BLOCK_PREFETCHING){
								printf("\n\nPrefetching next grad block id #%d (replacing grad block at index %d)...\n\n", next_grad_block_id, replacement_grad_layer_ind);
							}


							if (submitted_outbound_grad){
								// need to set dependency for the outbound stream to ensure this grad block has finished making its way home..
								cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
								if (!cur_stream_state){
									fprintf(stderr, "Error: failed to get stream state for grad block #%d...\n", k);
									return -1;
								}	
							

								// here cur stream state is either done with computation (if replacement ind is different than working ind, 
								// or it is waiting for the grad block to have finished making its way home if replacement ind is same as working ind)

								// prefetch next grad block...
								ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
								if (ret){
									fprintf(stderr, "Error: failed to submit dependency to prefetch next grad block...\n");
									return -1;
								}
							}

							sprintf(profile_msg, "Prefetching next grad block id #%d...", next_grad_block_id);
							dataflow_handle.profiler.range_push(profile_msg);
								
							ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, grad_blocks[replacement_grad_layer_ind] -> buffer, sys_grad_blocks[next_grad_block_id] -> buffer, aligned_block_bwd_size);
							if (ret){
								fprintf(stderr, "Error: failed to submit inbound transfer for grad block #%d...\n", next_grad_block_id);
								return -1;
							}

							ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[next_grad_block_id]), inbound_stream_id);
							if (ret){
								fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for next grad block...\n");
								return -1;
							}

							dataflow_handle.profiler.range_pop();

							if (replacement_grad_layer_ind > 0){
								replacement_grad_layer_ind--;
							}
							else{
								replacement_grad_layer_ind = num_dev_grad_blocks - 1;
							}
							next_grad_block_id--;
						}
					}


					// in case num_rounds_per_step > 1, we will follow up another forward pass
					// and don't want to lose the position of first layer, so only update if k > 0
					// even if only one round, we still need to know the index of first layer in order to 
					// know which grad block to use for the opt step...
					if (k > 0){
						if (working_layer_ind > 0){
							working_layer_ind -= 1;
						}
						else{
							working_layer_ind = num_dev_blocks - 1;
						}


						if (working_grad_block_ind > 0){
							working_grad_block_ind -= 1;
						}
						else{
							working_grad_block_ind = num_dev_grad_blocks - 1;
						}
					}
					
				}


				// NOW DOING BWD W....

				sprintf(profile_msg, "Embedding");
				dataflow_handle.profiler.range_push(profile_msg);
				
				for (int seq_group = num_seq_groups_per_round - 1; seq_group >= 0; seq_group--){
					
					sprintf(profile_msg, "Seq Group: %d", seq_group);
					dataflow_handle.profiler.range_push(profile_msg);

					for (int c = num_chunks_per_seq - 1; c >= 0; c--){

						chunk_id = seq_group * num_chunks_per_seq + c;


						sprintf(profile_msg, "Chunk: %d", chunk_id);
						dataflow_handle.profiler.range_push(profile_msg);

						if (TO_PRINT_SUBMITTING){
							printf("\n\nSubmitting embedding bwd_w for seq group #%d, chunk #%d...\n\n", seq_group, chunk_id);
						}

						// layer 0'ths output grad stream will be at index 1 (for given chunk)
						ret = dataflow_submit_transformer_embedding_bwd_w(&dataflow_handle, compute_stream_id,
														&(block_transitions[2 * chunk_id + 1]),
														grad_embedding_table);

						if (ret){
							fprintf(stderr, "Error: failed to submit transformer embedding bwd_w for seq group #%d, chunk #%d...\n", seq_group, chunk_id);
							return -1;
						}

						dataflow_handle.profiler.range_pop();
					}
					dataflow_handle.profiler.range_pop();
				}

				// pop from "Embedding"
				dataflow_handle.profiler.range_pop();

				// pop from pushing "Bwd"
				dataflow_handle.profiler.range_pop();

				// pop from "Round %d"
				dataflow_handle.profiler.range_pop();

				//printf("Finished enqueuing operations for round: %d\n\n", r);
			}

			// now that we are stepping we have the first num_dev_blocks layers on device...
			// and num_dev_grad_blocks grad blocks on device...

			// the 0th layer exists at index of working_layer_ind...
			next_layer_id = num_dev_blocks;
			replacement_layer_ind = working_layer_ind;
			next_grad_block_id = num_dev_grad_blocks;
			replacement_grad_layer_ind = working_grad_block_ind;

			// reset the opt layer indices...
			next_opt_layer_id = 0;
			replacement_opt_layer_ind = 0;
			working_opt_layer_ind = 0;

			sprintf(profile_msg, "Optimizer Step %d", t);
			dataflow_handle.profiler.range_push(profile_msg);
			// now perform opt step

			// need to fetch all of the opt state from home...
			if (t > 1){

				// ensure we are done with transitions and activations before fetching opt state...

				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
					return -1;
				}

				ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency for inbound stream to wait for opt state...\n");
					return -1;
				}

				sprintf(profile_msg, "Fetching first %d opt blocks...", num_dev_opt_blocks);
				dataflow_handle.profiler.range_push(profile_msg);

				for (int k = 0; k < num_dev_opt_blocks; k++){

					ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, opt_mean_blocks[k] -> buffer, sys_opt_mean_blocks[k] -> buffer, block_aligned_num_els * opt_mean_dt_size);
					if (ret){
						fprintf(stderr, "Error: failed to submit inbound transfer for opt mean block #%d...\n", k);
						return -1;
					}

					ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, opt_var_blocks[k] -> buffer, sys_opt_var_blocks[k] -> buffer, block_aligned_num_els * opt_var_dt_size);
					if (ret){
						fprintf(stderr, "Error: failed to submit inbound transfer for opt var block #%d...\n", k);
						return -1;
					}

					ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_opt_layer_ready[k]), inbound_stream_id);
					if (ret){
						fprintf(stderr, "Error: failed to submit host op to enqueue is_opt_layer_ready for opt block #%d...\n", k);
						return -1;
					}
					
					next_opt_layer_id++;
				}

				dataflow_handle.profiler.range_pop();

			}
			else{
				for (int k = 0; k < num_dev_opt_blocks; k++){
					sem_post(&(is_opt_layer_ready[k]));
					next_opt_layer_id++;
				}
			}

			// Can immeidately do the embedding and head opt steps...

			sprintf(profile_msg, "Embedding Opt Step %d", t);
			dataflow_handle.profiler.range_push(profile_msg);
			

			ret = dataflow_submit_default_adamw_step(&dataflow_handle, compute_stream_id,
																block_dt, block_bwd_dt, opt_mean_dt, opt_var_dt,
																embedding_num_els, t,
																lr, beta1, beta2, weight_decay, epsilon,
																embedding_table -> embedding_table, grad_embedding_table -> embedding_table,
																opt_mean_embedding_table -> embedding_table, opt_var_embedding_table -> embedding_table);
			if (ret){
				fprintf(stderr, "Error: failed to submit adam step for embedding...\n");
				return -1;
			}

			// clear the grad embedding table...
			ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, grad_embedding_table -> embedding_table, 0, grad_embedding_table -> embedding_table_size);
			if (ret){
				fprintf(stderr, "Error: failed to set mem for grad embedding table...\n");
				return -1;
			}


			// OPTIOANLLY sending back the updated embedding table to host for safekeeping/checkpointing...
			// can remove this is too much overhead...
			// (doesn't need to happen here and might be better to do it not in the critical path)

			cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
			if (!cur_stream_state){
				fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
				return -1;
			}

			ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
			if (ret){
				fprintf(stderr, "Error: failed to submit dependency to send params and opt state to host...\n");
				return -1;
			}
			
			sprintf(profile_msg, "Sending embedding table to host...");
			dataflow_handle.profiler.range_push(profile_msg);

			ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_embedding_table -> embedding_table, embedding_table -> embedding_table, sys_embedding_table -> embedding_table_size);
			if (ret){
				fprintf(stderr, "Error: failed to submit outbound transfer to send embedding table to host...\n");
				return -1;
			}

			if (TO_SAVE_UPDATED_PARAMS){
				ret = save_updated_params(&dataflow_handle, outbound_stream_id, t, 0, false, true, sys_embedding_table -> embedding_table, sys_embedding_table -> embedding_table_size);
				if (ret){
					fprintf(stderr, "Error: failed to save updated embedding table...\n");
					return -1;
				}
			}

			dataflow_handle.profiler.range_pop();

			// embedding opt step done...
			dataflow_handle.profiler.range_pop();




			sprintf(profile_msg, "Head Opt Step %d", t);
			dataflow_handle.profiler.range_push(profile_msg);

			ret = dataflow_submit_default_adamw_step(&dataflow_handle, compute_stream_id,
																block_dt, block_bwd_dt, opt_mean_dt, opt_var_dt,
																head_num_els, t,
																lr, beta1, beta2, weight_decay, epsilon,
																head -> buffer, grad_head -> buffer,
																opt_mean_head -> buffer, opt_var_head -> buffer);
			if (ret){
				fprintf(stderr, "Error: failed to submit adam step for head...\n");
				return -1;
			}

			// clear the grad head...
			ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, grad_head -> buffer, 0, combined_head_bwd_size);
			if (ret){
				fprintf(stderr, "Error: failed to set mem for grad head...\n");
				return -1;
			}


			// OPTIOANLLY sending back the updated head to host for safekeeping/checkpointing...
			// can remove this is too much overhead...
			// (doesn't need to happen here and might be better to do it not in the critical path)


			cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
			if (!cur_stream_state){
				fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
				return -1;
			}

			ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
			if (ret){
				fprintf(stderr, "Error: failed to submit dependency to send params and opt state to host...\n");
				return -1;
			}

			sprintf(profile_msg, "Sending head to host...");
			dataflow_handle.profiler.range_push(profile_msg);

			ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_head -> buffer, head -> buffer, combined_head_size);
			if (ret){
				fprintf(stderr, "Error: failed to submit outbound transfer to send head to host...\n");
				return -1;
			}

			if (TO_SAVE_UPDATED_PARAMS){
				ret = save_updated_params(&dataflow_handle, outbound_stream_id, t, 0, true, false, sys_head -> buffer, combined_head_size);
				if (ret){
					fprintf(stderr, "Error: failed to save updated head...\n");
					return -1;
				}
			}

			dataflow_handle.profiler.range_pop();

			// head opt step done...
			dataflow_handle.profiler.range_pop();


			// at this point the early layers (params and grads) should be on device
			// go in forwards direction loading in the opt value from sys while using the space not
			// needed for 

			// need to prefetch the embedding opt state and start opt step
			// (it's gradient is already on device)

			// after finalizing the head we can move on to the next step...

			// ensure we retrieve the resulting params and optimizer state back from device
			// that we will need for the end of the next step...

			// special case for t == 1 where we know opt state contains zero values...
			

			// at this point working_layer_ind should refer to location that holds layer 0 and moving forwards get us higher layers...
			// at this point working_grad_block_ind should refer to location that holds grad 0 and moving forwards get us higher grads...

			// assert(next_layer_id == num_dev_blocks);
			// assert(next_grad_block_id == num_dev_grad_blocks);
			// assert(next_opt_layer_id == num_dev_opt_blocks);


			// we ensured that working_layer_ind and working_grad_block_ind are referencing the location of layer 0 and grad 0
			// and that higher layers are in increasing indices (wrapped) in the blocks and grad_blocks arrays...

			// working_opt_layer_ind always starts at 0 and moves upwards...

			for (int k = 0; k < n_layers; k++){

				sprintf(profile_msg, "Opt Step %d: Block %d", t, k);
				dataflow_handle.profiler.range_push(profile_msg);				


				// ensure we have the layer, grad, and opt state state ready...

				if (TO_PRINT_OPT_STEP_WAITING){
					printf("\n\nOpt Step %d: Waiting for layer #%d to be ready (at index %d)...", t, k, working_layer_ind);
				}
				sprintf(profile_msg, "Opt Step %d: Waiting for layer #%d to be ready (at index %d)...", t, k, working_layer_ind);
				dataflow_handle.profiler.range_push(profile_msg);
				sem_wait(&(is_block_ready[k]));
				dataflow_handle.profiler.range_pop();
				working_block = blocks[working_layer_ind];

				if (TO_PRINT_OPT_STEP_WAITING){
					printf("\n\nOpt Step %d: Waiting for grad block #%d to be ready (at index %d)...", t, k, working_grad_block_ind);
				}

				sprintf(profile_msg, "Opt Step %d: Waiting for grad block #%d to be ready (at index %d)...", t, k, working_grad_block_ind);
				dataflow_handle.profiler.range_push(profile_msg);
				sem_wait(&(is_grad_block_ready[k]));

				int is_grad_block_ready_val = 0;
				sem_getvalue(&(is_grad_block_ready[k]), &is_grad_block_ready_val);

				assert(is_grad_block_ready_val == 0);

				dataflow_handle.profiler.range_pop();

				working_grad_block = grad_blocks[working_grad_block_ind];

				if (TO_PRINT_OPT_STEP_WAITING){
					printf("\n\nOpt Step %d: Waiting for opt block #%d to be ready (at index %d)...", t, k, working_opt_layer_ind);
				}

				sprintf(profile_msg, "Opt Step %d: Waiting for opt block #%d to be ready (at index %d)...", t, k, working_opt_layer_ind);
				dataflow_handle.profiler.range_push(profile_msg);
				sem_wait(&(is_opt_layer_ready[k]));
				dataflow_handle.profiler.range_pop();

				working_opt_mean_block = opt_mean_blocks[working_opt_layer_ind];
				working_opt_var_block = opt_var_blocks[working_opt_layer_ind];

				// ensure we zero out the opt state for the first step...
				if (t == 1){
					ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, working_opt_mean_block -> buffer, 0, block_aligned_num_els * opt_mean_dt_size);
					if (ret){
						fprintf(stderr, "Error: failed to set mem for opt mean block...\n");
						return -1;
					}

					ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id, working_opt_var_block -> buffer, 0, block_aligned_num_els * opt_var_dt_size);
					if (ret){
						fprintf(stderr, "Error: failed to set mem for opt var block...\n");
						return -1;
					}	
				}

				// now we can submit the opt step...

				sprintf(profile_msg, "Performing Adam: Block %d", k);
				dataflow_handle.profiler.range_push(profile_msg);

				if (TO_SAVE_GRAD_BLOCKS_PRE_STEP){
					ret = save_grad_blocks_pre_step(&dataflow_handle, compute_stream_id, t, k, false, false, working_grad_block -> buffer, aligned_block_bwd_size);
					if (ret){
						fprintf(stderr, "Error: failed to save grad blocks for step #%d, layer #%d...\n", t, k);
						return -1;
					}
				}

			
				ret = dataflow_submit_default_adamw_step(&dataflow_handle, compute_stream_id,
																block_dt, block_bwd_dt, opt_mean_dt, opt_var_dt,
																block_aligned_num_els, t,
																lr, beta1, beta2, weight_decay, epsilon,
																working_block -> buffer, working_grad_block -> buffer,
																working_opt_mean_block -> buffer, working_opt_var_block -> buffer);
				dataflow_handle.profiler.range_pop();

				if (ret){
					fprintf(stderr, "Error: failed to submit transformer block opt step for layer #%d...\n", k);
					return -1;
				}

				// Now need to send back the resulting params and opt state to host...
				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
					return -1;
				}

				ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency to send params and opt state to host...\n");
					return -1;
				}


				sprintf(profile_msg, "Sending params and opt state for layer #%d to host...", k);
				dataflow_handle.profiler.range_push(profile_msg);

				ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_blocks[k] -> buffer, working_block -> buffer, aligned_block_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit outbound transfer to send params and opt state to host...\n");
					return -1;
				}

				ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_home[k]), outbound_stream_id);
				if (ret){
					fprintf(stderr, "Error: failed to submit host op to enqueue is_block_home for layer #%d...\n", k);
					return -1;
				}

				ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_opt_mean_blocks[k] -> buffer, working_opt_mean_block -> buffer, block_aligned_num_els * opt_mean_dt_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit outbound transfer to send params and opt state to host...\n");
					return -1;
				}

				ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id, sys_opt_var_blocks[k] -> buffer, working_opt_var_block -> buffer, block_aligned_num_els * opt_var_dt_size);
				if (ret){
					fprintf(stderr, "Error: failed to submit outbound transfer to send params and opt state to host...\n");
					return -1;
				}

				dataflow_handle.profiler.range_pop();

				// special case for t == 1 where we don't fetch opt state from home...
				if (t == 1) {

					if (next_opt_layer_id < n_layers){
						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_opt_layer_ready[next_opt_layer_id]), outbound_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_opt_layer_ready for next opt block...\n");
							return -1;
						}
						next_opt_layer_id++;
					}
				}

				// now prefetch the next layer, grad, and/or opt state...
				if ((next_layer_id < n_layers) || (next_grad_block_id < n_layers) || ((t > 1) && (next_opt_layer_id < n_layers))){

					cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
					if (!cur_stream_state){
						fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
						return -1;
					}

					ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
					if (ret){
						fprintf(stderr, "Error: failed to submit dependency for inbound stream to wait for next layer...\n");
						return -1;
					}

				

					if (next_layer_id < n_layers){

						sprintf(profile_msg, "Prefetching next layer #%d in order to do opt step...", next_layer_id);
						dataflow_handle.profiler.range_push(profile_msg);

						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[replacement_layer_ind] -> buffer, sys_blocks[next_layer_id] -> buffer, aligned_block_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer for next layer...\n");
							return -1;
						}

						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[next_layer_id]), inbound_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_block_ready for next layer...\n");
							return -1;
						}

						replacement_layer_ind = (replacement_layer_ind + 1) % num_dev_blocks;
						next_layer_id++;

						dataflow_handle.profiler.range_pop();
					}

					if (next_grad_block_id < n_layers){

						sprintf(profile_msg, "Prefetching next grad block #%d in order to do opt step...", next_grad_block_id);
						dataflow_handle.profiler.range_push(profile_msg);

						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, grad_blocks[replacement_grad_layer_ind] -> buffer, sys_grad_blocks[next_grad_block_id] -> buffer, aligned_block_bwd_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer for next grad block...\n");
							return -1;
						}

						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_grad_block_ready[next_grad_block_id]), inbound_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_grad_block_ready for next grad block...\n");
							return -1;
						}

						replacement_grad_layer_ind = (replacement_grad_layer_ind + 1) % num_dev_grad_blocks;
						next_grad_block_id++;

						dataflow_handle.profiler.range_pop();
					}

					if ((t > 1) && (next_opt_layer_id < n_layers)){

						sprintf(profile_msg, "Prefetching next opt block #%d in order to do opt step...", next_opt_layer_id);
						dataflow_handle.profiler.range_push(profile_msg);

						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, opt_mean_blocks[replacement_opt_layer_ind] -> buffer, sys_opt_mean_blocks[next_opt_layer_id] -> buffer, block_aligned_num_els * opt_mean_dt_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer for next opt block...\n");
							return -1;
						}

						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, opt_var_blocks[replacement_opt_layer_ind] -> buffer, sys_opt_var_blocks[next_opt_layer_id] -> buffer, block_aligned_num_els * opt_var_dt_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer for next opt block...\n");
							return -1;
						}

						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_opt_layer_ready[next_opt_layer_id]), inbound_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_opt_layer_ready for next opt block...\n");
							return -1;
						}
						
						replacement_opt_layer_ind = (replacement_opt_layer_ind + 1) % num_dev_opt_blocks;
						next_opt_layer_id++;

						dataflow_handle.profiler.range_pop();
					}			
				}




				// Trying to figure out why this is needed...

				// makes sense for case of t == 1
				// where we reset opt_layer to 0 using compute stream and it might be set to ready from init
				// but unclear why this is needed every time...
				// however, incorrect without it...

				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
					return -1;
				}

				ret = dataflow_handle.submit_dependency(&dataflow_handle, compute_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency to send params and opt state to host...\n");
					return -1;
				}


				// done with opt step for block %d...
				dataflow_handle.profiler.range_pop();

				// now we can move on to the next layer...

				// if we on the last layer and we increment, then after this loop ends the working_layer_ind will
				// correspond the index holding the earlier block...
				working_layer_ind = (working_layer_ind + 1) % num_dev_blocks;

				// work upwards now that we are reversing direction again...
				working_grad_block_ind = (working_grad_block_ind + 1) % num_dev_grad_blocks;
				working_opt_layer_ind = (working_opt_layer_ind + 1) % num_dev_opt_blocks;	
			}


			// refetch all of the updated blocks that we need for the next forward pass...
			
			
			

			if (t < num_steps){

				cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id);
				if (!cur_stream_state){
					fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
					return -1;
				}

				ret = dataflow_handle.submit_dependency(&dataflow_handle, inbound_stream_id, cur_stream_state);
				if (ret){
					fprintf(stderr, "Error: failed to submit dependency for inbound stream to wait for next layer...\n");
					return -1;
				}

				sprintf(profile_msg, "Refetching blocks for next forward pass...");
				dataflow_handle.profiler.range_push(profile_msg);

				// the earliest block that will be on device after opt step is complete...
				int final_min_block_id = n_layers - num_dev_blocks;

				// now ensure we we blocks for the next forward pass...
				int final_min_layer_working_ind = working_layer_ind;

				// if there is some overlap (num_dev_blocks > .5 * n_layers)
				// then determine what blocks we actually need to fetch back from home..
				if (final_min_block_id < num_dev_blocks){
					int num_overlap = num_dev_blocks - final_min_block_id;

					if (final_min_block_id != final_min_layer_working_ind){
						// then we need to reshuffle the blocks currently on device to live at the proper locations
						// that align with workling_layer_ind starting at 0 every step...

						// copy the blocks currently on device to the temp blocks array...
						for (int i = 0; i < num_dev_blocks; i++){
							temp_blocks[i] = blocks[i];
						}

						// now we need to move the blocks that we need to keep on device to the proper locations...
						for (int i = 0; i < num_overlap; i++){
							blocks[final_min_block_id + i] = temp_blocks[(final_min_layer_working_ind + i) % num_dev_blocks];
						}
					}

					// these blocks are already ready...
					for (int k = final_min_block_id; k < final_min_block_id + num_overlap; k++){
						sem_post(&(is_block_ready[k]));
					}

					// we need to fetch the other blocks...
					for (int k = 0; k < final_min_block_id; k++){

						// first ensure the block is home...
						sem_wait(&(is_block_home[k]));

						sprintf(profile_msg, "Refetching block #%d...", k);
						dataflow_handle.profiler.range_push(profile_msg);

						// now fetch the block...
						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[k] -> buffer, sys_blocks[k] -> buffer, aligned_block_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer for next layer...\n");
							return -1;
						}

						// now enqueue the semaphore to wait for the block to be ready...
						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[k]), inbound_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_block_ready for next layer...\n");
							return -1;
						}

						dataflow_handle.profiler.range_pop();
					}
				}
				// otherwise we need to fetch all the early blocks
				else{
					// we need to fetch the other blocks...
					for (int k = 0; k < num_dev_blocks; k++){

						// first ensure the block is home...
						sem_wait(&(is_block_home[k]));

						int block_ready;
						sem_getvalue(&(is_block_ready[k]), &block_ready);
						assert(block_ready == 0);

						int block_home;
						sem_getvalue(&(is_block_home[k]), &block_home);
						assert(block_home == 0);

						sprintf(profile_msg, "Refetching block #%d...", k);
						dataflow_handle.profiler.range_push(profile_msg);

						// now fetch the block...
						ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id, blocks[k] -> buffer, sys_blocks[k] -> buffer, aligned_block_size);
						if (ret){
							fprintf(stderr, "Error: failed to submit inbound transfer for next layer...\n");
							return -1;
						}

						// now enqueue the semaphore to wait for the block to be ready...
						ret = dataflow_handle.submit_host_op(&dataflow_handle, post_sem_callback, (void *) &(is_block_ready[k]), inbound_stream_id);
						if (ret){
							fprintf(stderr, "Error: failed to submit host op to enqueue is_block_ready for next layer...\n");
							return -1;
						}

						dataflow_handle.profiler.range_pop();
					}
				}

				dataflow_handle.profiler.range_pop();
			}

			// consider sending all the results back to host as the end of the step...
			ret = dataflow_submit_end_step_metrics_host(&dataflow_handle, outbound_stream_id, 
													end_step_metrics, &(step_throughput_op_buffers[t - 1]));
			if (ret){
				fprintf(stderr, "Error: failed to submit end step metrics for step #%d...\n", t);
				return -1;
			}

			// make the compute stream (starting the next forward pass) wait for all of the outbound transfers to complete because
			// the opt space is aliased with the space need for transitons/activations/context needed in next immediate forward pass...

			cur_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, outbound_stream_id);
			if (!cur_stream_state){
				fprintf(stderr, "Error: failed to get stream state for compute stream...\n");
				return -1;
			}

			ret = dataflow_handle.submit_dependency(&dataflow_handle, compute_stream_id, cur_stream_state);
			if (ret){
				fprintf(stderr, "Error: failed to submit dependency to wait for outbound transfers to complete...\n");
				return -1;
			}
			
			// now we can submit the next forward pass...
			// first save down the updated params...
			if (TO_SAVE_UPDATED_PARAMS){
				for (int k = 0; k < n_layers; k++){
					ret = save_updated_params(&dataflow_handle, inbound_stream_id, t, k, false, false, sys_blocks[k] -> buffer, aligned_block_size);
					if (ret){
						fprintf(stderr, "Error: failed to save updated params for step #%d, layer #%d...\n", t, k);
						return -1;
					}
				}
			}

			// pop from "Optimizer Step %d"
			dataflow_handle.profiler.range_pop();	

			// pop from "Step %d"
			dataflow_handle.profiler.range_pop();

			//printf("Finished enqueuing operations for step: %d!\n\n", t);
		}
		


		// printf("\n\n\nFinished enqueueing all dataflow operations!\nWaiting to sync...\n\n");


		ret = dataflow_handle.sync_stream(&dataflow_handle, outbound_stream_id);
		if (ret){
			fprintf(stderr, "Error: failed to sync host ops stream at end of transformer...\n");
			return -1;
		}

		ret = dataflow_handle.profiler.stop();
		if (ret){
			fprintf(stderr, "Error: failed to stop profiling...\n");
			return -1;
		}


		//printf("All operations complete! Exiting...\n\n");

		float total_steps_time = 0;
		float total_steps_tok_per_sec = 0;
		float total_steps_flops = 0;
		float total_steps_mfu = 0;
		float total_steps_hfu = 0;
		float total_steps_recompute_pct = 0;
		float total_attn_flop_pct = 0;
		for (int t = NUM_STEPS_TO_SKIP_FOR_RECORDING; t < num_steps; t++){
			total_steps_time += step_throughput_op_buffers[t].duration_s;
			total_steps_tok_per_sec += step_throughput_op_buffers[t].tokens_per_second;
			total_steps_flops += step_throughput_op_buffers[t].achieved_flop_rate;
			total_steps_mfu += step_throughput_op_buffers[t].mfu;
			total_steps_hfu += step_throughput_op_buffers[t].hfu;
			total_steps_recompute_pct += step_throughput_op_buffers[t].total_recompute_flops / step_throughput_op_buffers[t].total_flops;
			total_attn_flop_pct += step_throughput_op_buffers[t].total_attn_flops / step_throughput_op_buffers[t].total_flops;
		}

		int num_recorded_steps = num_steps - NUM_STEPS_TO_SKIP_FOR_RECORDING;

		float avg_step_time = total_steps_time / num_recorded_steps;
		float avg_tok_per_sec = total_steps_tok_per_sec / num_recorded_steps;
		float avg_tflops = total_steps_flops / num_recorded_steps / 1e12;
		float avg_mfu = total_steps_mfu / num_recorded_steps;
		float avg_hfu = total_steps_hfu / num_recorded_steps;
		float avg_recompute_pct = total_steps_recompute_pct / num_recorded_steps;
		float avg_attn_flop_pct = total_attn_flop_pct / num_recorded_steps;
		FILE * f = fopen(output_filepath, "a");
		if (!f){
			fprintf(stderr, "Error: failed to open file: %s...\n", output_filepath);
			return -1;
		}

		fprintf(f, "%d,%d,%d,%d,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f\n", HOST_MEM_GB, DEV_MEM_GB, DEMO_SEQ_LEN, MODEL_CONFIG_SIZE_B, used_host_mem_gb, used_dev_mem_gb, (int) chunk_size, total_home_acts, num_inp_only_saved, num_inp_attn_saved, num_full_saved, total_dev_acts, num_rounds_per_step, seqs_per_step, avg_recompute_pct, avg_attn_flop_pct,avg_step_time, avg_tok_per_sec, avg_tflops, avg_mfu, avg_hfu);
		fclose(f);

		return 0;
	}
