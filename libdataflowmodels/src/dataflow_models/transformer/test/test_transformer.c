#include "dataflow_transformer.h"
#include "dataflow_seq_batch.h"
#include "cuda_dataflow_handle.h"
#include "register_ops.h"
int main(int argc, char * argv[]){

	int ret;

	DataflowDatatype block_dt = DATAFLOW_BF16;

	// for matmul accumulations...
	// on Geforce using FP16 gets double perf,
	// on datacenter cards should use DATAFLOW_FP32
	DataflowDatatype compute_dt = DATAFLOW_BF16;


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

	MoE_Config * moe_config = NULL;


	// setting to host page size.
	// really needs to be 256 in order to use tensor cores
	// depending on filesystem in order to use O_RDONLY | O_DIRECT, alignment may be different...
	
	
	// for now using 0 alignment to directly read from combined file...
	int pointer_alignment = 256;

	Transformer_Block * block = init_transformer_block(block_dt, compute_dt,
														norm_type, pos_emb_type, attn_type, mlp_type, activ_type,
														eps, theta,
														num_q_heads, num_kv_heads, head_dim,
														ffn_dim,
														moe_config,
														pointer_alignment);


	if (!block){
		fprintf(stderr, "Error: failed to init transformer block...\n");
		return -1;
	}

	uint64_t raw_size = get_transformer_block_raw_size(block);
	uint64_t aligned_size = get_transformer_block_aligned_size(block);


	printf("\nTransformer Block Sizes (bytes):\n\tRaw: %lu\n\tAligned (%d): %lu\n\n", raw_size, pointer_alignment, aligned_size);


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
	ret = init_seq_batch_offsets(seq_batch, total_tokens, num_seqs, &(block -> config), max_total_local_expert_tokens);
	if (ret){
		fprintf(stderr, "Error: failed to init seq_batch offsets...\n");
		return -1;
	}


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



	// 16 GB...
	void * host_mem;

	int host_alignment = 4096;
	size_t host_size_bytes = 1UL << 34;

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

	FILE * fp = fopen("data/token_ids_uint32.dat", "rb");
	if (!fp){
		fprintf(stderr, "Error: failed to open data/token_ids_uint32.dat...\n");
		return -1;
	}


	size_t read_els;
	read_els = fread(sys_token_ids, sizeof(uint32_t), total_tokens, fp);
	if (read_els != total_tokens){
		fprintf(stderr, "Error: failed to read token_id_uint32.dat, read_els: %zu, expected: %d\n", read_els, total_tokens);
		return -1;
	}
	fclose(fp);

	uint32_t * sys_labels = malloc(total_tokens * sizeof(uint32_t));

	fp = fopen("data/labels_uint32.dat", "rb");
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


	/* IF we are following up with embedding, ensure to wait for inbound stream....


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
	*/


	// For now just calling sync_stream to inspect outputs (with cuda-gdb...)
	printf("Waiting for data transfer of metadata buffer to complete...\n\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to sync inbound stream...\n");
		return -1;
	}

	printf("Succeeded!\n\nReady for embedding...\n");

	return 0;
}