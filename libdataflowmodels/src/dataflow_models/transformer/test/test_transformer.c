#include "dataflow_transformer.h"

int main(int argc, char * argv[]){

	DataflowDatatype block_dt = DATAFLOW_FP16;

	// for matmul accumulations...
	// on Geforce using FP16 gets double perf,
	// on datacenter cards should use DATAFLOW_FP32
	DataflowDatatype compute_dt = DATAFLOW_FP16;


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
	int pointer_alignment = 0;

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


	printf("Transformer Block Sizes (bytes):\n\tRaw: %lu\n\tAligned (%d): %lu\n\n", raw_size, pointer_alignment, aligned_size);

	return 0;
}