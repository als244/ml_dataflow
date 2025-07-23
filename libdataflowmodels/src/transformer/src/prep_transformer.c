#include "transformer/prep_transformer.h"

static void set_offset(uint64_t * cur_offset, uint64_t cur_size, uint64_t * raw_size, uint64_t * aligned_size, int pointer_alignment){

	if (cur_size == 0){
		*cur_offset = *aligned_size;
		return;
	}

	// set it equal to current aligned size
	*cur_offset = *aligned_size;

	// now increment raw size and aligned size
	*raw_size += cur_size;
	*aligned_size += cur_size;

	if ((pointer_alignment > 0) && (*aligned_size % pointer_alignment > 0)) {
		*aligned_size += pointer_alignment - (*aligned_size % pointer_alignment);
	}

	return;
}

static int set_transformer_block_weight_offsets(Transformer_Block_Config * config, 
													int pointer_alignment, uint64_t * ret_raw_size, uint64_t * ret_aligned_size){


	Transformer_Block_Weight_Offsets * weight_offsets = &(config -> weight_offsets);
	memset(weight_offsets, 0, sizeof(Transformer_Block_Weight_Offsets));


	uint64_t raw_size = 0;
	uint64_t aligned_size = 0;

	DataflowDatatype block_dt = config -> block_dt;

	uint64_t el_size = dataflow_sizeof_element(block_dt);

	if (el_size == 0){
		fprintf(stderr, "Error: cannot set block weight offsets. Element Size is 0 for el type of %s...\n", dataflow_datatype_as_string(block_dt));
		return -1;
	}

	uint64_t model_dim = (uint64_t) config -> model_dim;
	uint64_t kv_dim = (uint64_t) config -> kv_dim;


	uint64_t w_norm_size;

	switch (config -> normalization_type){
		case DATAFLOW_RMSNORM:
			w_norm_size = model_dim * el_size;
			break;
		default:
			fprintf(stderr, "Error: cannot set block weight offsets. Normalization type of %d is not supported yet...\n", config -> normalization_type);
			return -1;
	}


	uint64_t w_q_size;
	uint64_t w_k_size;
	uint64_t w_v_size;
	uint64_t w_o_size;

	switch (config -> attention_type){
		case DATAFLOW_EXACT_ATTENTION:
			w_q_size = model_dim * model_dim * el_size;
			w_k_size = model_dim * kv_dim * el_size;
			w_v_size = model_dim * kv_dim * el_size;
			w_o_size = model_dim * model_dim * el_size;
			break;
		default:
			fprintf(stderr, "Error: cannot set block weight offsets. Attention type of %d is not supported yet...\n", config -> attention_type);
			return -1;
	}

	// if non MOE, these will have been configured to be set to 1
	uint64_t num_global_routed_experts = (uint64_t) (config -> moe_config).num_global_routed_experts;
	uint64_t num_local_experts = (uint64_t) (config -> moe_config).num_local_experts;
	uint64_t w_router_size;

	uint64_t ffn_dim = (uint64_t) config -> ffn_dim;
	uint64_t w_1_size;
	uint64_t w_2_size;
	uint64_t w_3_size;

	switch (config -> mlp_type){
		case DATAFLOW_GATED_MLP:
			w_router_size = 0;
			w_1_size = model_dim * ffn_dim * el_size;
			w_2_size = ffn_dim * model_dim * el_size;
			w_3_size = model_dim * ffn_dim * el_size;
			break;
		case DATAFLOW_VANILLA_MLP:
			w_router_size = 0;
			w_1_size = model_dim * ffn_dim * el_size;
			w_2_size = ffn_dim * model_dim * el_size;
			w_3_size = 0;
			break;
		case DATAFLOW_MOE_MLP:
			w_router_size = model_dim * num_global_routed_experts * el_size;
			w_1_size = model_dim * ffn_dim * el_size;
			w_2_size = ffn_dim * model_dim * el_size;
			w_3_size = model_dim * ffn_dim * el_size;
			break;
		default:
			fprintf(stderr, "Error: cannot set block weight offsets. MLP type of %d is not supported yet...\n", config -> attention_type);
			return -1;
	}

	// num_local_experts == 1 for non moe
	weight_offsets -> w_1 = malloc(num_local_experts * sizeof(uint64_t));
	if (!(weight_offsets -> w_1)){
		fprintf(stderr, "Error: cannot set block weight offsets. malloc failed to alloc space for holding ffn offsets...\n");
		return -1;
	}
	weight_offsets -> w_2 = malloc(num_local_experts * sizeof(uint64_t));
	if (!(weight_offsets -> w_2)){
		fprintf(stderr, "Error: cannot set block weight offsets. malloc failed to alloc space for holding ffn offsets...\n");
		free(weight_offsets -> w_1);
		return -1;
	}
	weight_offsets -> w_3 = malloc(num_local_experts * sizeof(uint64_t));
	if (!(weight_offsets -> w_3)){
		fprintf(stderr, "Error: cannot set block weight offsets. malloc failed to alloc space for holding ffn offsets...\n");
		free(weight_offsets -> w_1);
		free(weight_offsets -> w_2);
		return -1;
	}

	set_offset(&(weight_offsets -> w_attn_norm), w_norm_size, &raw_size, &aligned_size, pointer_alignment);
	set_offset(&(weight_offsets -> w_q), w_q_size, &raw_size, &aligned_size, pointer_alignment);
	set_offset(&(weight_offsets -> w_k), w_k_size, &raw_size, &aligned_size, pointer_alignment);
	set_offset(&(weight_offsets -> w_v), w_v_size, &raw_size, &aligned_size, pointer_alignment);
	set_offset(&(weight_offsets -> w_o), w_o_size, &raw_size, &aligned_size, pointer_alignment);
	set_offset(&(weight_offsets -> w_ffn_norm), w_norm_size, &raw_size, &aligned_size, pointer_alignment);

	// For non-MoE this will have size = 0 => same offset as w_ffn_norm => w_router poitner set to null
	set_offset(&(weight_offsets -> w_router), w_router_size, &raw_size, &aligned_size, pointer_alignment);

	// now set all w_1, w_2, w_3
	// use ordering of w_1, w_3, and w_2 for
	// clarity on ordering of sequential depedendiences within block
	// but grouping expert all in same region

	for (int i = 0; i < num_local_experts; i++){
		set_offset(&((weight_offsets -> w_1)[i]), w_1_size, &raw_size, &aligned_size, pointer_alignment);
		set_offset(&((weight_offsets -> w_3)[i]), w_3_size, &raw_size, &aligned_size, pointer_alignment);
		set_offset(&((weight_offsets -> w_2)[i]), w_2_size, &raw_size, &aligned_size, pointer_alignment);
	}

	*ret_raw_size = raw_size;
	*ret_aligned_size = aligned_size;

	return 0;
}


Transformer_Block * init_transformer_block(int layer_id, DataflowDatatype block_dt, DataflowDatatype compute_dt,
						   DataflowNormalizationType normalization_type, 
						   DataflowPositionEmbeddingType position_embedding_type,
						   DataflowAttentionType attention_type,
						   DataflowMLPType mlp_type,
						   DataflowActivationType activation_type,
						   float eps, int theta,
						   int num_q_heads, int num_kv_heads, int head_dim,
						   int ffn_dim,
						   MoE_Config * moe_config,
						   int pointer_alignment) {

	int ret;

	Transformer_Block * block = malloc(sizeof(Transformer_Block));
	if (!block){
		fprintf(stderr, "Error: malloc failed to allocate block container...\n");
		return NULL;
	}
	memset(block, 0, sizeof(Transformer_Block));

	block -> layer_id = layer_id;


	(block -> config).block_dt = block_dt;
	(block -> config).compute_dt = compute_dt;


	(block -> config).normalization_type = normalization_type;
	(block -> config).position_embedding_type = position_embedding_type;
	(block -> config).attention_type = attention_type;
	(block -> config).mlp_type = mlp_type;

	(block -> config).eps = eps;
	(block -> config).theta = theta;

	(block -> config).num_q_heads = num_q_heads;
	(block -> config).num_kv_heads = num_kv_heads;
	(block -> config).head_dim = head_dim;

	(block -> config).model_dim = num_q_heads * head_dim;
	(block -> config).kv_dim = num_kv_heads * head_dim;


	(block -> config).ffn_dim = ffn_dim;

	if (moe_config){
		if (mlp_type != DATAFLOW_MOE_MLP){
			fprintf(stderr, "Error: specified a moe_config, but mlp type is not set to DATAFLOW_MOE_MLP...\n");
			free(block);
			return NULL;
		}


		if (moe_config -> num_global_routed_experts <= 0){
			fprintf(stderr, "Error: specified a moe_config, number of global routed experts must be > 0...\n");
			free(block);
			return NULL;
		}

		if (moe_config -> num_local_experts > (moe_config -> num_shared_experts + moe_config -> num_global_routed_experts)){
			fprintf(stderr, "Error: specified a moe_config, number of local experts must be <= number of shared experts + number of global routed experts...\n");
			free(block);
			return NULL;
		}


		if ((moe_config -> top_k_experts <= 0) || (moe_config -> top_k_experts > (moe_config -> num_shared_experts + moe_config -> num_global_routed_experts))){
			fprintf(stderr, "Error: specified a moe_config, but top_k experts must be [0, num_shared_experts + num_global_routed_experts], but is set to %d...\n", moe_config -> top_k_experts);
			free(block);
			return NULL;
		}

		(block -> config).moe_config.top_k_experts = moe_config -> top_k_experts;
		(block -> config).moe_config.num_shared_experts = moe_config -> num_shared_experts;
		(block -> config).moe_config.num_global_routed_experts = moe_config -> num_global_routed_experts;
		(block -> config).moe_config.num_local_experts = moe_config -> num_local_experts;

		(block -> config).moe_config.local_expert_inds = malloc(moe_config -> num_local_experts * sizeof(int));
		if (!(block -> config).moe_config.local_expert_inds){
			fprintf(stderr, "Error: specified a moe_config and malloc failed to allocate buffer to copy local expert inds...\n");
			free(block);
			return NULL;
		}

		memcpy((block -> config).moe_config.local_expert_inds, moe_config -> local_expert_inds, moe_config -> num_local_experts * sizeof(int));
	}
	else{
		if (mlp_type == DATAFLOW_MOE_MLP){
			fprintf(stderr, "Error: did not specify a moe_config, but mlp type is set to DATAFLOW_MOE_MLP...\n");
			free(block);
			return NULL;
		}

		(block -> config).moe_config.top_k_experts = 1;
		(block -> config).moe_config.num_shared_experts = 1;
		(block -> config).moe_config.num_global_routed_experts = 0;
		(block -> config).moe_config.num_local_experts = 1;
		(block -> config).moe_config.local_expert_inds = malloc(sizeof(int));
		(block -> config).moe_config.local_expert_inds[0] = 0;

	}


	// dont allow moe yet
	if (mlp_type == DATAFLOW_MOE_MLP){
		fprintf(stderr, "Error: MOE MLP not yet unsupported...\n");
		free((block -> config).moe_config.local_expert_inds);
		free(block);
		return NULL;
	}

	if (mlp_type == DATAFLOW_VANILLA_MLP){
		fprintf(stderr, "Error: VANILLA MLP not yet unsupported...\n");
		free((block -> config).moe_config.local_expert_inds);
		free(block);
		return NULL;
	}


	(block -> config).pointer_alignment = pointer_alignment;

	ret = set_transformer_block_weight_offsets(&(block -> config), 
												pointer_alignment, &((block -> config).block_raw_size), &((block -> config).block_aligned_size));

	if (ret){
		fprintf(stderr, "Error: could not set transformer block weights...\n");
		free((block -> config).moe_config.local_expert_inds);
		free(block);
		return NULL;
	}

	return block;
}

uint64_t get_transformer_block_raw_size(Transformer_Block * transformer_block) {
	return (transformer_block -> config).block_raw_size;
}

uint64_t get_transformer_block_aligned_size(Transformer_Block * transformer_block) {
	return (transformer_block -> config).block_aligned_size;
}


// now pass in a buffer of size >= size specified above
// and the pointers will be properly assigned (ensuring alignment)
int bind_transformer_block(void * buffer, Transformer_Block * transformer_block) {

	// set base buffer
	transformer_block -> buffer = buffer;

	Transformer_Block_Weight_Offsets * weight_offsets = &(transformer_block -> config).weight_offsets;


	transformer_block -> w_attn_norm = buffer + weight_offsets -> w_attn_norm;
	transformer_block -> w_q = buffer + weight_offsets -> w_q;
	transformer_block -> w_k = buffer + weight_offsets -> w_k;
	transformer_block -> w_v = buffer + weight_offsets -> w_v;
	transformer_block -> w_o = buffer + weight_offsets -> w_o;
	transformer_block -> w_ffn_norm = buffer + weight_offsets -> w_ffn_norm;

	// this will be set to 1 for non-MoE MLP types...
	int num_local_experts = (transformer_block -> config).moe_config.num_local_experts;
	transformer_block -> w_1 = malloc(num_local_experts * sizeof(void *));
	if (!transformer_block -> w_1){
		fprintf(stderr, "Error: cannot bind transformer block. malloc failed to alloc space for holding ffn pointers...\n");
		return -1;
	}

	transformer_block -> w_2 = malloc(num_local_experts * sizeof(void *));
	if (!transformer_block -> w_2){
		fprintf(stderr, "Error: cannot bind transformer block. malloc failed to alloc space for holding ffn pointers...\n");
		free(transformer_block -> w_1);
		return -1;
	}

	transformer_block -> w_3 = malloc(num_local_experts * sizeof(void *));
	if (!transformer_block -> w_3){
		fprintf(stderr, "Error: cannot bind transformer block. malloc failed to alloc space for holding ffn pointers...\n");
		free(transformer_block -> w_1);
		free(transformer_block -> w_2);
		return -1;
	}


	for (int i = 0; i < num_local_experts; i++){
		(transformer_block -> w_1)[i] = buffer + (weight_offsets -> w_1)[i];
		(transformer_block -> w_3)[i] = buffer + (weight_offsets -> w_3)[i];
		(transformer_block -> w_2)[i] = buffer + (weight_offsets -> w_2)[i];
	}

	return 0;
}

// the file consists of combined weights for block. 
// the block should have already been initialized and bound to buffer
int load_transformer_block(char * filename, Transformer_Block * transformer_block){

	FILE * file = fopen(filename, "rb");
	if (!file){
		fprintf(stderr, "Error: could not open file %s...\n", filename);
		return -1;
	}

	// For now, assuming loading into host....
	
	size_t nread = fread(transformer_block -> buffer, 1, transformer_block -> config.block_aligned_size, file);
	if (nread != transformer_block -> config.block_aligned_size){
		fprintf(stderr, "Error: could not read file %s. Expected %zu bytes (aligned size), but read %zu bytes (file size)...\n", filename, transformer_block -> config.block_aligned_size, nread);
		fclose(file);
		return -1;
	}

	fclose(file);

	return 0;
}

// int bind_transformer_block_activations(void * buffer, Seq_Batch * seq_batch, Transformer_Block * block, Transformer_Block_Activations * activation_buffer) {
// 	return -1;
// }