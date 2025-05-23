#ifndef DATAFLOW_MODELS_H
#define DATAFLOW_MODELS_H

#include "dataflow.h"

typedef enum dataflow_normalization_type {
	DATAFLOW_RMSNORM,
	// not supported yet
	DATAFLOW_LAYERNORM
} DataflowNormalizationType;

typedef enum dataflow_position_embedding_type {
	DATAFLOW_ROPE,
	DATAFLOW_NOPE,
	// not supported yet
	DATAFLOW_ALIBI
} DataflowPositionEmbeddingType;

typedef enum dataflow_actviation_type {
	DATAFLOW_SWIGLU,
	// not supported yet, could be epilogue of matmul
	DATAFLOW_RELU,
	// not supported yet, could be epilogue of matmul
	DATAFLOW_GELU
} DataflowActivationType;

typedef enum dataflow_attention_type {
	DATAFLOW_EXACT_ATTENTION,
	// DeepSeek NSA, not supported yet
	DATAFLOW_SPARSE_ATTENTION
} DataflowAttentionType;

typedef enum dataflow_mlp_type {
	DATAFLOW_GATED_MLP,
	// not supported yet
	DATAFLOW_VANILLA_MLP,
	// not supported yet
	DATAFLOW_MOE_MLP,
} DataflowMLPType;


typedef struct embedding_config {
	// number of unique tokens that
	// can be 1-hot encoded
	int vocab_size;
	// should = model_dim for transformers
	int embedding_size;
	DataflowDatatype embed_dt;
} Embedding_Config;

typedef struct moe_config {
	int top_k_experts;
	int num_shared_experts;
	// the total number of experts per-block part of 
	// model spec
	int num_global_routed_experts;
	// number of experts held within the block's model weights
	// this number includes both shared and routed experts
	// shared experts start from index 0 and go up to num_shared_experts - 1
	// routed experts start from index num_shared_experts and go up to 
	// num_shared_experts + num_global_routed_experts - 1
	int num_local_experts;
	// of size num_local_experts (if > 0, else Null)
	// and contains the expert indices (relative to shared + global experts)
	// held within this block's model weights
	int * local_expert_inds;
} MoE_Config;



#endif