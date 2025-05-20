#ifndef TRANSFORMER_STRUCTS_H
#define TRANSFORMER_STRUCTS_H

#include "dataflow.h"
#include "dataflow_models.h"

#include "seq_batch_structs.h"


typedef struct transformer_block_weight_offsets {
	uint64_t w_attn_norm;
	uint64_t w_q;
	uint64_t w_k;
	uint64_t w_v;
	uint64_t w_o;
	uint64_t w_ffn_norm;
	uint64_t w_router;
	// arrays of length num_local experts if DATAFLOW_MOE_MLP
	// otherwise arrays of length 1
	uint64_t * w_1;
	uint64_t * w_2;
	uint64_t * w_3;
} Transformer_Block_Weight_Offsets;

typedef struct transformer_block_config {
	DataflowDatatype block_dt;
	// for Matmul accumulation
	// if on Geforce using FP16 gives twice as much perf.
	DataflowDatatype compute_dt;
	DataflowNormalizationType normalization_type;
	DataflowPositionEmbeddingType position_embedding_type;
	DataflowAttentionType attention_type;
	DataflowMLPType mlp_type;
	DataflowActivationType activation_type;

	float eps;
	int theta;

	int num_q_heads;
	int num_kv_heads;
	int head_dim;

	// We set these
	// model_dim = num_q_heads * head_dim
	// kv_dim = num_kv_heads * head_dim
	int model_dim;
	int kv_dim;

	// if using mlp_type = DATAFLOW_MOE_MLP
	// this ffn_dim is the dimension of each expert
	int ffn_dim;

	// for models with mlp_type = DATAFLOW_MOE_MLP
	MoE_Config moe_config;
		
	// each unqiue weight pointer
	// must be a multiple of this value
	// (e.g. 256 to use tensor cores, or maybe 512 to load direct from SSD, or maybe 4k to start on unique sys ram pages, etc...)
	int pointer_alignment;

	Transformer_Block_Weight_Offsets weight_offsets;
	// the total amount of memory required
	// to hold all weights
	uint64_t block_raw_size;
	
	// the total amount of memory
	uint64_t block_aligned_size;
	
} Transformer_Block_Config;

typedef struct transformer_block {
	Transformer_Block_Config config;
	int layer_id;
	void * buffer;
	// offsets into buffer, based upon
	// config.weight_offsets
	void * w_attn_norm;
	void * w_q;
	void * w_k;
	void * w_v;
	void * w_o;
	void * w_ffn_norm;
	// if non-moe this is null
	void * w_router;
	// each of these 
	// are arrays of length = MAX(config.num_local_experts, 1)
	// (if non-moe, just use index 0)

	// if DATAFLOW_VANILLA_MLP, then w_3 should all be null
	void ** w_1;
	void ** w_2;
	void ** w_3;
} Transformer_Block;

typedef struct transformer_head {
	DataflowDatatype fwd_dt;
	DataflowDatatype bwd_dt;
	DataflowDatatype compute_dt;
	float eps;
	// contains the vocab size which 
	// has dimensionality out of w_out output
	Embedding_Config * embedding_config;
	void * buffer;
	void * w_head_norm;
	void * w_head;
} Transformer_Head;

typedef struct transformer_block_activations {
	// this holds pointer on device memory and correct pointers
	// are bound (size + alignment) during beginning of processing
	// the device buffers are passed in during binding process
	// and are configuraed to be hold the max # tokens per seq_batch
	Seq_Batch_Saved_Activations * working_activations;
	// similar to working_activations, but this is used as a temporary buffer
	Seq_Batch_Activation_Workspace * activation_workspace;
} Transformer_Block_Activations;


typedef struct transformer_head_activations {
	int num_tokens;
	int total_pred_tokens_in_step;

	void * buffer;
	// tempporary buffer for norm output
	void * head_norm_out;
	void * head_norm_rms_vals;
	void * head_out;

	// workspace for matmuls in head...
	void * kernelWorkspace;
	uint64_t kernelWorkspaceBytes;
} Transformer_Head_Activations;

typedef struct transformer_block_transition {
	Seq_Batch * seq_batch;
	void * X;
} Transformer_Block_Transition;

typedef struct transformer_embedding_table {
	Embedding_Config * config;
	uint64_t embedding_table_size;
	void * embedding_table;
} Transformer_Embedding_Table;

typedef struct transformer_model_input {
	Seq_Batch * seq_batch;
} Transformer_Model_Input;

typedef struct transformer_model_output {
	Seq_Batch * seq_batch;
	void * logits;
} Transformer_Model_Output;

#endif