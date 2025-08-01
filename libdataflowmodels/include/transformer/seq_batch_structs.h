#ifndef SEQ_BATCH_STRUCTS_H
#define SEQ_BATCH_STRUCTS_H

#include "dataflow.h"

typedef struct seq_batch Seq_Batch;


typedef struct seq_batch_metadata_offsets {
	
	// Embedding Stuff
	uint64_t token_ids;
	uint64_t sorted_token_ids;
	uint64_t sorted_token_mapping;
	uint64_t unique_token_sorted_inds_start;

	// Attention Stuff
	uint64_t seq_positions;
	uint64_t q_seq_offsets;
	uint64_t q_seq_lens;
	uint64_t k_seq_offsets;
	uint64_t k_seq_lens;

	// Loss Stuff
	uint64_t labels;
	uint64_t loss_vec;

	// MoE Stuff
	uint64_t host_expert_counts;

	uint64_t total_size;
} Seq_Batch_Metadata_Offsets;


typedef struct seq_batch_saved_activations_offsets {
	uint64_t x_inp;
	uint64_t attn_norm_rms_vals;
	uint64_t ffn_norm_rms_vals;
	uint64_t x_k_local;
	uint64_t x_v_local;

	// Save these guys no matter what...
	uint64_t x_routed;

	uint64_t chosen_experts;
	uint64_t token_expert_weights;

	// of size num_local_experts + 1
	// the +1 is used as counter for block completions within select experts kernel
	uint64_t expert_counts;

	// of size num_local_experts, excluse scan over expert counts
	// used when building mapping from experts to tokens
	uint64_t expert_counts_cumsum;

	// of size total_tokens * top_k
	uint64_t expert_mapping;

	uint64_t inp_only_cutoff;
	uint64_t softmax_lse;
	uint64_t x_attn_out;
	uint64_t inp_attn_only_cutoff;
	uint64_t x_q;
	uint64_t x_o;
	


	// MoE specific stuff
	// the individual partitioning of workspace for each expert
	// occurs dynamically after the select experts call
	int max_total_local_expert_tokens;
	// total number of shared and routed experts
	// each of these arrays are of size num_local_experts
	int num_local_experts;
	int top_k;

	// array of length num_local_experts
	uint64_t * x_1;
	uint64_t * x_3;

	uint64_t total_size;
} Seq_Batch_Saved_Activations_Offsets;




typedef struct seq_batch_recomputed_activations_offsets {
	uint64_t recomputed_attn_norm;
	uint64_t recomputed_ffn_norm;
	uint64_t total_size;
} Seq_Batch_Recomputed_Activations_Offsets;

typedef struct seq_batch_embedding_config {
	int total_tokens;
	int num_unique_tokens;
	// in order to do bwd pass we want to accumulate
	// gradients for each token id, so nice to have them
	// sorted for embedding_bwd_w


	// device pointers...

	// same form model_input
	uint32_t * token_ids;

	// can also use sorted_token_ids during fwd pass for
	// more efficient smem usage...
	uint32_t * sorted_token_ids;
	// mapping from sorted_token_ids to token_ids

	// this is mapping from index in sorted_token_ids to index in token_ids
	uint32_t * sorted_token_mapping;
	
	// length of num_unique_tokens + 1
	// each index represents the starting 
	// index of the token in sorted_token_ids
	// starts at 0 and ends at total_tokens - 1
	uint32_t * unique_token_sorted_inds_start;
} Seq_Batch_Embedding_Config;

typedef struct seq_batch_attention_config {
	// number of sequences part of this batch
	int num_seqs;

	// will be sum of q_seq_lens
	int total_q;
	// will be sum of k_seq_lens
	int total_k;

	// largest value from q_seq_lens
	int max_seqlen_q;

	// largest value from k_seq_lens
	int max_seqlen_k;

	// device pointers
	
	// for rope
	// of size total_q
	int * seq_positions;

	// of size num_seqs + 1
	// where index i represents starting
	// token offset of seq i. The value at q_seq_offsets[num_seqs] 
	// shoudl be q_seq_offsets[num_seqs - 1] + q_seq_lens[num_seqs - 1]
	int * q_seq_offsets;
	// of size num_seqs
	// q_seq_lens[i] represents total new queries to process for seq i,
	// starting at the corresponding offset, consecutively
	int * q_seq_lens;


	// of size num_seqs + 1; similar to q_seq_offsets, but now represents for kv cache
	// starting offsets. We can pass x_k, and x_v of different number of total
	// tokens (usually >= total_q) so during forwards pass the new queries
	// can utilize prior cached keys and values

	// during backwards pass if we start processing chunks from the end of the 
	// seq to the beginning, then we can also use this in order to  
	int * k_seq_offsets;
	// of size num_seqs; similar to q_seq_lens
	// Contains the number of keys we want to use for each sequence,
	// starting at the offset, consecutively
	int * k_seq_lens;
} Seq_Batch_Attention_Config;

typedef struct seq_batch_loss_config {

	// number of tokens to predict
	// for now assuming this is either 0, or seq_batch -> total_tokens...
	int num_tokens_to_predict;

	// of size num_tokens_to_predict
	// ignoring this for now...
	// uint32_t * inds_to_predict;

	// of size num_tokens_to_predict
	uint32_t * labels;

	// of size num_tokens_to_predict + 1
	// last index is the avg loss across all tokens in batch
	float * loss_vec;
} Seq_Batch_Loss_Config;

typedef struct seq_batch_recomputed_activations {
	uint64_t recomputedActivationsBufferBytes;
	void * recomputedActivationsBuffer;
	// during end of bwd_x (with still access to norm weights) we will populate these with recomputed RMS norms
    void * recomputed_attn_norm;
    void * recomputed_ffn_norm;

	uint64_t total_size;
} Seq_Batch_Recomputed_Activations;


typedef enum SavedActivationLevel {
	SAVED_ACTIVATION_LEVEL_NONE,
	SAVED_ACTIVATION_LEVEL_INP_ONLY,
	SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY,
	SAVED_ACTIVATION_LEVEL_FULL
} SavedActivationLevel;

typedef struct seq_batch_saved_activations {

	// the seq_batch this belongs to
	Seq_Batch * seq_batch;

	// the current saved activation level
	// this is set during populate_seq_batch_metadata_buffer
	// and used to determine what needs to be recomputed....
	SavedActivationLevel saved_activation_level;

	// the layer id
	int layer_id;

	// the buffer for the saved activations
	// might be bound to either host or device...
	void * savedActivationsBuffer;
	uint64_t savedActivationsBufferBytes;

	// This buffer should be the used as destination during
	// ring transfers, the input of each block is used during 
	// backprop so it should get saved down with the other 
	// activations...
	void * x_inp;

	// used during backprop
	float * attn_norm_rms_vals;
	// used during backprop
	float * ffn_norm_rms_vals;

	// These are the outputs of passing
	// normalized input through K and V weight
	// matrices, can use this to rebuild context
	// on host side if needed...
	void * x_k_local;
	void * x_v_local;

	// If saved_activation_level is SAVED_ACTIVATION_LEVEL_INP_ONLY, then
	// need to recompute fwd for attn

	// softmax_lse
	float * softmax_lse;
    void * x_attn_out;

	// If saved_activation_level is SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY, then
	// need to recompute fwd for x_q, x_o, and ffn

	void * x_q;
	void * x_o;
	

	void * x_routed;
	uint16_t * chosen_experts;
	float * token_expert_weights;

	// MoE specific stuff
	
    // if MoE, then this should be sent immediately after the select experts call
	// (after processing router)
	// and is needed to dynamically partition the expert workspace
	// of size num_local_experts + 1
	int * expert_counts;

	// these should also be sent immediately after the select experts call...
	int * expert_counts_cumsum;
	// num_tokens_per_expert result determines the boundaries of each expert...
	int * expert_mapping;

	// of size num_local_experts

	// need to save both in order to do correct
	// backprop through swiglu...
	void ** x_1;
	void ** x_3;

	uint64_t total_size;

	Seq_Batch_Recomputed_Activations * recomputed_activations;
} Seq_Batch_Saved_Activations;


typedef struct seq_batch_context {
	void * contextBuffer;
	uint64_t contextBufferBytes;

	int cur_tokens_populated;
	int total_context_tokens;

	// device pointers, but likely shared among multiple seq_batches
	// if it is a single seq broken into many chunks...

	// These are the matrices passed to attention
	// mechanism and may contain other state
	// (i.e. prior computed cached keys/values during fwd,
	// 		or accumulated gradients during backprop)
	void * x_k;
	void * x_v;
} Seq_Batch_Context;

typedef struct seq_batch_moe_config {
	int * host_expert_counts;
} Seq_Batch_MoE_Config;

struct seq_batch {
	
	int seq_id;
	int chunk_id;
	int total_tokens;
	int num_seqs;

	// host memory (part of pinned slab) that is reserved for token ids
	// and copied intoduring populate_seq_batch_metadata_buffer
	// useful for debuggingif we want to save down the raw token ids
	uint32_t * sys_token_ids;
	uint32_t * sys_labels;
	int * sys_seq_positions;

	Seq_Batch_Metadata_Offsets metadata_offsets;
	Seq_Batch_Saved_Activations_Offsets saved_activations_offsets;
	Seq_Batch_Recomputed_Activations_Offsets recomputed_activations_offsets;

	// the sum of size of device pointers from 
	void * devMetadataBuffer;
	uint64_t devMetadataBufferBytes;

	// the embedding config
	Seq_Batch_Embedding_Config embedding_config;

	// the attention config
	Seq_Batch_Attention_Config attention_config;

	// Can also add in metadata regarding MoE if needed...
	Seq_Batch_MoE_Config moe_config;

	// the loss config
	Seq_Batch_Loss_Config loss_config;

	// the global context if this is a single seq broken into many chunks...
	// (truncated backprop thorugh time...)
	// can be shared among multiple seq_batches
	Seq_Batch_Context * context;

	// the saved activations at each layer
	// array should be total_layers
	// should be bound to host memory,
	Seq_Batch_Saved_Activations * system_saved_activations;

	// the device memory saved activations buffers are passed in
	// as part of the transformer block and parameterized based on 
	// maximum seq_batch #tokens, and then bound during transformer block
	// start of processing...
};



// This can doesn't have to be tied to a seq_batch
// and can be parameterized based on maximum seq_batch #tokens...

// Can just keep a copy laying around and pass in as needed...

typedef struct seq_batch_activation_workspace {
	// the buffer for the activation workspace
	void * activationWorkspaceBuffer;
	uint64_t activationWorkspaceBytes;

	// used as temporary buffer during
	// norm outputs
	void * x_temp;

	// used as temporary output buffer during
	// MLP

	// could have more MoE specific temp buffer stuff here...

	// needs to be total_q * ffn_dim
	// used as buffer to hold the combined/activated
	// gate for x1 and x3...
	void * x_temp_mlp;

	// workspace for attention and matmuls in block...
	// needs to be zeroed out before attention fwd and bwd...
	void * kernelWorkspace;
	uint64_t kernelWorkspaceBytes;
} Seq_Batch_Activation_Workspace;


#endif // SEQ_BATCH_STRUCTS_H