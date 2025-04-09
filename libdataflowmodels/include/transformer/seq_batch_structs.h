#ifndef SEQ_BATCH_STRUCTS_H
#define SEQ_BATCH_STRUCTS_H

#include "dataflow.h"


typedef struct seq_batch Seq_Batch;
typedef struct seq_batch_embedding_config {

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
} Seq_Batch_Loss_Config;

typedef struct seq_batch_saved_activations {

	// the seq_batch this belongs to
	Seq_Batch * seq_batch;

	// the layer id
	int layer_id;

	// the buffer for the saved activations
	// might be bound to either host or device...
	void * savedActivationsBuffer;
	uint64_t savedActivationsBufferBytes;

	// used during backprop
	void * attn_norm_weighted_sums;
	void * attn_norm_rms_vals;
	void * x_q;

	// These are the outputs of passing
	// normalized input through K and V weight
	// matrices, can use this to rebuild context
	// on host side if needed...
	void * x_k_local;
	void * x_v_local;

	// softmax_lse
	void * softmax_lse;
    void * x_attn_out;
	void * x_o;
	// used during backprop
	void * ffn_norm_weighted_sums;
	void * ffn_norm_rms_vals;
	void ** x_1;
	void ** x_2;
	void ** x_3;
	void * x_layer_out;

	void * recomputedActivationsBuffer;
	uint64_t recomputedActivationsBufferBytes;
    // during end of bwd_x (with still access to norm weights) we will populate these with recomputed RMS norms
    void * recomputed_attn_norm;
    void * recomputed_ffn_norm;
} Seq_Batch_Saved_Activations;


typedef struct seq_batch_context {
	void * contextBuffer;
	uint64_t contextBufferBytes;

	// device pointers, but likely shared among multiple seq_batches
	// if it is a single seq broken into many chunks...

	// These are the matrices passed to attention
	// mechanism and may contain other state
	// (i.e. prior computed cached keys/values during fwd,
	// 		or accumulated gradients during backprop)
	void * x_k;
	void * x_v;
} Seq_Batch_Context;


struct seq_batch {
	
	int total_tokens;

	// the sum of size of device pointers from 
	void * devConfigBuffer;
	uint64_t devConfigBufferBytes;

	// the embedding config
	Seq_Batch_Embedding_Config embedding_config;

	// the attention config
	Seq_Batch_Attention_Config attention_config;

	// Can also add in metadata regarding MoE if needed...

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

typedef struct seq_batch_activation_workspace {
	// the buffer for the activation workspace
	void * activationWorkspaceBuffer;
	uint64_t activationWorkspaceBytes;

	// used as temporary buffer during
	// norm outputs
	void * x_temp;

	// used as temporary output buffer during
	// MLP

	// needs to be total_q * ffn_dim
	void * x_temp_mlp;
} Seq_Batch_Activation_Workspace;


#endif