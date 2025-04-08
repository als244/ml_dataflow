#ifndef SEQ_BATCH_STRUCTS_H
#define SEQ_BATCH_STRUCTS_H

#include "dataflow.h"

typedef struct seq_batch_config {
	int num_seqs;

	// will be sum of q_seq_lens
	int total_q;
	// will be sum of k_seq_lens
	int total_k;
	
	// for rope
	// of size total_q
	int * seq_positions;

	// for attn

	// of size num_seqs + 1
	// where index i represents starting
	// token offset of seq i. The value at q_seq_offsets[num_seqs] 
	// shoudl be q_seq_offsets[num_seqs - 1] + q_seq_lens[num_seqs - 1]
	int * q_seq_offsets;
	// of size num_seqs
	// q_seq_lens[i] represents total new queries to process for seq i,
	// starting at the corresponding offset, consecutively
	int * q_seq_lens;
	// largest value from q_seq_lens
	int max_seqlen_q;

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
	// largest value from k_seq_lens
	int max_seqlen_k;

	// Can also add in metadata regarding MoE if needed...
} Seq_Batch_Config;

typedef struct seq_batch_saved_activations {
	
	Seq_Batch_Config * config;
	void * buffer;
	uint64_t activationBufferBytes;

	// used during backprop
	void * attn_norm_weighted_sums;
	void * attn_norm_rms_vals;
	void * x_q;

	// These are the outputs of passing
	// normalized input through K and V weight
	// matrices
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
} Seq_Batch_Saved_Activations;


typedef struct seq_batch_context {
	Seq_Batch_Config * config;
	void * buffer;
	uint64_t contextBufferBytes;

	// Can use copy_to_seq_cache to move x_k_local (post rope)
	// and x_v_local to proper locations within these matrices
	// if there are multiple seqs and also prior caching involved...

	// These are the matrices passed to attention
	// mechanism and may contain other state
	// (i.e. prior computed cached keys/values during fwd,
	// 		or accumulated gradients during backprop)
	void * x_k;
	void * x_v;
} Seq_Batch_Context;

typedef struct seq_batch_activation_workspace {
	Seq_Batch_Config * config;

	void * buffer;
	uint64_t activationWorkspaceBytes;

	// used as temporary buffer during
	// norm outputs
	void * x_temp;

	// used as temporary output buffer during
	// MLP

	// needs to be total_q * ffn_dim
	void * x_temp_mlp;


    // during end of bwd_x (with still access to norm weights) we will populate these with recomputed RMS norms
    void * recomputed_attn_norm;
    void * recomputed_ffn_norm;

	// workspace for matmul and attention
	// for attention needs to be zeroed out...
	void * kernel_workspace;
	uint64_t kernel_workspaceBytes;
} Seq_Batch_Activation_Workspace;

#endif