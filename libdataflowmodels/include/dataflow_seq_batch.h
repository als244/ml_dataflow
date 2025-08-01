#ifndef DATAFLOW_SEQ_BATCH_H
#define DATAFLOW_SEQ_BATCH_H

#include "dataflow_models.h"
#include "transformer/transformer_structs.h"
#include "transformer/seq_batch_structs.h"


uint64_t get_seq_batch_metadata_buffer_size(int num_seqs, int total_tokens);
int init_seq_batch_offsets(Seq_Batch * seq_batch, int total_tokens, int num_seqs, Transformer_Block_Config * block_config, int max_total_local_expert_tokens);

int bind_seq_batch_metadata_buffer(Seq_Batch * seq_batch, void * metadata_buffer, uint64_t metadata_buffer_size);

// after bind has already been called...

// q/k seqlens and offsets are relative to the global context if this batch is just one chunk of longer sequence,
// or they are relative to the current batch's context (with size = total_tokens) if this batch contains multiple chunks of shorter sequences....
int populate_seq_batch_metadata_buffer(Dataflow_Handle * dataflow_handle, int inbound_stream_id, 
                                        Seq_Batch * seq_batch,
                                        void * sys_registered_metadata_buffer, uint64_t sys_registered_metadata_buffer_size,
                                        int seq_id, int chunk_id, int total_tokens, int num_seqs,
                                        uint32_t * sys_token_ids, uint32_t * sys_labels,
                                        int * sys_seq_positions, 
                                        int * sys_q_seq_offsets, int * sys_q_seq_lens,
                                        int * sys_k_seq_offsets, int * sys_k_seq_lens,
                                        int * sys_host_expert_count_buffer);




// These bindings happen dynamically...
// Can use information already in seq batch to help set other structs...

// the pointers to buffers will be likely coming from a fifo queue and bound when ready, 
// at which point the submit_block can be called...

uint64_t get_seq_batch_saved_activations_buffer_size(Seq_Batch * seq_batch, SavedActivationLevel saved_activation_level);
int bind_seq_batch_saved_activations_buffer(Seq_Batch * seq_batch, Seq_Batch_Saved_Activations * saved_activations, void * saved_activations_buffer, SavedActivationLevel saved_activation_level, uint64_t saved_activations_buffer_size,
                                            int layer_id);

uint64_t get_seq_batch_activation_workspace_buffer_size(Seq_Batch * seq_batch, Transformer_Block_Config * block_config);

uint64_t get_seq_batch_recomputed_activations_buffer_size(Seq_Batch * seq_batch);
int bind_seq_batch_recomputed_activations_buffer(Seq_Batch_Recomputed_Activations_Offsets * recomputed_activations_offsets, Seq_Batch_Recomputed_Activations * recomputed_activations, void * recomputed_activations_buffer, uint64_t recomputed_activations_buffer_size);

int set_moe_working_activations_offsets(Seq_Batch_Saved_Activations * working_activations, int num_tokens, int model_dim, int expert_dim, 
                                            int num_shared_experts, int num_routed_experts, int * host_expert_counts, size_t expert_dt_size);

#endif
