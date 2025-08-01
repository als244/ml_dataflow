#ifndef DATAFLOW_TRANSFORMER_H
#define DATAFLOW_TRANSFORMER_H

#include "dataflow_models.h"
#include "dataflow_ops.h"
#include "dataflow_seq_batch.h"

#include "transformer/transformer_structs.h"

Transformer_Model_Config * parse_config(char * config_path);

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
						   int pointer_alignment);

uint64_t get_transformer_block_raw_size(Transformer_Block * transformer_block);

uint64_t get_transformer_block_aligned_size(Transformer_Block * transformer_block);


// now pass in a buffer of size >= size specified above
// and the pointers will be properly assigned (ensuring alignment)
int bind_transformer_block(void * buffer, Transformer_Block * transformer_block);


// the file consists of combined weights for block. 
// the block should have already been initialized and bound to buffer
int load_transformer_block(char * filename, Transformer_Block * transformer_block);


// Need to set Seq Batch metadata...!
//int bind_transformer_block_activations(void * buffer, Seq_Batch * seq_batch, Transformer_Block * block, Transformer_Block_Activations * activation_buffer);


int dataflow_submit_transformer_embedding(Dataflow_Handle * dataflow_handle, int compute_stream_id,
											Transformer_Model_Input * model_input,
											Transformer_Embedding_Table * embedding_table,
											Transformer_Block_Transition * embedding_output);


int dataflow_submit_transformer_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
								Transformer_Block_Transition * block_input, 
								Transformer_Block * transformer_block, 
								Transformer_Block_Activations * activations, 
								Transformer_Block_Transition * block_output);

int dataflow_submit_transformer_head(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                        Transformer_Block_Transition * block_input, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations, 
                        Transformer_Model_Output * model_output,
						// during interference these would be NULL
						Transformer_Head * grad_transformer_head,
						Transformer_Block_Transition * grad_stream);
						

// Assumse that working_activations has been already populated in device memory with the appropriate
// saved activation level amount of data alreay ready to go (prefix of the this chunk). Assumes that
// there is enough memory to save the forward activations for the entire block (repeating the forward pass)
// and that the bwd_x function will be called next!
int dataflow_submit_transformer_block_recompute(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
												Transformer_Block * transformer_block,
												Seq_Batch * seq_batch,
												SavedActivationLevel saved_activation_level,
												Seq_Batch_Saved_Activations * working_activations, Seq_Batch_Context * fwd_context,
												Seq_Batch_Activation_Workspace * activation_workspace);


int dataflow_submit_transformer_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id,
								Transformer_Block * transformer_block, 
								Transformer_Block_Transition * inp_grad_stream, 
								Seq_Batch_Saved_Activations * fwd_activations, Seq_Batch_Context * fwd_context,
								Transformer_Block_Activations * grad_activations,
								Transformer_Block * grad_weights,
								Transformer_Block_Transition * next_grad_stream);;


int dataflow_submit_transformer_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                                Transformer_Block_Transition * grad_stream,
                                Seq_Batch_Saved_Activations * fwd_activations, 
                                Transformer_Block_Activations * grad_activations, 
                                Transformer_Block * grad_weights);



int dataflow_submit_transformer_embedding_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
											Transformer_Block_Transition * grad_stream,
											Transformer_Embedding_Table * grad_embedding_table);


// Just keeping these as seperate functions for now while we are testing...
int dataflow_submit_transformer_moe_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
		Transformer_Block_Transition * block_input, 
		Transformer_Block * transformer_block, 
		Transformer_Block_Activations * activations, 
		Transformer_Block_Transition * block_output);


int dataflow_submit_transformer_moe_block_recompute(Dataflow_Handle * dataflow_handle, int compute_stream_id, 
			Transformer_Block * transformer_block,
			Seq_Batch * seq_batch,
			SavedActivationLevel saved_activation_level,
			Seq_Batch_Saved_Activations * working_activations, Seq_Batch_Context * fwd_context,
			Seq_Batch_Activation_Workspace * activation_workspace);


int dataflow_submit_transformer_moe_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id,
		Transformer_Block * transformer_block, 
		Transformer_Block_Transition * inp_grad_stream, 
		Seq_Batch_Saved_Activations * fwd_activations, Seq_Batch_Context * fwd_context,
		Transformer_Block_Activations * grad_activations,
		Transformer_Block * grad_weights,
		Transformer_Block_Transition * next_grad_stream);


int dataflow_submit_transformer_moe_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
		Transformer_Block_Transition * grad_stream,
		Seq_Batch_Saved_Activations * fwd_activations, 
		Transformer_Block_Activations * grad_activations, 
		Transformer_Block * grad_weights);


#endif