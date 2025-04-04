#include "dataflow_transformer.h"

// ALL BAKED INTO 1 Large Function for now,
// but really should have subfunctions to do norms, attn, and mlp based on transformer block config...!

int dataflow_submit_transformer_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, int out_copy_stream_id, 
								Transformer_Block_Transition * block_input, 
								Transformer_Block * transformer_block, 
								Transformer_Block_Activations * activations, 
								Transformer_Block_Transition * block_output) {

    int ret;
    DataflowDatatype fwd_dt = transformer_block->config.block_dt;

	size_t x_el_size = dataflow_sizeof_element(fwd_dt);
    DataflowDatatype compute_dt = transformer_block->config.compute_dt;

	Seq_Batch_Config * batch_config = activations -> batch_config;


    int num_seqs = batch_config->num_seqs;
    int total_q = batch_config->total_q;
    int total_k = batch_config->total_k;
    
    int model_dim = transformer_block->config.model_dim;
    int kv_dim = transformer_block->config.kv_dim;
    int ffn_dim = transformer_block->config.ffn_dim;

	Seq_Batch_Saved_Activations * saved_activations = activations -> saved_activations;
	Seq_Batch_Activation_Workspace * activation_workspace = activations -> activation_workspace;

    uint64_t kernel_workspaceBytes = activation_workspace -> kernel_workspaceBytes;
    void * kernel_workspace = activation_workspace -> kernel_workspace;


	// Assume weights are in col-major format.

	// But we want to process activations in row-major

	// Note that matmul interface assumes col-major storage format

	// Also note that FP8 tensor cores only available in TN format

	// During FWD pass we normally want:


	// Thus to compute Y = X @ W, 
	// we can do Y^T = W^T @ X^T
	// where from matmul perspective ^T means we interpret as row-major
	// However we store W as col-major so we need to transpose it.

	// Also for M, K, N (assuming X: (m, k), W (k, n))
	// we set M = n, K = k, N = m

	// The BWD pass is different because if we want dX's to be in row major we need:

	// dX = dY @ W^T
	// => dX^T = W @ dY^T

	// so if we store W in col-major format we shouldn't transpose it..

	// Now for bwd we set
	// M = k, K = n, N = m
	// where m, k, n are from fwd values of X, W, and Y.

	int to_transa = 1;
	int to_transb = 0;


	printf("Submitting Attention RMS Norm...!\n");

	ret = dataflow_submit_rms_norm(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, (transformer_block -> config).eps, 
						transformer_block -> w_attn_norm, block_input -> X, activation_workspace -> x_temp, 
						saved_activations -> attn_norm_weighted_sums, saved_activations -> attn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit attention norm...\n");
		return -1;
	}	



	printf("Submitting Q, K, V matmuls...!\n");

	// Q Proj
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_q, activation_workspace -> x_temp, NULL, saved_activations -> x_q,
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit Q matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_k, activation_workspace -> x_temp, NULL, saved_activations -> x_k_local,
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit K matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_v, activation_workspace -> x_temp, NULL, saved_activations -> x_v_local,
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit V matmul proj...\n");
		return -1;
	}


	printf("Submitting RoPE...!\n");

	int num_q_heads = (transformer_block -> config).num_q_heads;
	int num_kv_heads = (transformer_block -> config).num_kv_heads;
	int head_dim = (transformer_block -> config).head_dim;


	uint64_t N = (uint64_t) total_q * (uint64_t) model_dim;

	ret = dataflow_submit_rope(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						N, model_dim, head_dim, num_kv_heads, (transformer_block -> config).theta,
						batch_config -> seq_positions, saved_activations -> x_q, saved_activations -> x_k_local);
	if (ret){
		fprintf(stderr, "Error: failed to submit rope...\n");
		return -1;
	}


	printf("Submitting Attention...!\n");

	// ensure workspace is zerod out beforehand....

	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, kernel_workspace, 0, kernel_workspaceBytes);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}

	void * q_seq_offsets = batch_config -> q_seq_offsets;
	void * q_seq_lens = batch_config -> q_seq_lens;
	int max_seqlen_q = batch_config -> max_seqlen_q;

	void * k_seq_offsets = batch_config -> k_seq_offsets;
	void * k_seq_lens = batch_config -> k_seq_lens;
	int max_seqlen_k = batch_config -> max_seqlen_k;

	Seq_Batch_Context * context = activations -> context;

	ret = dataflow_submit_attention(dataflow_handle, compute_stream_id,
						 fwd_dt, 
						 num_seqs, total_q, total_k,
						 q_seq_offsets, q_seq_lens, max_seqlen_q,
						 k_seq_offsets, k_seq_lens, max_seqlen_k,
						 num_q_heads, num_kv_heads, head_dim,
						 saved_activations -> x_q, context -> x_k, context -> x_v,
						 saved_activations -> x_attn_out, saved_activations -> softmax_lse, 
						 kernel_workspaceBytes, kernel_workspace);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention...\n");
		return -1;
	}


	printf("Submitting Attention Output Matmul...!\n");


	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q, 
					1.0, 1.0,
					transformer_block -> w_o, saved_activations -> x_attn_out, block_input -> X, saved_activations -> x_o,
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit o matmul proj...\n");
		return -1;
	}


	printf("Submitting FFN RMS Norm...!\n");

	ret = dataflow_submit_rms_norm(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, model_dim, (transformer_block -> config).eps, 
						transformer_block -> w_ffn_norm, saved_activations -> x_o, activation_workspace -> x_temp, 
						saved_activations -> ffn_norm_weighted_sums, saved_activations -> ffn_norm_rms_vals);

	if (ret){
		fprintf(stderr, "Error: failed to submit ffn norm...\n");
		return -1;
	}


	printf("Submitting FFN w1 and w3 matmuls...!\n");

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_1, activation_workspace -> x_temp, NULL, (saved_activations -> x_1)[0],
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w1 matmul proj...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q, 
					1.0, 0.0,
					transformer_block -> w_3, activation_workspace -> x_temp, NULL, (saved_activations -> x_3)[0],
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w3 matmul proj...\n");
		return -1;
	}


	printf("Submitting SwiGLU Activation...!\n");


	ret = dataflow_submit_swiglu(dataflow_handle, compute_stream_id, 
						fwd_dt, 
						total_q, ffn_dim, 
						(saved_activations -> x_1)[0], (saved_activations -> x_3)[0], activation_workspace -> x_temp_mlp);

	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu activation...\n");
		return -1;
	}


	printf("Submitting FFN w2 matmul...!\n");

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id, 
					fwd_dt, fwd_dt, fwd_dt, fwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q, 
					1.0, 1.0,
					transformer_block -> w_2, activation_workspace -> x_temp_mlp, saved_activations -> x_o, (saved_activations -> x_2)[0],
					kernel_workspaceBytes, kernel_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit w2 matmul proj...\n");
		return -1;
	}

	// copy to output

	uint64_t block_out_size = total_q * model_dim * x_el_size;

	ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, out_copy_stream_id,
										block_output -> X,
										(saved_activations -> x_2)[0], 
										block_out_size);

	if (ret){
		fprintf(stderr, "Error: failed to copy block output...\n");
		return -1;
	}




	return 0;

}


int submit_transformer_head(Dataflow_Handle * dataflow_handle, int compute_stream_id, int out_copy_stream_id,
                        Transformer_Block_Transition * block_input, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations, 
                        Transformer_Model_Output * model_output,
						// during interference these would be NULL
						Transformer_Head * grad_transformer_head,
						Transformer_Head_Activations * grad_head_activations,
						Transformer_Block_Transition * grad_stream,
						Transformer_Block_Transition * next_grad_stream) {

    int ret;

    // Get dimensions from embedding config
    int vocab_size = (transformer_head -> embedding_config).vocab_size;
    int embedding_size = (transformer_head -> embedding_config).embedding_size;

    // RMS Normalization
    ret = dataflow_submit_rms_norm(dataflow_handle, compute_stream_id,
                         transformer_head -> fwd_dt,
                         head_activations -> num_tokens,
                         embedding_size, transformer_head -> eps,
                         transformer_head -> w_head_norm,
                         block_input -> X,
                         head_activations -> head_norm_out,
                         head_activations -> head_norm_weighted_sums,
                         head_activations -> head_norm_rms_vals);
    if (ret) {
        fprintf(stderr, "Error: Failed to submit RMS normalization in transformer head...\n");
        return ret;
    }

    // Output projection
    // Input x_temp is in row-major format
    // Weights w_out are in col-major format
    // Want output in row-major format: head_out[num_tokens, vocab_size] = x_temp[num_tokens, embedding_size] @ w_out[embedding_size, vocab_size]
    // Use transa=1 for row-major input, transb=0 for col-major weights
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                       transformer_head -> fwd_dt,
                       transformer_head -> fwd_dt,
                       DATAFLOW_NONE,
                       transformer_head -> fwd_dt,
                       transformer_head -> compute_dt,
                       1, 0,  // transa=1 for row-major input, transb=0 for col-major weights
                       embedding_size,    // m = embedding_size (due to transa=1)
                       vocab_size,        // n = vocab_size
                       head_activations -> num_tokens,  // k = num_tokens (due to transa=1)
                       1.0, 0.0,
                       transformer_head -> w_head,       // Changed from w_out to w_head
                       head_activations -> head_norm_out,      // B[num_tokens, embedding_size] in row-major
                       NULL,
                       head_activations -> head_out,    // C[num_tokens, vocab_size] in row-major
                       head_activations -> kernel_workspace, head_activations -> kernel_workspaceBytes);  // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output projection in transformer head...\n");
        return ret;
    }

    // Apply softmax over vocabulary dimension
    // Each row (corresponding to a token) should sum to 1
    ret = dataflow_submit_softmax(dataflow_handle, compute_stream_id,
                        transformer_head -> fwd_dt,
                        transformer_head -> bwd_dt,  // Use bwd_dt for backward pass
                        head_activations -> num_tokens,  // Number of rows to process
                        vocab_size,                      // Size of softmax dimension
                        head_activations -> head_out,   // Input logits
                        model_output -> logits);        // Output probabilities to provided buffer
    if (ret) {
        fprintf(stderr, "Error: Failed to submit softmax in transformer head...\n");
        return ret;
    }


	if (!grad_transformer_head){
		return 0;
	}
	
	// STARTING BACKPROP HERE!

	// compute cross entropy loss
	// updates logits in-place

    // First compute cross entropy loss gradients
    // This will subtract 1.0 from the correct class logits in-place
    ret = dataflow_submit_cross_entropy_loss(dataflow_handle, compute_stream_id,
                                  transformer_head -> bwd_dt,
                                  head_activations -> num_tokens,  // Number of rows (tokens)
                                  vocab_size,                      // Number of columns (vocab size)
                                  model_output -> logits,         // Predicted logits
                                  model_output -> labels,
								  model_output -> loss);        // Ground truth labels
    if (ret) {
        fprintf(stderr, "Error: Failed to submit cross entropy loss in transformer head backward...\n");
        return ret;
    }

	// Now backpropagate through the output projection
    // Input logits is in row-major format after cross entropy
    // Weights w_head are in col-major format
    // Want output in row-major format: dx_temp = dlogits @ w_head^T
    // For backward pass with transa=0, transb=0:
    // M = fwd_K = embedding_size
    // K = fwd_N = vocab_size
    // N = fwd_M = num_tokens
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                       transformer_head -> bwd_dt,
                       transformer_head -> bwd_dt,
                       DATAFLOW_NONE,
                       transformer_head -> bwd_dt,
                       transformer_head -> compute_dt,
                       0, 0,  // transa=0 for row-major dlogits, transb=0 for col-major weights
                       embedding_size,                  // m = embedding_size (fwd_K)
                       head_activations -> num_tokens,  // n = num_tokens (fwd_M)
                       vocab_size,                      // k = vocab_size (fwd_N)
                       1.0, 0.0,
                       model_output -> logits,         // dlogits[num_tokens, vocab_size] in row-major
                       transformer_head -> w_head,      // w_head[embedding_size, vocab_size] in col-major
                       NULL,
                       grad_head_activations -> head_norm_out, // dx_temp[num_tokens, embedding_size] in row-major
                       grad_head_activations -> kernel_workspace, grad_head_activations -> kernel_workspaceBytes);  // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit bwd x head matmul in transformer head...\n");
        return ret;
    }

	 // 2. Output projection weight gradients
    // Forward: [num_tokens, embedding_size] @ [embedding_size, vocab_size] -> [num_tokens, vocab_size]
    // Backward for weights: dW = X^T @ dY
    // For matmul with transa=1, transb=0:
    // M = embedding_size (rows of dW)
    // K = num_tokens (reduction dim)
    // N = vocab_size (cols of dW)
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> bwd_dt,
                       grad_transformer_head -> compute_dt,
                       1, 0,  // transa=1 for X^T, transb=0 for dY
                       embedding_size, vocab_size, head_activations -> num_tokens,  // M, K, N
                       1.0, 1.0,  // Accumulate gradients
                       head_activations -> head_norm_out,           // Input activations [num_tokens, embedding_size]
                       grad_head_activations -> head_out,    // Gradient of output [num_tokens, vocab_size]
                       grad_transformer_head -> w_head,      // Previous gradient
                       grad_transformer_head -> w_head,      // Output gradient
                       grad_head_activations -> kernel_workspace, grad_head_activations -> kernel_workspaceBytes);                            // No workspace needed
    if (ret) {
        fprintf(stderr, "Error: Failed to submit output bwd weight gradient computation in transformer head...\n");
        return ret;
    }

	// Finally backpropagate through RMS normalization
    ret = dataflow_submit_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
                               transformer_head -> bwd_dt,
                               transformer_head -> bwd_dt,
                               head_activations -> num_tokens,
                               embedding_size,
                               transformer_head -> eps,
                               head_activations -> head_norm_weighted_sums,
                               head_activations -> head_norm_rms_vals,
                               transformer_head -> w_head_norm,
                               block_input -> X,         // Original input
                               grad_head_activations -> head_norm_out,      // Upstream gradient
                               grad_stream -> X);
	if (ret) {
        fprintf(stderr, "Error: Failed to submit bwd x rms norm in transformer head...\n");
        return ret;
    }

	ret = dataflow_submit_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
                               grad_transformer_head -> fwd_dt,
                               grad_transformer_head -> bwd_dt,
                               head_activations -> num_tokens,
                               embedding_size,
                               grad_transformer_head -> eps,
                               head_activations -> head_norm_rms_vals,  // RMS values from forward pass
                               head_activations -> head_out,            // Original input
                               grad_head_activations -> head_norm_out,         // Upstream gradient
                               grad_transformer_head -> w_head_norm);   // Output gradient
	if (ret) {
        fprintf(stderr, "Error: Failed to submit bwd w rms norm in transformer head...\n");
        return ret;
    }

	if (next_grad_stream){
		size_t x_el_size = dataflow_sizeof_element(grad_transformer_head -> bwd_dt);
		uint64_t block_out_size = head_activations -> num_tokens * embedding_size * x_el_size;

		// copy the grad_stream to the next_grad_stream
		ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, out_copy_stream_id,
										next_grad_stream -> X,
										grad_stream -> X, 
										block_out_size);

		if (ret){
			fprintf(stderr, "Error: failed to submit peer transfer for copying head gradient to next location...\n");
			return ret;
		}
	}

    return 0;
}

// dX_out is upstream gradient
// it is in col-major orientation

// now to do dX = matmul(dX_out, W^T)
// we remember that W is being stored in col-major as well
// thus we can correctly get dX in col major as:
// we can do matmul(W, dx_Out^T)
// This is because we will get a row major output of 
// dX^T = dX in col major

// To see how we get col major of dX recall:

// thus if we pass A = W in col-major = W^T effectively in row major
// and B = dX_Out in col-major,
// If we pass in M = w_in, K = w_out, N = x_in, which yields
// a (w_in, x_in) matrix which we interpret as col major dX




int submit_transformer_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id, int out_copy_stream_id,
								Transformer_Block * transformer_block, 
								Transformer_Block_Transition * inp_grad_stream, 
								Transformer_Block_Activations * activations, Transformer_Block_Transition * fwd_block_input,
								Transformer_Block_Activations * grad_activations,
								Transformer_Block * grad_weights, // for the norm weights while using streaming grad
								Transformer_Block_Transition * out_grad_stream) {

	int ret;
	DataflowDatatype bwd_dt = (transformer_block -> config).block_dt;
	DataflowDatatype compute_dt = (transformer_block -> config).compute_dt;
	Seq_Batch_Config * batch_config = grad_activations -> batch_config;

    int num_seqs = batch_config->num_seqs;
    int total_q = batch_config->total_q;
    int total_k = batch_config->total_k;
	
	int num_seqs = batch_config->num_seqs;
	int total_q = batch_config->total_q;
	int total_k = batch_config->total_k;
	
	int model_dim = (transformer_block -> config).model_dim;
	int kv_dim = (transformer_block -> config).kv_dim;
	int ffn_dim = (transformer_block -> config).ffn_dim;
	
	Seq_Batch_Activation_Workspace * bwd_activation_workspace = grad_activations -> activation_workspace;

	Seq_Batch_Saved_Activations * fwd_activations = activations -> saved_activations;
	Seq_Batch_Context * fwd_context = activations -> context;
	Seq_Batch_Saved_Activations * bwd_activations = grad_activations -> saved_activations;
	Seq_Batch_Context * bwd_context = grad_activations -> context;
	
	uint64_t kernel_workspaceBytes = bwd_activation_workspace -> kernel_workspaceBytes;
	void * kernel_workspace = bwd_activation_workspace -> kernel_workspace;

	int to_transa = 0;
	int to_transb = 0;

	// 1. Backprop through FFN w2 matmul
	// Forward: [num_tokens, ffn_dim] @ [ffn_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = ffn_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					ffn_dim, model_dim, total_q,  // M = ffn_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_2[0], inp_grad_stream -> X, NULL, bwd_activation_workspace -> x_temp_mlp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w2 backward matmul...\n");
		return -1;
	}

	// 2. Backprop through SwiGLU
	ret = dataflow_submit_swiglu_bwd_x(dataflow_handle, compute_stream_id,
						bwd_dt, bwd_dt,
						total_q, ffn_dim,
						fwd_activations -> x_1[0], fwd_activations -> x_3[0],
						bwd_activation_workspace -> x_temp_mlp,
						bwd_activations -> x_1[0], bwd_activations -> x_3[0]);
	if (ret) {
		fprintf(stderr, "Error: failed to submit swiglu backward...\n");
		return -1;
	}

	// 3. Backprop through w1 and w3 matmuls
	// Forward: [num_tokens, model_dim] @ [model_dim, ffn_dim] -> [num_tokens, ffn_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = ffn_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q,  // M = model_dim, K = ffn_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_1[0], bwd_activations -> x_1[0], NULL, bwd_activation_workspace -> x_temp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w1 backward matmul...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, ffn_dim, total_q,  // M = model_dim, K = ffn_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_3[0], bwd_activations -> x_3[0], NULL, bwd_activation_workspace -> x_temp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit w3 backward matmul...\n");
		return -1;
	}

	// 4. Backprop through FFN RMS Norm
	ret = dataflow_submit_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (transformer_block -> config).eps,
							fwd_activations -> ffn_norm_weighted_sums,
							fwd_activations -> ffn_norm_rms_vals,
							transformer_block -> w_ffn_norm,
							fwd_activations -> x_o,  // Input to norm
							bwd_activation_workspace -> x_temp,  // Upstream gradient
							inp_grad_stream -> X);                    // Final output gradient
	if (ret) {
		fprintf(stderr, "Error: failed to submit ffn norm backward...\n");
		return -1;
	}

	// 5. Now that we have the correct upstream gradient also do bwd_w for ffn norm
	ret = dataflow_submit_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
								bwd_dt, bwd_dt,
								total_q, model_dim, (transformer_block -> config).eps,
								fwd_activations -> ffn_norm_rms_vals,
								fwd_activations -> x_o,
								bwd_activation_workspace -> x_temp,
								grad_weights -> w_ffn_norm);
	if (ret){
		fprintf(stderr, "Error: failed to submit ffn norm weight gradient computation during bwd_x...\n");
		return -1;
	}

	// 6. Backprop through attention output projection
	// Forward: [num_tokens, model_dim] @ [model_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,  // M = model_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_o, bwd_activations -> x_o, NULL, bwd_activation_workspace -> x_temp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention output backward matmul...\n");
		return -1;
	}


	// ensure workspace is zerod out beforehand....

	ret = (dataflow_handle -> set_mem)(dataflow_handle, compute_stream_id, kernel_workspace, 0, kernel_workspaceBytes);
	if (ret){
		fprintf(stderr, "Error: unable to set attention workspace mem to 0 before submitting...\n");
		return -1;
	}

	// 7. Backprop through attention
	ret = dataflow_submit_attention_bwd(dataflow_handle, compute_stream_id,
							bwd_dt,
							num_seqs, total_q, total_k,
							batch_config -> q_seq_offsets,
							batch_config -> q_seq_lens,
							batch_config -> max_seqlen_q,
							batch_config -> k_seq_offsets,
							batch_config -> k_seq_lens,
							batch_config -> max_seqlen_k,
							(transformer_block -> config).num_q_heads,
							(transformer_block -> config).num_kv_heads,
							(transformer_block -> config).head_dim,
							fwd_activations -> x_q,       // Q input
							fwd_context -> x_k,  // K input (full context input keys)
							fwd_context -> x_v,  // V input (full context input values)
							fwd_activations -> x_attn_out,     // Attention output
							fwd_activations -> softmax_lse,// Softmax scaling factors
							bwd_activation_workspace -> x_temp,// Upstream gradient
							bwd_activations -> x_q,   // dQ output
							bwd_context -> x_k,  // dK output (full context key grads)
							bwd_context -> x_v, // dV output (full context grads)
							kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention backward...\n");
		return -1;
	}

	// Now bwd_activations -> x_k_locat should be pointer within bwd_context -> x_k
	// with the correct accumulated gradients...

	// 8. Backprop through RoPE
	ret = dataflow_submit_rope_bwd_x(dataflow_handle, compute_stream_id,
						bwd_dt,
						(uint64_t)total_q * (uint64_t)model_dim,
						model_dim,
						(transformer_block -> config).head_dim,
						(transformer_block -> config).num_kv_heads,
						(transformer_block -> config).theta,
						batch_config -> seq_positions,
						bwd_activations -> x_q,
						bwd_activations -> x_k_local);
	if (ret) {
		fprintf(stderr, "Error: failed to submit rope backward...\n");
		return -1;
	}

	// 9. Backprop through Q, K, V projections
	// Q Forward: [num_tokens, model_dim] @ [model_dim, model_dim] -> [num_tokens, model_dim]
	// Backward: dX = dY @ W^T
	// For Q:
	// M = output rows of dX = model_dim
	// K = output cols of dX = model_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, model_dim, total_q,  // M = model_dim, K = model_dim, N = num_tokens
					1.0, 0.0,
					transformer_block -> w_q, bwd_activations -> x_q, NULL, bwd_activation_workspace -> x_temp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit Q projection backward matmul...\n");
		return -1;
	}

	// For K/V:
	// Forward: [num_tokens, model_dim] @ [model_dim, kv_dim] -> [num_tokens, kv_dim]
	// Backward: dX = dY @ W^T
	// M = output rows of dX = model_dim
	// K = output cols of dX = kv_dim
	// N = batch dim = num_tokens
	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q,  // M = model_dim, K = kv_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_k, bwd_activations -> x_k_local, NULL, bwd_activation_workspace -> x_temp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit K projection backward matmul...\n");
		return -1;
	}

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
					bwd_dt, bwd_dt, DATAFLOW_NONE, bwd_dt,
					compute_dt,
					to_transa, to_transb,
					model_dim, kv_dim, total_q,  // M = model_dim, K = kv_dim, N = num_tokens
					1.0, 1.0,  // Add to previous gradient
					transformer_block -> w_v, bwd_activations -> x_v_local, NULL, bwd_activation_workspace -> x_temp,
					kernel_workspaceBytes, kernel_workspace);
	if (ret) {
		fprintf(stderr, "Error: failed to submit V projection backward matmul...\n");
		return -1;
	}

	// 10. Finally backprop through attention RMS norm
	ret = dataflow_submit_rms_norm_bwd_x(dataflow_handle, compute_stream_id,
							bwd_dt, bwd_dt,
							total_q, model_dim, (transformer_block -> config).eps,
							fwd_activations -> attn_norm_weighted_sums,
							fwd_activations -> attn_norm_rms_vals,
							transformer_block -> w_attn_norm,
							fwd_block_input -> X,  // Input to norm
							bwd_activation_workspace -> x_temp,  // Upstream gradient
							inp_grad_stream -> X);                    // Final output gradient
	if (ret) {
		fprintf(stderr, "Error: failed to submit attention norm backward...\n");
		return -1;
	}

	// While we have the correct upstream gradient, also do bwd_w for attn norm
	ret = dataflow_submit_rms_norm_bwd_w(dataflow_handle, compute_stream_id,
								bwd_dt, bwd_dt,
								total_q, model_dim, (transformer_block -> config).eps,
								fwd_activations -> attn_norm_rms_vals,
								fwd_block_input -> X,
								bwd_activation_workspace -> x_temp,
								grad_weights -> w_attn_norm);
	if (ret){
		fprintf(stderr, "Error: failed to submit attention norm weight gradient computation during bwd_w...\n");
		return -1;
	}

	// while we still the layer weights, recompute the RMS norms 
	// needed to do bwd_w computations...

	// recompute attn norm
	ret = dataflow_submit_rms_norm(dataflow_handle, compute_stream_id,
							bwd_dt, 
							total_q, model_dim, (transformer_block -> config).eps,
							transformer_block -> w_attn_norm,
							fwd_block_input -> X,
							bwd_activation_workspace -> recomputed_attn_norm,
							NULL, NULL);
	if (!ret){
		fprintf(stderr, "Error: failed to submit recompute attn norm...\n");
		return -1;
	}

	// recompute ffn norm
	ret = dataflow_submit_rms_norm(dataflow_handle, compute_stream_id,
							bwd_dt, 
							total_q, model_dim, (transformer_block -> config).eps,
							transformer_block -> w_ffn_norm,
							fwd_activations -> x_o,
							bwd_activation_workspace -> recomputed_ffn_norm,
							NULL, NULL);
	if (!ret){
		fprintf(stderr, "Error: failed to submit recompute ffn norm...\n");
		return -1;
	}

	if (out_grad_stream){

		uint64_t x_el_size = dataflow_sizeof_element(bwd_dt);

		uint64_t block_out_size = total_q * model_dim * x_el_size;
		
		ret = (dataflow_handle -> submit_peer_transfer)(dataflow_handle, out_copy_stream_id,
										out_grad_stream -> X,
										inp_grad_stream -> X, 
										block_out_size);
		if (ret){
			fprintf(stderr, "Error: failed to submit transfer to copy block output gradient...\n");
			return -1;
		}
	}

	return 0;
}

int submit_transformer_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                                Transformer_Block_Transition * grad_stream,
                                Transformer_Block_Activations * activations, 
                                Transformer_Block_Activations * grad_activations, 
                                Transformer_Block * grad_weights) {
    
    int ret;
    DataflowDatatype bwd_dt = (grad_weights -> config).block_dt;
	// just assume same for now...
    DataflowDatatype fwd_dt = bwd_dt;

    DataflowDatatype compute_dt = (grad_weights -> config).compute_dt;

    Seq_Batch_Config * batch_config = activations -> batch_config;
    int num_seqs = batch_config->num_seqs;
    int total_q = batch_config->total_q;
    int total_k = batch_config->total_k;
    
    int model_dim = (grad_weights -> config).model_dim;
    int kv_dim = (grad_weights -> config).kv_dim;
    int ffn_dim = (grad_weights -> config).ffn_dim;
    

    Seq_Batch_Saved_Activations * fwd_activations = activations -> saved_activations;
    Seq_Batch_Saved_Activations * bwd_activations = grad_activations -> saved_activations;
	Seq_Batch_Activation_Workspace * bwd_activation_workspace = grad_activations -> activation_workspace;
    
    uint64_t kernel_workspaceBytes = bwd_activation_workspace -> kernel_workspaceBytes;
    void * kernel_workspace = bwd_activation_workspace -> kernel_workspace;

    int to_transa = 1;
    int to_transb = 0;

	// 1.) Recompute-Swiglu in order to compute w2 grad
	ret = dataflow_submit_swiglu(dataflow_handle, compute_stream_id,
						fwd_dt,
						total_q, ffn_dim,
						fwd_activations -> x_1[0], fwd_activations -> x_3[0],
						bwd_activation_workspace -> x_temp_mlp);
	if (ret){
		fprintf(stderr, "Error: failed to submit swiglu when recomputing for w2 grad....\n");
		return -1;
	}

    // 2. FFN w2 weight gradients
    // dW2^T = x_temp_mlp @ dX_out^T
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    ffn_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    grad_stream -> X, bwd_activation_workspace -> x_temp_mlp, (grad_weights -> w_2)[0], (grad_weights -> w_2)[0],
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit w2 weight gradient computation...\n");
        return -1;
    }

    // 2. FFN w1 and w3 weight gradients after SwiGLU
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    ffn_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    (bwd_activations -> x_1)[0], bwd_activation_workspace -> recomputed_ffn_norm, (grad_weights -> w_1)[0], (grad_weights -> w_1)[0],
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit w1 weight gradient computation...\n");
        return -1;
    }

    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    ffn_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    (bwd_activations -> x_3)[0], bwd_activation_workspace -> recomputed_ffn_norm, (grad_weights -> w_3)[0], (grad_weights -> w_3)[0],
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit w3 weight gradient computation...\n");
        return -1;
    }

    // 3. FFN RMS Norm weight gradients -- already done in bwd_x while we hda correct streaming grad

    // 4. Attention output projection weight gradients
    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> x_o, fwd_activations -> x_attn_out, grad_weights -> w_o, grad_weights -> w_o,
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit attention output weight gradient computation...\n");
        return -1;
    }

    // 5. Q, K, V projection weight gradients

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    kv_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> x_v_local, bwd_activation_workspace -> recomputed_ffn_norm, grad_weights -> w_v, grad_weights -> w_v,
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit V projection weight gradient computation...\n");
        return -1;
    }

	ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    kv_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> x_k_local, bwd_activation_workspace -> recomputed_attn_norm, grad_weights -> w_k, grad_weights -> w_k,
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit K projection weight gradient computation...\n");
        return -1;
    }


    ret = dataflow_submit_matmul(dataflow_handle, compute_stream_id,
                    bwd_dt, bwd_dt, bwd_dt, bwd_dt,
                    compute_dt,
                    to_transa, to_transb,
                    model_dim, model_dim, total_q,
                    1.0, 1.0,  // Accumulate gradients
                    bwd_activations -> x_q, bwd_activation_workspace -> recomputed_ffn_norm, grad_weights -> w_q, grad_weights -> w_q,
                    kernel_workspaceBytes, kernel_workspace);
    if (ret) {
        fprintf(stderr, "Error: failed to submit Q projection weight gradient computation...\n");
        return -1;
    }

    // 6. Attention RMS Norm weight gradients
    // -- already done in bwd_x while we had the correct streaming grad

    return 0;
}


