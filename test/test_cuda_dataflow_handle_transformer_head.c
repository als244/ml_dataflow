#define __GNU_SOURCE
#include <stdio.h>

#include "dataflow.h"
#include "dataflow_utils.h"
#include "cuda_dataflow_handle.h"
#include "dataflow_ops.h"
#include "register_ops.h"

int main(int argc, char * argv[]){

	// Seed the random number generator with a constant value
    srand(42);
	
	int ret;

	Dataflow_Handle dataflow_handle;
	
	ComputeType compute_type = COMPUTE_CUDA;
	int device_id = 0;

	// In case we want to create multiple contexts per device, 
	// higher level can create multiple instances of dataflow handles...
	int ctx_id = 0;
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;

	int num_streams = 8;
	int opt_stream_prios[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	char * opt_stream_names[8] = {"Inbound (a)", "Compute (a)", "Outbound (a)", "Peer (a)", "Inbound (b)", "Compute (b)", "Outbound (b)", "Peer (b)"};


	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = dataflow_init_handle(&dataflow_handle, compute_type, device_id, 
			ctx_id, ctx_flags, 
			num_streams, opt_stream_prios, opt_stream_names); 
	
	if (ret){
		fprintf(stderr, "Error: failed to init cuda dataflow handle...\n");
		return -1;
	}

	// from backend/nvidia/src/ops/src/register_ops/register_ops.c	
	// handles registering external and native ops within cuda_dataflow_ops...
	int added_funcs = dataflow_register_default_ops(&dataflow_handle);
	printf("Registered %d default ops...\n", added_funcs);
	
	int alignment = 4096;

	void * host_mem;

	// 16 GB...
	size_t host_size_bytes = 1UL << 34;

	printf("Allocating host memory of size: %lu...\n", host_size_bytes);

	ret = posix_memalign(&host_mem, alignment, host_size_bytes);
	if (ret){
		fprintf(stderr, "Error: posix memalign failed...\n");
		return -1;
	}
	memset(host_mem, 0, host_size_bytes);


	printf("Registering host memory...\n");

	ret = dataflow_handle.enable_access_to_host_mem(&dataflow_handle, host_mem, host_size_bytes, 0);
	if (ret){
		fprintf(stderr, "Registration of host memory failed...\n");
		return -1;
	}

	// Llama 70B dimensions, taking example input after embedding stage and weights are loaded 
	// as first layer

	int vocab_size = 128256;
    int model_dim = 4096;

    float eps = 1e-5;

	DataflowDatatype fwd_dt = DATAFLOW_FP16;
	DataflowDatatype bwd_dt = DATAFLOW_FP16;
	DataflowDatatype compute_dt = DATAFLOW_FP16;


	uint64_t el_size = dataflow_sizeof_element(fwd_dt);


	printf("Loading Weight Matrices into host mem...\n");

	char * w_out_norm_f = "data/weights/8B/out_norm.dat";
	char * w_head_f = "data/weights/8B/head.dat";

	uint64_t norm_size = model_dim * el_size;
	uint64_t w_head_size = (uint64_t) model_dim * (uint64_t)vocab_size * (uint64_t) el_size;

	void * w_out_norm = host_mem;
	void * w_head = w_out_norm + norm_size;

	void * res;

	// NOTE: all of the weights are stored in column-major format 
	//			- (however this doesn't impact loading in as same total size, we just interpret differently)

	res = load_host_matrix_from_file(w_out_norm_f, 1, model_dim, fwd_dt, fwd_dt, w_out_norm);
	if (!res){
		fprintf(stderr, "Error: could not load in attn norm weight file...\n");
		return -1;
	}

	res = load_host_matrix_from_file(w_head_f, model_dim, vocab_size, fwd_dt, fwd_dt, w_head);
	if (!res){
		fprintf(stderr, "Error: could not load in w head proj weight file...\n");
		return -1;
	}

	// DEAL WITH EXAMPLE TRAINING SEQ BATCH...

	int num_tokens = 4096;

    size_t x_size = (uint64_t) num_tokens * (uint64_t) model_dim * (uint64_t) el_size;

    size_t labels_size = (uint64_t) num_tokens * (uint64_t) sizeof(uint32_t);

    size_t head_norm_size = (uint64_t) num_tokens * (uint64_t) model_dim * (uint64_t) el_size;

    size_t head_out_size = (uint64_t) num_tokens * (uint64_t) vocab_size * (uint64_t) el_size;

    size_t x_logits_size = (uint64_t) num_tokens * (uint64_t) vocab_size * (uint64_t) el_size;

	size_t loss_vec_size = ((uint64_t) num_tokens + 1) * sizeof(float);

	char * model_out_x_f = "data/8B/model_out_x.dat";
    char * labels_f = "data/8B/labels.dat";

	void * model_out_x = w_head + w_head_size;
    void * labels = model_out_x + x_size;

	res = load_host_matrix_from_file(model_out_x_f, num_tokens, model_dim, fwd_dt, fwd_dt, model_out_x);
	if (!res){
		fprintf(stderr, "Error: could not load in model out x file...\n");
		return -1;
	}

    res = load_host_matrix_from_file(labels_f, num_tokens, 1, fwd_dt, fwd_dt, labels);
    if (!res){
		fprintf(stderr, "Error: could not load in model out x file...\n");
		return -1;
	}

    void * x_norm_out = labels + labels_size;

    void * x_head_out = x_norm_out + head_norm_size;

    void * x_logits = x_head_out + head_out_size;

    void * x_logits_bwd = x_logits + x_logits_size;

    void * loss_vec = x_logits_bwd + x_logits_size;


	// 16 GB...	
	size_t dev_size_bytes = 1UL << 34;

	printf("Allocating device memory of size: %lu...\n", dev_size_bytes);

	void * dev_mem = dataflow_handle.alloc_mem(&dataflow_handle, dev_size_bytes);
	if (!dev_mem){
		fprintf(stderr, "Error: device memory allocation failed...\n");
		return -1;
	}

    // set all memory to 0
    ret = dataflow_handle.set_mem(&dataflow_handle, compute_stream_id_a, dev_mem, 0, dev_size_bytes);
    if (ret){
        fprintf(stderr, "Error: failed to set device memory to 0...\n");
        return -1;
    }


	// Transfer all required weights/inputs/metadata/workspace to device before submitting ops...

	// weights
	void * d_w_out_norm = dev_mem;
	void * d_w_head = d_w_out_norm + norm_size;

    // metadata
    void * d_labels = d_w_head + w_head_size;

    // activations
    void * d_model_out_x = d_labels + labels_size;
	void * d_x_norm_out = d_model_out_x + x_size;
	void * d_x_head_out = d_x_norm_out + head_norm_size;
	void * d_x_logits = d_x_head_out + head_out_size;
    void * d_x_logits_bwd = d_x_logits + x_logits_size;

	uint64_t workspaceBytes = 1UL << 24;
	void * d_workspace = d_x_logits_bwd + x_logits_size;


    // make this last to not mess up alignment...
    // loss
	void * d_loss_vec = d_workspace + workspaceBytes;


	printf("Transferring weights to device...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_w_out_norm, w_out_norm, norm_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for out norm weights...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_w_head, w_head, w_head_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for head weights...\n");
		return -1;
	}

	printf("Transferring batch to device...\n");

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_labels, labels, labels_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for labels...\n");
		return -1;
	}

	ret = dataflow_handle.submit_inbound_transfer(&dataflow_handle, inbound_stream_id_a, d_model_out_x, model_out_x, x_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed for model out x...\n");
		return -1;
	}

	printf("Syncing with device after transfer...\n");

	ret = dataflow_handle.sync_stream(&dataflow_handle, inbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer...\n");
		return -1;
	}

	int to_trans_a = 1;
	int to_trans_b = 0;


	printf("Submitting Output RMS Norm...!\n");

	ret = dataflow_submit_rms_norm(&dataflow_handle, compute_stream_id_a, 
						fwd_dt, 
						num_tokens, model_dim, eps, 
						d_x_norm_out, d_model_out_x, d_x_norm_out, 
						NULL, NULL // not saving weighted sums or rms vals for now...
                        );

	if (ret){
		fprintf(stderr, "Error: failed to submit out norm...\n");
		return -1;
	}	


	printf("Submitting head matmul...!\n");

	// Head Proj
	ret = dataflow_submit_matmul(&dataflow_handle, compute_stream_id_a, 
					fwd_dt, fwd_dt, DATAFLOW_NONE, fwd_dt,
					compute_dt,
					to_trans_a, to_trans_b,
					model_dim, model_dim, num_tokens,
					1.0, 0.0,
					d_w_head, d_x_norm_out, NULL, d_x_head_out,
					workspaceBytes, d_workspace);

	if (ret){
		fprintf(stderr, "Error: failed to submit q matmul proj...\n");
		return -1;
	}

    ret = dataflow_submit_softmax(&dataflow_handle, compute_stream_id_a, 
                        fwd_dt, bwd_dt,
                        num_tokens, vocab_size,
                        d_x_head_out, d_x_logits);
    if (ret){
        fprintf(stderr, "Error: failed to submit softmax...\n");
        return -1;
    }

    // make copy of logits to ensure derivs are correct...
    ret = dataflow_handle.submit_peer_transfer(&dataflow_handle, compute_stream_id_a, d_x_logits, d_x_logits_bwd, x_logits_size);
    if (ret){
        fprintf(stderr, "Error: failed to submit peer transfer...\n");
        return -1;
    }

    ret = dataflow_submit_cross_entropy_loss(&dataflow_handle, compute_stream_id_a, 
                        bwd_dt,
                        num_tokens, vocab_size,
                        d_x_logits_bwd, d_labels, d_loss_vec);
    if (ret){
        fprintf(stderr, "Error: failed to submit cross entropy loss...\n");
        return -1;
    }

	printf("Finished submitting all work, now submitting dependency for outbound transfer...\n");

	void * compute_stream_state = dataflow_handle.get_stream_state(&dataflow_handle, compute_stream_id_a);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = dataflow_handle.submit_dependency(&dataflow_handle, outbound_stream_id_a, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	printf("Submitting outbound transfers for all results...\n");


	// ATTENTION BLOCK!

	// Attn Norm...
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, x_norm_out, d_x_norm_out, norm_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for x_norm_out...\n");
		return -1;
	}

	// Head proj
	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, x_head_out, d_x_head_out, head_out_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for x_head_out...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, x_logits, d_x_logits, x_logits_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for x_logits...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, x_logits_bwd, d_x_logits_bwd, x_logits_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for x_logits_bwd...\n");
		return -1;
	}

	ret = dataflow_handle.submit_outbound_transfer(&dataflow_handle, outbound_stream_id_a, loss_vec, d_loss_vec, loss_vec_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer for loss_vec...\n");
		return -1;
	}

	ret = dataflow_handle.sync_stream(&dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}

	printf("Saving results in files...\n");


	// x_norm_out

	// x_head_out
	
    // x_logits

	// x_logits_bwd
	// loss_vec

	char * res_norm_out_f = "test_head/x_norm_out.dat";

	ret = save_host_matrix(res_norm_out_f, x_norm_out, num_tokens, model_dim, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_norm_out matrix...\n");
		return -1;
	}

	char * res_head_out_f = "test_head/x_head_out.dat";
	// Note: already applied rope to these out projections...

	ret = save_host_matrix(res_head_out_f, x_head_out, num_tokens, vocab_size, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_head_out matrix...\n");
		return -1;
	}

    char * res_logits_f = "test_head/x_logits.dat";

	ret = save_host_matrix(res_logits_f, x_logits, num_tokens, vocab_size, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_logits matrix...\n");
		return -1;
	}

    char * res_logits_bwd_f = "test_head/x_logits_bwd.dat";
    char * res_loss_vec_f = "test_head/loss_vec.dat";

	ret = save_host_matrix(res_logits_bwd_f, x_logits_bwd, num_tokens, vocab_size, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save x_logits_bwd matrix...\n");
		return -1;
	}

	ret = save_host_matrix(res_loss_vec_f, loss_vec, num_tokens + 1, 1, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save loss_vec matrix...\n");
		return -1;
	}

	printf("\n\n\nSuccessfully Performed Op...!!!\n");

	return 0;
}
