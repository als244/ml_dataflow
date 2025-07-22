#include "dataflow_ops.h"

/* HOST OPS */

// Require user to pass in host function pointer the backend implementation of the op
// The arguments passed in need to populated in memory at the the execution time of the op
// but this is uncertain, so it is user responssibility to pass in buffer that will be populated
// by these submission functions and shoudn't be overwritten until after the op has completed execution..


// GENERAL STUFF (used for optimizer stuff in those transformer variants)

int dataflow_submit_set_mem_host(Dataflow_Handle * handle, int stream_id, 
                        void * set_mem_host_func, Set_Mem_Host_Op_Args * op_buffer,
                        void * ptr, int value, size_t size_bytes){

    int ret;

    op_buffer -> ptr = ptr;
    op_buffer -> value = value;
    op_buffer -> size_bytes = size_bytes;

    ret = (handle -> submit_host_op)(handle, set_mem_host_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit set mem host op...\n");
        return -1;
    }

    return 0;
}


int dataflow_submit_add_host(Dataflow_Handle * handle, int stream_id, 
                        void * add_host_func, Add_Host_Op_Args * op_buffer,
                        DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                        int num_threads, int layer_id, size_t num_els, void * A, void * B, void * C,
                        float alpha, float beta){

    int ret;
    
    op_buffer -> A_dt = A_dt;
    op_buffer -> B_dt = B_dt;
    op_buffer -> C_dt = C_dt;

    op_buffer -> num_threads = num_threads;
    op_buffer -> layer_id = layer_id;
    op_buffer -> num_els = num_els;

    op_buffer -> A = A;
    op_buffer -> B = B;
    op_buffer -> C = C;

    op_buffer -> alpha = alpha;
    op_buffer -> beta = beta;

    ret = (handle -> submit_host_op)(handle, add_host_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit add host op...\n");
        return -1;
    }

    return 0;
}


// OPTIMIZER

int dataflow_submit_adam_step_host(Dataflow_Handle * handle, int stream_id, 
                        void * adam_host_func, Adam_Host_Op_Args * op_buffer,
						DataflowDatatype param_dt, DataflowDatatype grad_dt, 
                        DataflowDatatype mean_dt, DataflowDatatype var_dt,
                        int num_threads, int step_num, int layer_id, uint64_t num_els, 
                        float lr, float beta1, float beta2, float weight_decay, float epsilon,
                        void * param, void * grad, void * mean, void * var) {

    int ret;

    // NEEDS TO BE HANDLED DIFFERENTLY TO ENSURE OP ARG MEMORY IS NOT DEALLOCATED AFTER THIS FUNCTION CALL, 
    // BUT BEFORE THE HOST OP IS ACTUALLY CALLED...

    // different because other ops use the args immediately when submitted (either copied into driver for native, or used directly for external)

    op_buffer -> num_threads = num_threads;
    op_buffer -> step_num = step_num;
    op_buffer -> num_els = num_els;
    op_buffer -> layer_id = layer_id;

    op_buffer -> param_dt = param_dt;
    op_buffer -> grad_dt = grad_dt;
    op_buffer -> mean_dt = mean_dt;
    op_buffer -> var_dt = var_dt;

    op_buffer -> lr = lr;
    op_buffer -> beta1 = beta1;
    op_buffer -> beta2 = beta2;
    op_buffer -> weight_decay = weight_decay;
    op_buffer -> epsilon = epsilon;
    
   

    op_buffer -> param = param;
    op_buffer -> grad = grad;
    op_buffer -> mean = mean;
    op_buffer -> var = var;

    ret = (handle -> submit_host_op)(handle, adam_host_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit adam step host op...\n");
        return -1;
    }

    return 0;
    
}




// LOSS PRINTING


// Print Loss Host Op

int dataflow_submit_print_chunk_loss_host(Dataflow_Handle * handle, int stream_id,
									void * print_chunk_loss_host_func, Print_Chunk_Loss_Host_Op_Args * op_buffer,
									int step_num, int round_num, int seq_id, int chunk_id, int num_tokens, float * avg_loss_ref){

	int ret;
    
	op_buffer -> step_num = step_num;
	op_buffer -> round_num = round_num;
    op_buffer -> seq_id = seq_id;
    op_buffer -> chunk_id = chunk_id;
    op_buffer -> num_tokens = num_tokens;
    op_buffer -> avg_loss_ref = avg_loss_ref;


    ret = (handle -> submit_host_op)(handle, print_chunk_loss_host_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit print chunk loss host op...\n");
        return -1;
    }

    return 0;
}


int dataflow_submit_print_round_loss_host(Dataflow_Handle * handle, int stream_id,
									void * print_round_loss_host_func, Print_Round_Loss_Host_Op_Args * op_buffer,
									int step_num, int round_num, int num_seqs, int num_chunks, int total_tokens, float * per_chunk_avg_loss){

	int ret;

	op_buffer -> step_num = step_num;
	op_buffer -> round_num = round_num;
    op_buffer -> num_seqs = num_seqs;
	op_buffer -> num_chunks = num_chunks;
	op_buffer -> total_tokens = total_tokens;
	op_buffer -> per_chunk_avg_loss = per_chunk_avg_loss;

	ret = (handle -> submit_host_op)(handle, print_round_loss_host_func, op_buffer, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit print round loss host op...\n");
		return -1;
	}

	return 0;
}




// METRICS

// assumes that the op_buffer is already allocated and populated with the static model info...
int dataflow_submit_start_step_metrics_host(Dataflow_Handle * handle, int stream_id, 
                        void * start_step_metrics_func, Step_Throughput_Host_Op_Args * op_buffer,
						int step_num, int num_seqs, int * seqlens) {

    int ret;

    // NEEDS TO BE HANDLED DIFFERENTLY TO ENSURE OP ARG MEMORY IS NOT DEALLOCATED AFTER THIS FUNCTION CALL, 
    // BUT BEFORE THE HOST OP IS ACTUALLY CALLED...

    // different because other ops use the args immediately when submitted (either copied into driver for native, or used directly for external)

    op_buffer -> step_num = step_num;
    op_buffer -> num_seqs = num_seqs;
    memcpy(op_buffer -> seqlens, seqlens, num_seqs * sizeof(int));

    ret = (handle -> submit_host_op)(handle, start_step_metrics_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit start step metrics host op...\n");
        return -1;
    }

    return 0;
    
}


// assumes the same op_buffer that was used in the start_step_metrics_host op

int dataflow_submit_end_step_metrics_host(Dataflow_Handle * handle, int stream_id, 
                        void * end_step_metrics_func, Step_Throughput_Host_Op_Args * op_buffer){

    int ret;

    ret = (handle -> submit_host_op)(handle, end_step_metrics_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit end step metrics host op...\n");
        return -1;
    }
}

