#include "dataflow_ops.h"

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



int dataflow_submit_adam_step_host(Dataflow_Handle * handle, int stream_id, 
                        void * adam_host_func, Adam_Host_Op_Args * op_buffer,
						DataflowDatatype param_dt, DataflowDatatype grad_dt, 
                        DataflowDatatype mean_dt, DataflowDatatype var_dt,
                        int num_threads, int layer_id, uint64_t num_els, 
                        float lr, float beta1, float beta2, float weight_decay, float epsilon,
                        void * param, void * grad, void * mean, void * var) {

    int ret;

    // NEEDS TO BE HANDLED DIFFERENTLY TO ENSURE OP ARG MEMORY IS NOT DEALLOCATED AFTER THIS FUNCTION CALL, 
    // BUT BEFORE THE HOST OP IS ACTUALLY CALLED...

    // different because other ops use the args immediately when submitted (either copied into driver for native, or used directly for external)

    op_buffer -> num_threads = num_threads;
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


int dataflow_submit_set_mem_host(Dataflow_Handle * handle, int stream_id, 
                        void * set_mem_host_func, Set_Mem_Host_Op_Args * op_buffer,
                        void * ptr, size_t size_bytes, int value){

    int ret;

    op_buffer -> ptr = ptr;
    op_buffer -> size_bytes = size_bytes;
    op_buffer -> value = value;

    ret = (handle -> submit_host_op)(handle, set_mem_host_func, op_buffer, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit set mem host op...\n");
        return -1;
    }

    return 0;
}



