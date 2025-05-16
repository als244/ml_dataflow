#include "dataflow_ops.h"

int dataflow_submit_adam_step_host(Dataflow_Handle * handle, int stream_id, 
                        void * adam_host_func,
						DataflowDatatype param_dt, DataflowDatatype grad_dt, 
                        DataflowDatatype mean_dt, DataflowDatatype var_dt,
                        int layer_id, float lr, float beta1, float beta2, float weight_decay, float epsilon,
						int num_threads, uint64_t num_els, 
                        void * param, void * grad, void * mean, void * var) {

    int ret;

    Op adam_step_op;

    dataflow_set_default_adam_step_skeleton(&adam_step_op.op_skeleton, param_dt, grad_dt, mean_dt, var_dt);

    void ** op_args = adam_step_op.op_args;

    op_args[0] = &num_threads;
    op_args[1] = &num_els;
    op_args[2] = &layer_id;
    op_args[3] = &lr;
    op_args[4] = &beta1;
    op_args[5] = &beta2;
    op_args[6] = &weight_decay;
    op_args[7] = &epsilon;
    op_args[8] = &param;
    op_args[9] = &grad;
    op_args[10] = &mean;
    op_args[11] = &var;

    ret = (handle -> submit_host_op)(handle, adam_host_func, &adam_step_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit adam step host op...\n");
        return -1;
    }

    return 0;
    
}




