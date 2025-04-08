#include "dataflow_ops.h"


int dataflow_submit_default_select_experts(Dataflow_Handle * handle, int stream_id, 
                                DataflowDatatype fwd_dt,
                                int total_tokens, int n_experts, int top_k_experts,  
                                void * X_routed, void * token_expert_weights, 
                                uint16_t * chosen_experts, int * expert_counts, 
                                int * expert_counts_cumsum, int * num_routed_by_expert_workspace) {

    int ret;

    Op select_experts_op;

    dataflow_set_default_select_experts_skeleton(&select_experts_op.op_skeleton, fwd_dt);

    void ** op_args = select_experts_op.op_args;

    op_args[0] = &total_tokens;
    op_args[1] = &n_experts;
    op_args[2] = &top_k_experts;
    op_args[3] = &X_routed;
    op_args[4] = &token_expert_weights;
    op_args[5] = &chosen_experts;
    op_args[6] = &expert_counts;
    op_args[7] = &expert_counts_cumsum;
    op_args[8] = &num_routed_by_expert_workspace;

    ret = (handle -> submit_op)(handle, &select_experts_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit select experts op...\n");
        return -1;
    }

    return 0;
}