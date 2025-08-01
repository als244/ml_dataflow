#include "dataflow_ops.h"


int dataflow_submit_default_select_experts(Dataflow_Handle * handle, int stream_id, 
                                DataflowDatatype fwd_dt,
                                int total_tokens, int n_experts, int top_k_experts,  
                                void * X_routed, float * token_expert_weights, 
                                uint16_t * chosen_experts, int * expert_counts, 
                                int * expert_counts_cumsum,
                                int * host_expert_counts) {

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
    op_args[8] = &host_expert_counts;

    ret = (handle -> submit_op)(handle, &select_experts_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit select experts op...\n");
        return -1;
    }

    return 0;
}


int dataflow_submit_default_build_expert_mapping(Dataflow_Handle * handle, int stream_id, 
                                int total_tokens, int num_routed_experts, int num_selected_experts, 
                                uint16_t * chosen_experts, int * expert_counts_cumsum,
                                int * expert_mapping) {

    int ret;

    Op build_expert_mapping_op;

    dataflow_set_default_build_expert_mapping_skeleton(&build_expert_mapping_op.op_skeleton);

    void ** op_args = build_expert_mapping_op.op_args;

    op_args[0] = &total_tokens;
    op_args[1] = &num_routed_experts;
    op_args[2] = &num_selected_experts;
    op_args[3] = &chosen_experts;
    op_args[4] = &expert_counts_cumsum;
    op_args[5] = &expert_mapping;

    ret = (handle -> submit_op)(handle, &build_expert_mapping_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit build expert mapping op...\n");
        return -1;
    }

    return 0;

}

int dataflow_submit_default_prepare_expert_zone(Dataflow_Handle * handle, int stream_id, 
                                DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
                                int model_dim, void * X, 
								int expert_id, int * expert_counts, int * expert_counts_cumsum,
                                int * expert_mapping, 
								void * expert_zone){

    int ret;

    Op prepare_expert_zone_op;

    dataflow_set_default_prepare_expert_zone_skeleton(&prepare_expert_zone_op.op_skeleton, attn_datatype, expert_datatype);

    void ** op_args = prepare_expert_zone_op.op_args;

    op_args[0] = &model_dim;
    op_args[1] = &X;
    op_args[2] = &expert_id;
    op_args[3] = &expert_counts;
    op_args[4] = &expert_counts_cumsum;
    op_args[5] = &expert_mapping;
    op_args[6] = &expert_zone;

    ret = (handle -> submit_op)(handle, &prepare_expert_zone_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit prepare expert zone op...\n");
        return -1;
    }

    return 0;
}

int dataflow_submit_default_merge_expert_result(Dataflow_Handle * handle, int stream_id, 
								DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
								int num_tokens, int model_dim, int top_k_experts, 
								void * expert_zone, int expert_id, 
								int * expert_counts_cumsum, 
								int * expert_mapping,
								float * token_expert_weights,
								uint16_t * chosen_experts,
								void * X_combined){

    int ret;

    Op merge_expert_result_op;

    dataflow_set_default_merge_expert_result_skeleton(&merge_expert_result_op.op_skeleton, attn_datatype, expert_datatype);

    void ** op_args = merge_expert_result_op.op_args;

    op_args[0] = &num_tokens;
    op_args[1] = &model_dim;
    op_args[2] = &top_k_experts;
    op_args[3] = &expert_zone;
    op_args[4] = &expert_id;
    op_args[5] = &expert_counts_cumsum;
    op_args[6] = &expert_mapping;
    op_args[7] = &token_expert_weights;
    op_args[8] = &chosen_experts;
    op_args[9] = &X_combined;

    ret = (handle -> submit_op)(handle, &merge_expert_result_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit merge expert result op...\n");
        return -1;
    }

    return 0;
}

int	dataflow_submit_router_bwd_x(Dataflow_Handle * handle, int stream_id,
                            DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
                            int num_tokens, int model_dim, int num_routed_experts, int top_k_active,
                            int expert_id,
                            int * expert_counts_cumsum,
                            int * expert_mapping,
                            uint16_t * chosen_experts,
                            float * token_expert_weights,
                            void * expert_out, void * upstream_dX,
                            void * dX_routed, // populating column [i] of router derivs with dot product of expert output and loss gradient corresponding to tokens selected by this expert
                            void * dX_expert_out) { // repopulating with the rows from inp_grad_stream -> X corresponding to this expert...

    int ret;

    Op router_bwd_x_op;

    dataflow_set_default_router_bwd_x_skeleton(&router_bwd_x_op.op_skeleton, attn_datatype, expert_datatype);

    void ** op_args = router_bwd_x_op.op_args;

    op_args[0] = &num_tokens;
    op_args[1] = &model_dim;
    op_args[2] = &num_routed_experts;
    op_args[3] = &top_k_active;
    op_args[4] = &expert_id;
    op_args[5] = &expert_counts_cumsum;
    op_args[6] = &expert_mapping;
    op_args[7] = &chosen_experts;
    op_args[8] = &token_expert_weights;
    op_args[9] = &expert_out;
    op_args[10] = &upstream_dX;
    op_args[11] = &dX_routed;
    op_args[12] = &dX_expert_out;

    ret = (handle -> submit_op)(handle, &router_bwd_x_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit router bwd x op...\n");
        return -1;
    }

    return 0;
}

int dataflow_submit_router_gate_bwd_x(Dataflow_Handle * handle, int stream_id,
                                DataflowDatatype attn_datatype, DataflowDatatype expert_datatype,
                                int num_tokens, int num_routed_experts, int top_k_active,
                                uint16_t * chosen_experts,
                                float * token_expert_weights,
                                void * dX_routed){

    int ret;

    Op router_gate_bwd_x_op;

    dataflow_set_default_router_gate_bwd_x_skeleton(&router_gate_bwd_x_op.op_skeleton, attn_datatype, expert_datatype);

    void ** op_args = router_gate_bwd_x_op.op_args;

    op_args[0] = &num_tokens;
    op_args[1] = &num_routed_experts;
    op_args[2] = &top_k_active;
    op_args[3] = &chosen_experts;
    op_args[4] = &token_expert_weights;
    op_args[5] = &dX_routed;

    ret = (handle -> submit_op)(handle, &router_gate_bwd_x_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit router gate bwd x op...\n");
        return -1;
    }

    return 0;
}