#ifndef SET_OP_SKELETONS_H
#define SET_OP_SKELETONS_H

#include "dataflow.h"

// This function should just be called during the registration of the op
// Not performance senesitive so the strcmps are fine...

// switches into respective function based on op_base_name
// return 0 on success, -1 on failure
// depending on function might either use fwd_dt or bwd_dt or both
int dataflow_set_op_skeleton(Op_Skeleton * skeleton, char * op_name, DataflowDatatype fwd_dt, DataflowDatatype bwd_dt);


// During op submission, the function should directly call the appropriate setter...

// Matmul Helper
void dataflow_set_matmul_skeleton(Op_Skeleton * skeleton);

// Flash3 Attention Helpder
void dataflow_set_flash_attention_fwd_skeleton(Op_Skeleton * skeleton);
void dataflow_set_flash_attention_bwd_skeleton(Op_Skeleton * skeleton);
void dataflow_set_flash_attention_get_workspace_size_skeleton(Op_Skeleton * skeleton);



// Default Ops

void dataflow_set_default_embedding_table_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_default_embedding_table_bwd_w_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);

void dataflow_set_default_rms_norm_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_default_rms_norm_recompute_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_default_rms_norm_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void dataflow_set_default_rms_norm_bwd_w_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void dataflow_set_default_rms_norm_bwd_w_combine_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);
void dataflow_set_default_rms_norm_noscale_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_default_rms_norm_noscale_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);

void dataflow_set_default_rope_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_default_rope_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);
// void dataflow_set_default_copy_to_seq_context_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);


void dataflow_set_default_swiglu_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_default_swiglu_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);

void dataflow_set_default_softmax_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void dataflow_set_default_cross_entropy_loss_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);


void dataflow_set_default_adamw_step_skeleton(Op_Skeleton * skeleton, DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt);


// moE kernels underway...
void dataflow_set_default_select_experts_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);

#endif
