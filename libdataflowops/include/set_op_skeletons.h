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
void dataflow_set_flash3_attention_fwd_skeleton(Op_Skeleton * skeleton);
void dataflow_set_flash3_attention_bwd_skeleton(Op_Skeleton * skeleton);

void dataflow_set_embedding_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);

void dataflow_set_rms_norm_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_rms_norm_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void dataflow_set_rms_norm_bwd_w_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);

void dataflow_set_rope_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_rope_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);
void dataflow_set_copy_to_seq_context_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);

void dataflow_set_swiglu_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype);
void dataflow_set_swiglu_bwd_x_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);

void dataflow_set_softmax_skeleton(Op_Skeleton * skeleton, DataflowDatatype fwd_datatype, DataflowDatatype bwd_datatype);
void dataflow_set_cross_entropy_loss_skeleton(Op_Skeleton * skeleton, DataflowDatatype bwd_datatype);

#endif
