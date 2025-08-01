
// Making all functions available within one cubin file...

#include "preprocess/embedding_table.cu"
#include "preprocess/embedding_table_bwd_w.cu"

#include "norm/rms_norm.cu"
#include "norm/rms_norm_recompute.cu"
#include "norm/rms_norm_bwd_x.cu"
#include "norm/rms_norm_bwd_w.cu"
#include "norm/rms_norm_bwd_w_combine.cu"
#include "norm/rms_norm_noscale.cu"
#include "norm/rms_norm_noscale_bwd_x.cu"

#include "attention_misc/rope.cu"
#include "attention_misc/rope_bwd_x.cu"
#include "attention_misc/copy_to_seq_context.cu"

#include "optimizer/adamw_step.cu"

#include "moe/select_experts.cu"
#include "moe/build_expert_mapping.cu"
#include "moe/prepare_expert_zone.cu"
#include "moe/merge_expert_result.cu"
#include "moe/router_bwd_x.cu"
#include "moe/router_gate_bwd_x.cu"

#include "activations/swiglu.cu"
#include "activations/swiglu_bwd_x.cu"

#include "loss_misc/softmax.cu"
#include "loss_misc/cross_entropy.cu"
