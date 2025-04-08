#ifndef NVIDIA_OPS_CONFIG_H
#define NVIDIA_OPS_CONFIG_H

#include "dataflow.h"
#include "cuda_dataflow_handle.h"

#define ROUND_UP_TO_NEAREST_32(x) (((x) + 31) & ~31)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

int default_embedding_table_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);


int default_rms_norm_set_attribute_config(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function);
int default_rms_norm_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

int default_rms_norm_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int default_rms_norm_bwd_w_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

int default_rope_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int default_rope_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int default_copy_to_seq_context_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);


int default_swiglu_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int default_swiglu_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

int default_softmax_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int default_cross_entropy_loss_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

#endif