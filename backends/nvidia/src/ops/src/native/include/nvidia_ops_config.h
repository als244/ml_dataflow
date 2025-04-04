#ifndef NVIDIA_OPS_CONFIG_H
#define NVIDIA_OPS_CONFIG_H

#include "dataflow.h"
#include "cuda_dataflow_handle.h"

#define ROUND_UP_TO_NEAREST_32(x) (((x) + 31) & ~31)

int embedding_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);


int rms_norm_set_attribute_config(Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function);
int rms_norm_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

int rms_norm_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int rms_norm_bwd_w_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

int rope_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int rope_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int copy_to_seq_context_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);


int swiglu_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int swiglu_bwd_x_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

int softmax_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);
int cross_entropy_loss_set_launch_config(Cuda_Launch_Config * cuda_launch_config, Dataflow_Handle * dataflow_handle, Cuda_Function * cuda_function, Op * op);

#endif