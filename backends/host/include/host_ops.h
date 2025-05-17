#ifndef HOST_OPS_H
#define HOST_OPS_H

#include "dataflow.h"
#include "dataflow_ops.h"

#include <immintrin.h>  // For AVX512 intrinsics

#define USE_FP16_ARTIHMETIC_FOR_ADD 1


int set_mem_host(void * _set_mem_host_op_args);

int add_host(void * _add_host_op_args);

int adam_step_host(void * _adam_host_op_args);



#endif
