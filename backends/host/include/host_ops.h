#ifndef HOST_OPS_H
#define HOST_OPS_H

#include "dataflow.h"

#include <immintrin.h>  // For AVX512 intrinsics


typedef struct Adam_Host_Op_Args{
    DataflowDatatype param_dt;
    DataflowDatatype grad_dt;
    DataflowDatatype mean_dt;
    DataflowDatatype var_dt;
    int num_threads;
    int layer_id;
    uint64_t num_els;
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float epsilon;
    void * param;
    void * grad;
    void * mean;
    void * var;
} Adam_Host_Op_Args;

typedef struct set_mem_host_op_args{
    void * ptr;
    size_t size_bytes;
    int value;
} Set_Mem_Host_Op_Args;

typedef struct add_host_op_args{
    DataflowDatatype A_dt;
    DataflowDatatype B_dt;
    DataflowDatatype C_dt;
    void * A;
    void * B;
    void * C;
    int num_threads;
    int layer_id;
    size_t num_els;
} Add_Host_Op_Args;

int set_mem_host(void * _set_mem_host_op_args);

int add_host(void * _add_host_op_args);

int adam_step_host(void * _adam_host_op_args);



#endif
