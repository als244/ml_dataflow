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

typedef struct Host_Set_Mem_Args{
    void * ptr;
    size_t size_bytes;
    int value;
} Host_Set_Mem_Args;

typedef struct Host_Add_Args{
    DataflowDatatype A_dt;
    DataflowDatatype B_dt;
    DataflowDatatype C_dt;
    void * A;
    void * B;
    void * C;
    int num_threads;
    int layer_id;
    size_t num_els;
} Host_Add_Args;

int set_mem_host(void * _host_set_mem_args);

int add_host(void * _host_add_args);

int adam_step_host(void * _adam_host_op_args);



#endif
