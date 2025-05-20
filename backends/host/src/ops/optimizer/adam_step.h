#ifndef ADAM_STEP_H
#define ADAM_STEP_H

#include "host_ops.h"

typedef struct {
     // index of starting element
    uint64_t start_ind;
    // number of elements to process
    uint64_t num_els;
    // each of these are casted appropriately based
    // on the worker function
    void * param;
    void * grad;
    void * mean;
    void * var;
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float epsilon;
    uint64_t step_num;
} Adam_Worker_Args;

// Spwans num_threads to do the adam step, threads are pinned to the same node as calling thread

// from adam_step.c
// this is the fallback implementation for when avx2 is not supported...
int do_adam_step_host(DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt,
                             int num_threads,
                             uint64_t num_els, uint64_t step_num, float lr, float beta1, float beta2, float weight_decay, float epsilon,
                             void * param, void * grad, void * mean, void * var);


int do_adam_step_host_avx2(DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt,
                             int num_threads,
                             uint64_t num_els, uint64_t step_num, float lr, float beta1, float beta2, float weight_decay, float epsilon,
                             void * param, void * grad, void * mean, void * var);



// form adam_step_avx512.c

// in case of fp16 dtypes, this toggles if we want to do fp16 arithmetic for adam step
// saves on memory bandwidth because doing twice as many operations for same number 
// load insructions (for fp16: loading 512 elements instead of 256 elements)

// if this flag is toggled the fp16 elements are loaded and converted to fp32 before doing the adam step
// (better for precision, but likely not worth the extra memory bandwidth)
#define USE_AVX512_FP16_ARITHMETIC_FOR_ADAM_STEP 1

int do_adam_step_host_avx512(DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt,
                             int num_threads,
                             uint64_t num_els, uint64_t step_num, float lr, float beta1, float beta2, float weight_decay, float epsilon,
                             void * param, void * grad, void * mean, void * var);

#endif