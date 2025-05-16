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
} Adam_Worker_Args;

// Spwans num_threads to do the adam step, threads are pinned to the same node as calling thread

// form adam_step.c
int do_adam_step_host(DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt,
                             int num_threads,
                             uint64_t num_els, float lr, float beta1, float beta2, float weight_decay, float epsilon,
                             void * param, void * grad, void * mean, void * var);

// form adam_step_avx512.c
int do_adam_step_host_avx512(DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt,
                             int num_threads,
                             uint64_t num_els, float lr, float beta1, float beta2, float weight_decay, float epsilon,
                             void * param, void * grad, void * mean, void * var);