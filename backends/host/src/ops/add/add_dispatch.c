#include "add.h"

int add_host(void * _add_host_op_args){
    Add_Host_Op_Args * args = (Add_Host_Op_Args *) _add_host_op_args;

    // need to dispatch to a scalar, avx2, avx512, etc...

    void * A = args -> A;
    void * B = args -> B;
    void * C = args -> C;

    size_t num_els = args -> num_els;

    int num_threads = args -> num_threads;
    int layer_id = args -> layer_id;

    DataflowDatatype A_dt = args -> A_dt;
    DataflowDatatype B_dt = args -> B_dt;
    DataflowDatatype C_dt = args -> C_dt;

    float alpha = args -> alpha;
    float beta = args -> beta;


    
    // Now call the add dispatcher...

    printf("[AddDispatcher] Accumulating Gradients for Layer ID: %d...\n\n", layer_id);

	if (__builtin_cpu_supports("avx512f")){
        return do_add_host_avx512(A_dt, B_dt, C_dt, num_threads, num_els, A, B, C, alpha, beta);
    }
    else{
        return do_add_host(A_dt, B_dt, C_dt, num_threads, num_els, A, B, C, alpha, beta);
    }
    
    // not getting here    


    return 0;
}