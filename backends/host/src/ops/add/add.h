#include "host_ops.h"

typedef struct {
     // index of starting element
    uint64_t start_ind;
    // number of elements to process
    uint64_t num_els;
    // each of these are casted appropriately based
    // on the worker function
    void * A;
    void * B;
    void * C;
    float alpha;
    float beta;
} Add_Worker_Args;

int do_add_host(DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                int num_threads, size_t num_els, void * A, void * B, void * C,
                float alpha, float beta);


// in case of fp16 dtypes, this toggles if we want to do fp16 arithmetic for additoins
// saves on memory bandwidth because doing twice as many operations for same number 
// load insructions (for fp16: loading 512 elements instead of 256 elements)

// if this flag is toggled the fp16 elements are loaded and converted to fp32 before doing the add
// (better for precision, but likely not worth the extra memory bandwidth)
#define USE_AVX512_FP16_ARITHMETIC_FOR_ADD 1

int do_add_host_avx512(DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                int num_threads, size_t num_els, void * A, void * B, void * C,
                float alpha, float beta);

