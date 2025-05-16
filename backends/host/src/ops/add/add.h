#include "host_ops.h"

int do_add_host(DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                int num_threads, size_t num_els, void * A, void * B, void * C);

int do_add_host_avx512(DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                int num_threads, size_t num_els, void * A, void * B, void * C);

