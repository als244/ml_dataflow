#ifndef MATMUL_HELPER_H
#define MATMUL_HELPER_H

#include "dataflow.h"
#include "cuda_dataflow_handle.h" // for Cuda_Function

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif


#include <cublasLt.h>

typedef struct cublas_matmul_op_extra {
	cublasLtHandle_t cublas_handle;
	Dataflow_Table cublas_matmul_algo_table;
	uint64_t num_algos_saved;
	uint64_t num_matmuls_called;
	uint64_t num_algo_hits;
} Cublas_Matmul_Op_Extra;

// responsible for setting cuda function extra
int cublas_matmul_init(Dataflow_Handle * dataflow_handle, void * op_table_value);


// If A, C, D are all stored in Row-Major
// And B is stored in Col-Major. If so, it compute:
// D = alpha * AB + beta * C

// If B is stored in Row-Major that implies it computes:
// D = alpha * AB^T + beta * C
int cublas_matmul(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra);

#endif
