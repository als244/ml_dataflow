#include "dataflow.h"

#include <immintrin.h>  // For AVX512 intrinsics

int adam_step_host(void * _op);
