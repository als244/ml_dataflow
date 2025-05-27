/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>


#define CUDA_CHECK(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

#endif
