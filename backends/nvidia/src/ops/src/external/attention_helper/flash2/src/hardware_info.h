/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#include "cuda_check.h"

inline int get_current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}
