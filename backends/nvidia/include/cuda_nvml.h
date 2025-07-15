#ifndef CUDA_NVML_H
#define CUDA_NVML_H

#include "dataflow_common.h"

#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif

#ifndef NVML_H
#define NVML_H
#include <nvml.h>
#endif

int cuda_nvml_init();

int cuda_nvml_get_pcie_info(int device_id, unsigned int * pcie_link_width, unsigned int * pcie_link_gen);

int cuda_nvml_shutdown();