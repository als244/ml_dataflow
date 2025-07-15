#include "cuda_nvml.h"

int cuda_nvml_init() {

    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return -1;
    }

    return 0;
}

int cuda_nvml_get_pcie_info(int device_id, unsigned int * pcie_link_width, unsigned int * pcie_link_gen) {

    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(device_id, &device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get device handle: %s\n", nvmlErrorString(result));
        return -1;
    }

    // Get PCIe link width
    result = nvmlDeviceGetPcieLinkWidth(device, pcie_link_width);
    if (NVML_SUCCESS != result) {
        fprintf(stderr, "Failed to get PCIe link width for device %d: %s\n", device_id, nvmlErrorString(result));
        return -1;
    }

    // Get PCIe link speed (generation)
    result = nvmlDeviceGetPcieLinkGen(device, pcie_link_gen);
    if (NVML_SUCCESS != result) {
        fprintf(stderr, "Failed to get PCIe link speed for device %d: %s\n", device_id, nvmlErrorString(result));
        return -1;
    }

    return 0;
}

int cuda_nvml_shutdown() {

    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to shutdown NVML: %s\n", nvmlErrorString(result));
        return -1;
    }

    return 0;
}