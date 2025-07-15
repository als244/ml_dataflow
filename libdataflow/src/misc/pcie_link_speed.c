

float get_home_link_speed_bytes_per_sec(unsigned int pcie_link_width, unsigned int pcie_link_gen) {

    // PCIe 6.0: 8 GB/s per lane
    // PCIe 5.0: 4 GB/s per lane
    // PCIe 4.0: 2 GB/s per lane
    // PCIe 3.0: 1 GB/s per lane
    // PCIe 2.0: 0.5 GB/s per lane
    // PCIe 1.0: 0.25 GB/s per lane

    float bytes_per_sec =  0.25f * (float) (1UL << 30) * (float) (1UL << (pcie_link_gen - 1)) * (float) pcie_link_width;

    return bytes_per_sec;
}