// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_<120, cutlass::half_t, 128, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::half_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
