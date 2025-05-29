// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_combine_launch_template.h"

template void run_mha_fwd_combine_<80, float, float, 64>(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
template void run_mha_fwd_combine_<80, float, float, 128>(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

template void run_mha_fwd_combine_<80, cutlass::half_t, float, 64>(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
template void run_mha_fwd_combine_<80, cutlass::half_t, float, 128>(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

template void run_mha_fwd_combine_<80, cutlass::bfloat16_t, float, 64>(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
template void run_mha_fwd_combine_<80, cutlass::bfloat16_t, float, 128>(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
