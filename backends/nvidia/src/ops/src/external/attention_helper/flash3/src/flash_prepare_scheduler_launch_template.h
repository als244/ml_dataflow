/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#ifndef FLASH_PREPARE_SCHEDULER_LAUNCH_TEMPLATE_H
#define FLASH_PREPARE_SCHEDULER_LAUNCH_TEMPLATE_H

#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"

#include "cutlass/arch/grid_dependency_control.h"

#include "flash.h"

namespace flash {

// Declaration of the CUDA kernel
template<int compiler_arch>
__global__ void prepare_varlen_num_blocks_kernel(
        int seqlen_q_static, int seqlen_k_static, int seqlen_k_new_static,
        int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
        int const* const seqused_q, int const* const seqused_k, int const* const leftpad_k_ptr,
        int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static,
        cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        int* const num_splits_dynamic_ptr,
        bool enable_pdl);

} // namespace flash

template<int compiler_arch>
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl);

#endif // FLASH_PREPARE_SCHEDULER_LAUNCH_TEMPLATE_H