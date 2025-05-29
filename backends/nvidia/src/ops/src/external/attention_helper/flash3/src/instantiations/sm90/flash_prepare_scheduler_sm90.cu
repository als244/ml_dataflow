// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_prepare_scheduler_launch_template.h"

template<>
void run_prepare_varlen_num_blocks_<90>(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl){
    int qhead_per_khead = !packgqa ? 1 : cutlass::ceil_div(params.h, params.h_k);
    // Max 1024 threads per block. Max 32 warps. (1024 / 32 = 32 batches per warp, but code uses NumThreadsPerWarp - 1)
    // (NumThreadsPerWarp - 1) = 31 batches per warp.
    // 1024 threads / 32 threads/warp = 32 warps.
    // Max batches this kernel can handle: 32 warps * 31 batches/warp = 992.
    // Ensure block size does not exceed device limits and is appropriate.
    dim3 gridDim(1);
    dim3 blockDim(1024); // Or a more configurable/calculated block size, e.g., params.num_sm * CUDART_WARP_SIZE
                         // Or, if num_batch is small, can reduce blockDim to (ceil_div(params.b, kNumBatchPerWarp) * NumThreadsPerWarp)
                         // For simplicity, using 1024 as in the original code.
                         // Consider the maximum number of batches to process.
    if (params.b == 0) return; // Nothing to do.

    flash::prepare_varlen_num_blocks_kernel<<<gridDim, blockDim, 0, stream>>>(
        params.seqlen_q, params.seqlen_k, params.seqlen_knew,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k, params.leftpad_k,
        params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
        cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
        params.tile_count_semaphore,
        params.num_splits_dynamic_ptr, enable_pdl);
}
