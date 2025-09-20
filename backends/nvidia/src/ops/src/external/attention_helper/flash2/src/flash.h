/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    // MODIFIED! Added this in order to handle green context (using less SMs than device, when deterministic backwards is running
    // (modified run_flash_bwd_seqk_parallel within flash_bwd_launch_template.h
    int num_sm;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int Arch, typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<int Arch, typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<int Arch, typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

template<int Arch, typename T> void run_bwd_agg_expanded_kv_(cudaStream_t stream, 
                                                    int num_seqs, int * k_seq_offsets, int * k_seq_lens, int max_k_seq_len, 
                                                    int head_dim, int n_q_heads, int n_kv_heads,
                                                    void * new_dk_expanded, void * new_dv_expanded,
                                                    void * orig_dk, void * orig_dv);

}  // namespace FLASH_NAMESPACE





// PREPRARING FOR EXPORT

// MUST BE EXACT SAME ORDERING AS IN
// DATAFLOW_DATATYPE
typedef enum {
    DATAFLOW_NONE,
    DATAFLOW_VOID,
    DATAFLOW_FP64,
    DATAFLOW_FP32,
    DATAFLOW_FP16,
    DATAFLOW_BF16,
    DATAFLOW_FP8E4M3,
    DATAFLOW_FP8E5M2,
    DATAFLOW_UINT64,
    DATAFLOW_UINT32,
    DATAFLOW_UINT16,
    DATAFLOW_UINT8,
    DATAFLOW_LONG,
    DATAFLOW_INT,
    DATAFLOW_BOOL,
    DATAFLOW_FP64_SCALAR,
    DATAFLOW_FP32_SCALAR,
    DATAFLOW_FP16_SCALAR,
    DATAFLOW_BF16_SCALAR,
    DATAFLOW_FP8E4M3_SCALAR,
    DATAFLOW_FP8E5M2_SCALAR,
    DATAFLOW_UINT64_SCALAR,
    DATAFLOW_UINT32_SCALAR,
    DATAFLOW_UINT16_SCALAR,
    DATAFLOW_UINT8_SCALAR,
    DATAFLOW_LONG_SCALAR,
    DATAFLOW_INT_SCALAR,
    DATAFLOW_BOOL_SCALAR
} DataflowDatatype;

// C functions
extern "C" {

    int flash2_get_fwd_workspace_size(int flash_dtype_as_int, int arch, int num_sm, 
                                            int num_q_heads, int num_kv_heads, int head_dim, 
                                            int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                            int is_causal,
                                            uint64_t * ret_workspace_size);

    int flash2_get_bwd_workspace_size(int flash_dtype_as_int, int arch, int num_sm, 
                                            int num_q_heads, int num_kv_heads, int head_dim, 
                                            int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                            int is_causal,
                                            uint64_t * ret_workspace_size);
    
    int flash2_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                        int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, float * softmax_lse,
                        int is_causal,
                        uint64_t workspaceBytes, void * workspace);


    int flash2_bwd_wrapper(CUstream stream, int arch, int num_sm,
                            int flash_dtype_as_int, 
                            int num_seqs, int total_q, int total_k, 
                            int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                            int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                            int num_q_heads, int num_kv_heads, int head_dim, 
                            void * x_q, void * x_k, void * x_v, 
                            void * x_attn_out, float * softmax_lse, 
                            void * dx_out, 
                            void * dx_q, void * dx_k, void * dx_v,
                            int is_causal,
                            uint64_t workspaceBytes, void * workspace);
}