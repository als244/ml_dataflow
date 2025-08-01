

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cuda_check.h"

#include "namespace_config.h"
#include "bwd_agg_expanded_kv.h"

namespace FLASH_NAMESPACE {

template<>
void run_bwd_agg_expanded_kv_<120, __nv_bfloat16>(cudaStream_t stream, 
                                                    int num_seqs, int * k_seq_offsets, int * k_seq_lens, int max_k_seq_len, 
                                                    int head_dim, int n_q_heads, int n_kv_heads,
                                                    void * new_dk_expanded, void * new_dv_expanded,
                                                    void * orig_dk, void * orig_dv) {


    dim3 grid(max_k_seq_len, num_seqs);
    dim3 block(128);

    int kv_dim = n_kv_heads * head_dim;
    int model_dim = n_q_heads * head_dim;

    // do aggregation in fp32, then convert back to bf16
    // retrieve original dk and dv in fp32,
    // read in new dk, dv expanded in bf16
    int smem_size = 2 * sizeof(float) * kv_dim + 2 * sizeof(__nv_bfloat16) * model_dim;

    bwd_agg_expanded_kv_bf16<120><<<grid, block, smem_size, stream>>>(num_seqs, k_seq_offsets, k_seq_lens, 
                                                                           head_dim, n_q_heads, n_kv_heads,
                                                                           (__nv_bfloat16 *) new_dk_expanded, (__nv_bfloat16 *) new_dv_expanded,
                                                                           (__nv_bfloat16 *) orig_dk, (__nv_bfloat16 *) orig_dv);
}

} // namespace FLASH_NAMESPACE