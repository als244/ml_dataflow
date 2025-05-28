/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/


#include "cutlass/numeric_types.h"

#include "flash.h"
#include "static_switch.h"
#include "cuda_check.h"


#define ROUND_UP_TO_128(x) (((x) + 127) & ~127)

#define NEG_INF_FP32 0xFF800000


inline int round_up_headdim(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}


namespace FLASH_NAMESPACE {


inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}


int run_mha_fwd(int major_arch,Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    switch (major_arch) {
        case 80:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                            run_mha_fwd_<80, elem_type, kHeadDim, Is_causal>(params, stream);
                        } else {
                            run_mha_fwd_splitkv_dispatch<80, elem_type, kHeadDim, Is_causal>(params, stream);
                        }
                    });
                });
            });
            break;
        case 90:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                            run_mha_fwd_<90, elem_type, kHeadDim, Is_causal>(params, stream);
                        } else {
                            run_mha_fwd_splitkv_dispatch<90, elem_type, kHeadDim, Is_causal>(params, stream);
                        }
                    });
                });
            });
            break;
        case 100:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                            run_mha_fwd_<100, elem_type, kHeadDim, Is_causal>(params, stream);
                        } else {
                            run_mha_fwd_splitkv_dispatch<100, elem_type, kHeadDim, Is_causal>(params, stream);
                        }
                    });
                });
            });
            break;
        case 120:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                            run_mha_fwd_<120, elem_type, kHeadDim, Is_causal>(params, stream);
                        } else {
                            run_mha_fwd_splitkv_dispatch<120, elem_type, kHeadDim, Is_causal>(params, stream);
                        }
                    });
                });
            });
            break;
        default:
            fprintf(stderr, "Error: major_arch %d not supported in flash2...\n", major_arch);
            return -1;
    }
    return 0;
}

int run_mha_bwd(int major_arch, Flash_bwd_params &params, cudaStream_t stream) {
    switch (major_arch) {
        case 80:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        run_mha_bwd_<80, elem_type, kHeadDim, Is_causal>(params, stream);
                    });
                });
            });
            break;
        case 90:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        run_mha_bwd_<90, elem_type, kHeadDim, Is_causal>(params, stream);
                    });
                });
            });
            break;
        case 100:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        run_mha_bwd_<100, elem_type, kHeadDim, Is_causal>(params, stream);
                    });
                });
            });
            break;
        case 120:
            FP16_SWITCH(!params.is_bf16, [&] {
                HEADDIM_SWITCH(params.d, [&] {
                    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                        run_mha_bwd_<120, elem_type, kHeadDim, Is_causal>(params, stream);
                    });
                });
            });
            break;
        default:
            fprintf(stderr, "Error: major_arch %d not supported in flash2...\n", major_arch);
            return -1;
    }
    return 0;
    
}

int run_bwd_agg_expanded_kv(int major_arch, int is_bf16, cudaStream_t stream,
                            int num_seqs, int * k_seq_offsets, int * k_seq_lens, int max_k_seq_len,
                            int head_dim, int n_q_heads, int n_kv_heads,
                            void * new_dk_expanded, void * new_dv_expanded,
                            void * orig_dk, void * orig_dv) {
    switch (major_arch) {
        case 80:
            if (is_bf16) {
                run_bwd_agg_expanded_kv_<80, __nv_bfloat16>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            } else {
                run_bwd_agg_expanded_kv_<80, __half>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            }
            break;
        case 90:
            if (is_bf16) {
                run_bwd_agg_expanded_kv_<90, __nv_bfloat16>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            } else {
                run_bwd_agg_expanded_kv_<90, __half>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            }
            break;
        case 100:
            if (is_bf16) {
                run_bwd_agg_expanded_kv_<100, __nv_bfloat16>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            } else {
                run_bwd_agg_expanded_kv_<100, __half>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            }
            break;
        case 120:
            if (is_bf16) {
                run_bwd_agg_expanded_kv_<120, __nv_bfloat16>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            } else {
                run_bwd_agg_expanded_kv_<120, __half>(stream, num_seqs, k_seq_offsets, k_seq_lens, max_k_seq_len, head_dim, n_q_heads, n_kv_heads,
                new_dk_expanded, new_dv_expanded, orig_dk, orig_dv);
            }
            break;
        default:
            fprintf(stderr, "Error: major_arch %d not supported in flash2...\n", major_arch);
            return -1;
    }
    
    return 0;
}

extern "C" {
    
    // Note: must have already called set_flash2_fwd_params()
    //   (or set fully yourself)
    int set_flash2_fwd_workspace(Flash_fwd_params &params,
                                    void * attn_workspace,
                                    uint64_t * ret_used_workspace_size){

        // If running split_kv version, then we need temp space for lse_accum and oaccum...
        
        *ret_used_workspace_size = 0;

        return 0;
    }


    // Note: must have already called set_flash2_fwd_params()
    //   (or set fully yourself)
    int set_flash2_bwd_workspace(Flash_bwd_params &params,
                                    int num_sms,
                                    int total_k,
                                    void * attn_bwd_workspace,
                                    uint64_t * ret_used_workspace_size, void ** ret_set_to_zero_start, size_t * ret_set_to_zero_size){


       

        uint64_t used_workspace_size = 0;

        void * cur_attn_workspace = attn_bwd_workspace;

        // FOLLOWING WHAT WAS DONE IN ORGINAL SOURCE...

        int total_q = params.total_q;

        int num_q_heads = params.h;
        int num_kv_heads = params.h_k;

        int head_dim_rounded = params.d_rounded;

        int num_seqs = params.b;

        int seqlen_q = params.seqlen_q;
        int seqlen_k = params.seqlen_k;



        uint64_t softmax_size = num_q_heads * (total_q + 128 * num_seqs) * sizeof(float);

        params.dsoftmax_sum = cur_attn_workspace;

        cur_attn_workspace += softmax_size;
        used_workspace_size += softmax_size;


        if (num_q_heads != num_kv_heads){
            uint64_t kv_dtype_size = 2;
            uint64_t dkv_expanded_size = total_k * num_q_heads * head_dim_rounded * kv_dtype_size;
            
            params.dk_accum_ptr = cur_attn_workspace;
            cur_attn_workspace += dkv_expanded_size;
            used_workspace_size += dkv_expanded_size;

            params.dv_accum_ptr = cur_attn_workspace;
            cur_attn_workspace += dkv_expanded_size;
            used_workspace_size += dkv_expanded_size;
        }


        void * set_to_zero_start = cur_attn_workspace;
        uint64_t set_to_zero_size = 0;

        const int nsplits = (num_sms + num_seqs * num_q_heads - 1) / (num_seqs * num_q_heads);

        uint64_t dq_accum_size = nsplits * (total_q + 128 * num_seqs) * num_q_heads * head_dim_rounded * sizeof(float);

        params.dq_accum_split_stride = (total_q + 128 * num_seqs) * num_q_heads * head_dim_rounded;

        params.dq_accum_ptr = cur_attn_workspace;
        cur_attn_workspace += dq_accum_size;
        used_workspace_size += dq_accum_size;

        set_to_zero_size += dq_accum_size;

        if (ret_used_workspace_size){
            *ret_used_workspace_size = used_workspace_size;
        }
        
        if (ret_set_to_zero_start){
            *ret_set_to_zero_start = set_to_zero_start;
        }

        if (ret_set_to_zero_size){
            *ret_set_to_zero_size = set_to_zero_size;
        }

        return 0;
    }

    int set_flash2_fwd_params(Flash_fwd_params &params,
                        int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                        int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, void * softmax_lse, 
                        int is_causal) {

        int model_dim = num_q_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        params.is_bf16 = false;

        DataflowDatatype flash_dt = (DataflowDatatype) flash_dtype_as_int;

        if (flash_dt == DATAFLOW_BF16){
            params.is_bf16 = true;
        }
        else{
            if (flash_dt != DATAFLOW_FP16){
                fprintf(stderr, "Error: dtype of DataflowDatatype enum val of %d not supported in flash2...\n", flash_dtype_as_int);
                return -1;
            }
        }

        params.total_q = total_q;

        params.seqlen_q = max_seqlen_q;
        params.seqlen_q_rounded = ROUND_UP_TO_128(max_seqlen_q); 

        // Think it is ok to set this 0 and not take in max_seqlen_k...
        params.seqlen_k = max_seqlen_k;
        params.seqlen_k_rounded = ROUND_UP_TO_128(max_seqlen_k);

        params.seqlen_knew = 0;

        // these are used for split_kv, particularly for during decooding...
        params.oaccum_ptr = NULL;
        params.softmax_lseaccum_ptr = NULL; 
       

        params.q_ptr = x_q;
        params.k_ptr = x_k;
        params.v_ptr = x_v;
        params.o_ptr = x_attn_out;

        params.q_row_stride = model_dim;
        params.k_row_stride = kv_dim;
        params.v_row_stride = kv_dim;
        params.o_row_stride = model_dim;

        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;

        params.cu_seqlens_q = q_seq_offsets;
        params.cu_seqlens_k = k_seq_offsets;
        params.leftpad_k = NULL;

        params.seqused_k = k_seq_lens;

        params.p_ptr = NULL;
        params.knew_ptr = NULL;
        params.vnew_ptr = NULL;
        
        params.knew_batch_stride = 0;
        params.knew_row_stride = 0;
        params.knew_head_stride = 0;
        params.vnew_row_stride = 0;
        params.vnew_head_stride = 0;
        params.vnew_batch_stride = 0;


        params.q_batch_stride = 0;
        params.k_batch_stride = 0;
        params.v_batch_stride = 0;
        params.o_batch_stride = 0;

        params.blockmask = NULL;

        

        // Need to determine what to do here
        // (if dropout is non-zero...)
        params.rng_state = NULL;    


        params.softmax_lse_ptr = softmax_lse;

        int head_dim_rounded = round_up_headdim(head_dim);
        
        params.b = num_seqs;
        params.h = num_q_heads;
        params.h_k = num_kv_heads;
        params.h_h_k_ratio = params.h / params.h_k;
        params.d = head_dim;
        params.d_rounded = head_dim_rounded;

        params.scale_softmax = 1.0 / sqrtf((float) head_dim);
        params.softcap = 0.0f;
        params.scale_softmax_log2 = params.scale_softmax * (float) M_LOG2E;

        params.cache_batch_idx = NULL;
        params.block_table = NULL;
        params.block_table_batch_stride = 0;
        params.page_block_size = 0;

        params.p_dropout = 1.0f;

        params.p_dropout_in_uint8_t = (uint8_t) 255;

        params.rp_dropout = 1.0f;

        params.scale_softmax_rp_dropout = params.scale_softmax * params.rp_dropout;
    
        // Having API either be causal or full...
        params.is_causal = is_causal;

        // These are from flash3 lib but make sense here too...
        if (is_causal){
            params.window_size_left = max_seqlen_k - 1;
            params.window_size_right = 0;
        }
        else{
            params.window_size_left = -1;
            params.window_size_right = -1;
        }
        
        params.rotary_dim = 0;
        params.rotary_cos_ptr = NULL;
        params.rotary_sin_ptr = NULL;
        params.is_rotary_interleaved = false;
        

        params.alibi_slopes_ptr = NULL;
        params.alibi_slopes_batch_stride = 0;
        
        params.num_splits = 0;

        params.unpadded_lse = true;
        params.is_seqlens_k_cumulative = true;
        params.seqlenq_ngroups_swapped = false;

        return 0;
    }



    // if TYPE FP8, output must be BF16
    
    // To compute required size of attn_workspace:

    // attn_workspace_size = 0

    // Occum and LSE accum:
    // If num_splits > 1:
    //      attn_workspace_size += num_splits * sizeof(float) * num_q_heads * total_q * (1 + head_dim)
    
    // Tile count sem: 
    // If arch >= 90 || num_splits > 1:
    //      attn_workspace_size += sizeof(int)

    // Dynamic split ptr for each seq:
    // If num_seqs <= 992:
    //      attn_workspace_size += num_seqs * sizeof(int)

    int flash2_fwd_wrapper(CUstream stream, int arch, int num_sm,
                        int flash_dtype_as_int,
                        int num_seqs, int total_q, int total_k,
                        int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                        int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                        int num_q_heads, int num_kv_heads, int head_dim,
                        void * x_q, void * x_k, void * x_v,
                        void * x_attn_out, float * softmax_lse,
                        int is_causal,
                        uint64_t workspaceBytes, void * workspace) {

        
        int ret;

        Flash_fwd_params params;
        memset(&params, 0, sizeof(Flash_fwd_params));    

        ret = set_flash2_fwd_params(params,
                                    arch, num_sm,
                                    flash_dtype_as_int,
                                    num_seqs, total_q, total_k,
                                    q_seq_offsets, q_seq_lens, max_seqlen_q,
                                    k_seq_offsets, k_seq_lens, max_seqlen_k,
                                    num_q_heads, num_kv_heads, head_dim,
                                    x_q, x_k, x_v,
                                    x_attn_out, (void *) softmax_lse,
                                    is_causal);

        if (ret){
            fprintf(stderr, "Error: setting flash2 fwd params failed...\n");
            return -1;
        }

        uint64_t used_workspace_size = 0;
        ret = set_flash2_fwd_workspace(params, workspace, &used_workspace_size);
        if (ret){
            fprintf(stderr, "Error: setting flash2_fwd params failed...\n");
            return -1;
        }

        if (used_workspace_size > workspaceBytes){
            fprintf(stderr, "Error: attention fwd failed because not enough workspace. Supplied %lu bytes, but requires %lu...\n", workspaceBytes, used_workspace_size);
            return -1;
        }

        int major_arch = arch - (arch % 10);
        
       
        return run_mha_fwd(major_arch, params, stream);
    }

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
                            uint64_t workspaceBytes, void * workspace) {
        

        // ensure valid datatype...
        DataflowDatatype flash_dt = (DataflowDatatype) flash_dtype_as_int;
        if ((flash_dt != DATAFLOW_FP16) && (flash_dt != DATAFLOW_BF16)){
           fprintf(stderr, "Error: flash2 bwd only supports FP16 or BF16 bwds...\n");
           return -1;
        }
        

        // Set Same Params as FWD...
        int ret;

        Flash_bwd_params params;
        memset(&params, 0, sizeof(Flash_bwd_params));   

        ret = set_flash2_fwd_params(params,
                                    arch, num_sm,
                                    flash_dtype_as_int,
                                    num_seqs, total_q, total_k,
                                    q_seq_offsets, q_seq_lens, max_seqlen_q,
                                    k_seq_offsets, k_seq_lens, max_seqlen_k,
                                    num_q_heads, num_kv_heads, head_dim,
                                    x_q, x_k, x_v,
                                    x_attn_out, (void *)softmax_lse,
                                    is_causal);

        if (ret){
            fprintf(stderr, "Error: setting flash2 fwd params during bwd failed...\n");
            return -1;
        }


        // NOW SET BWD UNIQUE PARAMS...
        
        int model_dim = num_q_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        params.do_ptr = dx_out;
        params.do_row_stride = model_dim;
        params.do_head_stride = head_dim;
        params.do_batch_stride = 0;
        
        params.dq_ptr = dx_q;
        params.dq_row_stride = model_dim;
        params.dq_head_stride = head_dim;
        params.dq_batch_stride = 0;

        params.dk_ptr = dx_k;
    
        //params.dk_row_stride = kv_dim;
        params.dk_row_stride = model_dim;
        params.dk_head_stride = head_dim;
        params.dk_batch_stride = 0;

        params.dv_ptr = dx_v;
        //params.dv_row_stride = kv_dim;
        params.dv_row_stride = model_dim;
        params.dv_head_stride = head_dim;
        params.dv_batch_stride = 0;


        params.deterministic = true;


        uint64_t used_workspace_size = 0;
        void * set_to_zero_start = NULL;
        size_t set_to_zero_size = 0;
        ret = set_flash2_bwd_workspace(params, num_sm, total_k, workspace, &used_workspace_size, &set_to_zero_start, &set_to_zero_size);
        if (ret){
            fprintf(stderr, "Error: setting flash2 bwd workspace failed...\n");
            return -1;
        }

        if (used_workspace_size > workspaceBytes){
            fprintf(stderr, "Error: attention bwd failed because not enough workspace. Supplied %lu bytes, but requires %lu...\n", workspaceBytes, used_workspace_size);
            return -1;
        }

        CUresult res;
        if (set_to_zero_start){
            res = cuMemsetD8Async((CUdeviceptr) set_to_zero_start, 0, set_to_zero_size, stream);
            if (res != CUDA_SUCCESS){
                fprintf(stderr, "Error: cuMemset within flash2_bwd_wrapper failed...\n");
                return -1;
            }
        }

        if (num_q_heads != num_kv_heads){
            params.dk_ptr = params.dk_accum_ptr;
            params.dv_ptr = params.dv_accum_ptr;
            params.dk_accum_ptr = NULL;
            params.dv_accum_ptr = NULL;
        }  

        int major_arch = arch - (arch % 10);
        
        ret = run_mha_bwd(major_arch, params, stream);
        if (ret){
            fprintf(stderr, "Error: running flash2 bwd failed...\n");
            return -1;
        }

        if (num_q_heads != num_kv_heads){
            // TODO:
            // run kernel to combine the expanded dk and dv (which are now of dims (total_k, num_q_heads, head_dim))
            // and add them to the original dk and dv (which are of dims (total_k, num_kv_heads, head_dim))

            // at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
            // at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});

            ret = run_bwd_agg_expanded_kv(major_arch, params.is_bf16, stream,
                            num_seqs, k_seq_offsets, k_seq_lens, max_seqlen_k,
                            head_dim, num_q_heads, num_kv_heads,
                            params.dk_ptr, params.dv_ptr,
                            dx_k, dx_v);
            if (ret){
                fprintf(stderr, "Error: running bwd_agg_expanded_kv failed...\n");
                return -1;
            }

            CUDA_KERNEL_LAUNCH_CHECK();
        }

        return 0;

    }
}
}