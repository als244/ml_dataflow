/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include "cutlass/numeric_types.h"

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"


#define ROUND_UP_TO_128(x) (((x) + 127) & ~127)


inline int round_up_headdim(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}


inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
}

inline int get_num_splits(Flash_fwd_params const& params) {
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
}

void run_mha_fwd_combine_80(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl) {
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<80, float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<80, float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<80, cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<80, cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<80, cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<80, cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
}

void run_mha_fwd_combine_90(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl) {
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<90, float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<90, float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<90, cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<90, cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<90, cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<90, cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        if (!params.is_e4m3) {
                            if (params.is_bf16) {
                                if (params.d <= 64) {
                                    if (params.dv > 256 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else if (params.dv > 64 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 96) { run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 128) { run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 256) { run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            } else {
                                if (params.d <= 64) {
                                    if (params.dv > 256 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else if (params.dv > 64 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 96) { run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 128) { run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                if (params.d <= 256) { run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            }
                        } else {
                            if (params.d <= 64) { run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            if (params.d <= 96) { run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            if (params.d <= 128) { run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            if (params.d <= 192) {
                                if (params.dv <= 128 && Arch == 90) {
                                    run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                } else {
                                    run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                }
                            }
                            if (params.d <= 256) { run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                        }
                    });
                });
            });
        });
    });

    if (params.num_splits > 1){
        if (params.arch == 90){
            run_mha_fwd_combine_90(params, stream, true);
        } else if (params.arch >= 80 && params.arch < 90) {
            run_mha_fwd_combine_80(params, stream, false);
        }
        else{
            fprintf(stderr, "Unsupported architecture for run_mha_fwd_combine in flash3: %d\n", params.arch);
        }
    }
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            if (!params.is_bf16) {
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap>(params, stream); }
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap>(params, stream); }
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap>(params, stream); }
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap>(params, stream); }
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap>(params, stream); }
            } else {
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
            }
        });
    });
}

extern "C" {

    int set_flash3_fwd_params(Flash_fwd_params &params,
                                int arch, int num_sm,
                                int flash_dtype_as_int,
                                int num_seqs, int total_q, int total_k,
                                int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q,
                                int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k,
                                int num_q_heads, int num_kv_heads, int head_dim,
                                void * x_q, void * x_k, void * x_v,
                                void * x_attn_out, float * softmax_lse, 
                                int is_causal) {

        int model_dim = num_q_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        params.is_fp32 = false;
        params.is_bf16 = false;
        params.is_e4m3 = false;

        DataflowDatatype flash_dt = (DataflowDatatype) flash_dtype_as_int;

        if (flash_dt == DATAFLOW_FP32){
            params.is_fp32 = true;
        }
        else if (flash_dt == DATAFLOW_BF16){
            params.is_bf16 = true;
        }
        else if (flash_dt == DATAFLOW_FP8E4M3){
            params.is_e4m3 = true;
        }
        else{
            if (flash_dt != DATAFLOW_FP16){
                fprintf(stderr, "Error: dtype of DataflowDatatype enum val of %d not supported in flash3...\n", flash_dtype_as_int);
            return -1;
            }
        }

        params.total_q = total_q;
        params.total_k = total_k;
        params.total_knew = 0;

        params.seqlen_q = max_seqlen_q;
        params.seqlen_q_rounded = ROUND_UP_TO_128(max_seqlen_q); 

        // Think it is ok to set this 0 and not take in max_seqlen_k...
        params.seqlen_k = max_seqlen_k;
        params.seqlen_k_rounded = ROUND_UP_TO_128(max_seqlen_k);

        params.seqlen_knew = 0;

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

        params.v_dim_stride = 1;

        params.cu_seqlens_q = q_seq_offsets;
        params.cu_seqlens_k = k_seq_offsets;
        params.cu_seqlens_knew = NULL;
        params.leftpad_k = NULL;


        params.seqused_q = q_seq_lens;
        params.seqused_k = k_seq_lens;

        params.knew_ptr = NULL;
        params.vnew_ptr = NULL;

        params.knew_batch_stride = 0;
        params.knew_row_stride = 0;
        params.knew_head_stride = 0;
        params.vnew_row_stride = 0;
        params.vnew_head_stride = 0;
        params.vnew_batch_stride = 0;

        params.q_descale_ptr = NULL;
        params.k_descale_ptr = NULL;
        params.v_descale_ptr = NULL;

        params.q_descale_batch_stride = 0;
        params.q_descale_head_stride = 0;
        params.k_descale_batch_stride = 0;
        params.k_descale_head_stride = 0;
        params.v_descale_batch_stride = 0;
        params.v_descale_head_stride = 0;

        params.q_batch_stride = 0;
        params.k_batch_stride = 0;
        params.v_batch_stride = 0;
        params.o_batch_stride = 0;



        params.qv_ptr = NULL;
        params.qv_batch_stride = 0;
        params.qv_row_stride = 0;
        params.qv_head_stride = 0;

        params.kv_batch_idx = NULL;
        params.page_table = NULL;
        params.page_table_batch_stride = 0;
        params.page_size = 0;
        params.num_pages = 0;
        params.pagedkv_tma = false;

        // Need to determine what to do here
        // (if dropout is non-zero...)
        params.rng_state = NULL;    

        // Will over-write if split
        params.oaccum_batch_stride = 0;
        params.oaccum_split_stride = 0;
        params.oaccum_row_stride = 0;
        params.oaccum_head_stride = 0;

        params.lseaccum_batch_stride = 0;
        params.lseaccum_split_stride = 0;
        params.lseaccum_head_stride = 0;


        params.softmax_lse_ptr = softmax_lse;

        int head_dim_rounded = round_up_headdim(head_dim);

        params.b = num_seqs;
        params.b_k = num_seqs;
        params.h = num_q_heads;
        params.h_k = num_kv_heads;
        params.d = head_dim;
        params.d_rounded = head_dim_rounded;
        params.dv = head_dim;
        params.dv_rounded = head_dim_rounded;

        params.scale_softmax = 1.0 / sqrtf((float) head_dim);
        params.softcap = 0.0f;

        params.p_dropout = 1.0f;

        params.p_dropout_in_uint8_t = (uint8_t) 255;

        params.rp_dropout = 1.0f;

        params.is_causal = is_causal;
        params.is_local = !is_causal;

        if (is_causal){
            params.window_size_left = max_seqlen_k - 1;
            params.window_size_right = 0;
        }
        else{
            // these might have -1...
            params.window_size_left = max_seqlen_k - 1;
            params.window_size_right = max_seqlen_q - 1;
        }

        params.rotary_dim = 0;
        params.rotary_cos_ptr = NULL;
        params.rotary_sin_ptr = NULL;
        params.seqlens_rotary = NULL;
        params.is_rotary_interleaved = false;


        params.arch = arch;
        params.num_sm = num_sm;

        return 0;
    }

    // Note: must have already called set_flash3_fwd_params()
    //   (or set fully yourself)
    int set_flash3_fwd_workspace(Flash_fwd_params &params,
                                void * attn_workspace,
                                uint64_t * ret_used_workspace_size, void ** ret_set_to_zero_start, size_t * ret_set_to_zero_size){


        int total_q = params.total_q;
        int head_dim = params.d;
        int num_q_heads = params.h;

        int model_dim = head_dim * num_q_heads;

        uint64_t used_workspace_size = 0;

        void * cur_attn_workspace = attn_workspace;

        // FOLLOWING WHAT WAS DONE IN ORGINAL SOURCE...

        params.num_splits_dynamic_ptr = (int *) 1;

        int num_splits = get_num_splits(params);
        params.num_splits = num_splits;

        if (params.num_splits > 1){

            // (num_splits, num_heads, total_q, headdim)
            params.oaccum_ptr = (float *) cur_attn_workspace;
            cur_attn_workspace += (num_splits * num_q_heads * total_q * head_dim * sizeof(float));
            used_workspace_size += (num_splits * num_q_heads * total_q * head_dim * sizeof(float));

            params.oaccum_split_stride = params.num_splits;
            params.oaccum_row_stride = model_dim;
            params.oaccum_head_stride = head_dim;

            // (num_splits, num_heads, total_q)
            params.softmax_lseaccum_ptr = (float *) cur_attn_workspace;
            cur_attn_workspace += (num_splits * num_q_heads * total_q * sizeof(float));
            used_workspace_size += (num_splits * num_q_heads * total_q * sizeof(float));
            params.lseaccum_split_stride = params.num_splits;
            params.lseaccum_head_stride = head_dim;
        }


        params.pack_gqa = get_pack_gqa(params);

        int to_use_dynamic_split = 0;

        // Harcoded number from original source
        if (params.b <= 992){
            to_use_dynamic_split = 1;
        } 

        int needs_sem = 0;
        if ((params.arch >= 90) || (params.num_splits > 1)){
            needs_sem = 1;
        }


        params.tile_count_semaphore = NULL;

        // reset back to null now
        params.num_splits_dynamic_ptr = NULL;

        void * set_to_zero_start = cur_attn_workspace;
        size_t set_to_zero_size = 0;

        if ((needs_sem) || (to_use_dynamic_split)) {
            if (needs_sem){
                if (!to_use_dynamic_split){
                    // ensure tile count semaphore set to zero
                    // should happen before call to this function
                }
                // only 1 int
                params.tile_count_semaphore = (int *) cur_attn_workspace;
                cur_attn_workspace += sizeof(int);
                used_workspace_size += sizeof(int);
                set_to_zero_size += sizeof(int);
            }

            if (to_use_dynamic_split){
                // need params.b integers here or params.b - 1..?
                // is this +1 a bug if the sched doesn't need sem...?
                // they initialzed buffer as needs_sem + use_dynamic * params.b
                // assuming bug...
                // params.num_splits_dynamic_ptr = ((int *) cur_attn_workspace + 1);
                params.num_splits_dynamic_ptr = (int *) cur_attn_workspace;
                cur_attn_workspace += params.b * sizeof(int);
                used_workspace_size += params.b * sizeof(int);
                set_to_zero_size += params.b * sizeof(int);
            }
        }


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

    int flash3_get_fwd_workspace_size(int flash_dtype_as_int, int arch, int num_sm, 
                                            int num_q_heads, int num_kv_heads, int head_dim, 
                                            int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                            int is_causal,
                                            uint64_t * ret_workspace_size){

        int ret;

        // To get the workspace size we need to determine num_splits
        // so easier to just populate a params struct with the necessary fields...
        Flash_fwd_params params;
        memset(&params, 0, sizeof(Flash_fwd_params));

        // Pass in dummy values for pointers, we don't care about assignment, 
        // just want to set params we can call set workspace to determine the workspace size...

        // Pass in worst-case values for total_q (== max_chunk_size) and total_k (== max_seq_len)
        
        int dummy_ptr;

        ret = set_flash3_fwd_params(params,
                                    arch, num_sm,
                                    flash_dtype_as_int,
                                    max_seqs_in_chunk, max_chunk_size, max_seq_len,
                                    &dummy_ptr, &dummy_ptr, max_chunk_size,
                                    &dummy_ptr, &dummy_ptr, max_seq_len,
                                    num_q_heads, num_kv_heads, head_dim,
                                    (void *) &dummy_ptr, (void *) &dummy_ptr, (void *) &dummy_ptr,
                                    (void *) &dummy_ptr, (float *) &dummy_ptr,
                                    is_causal);

        if (ret) {
            fprintf(stderr, "Error: unable to get flash3_get_fwd_workspace_size\n");
            return -1;
        }

        // now call set_workspace to discover what the required workspace size is...

        uint64_t workspace_size;

        ret = set_flash3_fwd_workspace(params, (void *) &dummy_ptr, &workspace_size, NULL, NULL);
        if (ret) {
            fprintf(stderr, "Error: unable to get flash3_get_fwd_workspace_size\n");
            return -1;
        }
        
        *ret_workspace_size = workspace_size;
        return 0;
    }
    



    // Note: must have already called set_flash3_fwd_params()
    //   (or set fully yourself)
    int set_flash3_bwd_workspace(Flash_bwd_params &params,
                                    void * attn_bwd_workspace,
                                    uint64_t * ret_used_workspace_size, void ** ret_set_to_zero_start, size_t * ret_set_to_zero_size){


       

        uint64_t used_workspace_size = 0;

        void * cur_attn_workspace = attn_bwd_workspace;

        // FOLLOWING WHAT WAS DONE IN ORGINAL SOURCE...

        int total_q = params.total_q;
        int total_k = params.total_k;

        int num_q_heads = params.h;
        int num_kv_heads = params.h_k;

        int num_seqs = params.b;

        int seqlen_q = params.seqlen_q;
        int seqlen_k = params.seqlen_k;


        int arch = params.arch;
        int is_causal = params.is_causal;
        int is_local = params.is_local;
        float softcap = params.softcap;

        int const head_dim_rounded = params.d_rounded;

        int const kBlockM_sm90 = head_dim_rounded <= 64 ? (is_causal && softcap > 0.0 ? 96 : 128)
                                    : (head_dim_rounded <= 96 ? 64
                                    : (head_dim_rounded <= 128 ? (is_causal || is_local || softcap > 0.0 ? 64 : 80)
                                    : 64));
        int const kBlockM_sm80 = head_dim_rounded <= 64 ? 128 : 64;
        int const kBlockM_sm86 = head_dim_rounded <= 192 ? 64 : 32;
        int const kBlockM = arch >= 90 ? kBlockM_sm90 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
        int const kBlockN_sm90 = head_dim_rounded <= 128 ? 128 : (head_dim_rounded <= 192 ? 96 : 80);
        int const kBlockN_sm80 = head_dim_rounded <= 128 ? 128 : (head_dim_rounded <= 192 ? 80 : 64);
        int const kBlockN_sm86 = head_dim_rounded <= 64 ? 128 : (head_dim_rounded <= 96 ? 128
                                    : (head_dim_rounded <= 128 ? 96
                                    : (head_dim_rounded <= 192 ? 64 : 64)));
        int const kBlockN = arch >= 90 ? kBlockN_sm90 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
        auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
        int const total_q_padded_rounded = round_multiple(total_q + num_seqs * kBlockM, kBlockM);
        int const total_k_padded_rounded = round_multiple(total_k + num_seqs * kBlockN, kBlockN);


        // Values to Set...

        // - softmax_d: (num_q_heads, total_q_padded_rounded), dtype=float32
        // - softmax_lse_log2: (num_q_heads, total_q_padded_rounded), dtype=float32

        // - dq_accum: (num_q_heads, total_q_padded_rounded * head_dim_rounded), dtype=float32
        
        // if (num_q_heads != num_kv_heads):
        //      - dk_accum: (num_kv_heads, total_k_padded_rounded, head_dim_rounded), dtype=float32
        //      - dv_accum: (num_kv_heads, total_k_padded_rounded, head_dim_rounded), dtype=float32
        

        // - dq_semaphore: ( (max_seqlen_q + kBlockM - 1) / (kBlockM), num_seqs, num_q_heads), dtype=int32
        // if (num_q_heads != num_kv_heads) & deterministic:
        //      - dk_semaphore: (max_seqlen_k + kBlockN - 1) / kBlockN, num_seqs, num_heads_kv), dtype=int32
        //      - dv_semaphore: (max_seqlen_k + kBlockN - 1) / kBlockN, num_seqs, num_heads_kv), dtype=int32



        uint64_t softmax_size = num_q_heads * total_q_padded_rounded * sizeof(float);

        params.dsoftmax_sum = cur_attn_workspace;

        cur_attn_workspace += softmax_size;
        used_workspace_size += softmax_size;

        params.softmax_lse_log2_ptr = cur_attn_workspace;
            
        cur_attn_workspace += softmax_size;
        used_workspace_size += softmax_size;

     

        uint64_t dq_accum_size = num_q_heads * total_q_padded_rounded * head_dim_rounded * sizeof(float);
        params.dq_accum_ptr = cur_attn_workspace;
        cur_attn_workspace += dq_accum_size;
        used_workspace_size += dq_accum_size;
        

        void * set_to_zero_start = cur_attn_workspace;
        size_t set_to_zero_size = 0;

        params.dk_accum_ptr = NULL;
        params.dv_accum_ptr = NULL;
        if (num_q_heads != num_kv_heads) {

            uint64_t dkv_accum_size = num_kv_heads * total_k_padded_rounded * head_dim_rounded * sizeof(float);
            params.dk_accum_ptr = cur_attn_workspace;

            cur_attn_workspace += dkv_accum_size;
            used_workspace_size += dkv_accum_size;
            set_to_zero_size += dkv_accum_size;

            params.dv_accum_ptr = cur_attn_workspace;
            cur_attn_workspace += dkv_accum_size;
            used_workspace_size += dkv_accum_size;
            set_to_zero_size += dkv_accum_size;
        }
        

        
        uint64_t dq_sem_size =  ((seqlen_q + kBlockM - 1) / (kBlockM)) * num_seqs * num_q_heads * sizeof(int);


        params.dq_semaphore = (int *) cur_attn_workspace;
        cur_attn_workspace += dq_sem_size;
        used_workspace_size += dq_sem_size;
        set_to_zero_size += dq_sem_size;

        params.dk_semaphore = NULL;
        params.dv_semaphore = NULL;
        if ((num_q_heads != num_kv_heads) && (params.deterministic)){

            uint64_t dkv_sem_size = ((seqlen_k + kBlockN - 1) / kBlockN) * num_seqs * num_kv_heads * sizeof(int);

            params.dk_semaphore = (int *) cur_attn_workspace;

            cur_attn_workspace += dkv_sem_size;
            used_workspace_size += dkv_sem_size;
            set_to_zero_size += dkv_sem_size;

            params.dv_semaphore = (int *) cur_attn_workspace;
            cur_attn_workspace += dkv_sem_size;
            used_workspace_size += dkv_sem_size;
            set_to_zero_size += dkv_sem_size;
        }
        
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


    int flash3_get_bwd_workspace_size(int flash_dtype_as_int, int arch, int num_sm, 
                                            int num_q_heads, int num_kv_heads, int head_dim, 
                                            int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
                                            int is_causal,
                                            uint64_t * ret_workspace_size){

        int ret;

        DataflowDatatype dtype = (DataflowDatatype) flash_dtype_as_int;

        if ((dtype != DATAFLOW_FP16) && (dtype != DATAFLOW_BF16)){
            fprintf(stderr, "Error: cannot get flash3 bwd workspace size for dtype (enum value %d), flash3 bwd only supports FP16 or BF16 bwds...\n", dtype);
            return -1;
        }

        // To get the workspace size we need to determine num_splits
        // so easier to just populate a params struct with the necessary fields...
        Flash_bwd_params params;
        memset(&params, 0, sizeof(Flash_bwd_params));

        int dummy_ptr;

        ret = set_flash3_fwd_params(params,
                                    arch, num_sm,
                                    flash_dtype_as_int,
                                    max_seqs_in_chunk, max_chunk_size, max_seq_len,
                                    &dummy_ptr, &dummy_ptr, max_chunk_size,
                                    &dummy_ptr, &dummy_ptr, max_seq_len,
                                    num_q_heads, num_kv_heads, head_dim,
                                    (void *) &dummy_ptr, (void *) &dummy_ptr, (void *) &dummy_ptr,
                                    (void *) &dummy_ptr, (float *) &dummy_ptr,
                                    is_causal);

        if (ret) {
            fprintf(stderr, "Error: unable to get flash3_get_bwd_workspace_size\n");
            return -1;
        }

        uint64_t workspace_size;

        ret = set_flash3_bwd_workspace(params, (void *) &dummy_ptr, &workspace_size, NULL, NULL);
        if (ret) {
            fprintf(stderr, "Error: unable to get flash3_get_bwd_workspace_size\n");
            return -1;
        }

        *ret_workspace_size = workspace_size;
        return 0;
    }

    // if TYPE FP8, output must be BF16

    int flash3_fwd_wrapper(CUstream stream, int arch, int num_sm,
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

        ret = set_flash3_fwd_params(params,
                                    arch, num_sm,
                                    flash_dtype_as_int,
                                    num_seqs, total_q, total_k,
                                    q_seq_offsets, q_seq_lens, max_seqlen_q,
                                    k_seq_offsets, k_seq_lens, max_seqlen_k,
                                    num_q_heads, num_kv_heads, head_dim,
                                    x_q, x_k, x_v,
                                    x_attn_out, softmax_lse,
                                    is_causal);

        if (ret){
            fprintf(stderr, "Error: setting flash3 fwd params failed...\n");
            return -1;
        }



        uint64_t used_workspace_size = 0;
        void * set_to_zero_start = NULL;
        size_t set_to_zero_size = 0;
        ret = set_flash3_fwd_workspace(params, workspace, &used_workspace_size, &set_to_zero_start, &set_to_zero_size);
        if (ret){
            fprintf(stderr, "Error: setting flash3_fwd params failed...\n");
            return -1;
        }

        if (used_workspace_size > workspaceBytes){
            fprintf(stderr, "Error: attention fwd failed because not enough workspace. Supplied %lu bytes, but requires %lu...\n", workspaceBytes, used_workspace_size);
            return -1;
        }


        CUresult res;
        if (set_to_zero_start){
            res = cuMemsetD8Async((CUdeviceptr) set_to_zero_start, 0, set_to_zero_size, stream);
            if (res != CUDA_SUCCESS){
                fprintf(stderr, "Error: cuMemset within flash3_fwd_wrapper failed...\n");
                return -1;
            }
        }

        // copying from Original source...
        if (params.num_splits_dynamic_ptr){

            auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
            auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, params.num_splits > 1, params.softcap > 0.f, params.knew_ptr);
            int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
            int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
            
            if (params.arch == 90){
                run_prepare_varlen_num_blocks_<90>(params, stream, params.pack_gqa, kBlockM, kBlockN, true /*enable_pdl*/);
                CHECK_CUDA_KERNEL_LAUNCH();
            } else if (params.arch >= 80 && params.arch < 90) {
                run_prepare_varlen_num_blocks_<80>(params, stream, params.pack_gqa, kBlockM, kBlockN, false /*enable_pdl*/);
                CHECK_CUDA_KERNEL_LAUNCH();
            } else {
                fprintf(stderr, "Unsupported architecture for prepare_varlen_num_blocks in flash3: %d\n", params.arch);
                return -1;
            }
        }

        // ^ did sched metadata above
        params.skip_scheduler_metadata_computation = true;
        
        // Also calls combine at end of function if 
        // num_splits > 1
        run_mha_fwd(params, stream);

        return 0;
    }

    int flash3_bwd_wrapper(CUstream stream, int arch, int num_sm,
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
           fprintf(stderr, "Error: flash3 bwd only supports FP16 or BF16 bwds...\n");
           return -1;
        }
        

        // Set Same Params as FWD...
        int ret;

        Flash_bwd_params params;
        memset(&params, 0, sizeof(Flash_bwd_params));   

        ret = set_flash3_fwd_params(params,
                                    arch, num_sm,
                                    flash_dtype_as_int,
                                    num_seqs, total_q, total_k,
                                    q_seq_offsets, q_seq_lens, max_seqlen_q,
                                    k_seq_offsets, k_seq_lens, max_seqlen_k,
                                    num_q_heads, num_kv_heads, head_dim,
                                    x_q, x_k, x_v,
                                    x_attn_out, softmax_lse,
                                    is_causal);

        if (ret){
            fprintf(stderr, "Error: setting flash3 fwd params during bwd failed...\n");
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
        params.dk_row_stride = kv_dim;
        params.dk_head_stride = head_dim;
        params.dk_batch_stride = 0;

        params.dv_ptr = dx_v;
        params.dv_row_stride = kv_dim;
        params.dv_head_stride = head_dim;
        params.dv_batch_stride = 0;


        params.deterministic = true;


        uint64_t used_workspace_size = 0;
        void * set_to_zero_start = NULL;
        size_t set_to_zero_size = 0;
        ret = set_flash3_bwd_workspace(params, workspace, &used_workspace_size, &set_to_zero_start, &set_to_zero_size);
        if (ret){
            fprintf(stderr, "Error: setting flash3 bwd workspace failed...\n");
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
                fprintf(stderr, "Error: cuMemset within flash3_bwd_wrapper failed...\n");
                return -1;
            }
        }   

        run_mha_bwd(params, stream);

        return 0;

    }
}