#define _GNU_SOURCE
#include "add.h"

static inline float scalar_u16_to_fp32_fp16(uint16_t val_fp16_bits) {
    _Float16 f16_val; // Assumes compiler support for _Float16 with -mavx512fp16
    memcpy(&f16_val, &val_fp16_bits, sizeof(f16_val));
    return (float)f16_val;
}

static inline uint16_t scalar_fp32_to_u16_fp16(float val_fp32) {
    _Float16 f16_val = (_Float16)val_fp32; // Assumes compiler support for _Float16
    uint16_t u16_bits;
    memcpy(&u16_bits, &f16_val, sizeof(u16_bits));
    return u16_bits;
}


static inline float scalar_u16_to_fp32_bf16(uint16_t val_bf16_bits) {
    uint32_t val_fp32_bits = ((uint32_t)val_bf16_bits) << 16;
    float fp32_val;
    memcpy(&fp32_val, &val_fp32_bits, sizeof(fp32_val));
    return fp32_val;
}

static inline uint16_t scalar_fp32_to_u16_bf16(float val_fp32) {
    uint32_t u32_in;
    memcpy(&u32_in, &val_fp32, sizeof(u32_in));

    if (isnan(val_fp32)) {
        return ((u32_in & 0x80000000) >> 16) | 0x7FC0; // Preserve sign, return a QNaN pattern
    }
    if (isinf(val_fp32)) {
        return ((u32_in & 0x80000000) >> 16) | 0x7F80; // Preserve sign, Inf
    }

    // Round to Nearest, Ties to Even
    uint32_t sign = u32_in & 0x80000000;
    uint32_t u32_abs = u32_in & 0x7FFFFFFF; // Work with magnitude

    // Special case for 0.0
    if (u32_abs == 0) {
        return (uint16_t)(sign >> 16);
    }
    
    uint16_t truncated_val_abs = (uint16_t)(u32_abs >> 16);
    uint32_t remainder = u32_abs & 0xFFFF;

    if (remainder > 0x8000) { // Greater than halfway
        truncated_val_abs++;
    } else if (remainder == 0x8000) { // Exactly halfway
        if ((truncated_val_abs & 1) != 0) { // If truncated value's LSB is 1 (odd), round up (to make it even)
            truncated_val_abs++;
        }
    }
    // Else (less than halfway, or exactly halfway and truncated is even), keep truncated.

    // Check for overflow after rounding (exponent becomes all 1s)
    // This means the magnitude (without sign) would look like BF16 infinity.
    if ((truncated_val_abs & 0x7F80) == 0x7F80) {
         // If original was not Inf (already handled), this is an overflow due to rounding
        if (!isinf(val_fp32)) { // Check original again to be safe
            return (uint16_t)(sign >> 16) | 0x7F80; // Overflow to Inf
        }
    }
    return (uint16_t)(sign >> 16) | (truncated_val_abs & 0x7FFF); // Combine sign with rounded magnitude
}



static void * thread_func_add_avx2_fp32(void * _add_worker_args){
    Add_Worker_Args * args = (Add_Worker_Args *) _add_worker_args;

    float * const restrict orig_A = (float *)args->A;
    float * const restrict orig_B = (float *)args->B;
    float * const restrict orig_C = (float *)args->C; // Destination

    const float alpha_val = args->alpha;
    const float beta_val = args->beta;
    const size_t start_ind = args->start_ind;
    const size_t num_els = args->num_els;

    if (num_els == 0) {
        return NULL;
    }

    const bool A_is_C = (orig_A == orig_C);
    const bool B_is_C = (orig_B == orig_C);
    const bool A_is_B = (orig_A == orig_B);

    const int vec_len = 8; // 8 float elements in a YMM register
    const size_t num_vectors = num_els / vec_len;

    const __m256 vec_alpha = _mm256_set1_ps(alpha_val);
    const __m256 vec_beta = _mm256_set1_ps(beta_val);

    const bool alpha_is_one = (alpha_val == 1.0f);
    const bool beta_is_one = (beta_val == 1.0f);

    for (size_t i = 0; i < num_vectors; ++i) {
        size_t current_block_abs_offset = start_ind + i * vec_len;
        
        float * const ptr_C_block_dst = orig_C + current_block_abs_offset;
        const float * ptr_A_block_src;
        const float * ptr_B_block_src;

        __m256 vec_a, vec_b, result_vec;

        if (A_is_C) {
            ptr_A_block_src = ptr_C_block_dst;
        } else {
            ptr_A_block_src = orig_A + current_block_abs_offset;
        }
        vec_a = _mm256_loadu_ps(ptr_A_block_src);

        if (A_is_B) {
            vec_b = vec_a;
        } else {
            if (B_is_C) {
                ptr_B_block_src = ptr_C_block_dst;
            } else {
                ptr_B_block_src = orig_B + current_block_abs_offset;
            }
            vec_b = _mm256_loadu_ps(ptr_B_block_src);
        }

        if (alpha_is_one && beta_is_one) {
            result_vec = _mm256_add_ps(vec_a, vec_b);
        } else if (alpha_is_one) {
            result_vec = _mm256_fmadd_ps(vec_beta, vec_b, vec_a); // beta*B + A
        } else if (beta_is_one) {
            result_vec = _mm256_fmadd_ps(vec_alpha, vec_a, vec_b); // alpha*A + B
        } else {
            __m256 term_a = _mm256_mul_ps(vec_alpha, vec_a);
            result_vec = _mm256_fmadd_ps(vec_beta, vec_b, term_a); // beta*B + alpha*A
        }
        _mm256_storeu_ps(ptr_C_block_dst, result_vec);
    }

    const size_t processed_elements = num_vectors * vec_len;
    for (size_t i = processed_elements; i < num_els; ++i) {
        size_t current_scalar_abs_offset = start_ind + i;
        
        float * const ptr_C_scalar_dst = orig_C + current_scalar_abs_offset;
        const float * ptr_A_scalar_src;
        const float * ptr_B_scalar_src;

        float val_a, val_b;

        if (A_is_C) {
            ptr_A_scalar_src = ptr_C_scalar_dst;
        } else {
            ptr_A_scalar_src = orig_A + current_scalar_abs_offset;
        }
        val_a = *ptr_A_scalar_src;

        if (A_is_B) {
            val_b = val_a;
        } else {
            if (B_is_C) {
                ptr_B_scalar_src = ptr_C_scalar_dst;
            } else {
                ptr_B_scalar_src = orig_B + current_scalar_abs_offset;
            }
            val_b = *ptr_B_scalar_src;
        }
        
        if (alpha_is_one && beta_is_one) {
             *ptr_C_scalar_dst = val_a + val_b;
        } else {
             *ptr_C_scalar_dst = alpha_val * val_a + beta_val * val_b;
        }
    }
    return NULL;
}

static void * thread_func_add_avx2_bf16(void * _add_worker_args){
    Add_Worker_Args * args = (Add_Worker_Args *) _add_worker_args;

    uint16_t * const restrict orig_A_bf16 = (uint16_t *)args->A;
    uint16_t * const restrict orig_B_bf16 = (uint16_t *)args->B;
    uint16_t * const restrict orig_C_bf16 = (uint16_t *)args->C;

    const float alpha_val = args->alpha;
    const float beta_val = args->beta;
    const size_t start_ind = args->start_ind;
    const size_t num_els = args->num_els;

    if (num_els == 0) {
        return NULL;
    }

    const bool A_is_C = (orig_A_bf16 == orig_C_bf16);
    const bool B_is_C = (orig_B_bf16 == orig_C_bf16);
    const bool A_is_B = (orig_A_bf16 == orig_B_bf16);

    const int vec_len = 8; // Process 8 elements (bf16 -> fp32) at a time
    const size_t num_vectors = num_els / vec_len;

    const __m256 vec_alpha_ps = _mm256_set1_ps(alpha_val);
    const __m256 vec_beta_ps = _mm256_set1_ps(beta_val);

    const bool alpha_is_one = (alpha_val == 1.0f);
    const bool beta_is_one = (beta_val == 1.0f);

    float temp_fp32_result_buffer[8]; // For scalar conversion store

    for (size_t i = 0; i < num_vectors; ++i) {
        size_t current_block_abs_offset = start_ind + i * vec_len;
        
        uint16_t * const ptr_C_block_dst_bf16 = orig_C_bf16 + current_block_abs_offset;
        const uint16_t * ptr_A_block_src_bf16;
        const uint16_t * ptr_B_block_src_bf16;

        __m128i vec_a_bf16_bits, vec_b_bf16_bits; 

        if (A_is_C) {
            ptr_A_block_src_bf16 = ptr_C_block_dst_bf16;
        } else {
            ptr_A_block_src_bf16 = orig_A_bf16 + current_block_abs_offset;
        }
        vec_a_bf16_bits = _mm_loadu_si128((const __m128i*)ptr_A_block_src_bf16);

        if (A_is_B) {
            vec_b_bf16_bits = vec_a_bf16_bits;
        } else {
            if (B_is_C) {
                ptr_B_block_src_bf16 = ptr_C_block_dst_bf16;
            } else {
                ptr_B_block_src_bf16 = orig_B_bf16 + current_block_abs_offset;
            }
            vec_b_bf16_bits = _mm_loadu_si128((const __m128i*)ptr_B_block_src_bf16);
        }
        
        // Convert 8 BF16s from __m128i to 8 FP32s in __m256
        __m256i vec_a_u32 = _mm256_cvtepu16_epi32(vec_a_bf16_bits);
        __m256i vec_b_u32 = _mm256_cvtepu16_epi32(vec_b_bf16_bits);

        __m256i vec_a_fp32_bits = _mm256_slli_epi32(vec_a_u32, 16);
        __m256i vec_b_fp32_bits = _mm256_slli_epi32(vec_b_u32, 16);

        __m256 vec_a_ps = _mm256_castsi256_ps(vec_a_fp32_bits);
        __m256 vec_b_ps = _mm256_castsi256_ps(vec_b_fp32_bits);
        
        __m256 result_vec_ps;

        if (alpha_is_one && beta_is_one) {
            result_vec_ps = _mm256_add_ps(vec_a_ps, vec_b_ps);
        } else if (alpha_is_one) {
            result_vec_ps = _mm256_fmadd_ps(vec_beta_ps, vec_b_ps, vec_a_ps);
        } else if (beta_is_one) {
            result_vec_ps = _mm256_fmadd_ps(vec_alpha_ps, vec_a_ps, vec_b_ps);
        } else {
            __m256 term_a_ps = _mm256_mul_ps(vec_alpha_ps, vec_a_ps);
            result_vec_ps = _mm256_fmadd_ps(vec_beta_ps, vec_b_ps, term_a_ps);
        }
        
        // Convert 8 FP32s in result_vec_ps back to 8 BF16s and store
        // No direct AVX2 intrinsic for FP32->BF16 with rounding.
        // Store __m256 to a temporary buffer and convert scalar one-by-one.
        _mm256_storeu_ps(temp_fp32_result_buffer, result_vec_ps);
        for (int k=0; k < vec_len; ++k) {
            ptr_C_block_dst_bf16[k] = scalar_fp32_to_u16_bf16(temp_fp32_result_buffer[k]);
        }
    }

    const size_t processed_elements = num_vectors * vec_len;
    for (size_t i = processed_elements; i < num_els; ++i) {
        size_t current_scalar_abs_offset = start_ind + i;

        uint16_t * const ptr_C_scalar_dst_bf16 = orig_C_bf16 + current_scalar_abs_offset;
        const uint16_t * ptr_A_scalar_src_bf16;
        const uint16_t * ptr_B_scalar_src_bf16;
        
        uint16_t val_a_bf16_bits;
        float val_a_ps;

        if (A_is_C) {
            ptr_A_scalar_src_bf16 = ptr_C_scalar_dst_bf16;
        } else {
            ptr_A_scalar_src_bf16 = orig_A_bf16 + current_scalar_abs_offset;
        }
        val_a_bf16_bits = *ptr_A_scalar_src_bf16;
        val_a_ps = scalar_u16_to_fp32_bf16(val_a_bf16_bits);
        
        float val_b_ps;
        if (A_is_B) {
            val_b_ps = val_a_ps;
        } else {
            uint16_t val_b_bf16_bits;
            if (B_is_C) {
                ptr_B_scalar_src_bf16 = ptr_C_scalar_dst_bf16;
            } else {
                ptr_B_scalar_src_bf16 = orig_B_bf16 + current_scalar_abs_offset;
            }
            val_b_bf16_bits = *ptr_B_scalar_src_bf16;
            val_b_ps = scalar_u16_to_fp32_bf16(val_b_bf16_bits);
        }
        
        float result_ps;
        if (alpha_is_one && beta_is_one) {
            result_ps = val_a_ps + val_b_ps;
        } else {
            result_ps = alpha_val * val_a_ps + beta_val * val_b_ps;
        }
        *ptr_C_scalar_dst_bf16 = scalar_fp32_to_u16_bf16(result_ps);
    }
    return NULL;
}

static void * thread_func_add_avx2_fp16(void * _add_worker_args){
    Add_Worker_Args * args = (Add_Worker_Args *) _add_worker_args;

    uint16_t * const restrict orig_A_ph = (uint16_t *)args->A; // ph for "packed half-precision"
    uint16_t * const restrict orig_B_ph = (uint16_t *)args->B;
    uint16_t * const restrict orig_C_ph = (uint16_t *)args->C;

    const float alpha_val = args->alpha;
    const float beta_val = args->beta;
    const size_t start_ind = args->start_ind;
    const size_t num_els = args->num_els;

    if (num_els == 0) {
        return NULL;
    }

    const bool A_is_C = (orig_A_ph == orig_C_ph);
    const bool B_is_C = (orig_B_ph == orig_C_ph);
    const bool A_is_B = (orig_A_ph == orig_B_ph);

    const int vec_len = 8; // Process 8 elements (fp16 -> fp32) at a time using F16C
    const size_t num_vectors = num_els / vec_len;

    const __m256 vec_alpha_ps = _mm256_set1_ps(alpha_val);
    const __m256 vec_beta_ps = _mm256_set1_ps(beta_val);
    
    const bool alpha_is_one = (alpha_val == 1.0f);
    const bool beta_is_one = (beta_val == 1.0f);

    for (size_t i = 0; i < num_vectors; ++i) {
        size_t current_block_abs_offset = start_ind + i * vec_len;
        
        uint16_t * const ptr_C_block_dst_ph = orig_C_ph + current_block_abs_offset;
        // Load 8 fp16 values (128 bits)
        const uint16_t * ptr_A_block_src_ph;
        const uint16_t * ptr_B_block_src_ph;

        __m128i vec_a_ph_bits, vec_b_ph_bits; 

        if (A_is_C) {
            ptr_A_block_src_ph = ptr_C_block_dst_ph;
        } else {
            ptr_A_block_src_ph = orig_A_ph + current_block_abs_offset;
        }
        // Load 8 uint16_t values into an __m128i register
        vec_a_ph_bits = _mm_loadu_si128((const __m128i*)ptr_A_block_src_ph);

        if (A_is_B) {
            vec_b_ph_bits = vec_a_ph_bits;
        } else {
            if (B_is_C) {
                ptr_B_block_src_ph = ptr_C_block_dst_ph;
            } else {
                ptr_B_block_src_ph = orig_B_ph + current_block_abs_offset;
            }
            vec_b_ph_bits = _mm_loadu_si128((const __m128i*)ptr_B_block_src_ph);
        }
        
        // Convert 8 packed FP16 in __m128i to 8 packed FP32 in __m256
        __m256 vec_a_ps = _mm256_cvtph_ps(vec_a_ph_bits);
        __m256 vec_b_ps = _mm256_cvtph_ps(vec_b_ph_bits);
        
        __m256 result_vec_ps;

        if (alpha_is_one && beta_is_one) {
            result_vec_ps = _mm256_add_ps(vec_a_ps, vec_b_ps);
        } else if (alpha_is_one) {
            result_vec_ps = _mm256_fmadd_ps(vec_beta_ps, vec_b_ps, vec_a_ps);
        } else if (beta_is_one) {
            result_vec_ps = _mm256_fmadd_ps(vec_alpha_ps, vec_a_ps, vec_b_ps);
        } else {
            __m256 term_a_ps = _mm256_mul_ps(vec_alpha_ps, vec_a_ps);
            result_vec_ps = _mm256_fmadd_ps(vec_beta_ps, vec_b_ps, term_a_ps);
        }

        // Convert 8 packed FP32 in __m256 back to 8 packed FP16 in __m128i
        // _MM_FROUND_TO_NEAREST_INT often used as default rounding for F16C conversions
        __m128i result_vec_ph_bits = _mm256_cvtps_ph(result_vec_ps, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)ptr_C_block_dst_ph, result_vec_ph_bits);
    }

    const size_t processed_elements = num_vectors * vec_len;
    for (size_t i = processed_elements; i < num_els; ++i) {
        size_t current_scalar_abs_offset = start_ind + i;

        uint16_t * const ptr_C_scalar_dst_ph = orig_C_ph + current_scalar_abs_offset;
        const uint16_t * ptr_A_scalar_src_ph;
        const uint16_t * ptr_B_scalar_src_ph;

        uint16_t val_a_ph_bits;
        float val_a_ps;

        if (A_is_C) {
            ptr_A_scalar_src_ph = ptr_C_scalar_dst_ph;
        } else {
            ptr_A_scalar_src_ph = orig_A_ph + current_scalar_abs_offset;
        }
        val_a_ph_bits = *ptr_A_scalar_src_ph;
        val_a_ps = scalar_u16_to_fp32_fp16(val_a_ph_bits);
        
        float val_b_ps;
        if (A_is_B) {
            val_b_ps = val_a_ps;
        } else {
            uint16_t val_b_ph_bits;
            if (B_is_C) {
                ptr_B_scalar_src_ph = ptr_C_scalar_dst_ph;
            } else {
                ptr_B_scalar_src_ph = orig_B_ph + current_scalar_abs_offset;
            }
            val_b_ph_bits = *ptr_B_scalar_src_ph;
            val_b_ps = scalar_u16_to_fp32_fp16(val_b_ph_bits);
        }
        
        float result_ps;
        if (alpha_is_one && beta_is_one) {
            result_ps = val_a_ps + val_b_ps;
        } else {
            result_ps = alpha_val * val_a_ps + beta_val * val_b_ps;
        }
        *ptr_C_scalar_dst_ph = scalar_fp32_to_u16_fp16(result_ps);
    }
    return NULL;
}


static void * thread_func_add_avx2_fp8e4m3(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    uint8_t * A = add_worker_args->A;
    uint8_t * B = add_worker_args->B;
    uint8_t * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    fprintf(stderr, "Error: add_avx512_fp8e4m3 add not supported yet...\n");
    return NULL;

}

static void * thread_func_add_avx2_fp8e5m2(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    uint8_t * A = add_worker_args->A;
    uint8_t * B = add_worker_args->B;
    uint8_t * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    fprintf(stderr, "Error: add_avx512_fp8e5m2 add not supported yet...\n");
    return NULL;

}





int do_add_host_avx2(DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                int num_threads, size_t num_els, void * A, void * B, void * C,
                float alpha, float beta) { 
    

    // for now only support when all same dtype..
    if (A_dt != B_dt || A_dt != C_dt){
        fprintf(stderr, "Error: all dtypes must be the same for avx2 add...\n");
        return -1;
    }

    void * (*add_func)(void * _add_worker_args);

    if (A_dt == DATAFLOW_FP32){
        add_func = thread_func_add_avx2_fp32;
    }
    else if (A_dt == DATAFLOW_BF16){
        add_func = thread_func_add_avx2_bf16;
    }
    else if (A_dt == DATAFLOW_FP16){
        add_func = thread_func_add_avx2_fp16;
    }
    /* Not ready for fp8 yet
    else if (A_dt == DATAFLOW_FP8E4M3){
        add_func = thread_func_add_avx2_fp8e4m3;
    }
    else if (A_dt == DATAFLOW_FP8E5M2){
        add_func = thread_func_add_avx2_fp8e5m2;
    }
    */
    else{
        fprintf(stderr, "Error: unsupported dtype for avx2 add...\n");
        return -1;
    }

    if (!add_func){
        fprintf(stderr, "Error: failed to get thread function for avx2 add...\n");
        return -1;
    }

    if (num_threads <= 1){
        Add_Worker_Args add_worker_args;
        add_worker_args.A = A;
        add_worker_args.B = B;
        add_worker_args.C = C;
        add_worker_args.start_ind = 0;
        add_worker_args.num_els = num_els;
        add_worker_args.alpha = alpha;
        add_worker_args.beta = beta;

        add_func(&add_worker_args);
    }


    // Ensure NUMA affinity...

    pthread_attr_t attr;
    int ret_attr_init = pthread_attr_init(&attr);
    if (ret_attr_init != 0) {
        perror("pthread_attr_init failed");
        // Decide if this is a fatal error or if you want to proceed without affinity
        // For now, let's proceed but affinity won't be set.
    }

    cpu_set_t *cpu_mask_for_node = NULL;
    int affinity_set_successfully = 0;
    int max_cpus = 0; // Will store the number of possible CPUs

    if (numa_available()) {
 
        int current_cpu = sched_getcpu();
        if (current_cpu < 0) {
            perror("Warning: sched_getcpu failed");
        } else {
            int target_numa_node = numa_node_of_cpu(current_cpu);
            if (target_numa_node < 0) {
                perror("Warning: numa_node_of_cpu failed");
            } else {
                printf("Calling thread is on CPU %d, NUMA node %d. Attempting to pin worker threads to this node.\n", current_cpu, target_numa_node);
                
                max_cpus = numa_num_possible_cpus();
                if (max_cpus <= 0) {
                     fprintf(stderr, "Warning: numa_num_possible_cpus() returned invalid value %d\n", max_cpus);
                } else {
                    cpu_mask_for_node = CPU_ALLOC(max_cpus);
                    if (cpu_mask_for_node == NULL) {
                        perror("Warning: CPU_ALLOC failed");
                    } else {
                        size_t cpuset_size = CPU_ALLOC_SIZE(max_cpus);
                        CPU_ZERO_S(cpuset_size, cpu_mask_for_node);

                        struct bitmask *numa_cpumask = numa_allocate_cpumask();
                        if (numa_cpumask == NULL) {
                            perror("Warning: numa_allocate_cpumask failed");
                            CPU_FREE(cpu_mask_for_node);
                            cpu_mask_for_node = NULL;
                        } else {
                            if (numa_node_to_cpus(target_numa_node, numa_cpumask) < 0) {
                                perror("Warning: numa_node_to_cpus failed");
                            } else {
                                for (int k = 0; k < max_cpus; ++k) {
                                    if (numa_bitmask_isbitset(numa_cpumask, k)) {
                                        CPU_SET_S(k, cpuset_size, cpu_mask_for_node);
                                    }
                                }
                                affinity_set_successfully = 1; // Mark that we have a valid mask
                            }
                            numa_free_cpumask(numa_cpumask);
                        }
                    }
                }
            }
        }
    }

    if (affinity_set_successfully && ret_attr_init == 0 && cpu_mask_for_node != NULL) {
        size_t cpuset_size = CPU_ALLOC_SIZE(max_cpus);
        int ret_set_affinity = pthread_attr_setaffinity_np(&attr, cpuset_size, cpu_mask_for_node);
        if (ret_set_affinity != 0) {
            perror("Warning: pthread_attr_setaffinity_np failed. Threads will use default affinity.");
            affinity_set_successfully = 0; // Revert status
        } else {
            printf("Successfully prepared to set worker thread affinity to NUMA node.\n");
        }
    } else {
        affinity_set_successfully = 0; // Ensure it's clear affinity isn't being applied
    }

    pthread_t * add_workers = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    if (!add_workers){
        fprintf(stderr, "Error: failed to allocate add workers...\n");
        if (ret_attr_init == 0) {
            pthread_attr_destroy(&attr);
        }
        if (cpu_mask_for_node) {
            CPU_FREE(cpu_mask_for_node);
        }
        return -1;
    }

    Add_Worker_Args * add_worker_args_array = (Add_Worker_Args *) malloc(num_threads * sizeof(Add_Worker_Args));
    if (!add_worker_args_array){
        fprintf(stderr, "Error: failed to allocate add worker args...\n");
        free(add_workers);
        if (ret_attr_init == 0) {
            pthread_attr_destroy(&attr);
        }
        if (cpu_mask_for_node) {
            CPU_FREE(cpu_mask_for_node);
        }
        return -1;
    }

    // Same arguments for all workers
    for (int i = 0; i < num_threads; i++){
        add_worker_args_array[i].A = A;
        add_worker_args_array[i].B = B;
        add_worker_args_array[i].C = C;
        add_worker_args_array[i].alpha = alpha;
        add_worker_args_array[i].beta = beta;
    }
    
    // Give slices to each worker...

    uint64_t base_chunk_size = num_els / num_threads;
    uint64_t remainder_elements = num_els % num_threads;
    uint64_t current_start_index = 0;
    uint64_t elements_for_this_thread;

    for (int i = 0; i < num_threads; i++){
        
        add_worker_args_array[i].start_ind = current_start_index;
        elements_for_this_thread = base_chunk_size + (i < remainder_elements ? 1 : 0);
        add_worker_args_array[i].num_els = elements_for_this_thread;
        current_start_index += elements_for_this_thread;

        // Start worker
        int ret_create;
        // Only use attr if successfully initialized and affinity mask is ready
        if (ret_attr_init == 0 && affinity_set_successfully) { 
            ret_create = pthread_create(&add_workers[i], &attr, add_func, &add_worker_args_array[i]);
        } else {
            ret_create = pthread_create(&add_workers[i], NULL, add_func, &add_worker_args_array[i]);
        }
        if (ret_create != 0) {
            fprintf(stderr, "[Add] Error creating thread %d: %s\n", i, strerror(ret_create));
            // Handle thread creation error: cleanup already created threads, allocated memory etc.
            // For simplicity here, we'll just print and continue, but in robust code, you'd manage this.
            // For now, just free and exit might be simpler than partial cleanup.
            for(int k=0; k<i; ++k) {
                pthread_cancel(add_workers[k]);
            }
            for(int k=0; k<i; ++k) {
                pthread_join(add_workers[k], NULL);
            }
            free(add_workers);
            free(add_worker_args_array);
            if (ret_attr_init == 0) {
                pthread_attr_destroy(&attr);
            }
            if (cpu_mask_for_node) {
                CPU_FREE(cpu_mask_for_node);
            }
            return -1;
        }
    }
    
    // Clean up pthread attributes if they were initialized
    if (ret_attr_init == 0) {
        pthread_attr_destroy(&attr);
    }
    // Free the cpu_set_t if it was allocated
    if (cpu_mask_for_node) {
        CPU_FREE(cpu_mask_for_node);
    }

    // Wait for all workers to finish
    for (int i = 0; i < num_threads; i++){
        pthread_join(add_workers[i], NULL);
    }

    // Free allocated memory
    free(add_workers);
    free(add_worker_args_array);
    

    return 0;
}