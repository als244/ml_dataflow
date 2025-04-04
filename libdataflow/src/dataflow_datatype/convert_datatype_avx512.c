#include "datatype_conversion.h"


// BF16 load/store wrappers (if not provided by your toolchain).
#ifndef _mm256_loadu_bf16
static inline __m256bh my_mm256_loadu_bf16(const void *addr) {
    return (__m256bh)_mm256_loadu_si256((const __m256i *)addr);
}
#define _mm256_loadu_bf16(addr) my_mm256_loadu_bf16(addr)
#endif

#ifndef _mm256_storeu_bf16
static inline void my_mm256_storeu_bf16(void *addr, __m256bh a) {
    _mm256_storeu_si256((__m256i *)addr, (__m256i)a);
}
#define _mm256_storeu_bf16(addr, a) my_mm256_storeu_bf16(addr, a)
#endif



static void *thread_func_fp32_to_fp16_avx512(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start;
    size_t end = targ->end;
    
    float * src = (float *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    size_t i;
    const size_t vec_size = 16;

    for (i = start; i + vec_size <= end; i += vec_size) {
        __m512 fp32_vec = _mm512_loadu_ps(src + i);
        __m256i half_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), half_vec);
    }

    for (; i < end; i++) {
        dst[i] = solo_fp32_to_fp16(src[i]);
    }

    return NULL;
}

static void *thread_func_fp32_to_bf16_avx512(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start;
    size_t end = targ->end;

    float * src = (float *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    size_t i;
    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        __m512 fp32_vec = _mm512_loadu_ps(src + i);
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        _mm256_storeu_bf16(dst + i, bf16_vec);
    }
    for (; i < end; i++) {
        dst[i] = solo_fp32_to_bf16(src[i]);
    }
    return NULL;
}


static void *thread_func_fp16_to_fp32_avx512(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start;
    size_t end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    float * dst = (float *) targ -> dst;

    const size_t vec_size = 16;  // Process 16 elements at a time.
    
    // Vectorized loop.
    for (i = start; i + vec_size <= end; i += vec_size) {
        __m256i half_vec = _mm256_loadu_si256((const __m256i*)(src + i));
        __m512 fp32_vec = _mm512_cvtph_ps(half_vec);
        _mm512_storeu_ps(dst + i, fp32_vec);
    }
    // Scalar fallback.
    for (; i < end; i++) {
        dst[i] = solo_fp16_to_fp32(src[i]);
    }
    return NULL;
}


static void *thread_func_fp16_to_bf16_avx512(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start;
    size_t end = targ->end;

    uint16_t * src = (uint16_t *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    size_t i;
    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        // Load 16 FP16 values.
        __m256i half_vec = _mm256_loadu_si256((const __m256i*)(src + i));
        // Convert FP16 to FP32.
        __m512 fp32_vec = _mm512_cvtph_ps(half_vec);
        // Convert FP32 to BF16.
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        // Store 16 BF16 values.
        _mm256_storeu_bf16(dst + i, bf16_vec);
    }
    for (; i < end; i++) {
        dst[i] = solo_fp16_to_bf16(src[i]);
    }
    return NULL;
}


static void *thread_func_bf16_to_fp32_avx512(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start;
    size_t end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    float * dst = (float *) targ -> dst;

    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        __m256bh bf16_vec = _mm256_loadu_bf16(src + i);
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);
        _mm512_storeu_ps(dst + i, fp32_vec);
    }
    for (; i < end; i++) {
        dst[i] = solo_bf16_to_fp32(src[i]);
    }
    return NULL;
}


static void *thread_func_bf16_to_fp16_avx512(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        // Load 16 BF16 values.
        __m256bh bf16_vec = _mm256_loadu_bf16(src + i);
        // Convert BF16 to FP32.
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);
        // Convert FP32 to FP16.
        __m256i half_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), half_vec);
    }
    for (; i < end; i++) {
        dst[i] = solo_bf16_to_fp16(src[i]);
    }
    return NULL;
}


DataflowConversionType get_conversion_type_avx512(DataflowDatatype to_dt, DataflowDatatype from_dt){

    switch (from_dt){
        case DATAFLOW_FP32:
            switch (to_dt){
                case DATAFLOW_FP32:
                    return DATAFLOW_CONVERT_SAME;
                case DATAFLOW_FP16:
                    return DATAFLOW_CONVERT_FP32_TO_FP16;
                case DATAFLOW_BF16:
                    return DATAFLOW_CONVERT_FP32_TO_BF16;
                default:
                    return DATAFLOW_CONVERT_NOT_AVAILABLE;
            }
            break;
        case DATAFLOW_FP16:
            switch (to_dt){
                case DATAFLOW_FP32:
                    return DATAFLOW_CONVERT_FP16_TO_FP32;
                case DATAFLOW_FP16:
                    return DATAFLOW_CONVERT_SAME;
                case DATAFLOW_BF16:
                    return DATAFLOW_CONVERT_FP16_TO_BF16;
                default:
                    return DATAFLOW_CONVERT_NOT_AVAILABLE;
            }
            break;
        case DATAFLOW_BF16:
            switch (to_dt){
                case DATAFLOW_FP32:
                    return DATAFLOW_CONVERT_BF16_TO_FP32;
                case DATAFLOW_FP16:
                    return DATAFLOW_CONVERT_BF16_TO_FP16;
                case DATAFLOW_BF16:
                    return DATAFLOW_CONVERT_SAME;
                default:
                    return DATAFLOW_CONVERT_NOT_AVAILABLE;
            }
            break;
        default:
            return DATAFLOW_CONVERT_NOT_AVAILABLE;
    }

    // not getting here
    return DATAFLOW_CONVERT_NOT_AVAILABLE;
}



int convert_datatype_avx512(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads) {

    if (to_dt == from_dt){
        size_t el_size = dataflow_sizeof_element(to_dt);
        memcpy(to, from, el_size * n);
        return 0;
    }

    DataflowConversionType conversion_type = get_conversion_type_avx512(to_dt, from_dt);

	void * (*thread_conv_func)(void *);

	switch (conversion_type){

		case (DATAFLOW_CONVERT_FP32_TO_FP16):
			thread_conv_func = &thread_func_fp32_to_fp16_avx512;
			break;
		case (DATAFLOW_CONVERT_FP32_TO_BF16):
			thread_conv_func = &thread_func_fp32_to_bf16_avx512;
			break;
		case (DATAFLOW_CONVERT_FP16_TO_FP32):
			thread_conv_func = &thread_func_fp16_to_fp32_avx512;
			break;
		case (DATAFLOW_CONVERT_FP16_TO_BF16):
			thread_conv_func = &thread_func_fp16_to_bf16_avx512;
			break;
		case (DATAFLOW_CONVERT_BF16_TO_FP32):
			thread_conv_func = &thread_func_bf16_to_fp32_avx512;
			break;
		case (DATAFLOW_CONVERT_BF16_TO_FP16):
			thread_conv_func = &thread_func_bf16_to_fp16_avx512;
			break;
		default:
			thread_conv_func = NULL;
			break;
	}

	if (!thread_conv_func){
        fprintf(stderr, "Error: Avx conversion from dtype %s to type %s unavailable\n", dataflow_datatype_as_string(from_dt), dataflow_datatype_as_string(to_dt));
        return -1;
    }


	if (num_threads <= 1) {
        thread_conv_args args = {from, to, 0, n};
        thread_conv_func(&args);
        return 0;
    }


    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_conv_args *targs = malloc(num_threads * sizeof(thread_conv_args));
    if (!threads || !targs) {
    	fprintf(stderr, "Error: Could not alloc space for threads or args...\n");
    	return -1;
    }
    


    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = from;
        targs[t].dst = to;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_func, &targs[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(targs);

    return 0;

}