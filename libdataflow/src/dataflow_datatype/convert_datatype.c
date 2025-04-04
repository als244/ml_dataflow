#include "datatype_conversion.h"


static void *thread_func_fp32_to_fp16(void *arg) {
    
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    float * src = (float *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    for (i = start; i < end; i++) {
        dst[i] = solo_fp32_to_fp16(src[i]);
    }

    return NULL;
}

static void *thread_func_fp32_to_bf16(void *arg) {
    
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    float * src = (float *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    for (i = start; i < end; i++) {
        dst[i] = solo_fp32_to_bf16(src[i]);
    }

    return NULL;
}


static void *thread_func_fp16_to_fp32(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    float * dst = (float *) targ -> dst;

    for (i = start; i < end; i++) {
        dst[i] = solo_fp16_to_fp32(src[i]);
    }

    return NULL;
}


static void *thread_func_fp16_to_bf16(void *arg) {
    
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    for (i = start; i < end; i++) {
        dst[i] = solo_fp16_to_bf16(src[i]);
    }

    return NULL;
}


static void *thread_func_bf16_to_fp32(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    float * dst = (float *) targ -> dst;

    for (i = start; i < end; i++) {
        dst[i] = solo_bf16_to_fp32(src[i]);
    }

    return NULL;
}


static void *thread_func_bf16_to_fp16(void *arg) {
    thread_conv_args *targ = (thread_conv_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;

    uint16_t * src = (uint16_t *) targ -> src;
    uint16_t * dst = (uint16_t *) targ -> dst;

    for (i = start; i < end; i++) {
        dst[i] = solo_bf16_to_fp16(src[i]);
    }

    return NULL;
}

DataflowConversionType get_conversion_type(DataflowDatatype to_dt, DataflowDatatype from_dt){

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


int convert_datatype(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads) {

    if (to_dt == from_dt){
        size_t el_size = dataflow_sizeof_element(to_dt);
        memcpy(to, from, el_size * n);
        return 0;
    }

    DataflowConversionType conversion_type = get_conversion_type(to_dt, from_dt);

	void * (*thread_conv_func)(void *);

	switch (conversion_type){

		case (DATAFLOW_CONVERT_FP32_TO_FP16):
			thread_conv_func = &thread_func_fp32_to_fp16;
			break;
		case (DATAFLOW_CONVERT_FP32_TO_BF16):
			thread_conv_func = &thread_func_fp32_to_bf16;
			break;
		case (DATAFLOW_CONVERT_FP16_TO_FP32):
			thread_conv_func = &thread_func_fp16_to_fp32;
			break;
		case (DATAFLOW_CONVERT_FP16_TO_BF16):
			thread_conv_func = &thread_func_fp16_to_bf16;
			break;
		case (DATAFLOW_CONVERT_BF16_TO_FP32):
			thread_conv_func = &thread_func_bf16_to_fp32;
			break;
		case (DATAFLOW_CONVERT_BF16_TO_FP16):
			thread_conv_func = &thread_func_bf16_to_fp16;
			break;
		default:
			thread_conv_func = NULL;
			break;
	}

	if (!thread_conv_func){
		fprintf(stderr, "Error: Conversion from dtype %s to type %s unavailable\n", dataflow_datatype_as_string(from_dt), dataflow_datatype_as_string(to_dt));
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