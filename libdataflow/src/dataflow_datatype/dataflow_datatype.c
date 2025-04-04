#include "dataflow_common.h"
#include "datatype_conversion.h"

size_t dataflow_sizeof_element(DataflowDatatype arr_dtype){

    switch(arr_dtype){

        case DATAFLOW_VOID:
            return 0;
        case DATAFLOW_FP64:
            return 8;
        case DATAFLOW_FP32:
            return 4;
        case DATAFLOW_FP16:
            return 2;
        case DATAFLOW_BF16:
            return 2;
        case DATAFLOW_FP8E4M3:
            return 1;
        case DATAFLOW_FP8E5M2:
            return 1;
        case DATAFLOW_UINT64:
            return 8;
        case DATAFLOW_UINT32:
            return 4;
        case DATAFLOW_UINT16:
            return 2;
        case DATAFLOW_UINT8:
            return 1;
        case DATAFLOW_LONG:
            return sizeof(long);
        case DATAFLOW_INT:
            return sizeof(int);
        default:
            return 0;
    }

    // not getting here
    return 0;
}


char * dataflow_datatype_as_string(DataflowDatatype dtype) {

    switch(dtype){
        case DATAFLOW_VOID:
            return "VOID";
        case DATAFLOW_FP64:
            return "FP64";
        case DATAFLOW_FP32:
            return "FP32";
        case DATAFLOW_FP16:
            return "FP16";
        case DATAFLOW_BF16:
            return "BF16";
        case DATAFLOW_FP8E4M3:
            return "FP8E4M3";
        case DATAFLOW_FP8E5M2:
            return "FP8E5M2";
        case DATAFLOW_UINT64:
            return "UINT64";
        case DATAFLOW_UINT32:
            return "UINT32";
        case DATAFLOW_UINT16:
            return "UINT16";
        case DATAFLOW_UINT8:
            return "UINT8";
        case DATAFLOW_LONG:
            return "LONG";
        case DATAFLOW_INT:
            return "INT";
        default:
            return "UNKNOWN";
    }

    // not getting here
    return "UNKNOWN";
}


int dataflow_convert_datatype(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads) {

	void * (*thread_conv_func)(void *);

    if (__builtin_cpu_supports("avx512f")){
        return convert_datatype_avx512(to, from, to_dt, from_dt, n, num_threads);
    }

    return convert_datatype(to, from, to_dt, from_dt, n, num_threads);
}