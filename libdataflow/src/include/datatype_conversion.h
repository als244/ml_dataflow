#ifndef DATATYPE_CONVERSION_H
#define DATATYPE_CONVERSION_H

#include "dataflow_common.h"

typedef enum {
    DATAFLOW_CONVERT_FP32_TO_FP16,
    DATAFLOW_CONVERT_FP32_TO_BF16,
    DATAFLOW_CONVERT_FP16_TO_FP32,
    DATAFLOW_CONVERT_FP16_TO_BF16,
    DATAFLOW_CONVERT_BF16_TO_FP32,
    DATAFLOW_CONVERT_BF16_TO_FP16,
    DATAFLOW_CONVERT_SAME,
    DATAFLOW_CONVERT_NOT_AVAILABLE
} DataflowConversionType;


typedef struct {
    void *src;
    void *dst;
    size_t start;
    size_t end;
} thread_conv_args;


int convert_datatype(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads);

int convert_datatype_avx512(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads);

extern size_t dataflow_sizeof_element(DataflowDatatype arr_dtype);

extern char * dataflow_datatype_as_string(DataflowDatatype dtype);

#endif