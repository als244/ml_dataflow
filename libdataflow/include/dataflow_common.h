#ifndef DATAFLOW_COMMON_H
#define DATAFLOW_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#include <assert.h>
#include <errno.h>

#include <semaphore.h>
#include <string.h>
#include <immintrin.h>

// path math
#include <linux/limits.h>

// loading shared library and using symbols
#include <dlfcn.h>

// dealwith with affinity/priority
#include <sched.h>

// dealing with numa
#include <numa.h>

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

typedef int (*Item_Cmp)(void * item, void * other_item);
typedef uint64_t (*Hash_Func)(void * item, uint64_t table_size);

#define MY_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MY_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MY_CEIL(a, b) ((a + b - 1) / b)

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

float solo_bf16_to_fp32(uint16_t a);

uint16_t solo_fp32_to_bf16(float f);

float solo_fp16_to_fp32(uint16_t h);

uint16_t solo_fp32_to_fp16(float f);

uint16_t solo_fp16_to_bf16(uint16_t h);

uint16_t solo_bf16_to_fp16(uint16_t a);


// sha256 encoding of each op skeleton
#define DATAFLOW_OP_IDENTIFIER_FINGERPRINT_TYPE 0
#define DATAFLOW_OP_IDENTIFIER_FINGERPRINT_NUM_BYTES 32


#endif
