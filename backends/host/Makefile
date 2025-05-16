CC = gcc
DEV_CFLAGS = -g -fPIC
BUILD_CFLAGS = -O3 -fPIC

CFLAGS = ${DEV_CFLAGS}

DATAFLOW_INCLUDE_DIR = ${CURDIR}/../../libdataflow/include

HOST_OPS_INCLUDE_DIR = ${CURDIR}/include

ALL_INCLUDES = -I ${DATAFLOW_INCLUDE_DIR} -I ${HOST_OPS_INCLUDE_DIR}

ALL_LIB_DEPENDS = -pthread -lm -lnuma

HOST_OPS_SRC_DIR = src/ops

HOST_OPS_OBJ_DIR = ${HOST_OPS_SRC_DIR}/objs

OPTIMIZER_SRC_DIR = ${HOST_OPS_SRC_DIR}/optimizer

HOST_OPS_OBJS = ${HOST_OPS_OBJ_DIR}/adam_dispatch.o ${HOST_OPS_OBJ_DIR}/adam_step.o ${HOST_OPS_OBJ_DIR}/adam_step_avx512.o

HOST_OPS_LIB = lib/libdataflow_hostops.so

AVX512_FLAGS = -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512bf16 -mavx512fp16

all: create_dirs ${HOST_OPS_LIB}

## ensure folder for build objs
create_dirs:
	mkdir -p ${HOST_OPS_OBJ_DIR}

## library for training programs to link to
${HOST_OPS_LIB}: ${HOST_OPS_OBJS}
	${CC} ${CFLAGS} -shared ${ALL_INCLUDES} $^ -o $@ ${ALL_LIB_DEPENDS}


## optimizer ops

## API that takes in 'op' struct as argument
${HOST_OPS_OBJ_DIR}/adam_dispatch.o: ${OPTIMIZER_SRC_DIR}/adam_dispatch.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

## fallback implementations
${HOST_OPS_OBJ_DIR}/adam_step.o: ${OPTIMIZER_SRC_DIR}/adam_step.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

## main avx512 implementations
${HOST_OPS_OBJ_DIR}/adam_step_avx512.o: ${OPTIMIZER_SRC_DIR}/adam_step_avx512.c
	${CC} ${CFLAGS} ${AVX512_FLAGS} ${ALL_INCLUDES} -c $< -o $@

clean:
	rm -rf ${HOST_OPS_OBJ_DIR}/*.o ${HOST_OPS_LIB}