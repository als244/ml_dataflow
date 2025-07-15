CC = gcc
CFLAGS = -g -fPIC

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs
CUDA_LIB_DEPENDS = -L${CUDA_LIB_DIR} -lcuda -lcublasLt -lnvidia-ml

DATAFLOW_INCLUDE_DIR = ../../../../libdataflow/include
DATAFLOW_LIB_DIR = ${CURDIR}/../../../../libdataflow/lib
DATAFLOW_LIB_DEPENDS = -ldataflow
SELF_INCLUDE_DIR = ../../include

ALL_INCLUDES = -I${SELF_INCLUDE_DIR} -I${DATAFLOW_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}

ALL_OBJS = objs/cuda_drv.o objs/cuda_nvml.o objs/cuda_profiler.o objs/cuda_dataflow_handle.o

OTHER_LIB_DEPENDS = -pthread -ldl ${CUDA_LIB_DEPENDS} -L${DATAFLOW_LIB_DIR} -ldataflow -Wl,-rpath,${DATAFLOW_LIB_DIR} -Wl,-rpath,${CUDA_LIB_DIR} -Wl,--enable-new-dtags

LIB_NAME = libcuda_dataflow_handle.so

../../lib/${LIB_NAME}: ${ALL_OBJS}
	${CC} ${CFLAGS} ${ALL_INCLUDES} -shared -o $@ $^ ${OTHER_LIB_DEPENDS}


## Interface to Cuda Driver

objs/cuda_drv.o: src/cuda_drv.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

## Interface to NVML

objs/cuda_nvml.o: src/cuda_nvml.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

## Cuda Profiler
objs/cuda_profiler.o: src/cuda_profiler.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@


## Cuda Dataflow Handle Implementation
objs/cuda_dataflow_handle.o: src/cuda_dataflow_handle.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@


## Cleaning

clean:
	rm -f ${ALL_OBJS} ../../lib/${LIB_NAME}



