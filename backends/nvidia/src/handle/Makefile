CC = gcc
CFLAGS = -g -fPIC

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64
CUDA_LIB_DEPENDS = -lcuda -lcublasLt

DATAFLOW_INCLUDE_DIR = ../../../../libdataflow/include
DATAFLOW_LIB_DIR = ${CURDIR}/../../../../libdataflow/lib
DATAFLOW_LIB_DEPENDS = -ldataflow
SELF_INCLUDE_DIR = ../../include

ALL_INCLUDES = -I${SELF_INCLUDE_DIR} -I${DATAFLOW_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}

ALL_OBJS = objs/cuda_drv.o objs/cuda_dataflow_handle.o

OTHER_LIB_DEPENDS = ${CUDA_LIB_DEPENDS} -L${DATAFLOW_LIB_DIR} -ldataflow -Wl,-rpath,${DATAFLOW_LIB_DIR} -Wl,-rpath,${CUDA_LIB_DIR} -Wl,--enable-new-dtags

LIB_NAME = libcuda_dataflow_handle.so

../../lib/${LIB_NAME}: ${ALL_OBJS}
	${CC} ${CFLAGS} ${ALL_INCLUDES} -shared -o $@ $^ ${OTHER_LIB_DEPENDS}


## Interface to Cuda Driver

objs/cuda_drv.o: src/cuda_drv.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@


## Cuda Dataflow Handle Implementation
objs/cuda_dataflow_handle.o: src/cuda_dataflow_handle.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@


## Cleaning

clean:
	rm -f ${ALL_OBJS} ../../lib/${LIB_NAME}



