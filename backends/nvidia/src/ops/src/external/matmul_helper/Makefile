CC = gcc
CFLAGS = -g -fPIC -shared


PROJ_COMMON_HEADER = ../../../../../include
DATAFLOW_HANDLE_HEADER = ../../../../dataflow_handle/include
CUDA_DATAFLOW_HANDLE_HEADER = ../../../../dataflow_handle/nvidia/include
DATATYPE_HEADER = ../../../../datatype/include
TABLE_HEADER = ../../../../table/include
FINGERPRINT_HEADER = ../../../../fingerprint/include
OPS_HEADER = ../../../include
MATMUL_HELPER_HEADER = include

MATMUL_HELPER_LIB_DIR = ./lib
MATMUL_HELPER_LIB = ${MATMUL_HELPER_LIB_DIR}/libmatmulwrapper.so

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64
CUDA_LIB_DEPENDS = -lcuda -lcublasLt


MATMUL_HELPER_SRC = src/matmul_helper.c
MATMUL_HELPER_OBJ_DEPENDS = ../../../../datatype/objs/dataflow_datatype.o ../../../../datatype/objs/solo_conversion.o

all: ${MATMUL_HELPER_LIB}

${MATMUL_HELPER_LIB}: ${MATMUL_HELPER_SRC} ${MATMUL_HELPER_OBJ_DEPENDS}
	${CC} ${CFLAGS} -I${MATMUL_HELPER_HEADER} -I${CUDA_DATAFLOW_HANDLE_HEADER} -I${OPS_HEADER} -I${DATAFLOW_HANDLE_HEADER} -I${TABLE_HEADER} -I${FINGERPRINT_HEADER} -I${DATATYPE_HEADER} -I${PROJ_COMMON_HEADER} -I${CUDA_INCLUDE_DIR} $^ -o $@ -L${CUDA_LIB_DIR} ${CUDA_LIB_DEPENDS} -Wl,-rpath,${CUDA_LIB_DIR}

clean:
	rm -f ${MATMUL_HELPER_LIB}
