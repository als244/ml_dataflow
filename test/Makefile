CC = gcc
CFLAGS = -g

CUDA_INCLUDE_DIR = /usr/local/cuda/include

DATAFLOW_LIB_INCLUDE_DIR = ../libdataflow/include
DATAFLOW_LIB_DIR = ${CURDIR}/../libdataflow/lib
DATAFLOW_LIB = -ldataflow

CUDA_DATAFLOW_BACKEND_INCLUDE_DIR = ../backends/nvidia/include
CUDA_DATAFLOW_BACKEND_LIB_DIR = ${CURDIR}/../backends/nvidia/lib
CUDA_DATAFLOW_BACKEND_LIBS = -lcuda_dataflow_handle -lcuda_register_default_ops

DATAFLOW_OPS_INCLUDE_DIR = ../libdataflowops/include
DATAFLOW_OPS_LIB_DIR = ${CURDIR}/../libdataflowops/lib
DATAFLOW_OPS_LIB = -ldataflowops

ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${CUDA_DATAFLOW_BACKEND_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR} -I${DATAFLOW_OPS_INCLUDE_DIR}

ALL_LIBS = -L${DATAFLOW_LIB_DIR} -L${CUDA_DATAFLOW_BACKEND_LIB_DIR} -L${DATAFLOW_OPS_LIB_DIR} ${DATAFLOW_LIB} ${CUDA_DATAFLOW_BACKEND_LIBS} ${DATAFLOW_OPS_LIB} -Wl,-rpath,${DATAFLOW_LIB_DIR} -Wl,-rpath,${CUDA_DATAFLOW_BACKEND_LIB_DIR} -Wl,-rpath,${DATAFLOW_OPS_LIB_DIR}

EXECS = test_cuda_dataflow_handle_transformer_head test_cuda_dataflow_handle_transformer_block

all: ${EXECS}

test_cuda_dataflow_handle_transformer_head: test_cuda_dataflow_handle_transformer_head.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

test_cuda_dataflow_handle_transformer_block: test_cuda_dataflow_handle_transformer_block.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

clean:
	rm -f ${EXECS}


