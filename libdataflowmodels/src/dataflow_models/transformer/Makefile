CC = gcc
CFLAGS = -g -fPIC

DATAFLOW_LIB_INCLUDE_DIR = ../../../../libdataflow/include
DATAFLOW_LIB_DIR = ${CURDIR}/../../../../libdataflow/lib
DATAFLOW_LIB = -ldataflow

DATAFLOW_OP_INCLUDE_DIR = ../../../../libdataflowops/include
DATAFLOW_OP_LIB_DIR = ${CURDIR}/../../../../libdataflowops/lib
DATAFLOW_OP_LIB = -ldataflowops

CUDA_DATAFLOW_BACKEND_INCLUDE_DIR = ../../../../backends/nvidia/include
CUDA_DATAFLOW_BACKEND_LIB_DIR = ${CURDIR}/../../../../backends/nvidia/lib
CUDA_DATAFLOW_BACKEND_LIBS = -lcuda_dataflow_handle -lcuda_register_default_ops

DATAFLOW_MODELS_INCLUDE_DIR = ../../../include
ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${CUDA_DATAFLOW_BACKEND_INCLUDE_DIR} -I${DATAFLOW_OP_INCLUDE_DIR} -I${DATAFLOW_MODELS_INCLUDE_DIR}

ALL_LIBS =  -lm -ldl -pthread -L${DATAFLOW_LIB_DIR} -L${CUDA_DATAFLOW_BACKEND_LIB_DIR} -L${DATAFLOW_OP_LIB_DIR} ${DATAFLOW_LIB} ${CUDA_DATAFLOW_BACKEND_LIBS} ${DATAFLOW_OP_LIB} -Wl,-rpath,${DATAFLOW_LIB_DIR} -Wl,-rpath,${CUDA_DATAFLOW_BACKEND_LIB_DIR} -Wl,-rpath,${DATAFLOW_OP_LIB_DIR}

all: test/test_transformer

test/test_transformer: test/test_transformer.c objs/prep_transformer.o objs/dataflow_transformer.o
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

objs/dataflow_transformer.o: src/dataflow_transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c -o $@ $<

objs/prep_transformer.o: src/prep_transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c -o $@ $<

clean:
	rm -f test/test_transformer objs/dataflow_transformer.o objs/prep_transformer.o
