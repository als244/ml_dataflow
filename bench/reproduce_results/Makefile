CC = gcc
CFLAGS = -g

CUDA_INCLUDE_DIR = /usr/local/cuda/include

CUDA_DATAFLOW_BACKEND_INCLUDE_DIR = ../../backends/nvidia/include
CUDA_DATAFLOW_BACKEND_LIB_DIR = ${CURDIR}/../../backends/nvidia/lib
CUDA_DATAFLOW_BACKEND_LIBS = -lcuda_dataflow_handle -lcuda_register_default_ops


DATAFLOW_LIB_INCLUDE_DIR = ../../libdataflow/include
DATAFLOW_LIB_DIR = ${CURDIR}/../../libdataflow/lib
DATAFLOW_LIB = -ldataflow

DATAFLOW_OPS_INCLUDE_DIR = ../../libdataflowops/include
DATAFLOW_OPS_LIB_DIR = ${CURDIR}/../../libdataflowops/lib
DATAFLOW_OPS_LIB = -ldataflowops


DATAFLOW_MODELS_INCLUDE_DIR = ../../libdataflowmodels/include
DATAFLOW_MODELS_LIB_DIR = ${CURDIR}/../../libdataflowmodels/lib
DATAFLOW_MODELS_LIB = -ldataflowmodels


HOST_OPS_INCLUDE_DIR = ../../backends/host/include
HOST_OPS_LIB_DIR = ${CURDIR}/../../backends/host/lib
HOST_OPS_LIB = -ldataflow_hostops

ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${CUDA_DATAFLOW_BACKEND_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR} -I${DATAFLOW_OPS_INCLUDE_DIR} -I${DATAFLOW_MODELS_INCLUDE_DIR} -I${HOST_OPS_INCLUDE_DIR}
ALL_LIBS = -lm -pthread -ldl -L${DATAFLOW_LIB_DIR} -L${CUDA_DATAFLOW_BACKEND_LIB_DIR} -L${DATAFLOW_OPS_LIB_DIR} -L${DATAFLOW_MODELS_LIB_DIR} -L${HOST_OPS_LIB_DIR} ${DATAFLOW_LIB} ${CUDA_DATAFLOW_BACKEND_LIBS} ${DATAFLOW_OPS_LIB} ${DATAFLOW_MODELS_LIB} ${HOST_OPS_LIB} -Wl,-rpath,${DATAFLOW_LIB_DIR} -Wl,-rpath,${CUDA_DATAFLOW_BACKEND_LIB_DIR} -Wl,-rpath,${DATAFLOW_OPS_LIB_DIR} -Wl,-rpath,${DATAFLOW_MODELS_LIB_DIR} -Wl,-rpath,${HOST_OPS_LIB_DIR}

EXECS = transformerRecordThroughput

all: ${EXECS}

## different CLI to take as input a csv to append to....
transformerRecordThroughput: transformer_record_throughput.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

clean:
	rm -f ${EXECS}

