CC = gcc
CFLAGS = -g -fPIC

DATAFLOW_MODELS_INCLUDE_DIR = ../include

DATAFLOW_LIB_INCLUDE_DIR = ../libdataflow/include
DATAFLOW_LIB_DIR = ${CURDIR}/../libdataflow/lib
DATAFLOW_LIB = -ldataflow

DATAFLOW_OP_INCLUDE_DIR = ../libdataflowops/include
DATAFLOW_OP_LIB_DIR = ${CURDIR}/../libdataflowops/lib
DATAFLOW_OP_LIB = -ldataflowops

ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${DATAFLOW_OP_INCLUDE_DIR} -I${DATAFLOW_MODELS_INCLUDE_DIR}

ALL_LIBS =  -lm -ldl -pthread -L${DATAFLOW_LIB_DIR} -L${DATAFLOW_OP_LIB_DIR} ${DATAFLOW_LIB} ${DATAFLOW_OP_LIB} -Wl,-rpath,${DATAFLOW_LIB_DIR} -Wl,-rpath,${DATAFLOW_OP_LIB_DIR}

TRANSFORMER_OBJS = src/transformer/objs/dataflow_transformer.o src/transformer/objs/prep_seq_batch.o src/transformer/objs/prep_transformer.o

DATAFLOW_MODELS_OBJS = ${TRANSFORMER_OBJS}

DATAFLOW_MODELS_LIB = libdataflowmodels.so

all: create_dirs lib/${DATAFLOW_MODELS_LIB}

create_dirs:
	mkdir -p lib

lib/${DATAFLOW_MODELS_LIB}: make_transformer_objs ${DATAFLOW_MODELS_OBJS}
	${CC} ${CFLAGS} -shared ${ALL_INCLUDES} ${DATAFLOW_MODELS_OBJS} -o $@ ${ALL_LIBS}

make_transformer_objs:
	${MAKE} -C src/transformer

clean:
	${MAKE} -C src/transformer clean && rm -f lib/${DATAFLOW_MODELS_LIB}

	