CC = gcc
CFLAGS = -g -fPIC

DATAFLOW_MODELS_INCLUDE_DIR = ../../include
DATAFLOW_LIB_INCLUDE_DIR = ../../../libdataflow/include
DATAFLOW_OP_INCLUDE_DIR = ../../../libdataflowops/include

ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${DATAFLOW_OP_INCLUDE_DIR} -I${DATAFLOW_MODELS_INCLUDE_DIR}

TRANSFORMER_OBJS = objs/dataflow_transformer.o objs/prep_seq_batch.o objs/prep_transformer.o

all: create_dirs ${TRANSFORMER_OBJS}

create_dirs:
	mkdir -p objs

objs/dataflow_transformer.o: src/dataflow_transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c -o $@ $<

objs/prep_seq_batch.o: src/prep_seq_batch.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c -o $@ $<

objs/prep_transformer.o: src/prep_transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c -o $@ $<

clean:
	rm -f objs/dataflow_transformer.o objs/prep_transformer.o objs/prep_seq_batch.o
