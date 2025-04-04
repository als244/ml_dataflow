CC = gcc
CFLAGS = -g -std=c99


DATAFLOW_LIB = -ldataflow
DATAFLOW_LIB_DIR = ${CURDIR}/../lib

DATAFLOW_INCLUDE_DIR = ../include

ALL_INCLUDES = -I${DATAFLOW_INCLUDE_DIR}

ALL_LIBS = -L${DATAFLOW_LIB_DIR} ${DATAFLOW_LIB} -Wl,--enable-new-dtags -Wl,-rpath,${DATAFLOW_LIB_DIR}

test_convert: test_convert.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

clean:
	rm -f test_convert


