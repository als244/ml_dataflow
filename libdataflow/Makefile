CC = gcc
CFLAGS = -fPIC -g -std=c99

SELF_INCLUDE_DIR = include
INTERNAL_INCLUDE = src/include

ALL_INCLUDES = -I${SELF_INCLUDE_DIR} -I$(INTERNAL_INCLUDE)




AVX512_INTRIN_FLAGS = -mavx512f -mavx512fp16 -mavx512bf16


## All objects

TABLE_OBJS = objs/dataflow_table.o
FINGERPRINT_OBJS = objs/dataflow_fingerprint.o
DATATYPE_OBJS = objs/dataflow_datatype.o objs/solo_conversion.o objs/convert_datatype.o objs/convert_datatype_avx512.o
MISC_OBJS = objs/create_host_matrix.o

ALL_OBJS = ${TABLE_OBJS} ${FINGERPRINT_OBJS} ${DATATYPE_OBJS} ${MISC_OBJS}

OTHER_LIB_DEPENDS = -pthread -lm -lcrypto

LIB_NAME = libdataflow.so


lib/${LIB_NAME}: ${ALL_OBJS}
	${CC} ${CFLAGS} ${ALL_INCLUDES} -shared -o $@ $^ ${OTHER_LIB_DEPENDS}


## Table Objects

objs/dataflow_table.o: src/dataflow_table/dataflow_table.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@


## Fingerprint Objects
objs/dataflow_fingerprint.o: src/dataflow_fingerprint/dataflow_fingerprint.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@


## DataType Objects

objs/dataflow_datatype.o: src/dataflow_datatype/dataflow_datatype.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

objs/solo_conversion.o: src/dataflow_datatype/solo_conversion.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

objs/convert_datatype.o: src/dataflow_datatype/convert_datatype.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@

objs/convert_datatype_avx512.o: src/dataflow_datatype/convert_datatype_avx512.c
	${CC} ${CFLAGS} ${AVX512_INTRIN_FLAGS} ${ALL_INCLUDES} -c $< -o $@



## Misc Objects

objs/create_host_matrix.o: src/misc/create_host_matrix.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} -c $< -o $@



## Cleaning

clean:
	rm -f ${ALL_OBJS} lib/${LIB_NAME}