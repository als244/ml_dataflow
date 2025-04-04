
CC = gcc
C_FLAGS = -g

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs
CUDA_LIB = -lcuda

FLASH3_INCLUDE_DIR = ../include

PWD = ${CURDIR}

FLASH3_LIB_DIR = ${PWD}/../lib
FLASH3_LIB = -lflash3

ALL_INCLUDES = -I${FLASH3_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}
ALL_LIBS = -L${FLASH3_LIB_DIR} ${FLASH3_LIB} -L${CUDA_LIB_DIR} ${CUDA_LIB} -Wl,-rpath,${FLASH3_LIB_DIR}

all: test_libflash3

test_libflash3: test_libflash3.c
	${CC} ${C_FLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

clean:
	rm -f test_libflash3

