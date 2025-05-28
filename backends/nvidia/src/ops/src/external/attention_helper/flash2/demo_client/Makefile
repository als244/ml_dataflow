
CC = gcc
C_FLAGS = -g

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs
CUDA_LIB = -lcuda

FLASH2_INCLUDE_DIR = ../include

PWD = ${CURDIR}

FLASH2_LIB_DIR = ${PWD}/../lib
FLASH2_LIB = -lflash2

ALL_INCLUDES = -I${FLASH2_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}
ALL_LIBS = -L${FLASH2_LIB_DIR} ${FLASH2_LIB} -L${CUDA_LIB_DIR} ${CUDA_LIB} -Wl,-rpath,${FLASH2_LIB_DIR}

all: test_libflash2

test_libflash2: test_libflash2.c
	${CC} ${C_FLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

clean:
	rm -f test_libflash2

