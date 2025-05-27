## C++ Compiler
CXX = g++
CXX_FLAGS = -std=c++17 -g -fPIC -O3 -Wall

## NVCC Compilation
## SYSTEM-DEPENDENT
## (may be in weird location if python installation of CUDA
## within some environment...)
NVCC = nvcc
NVCC_FLAGS = -Xcompiler=-fPIC -O3 -std=c++17 --ftemplate-backtrace-limit=0 --use_fast_math --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG

## Flags for shared library
WRAPPER_LIB_DEPENDS = -pthread -lm -ldl

## SYSTEM-DEPENDENT PATHS
CUDA_INCLUDE_DIR = /usr/local/cuda/include
## required for cudaGetDevice/cudaSetAttribute/cudaGetError, etc...
## all embedded within various files

## using -rpath to hardcode library path so clients
## do not need to be aware of this linkage
CUDA_RUNTIME_LIB_DIR = /usr/local/cuda/lib64
CUDA_RUNTIME_LIB = -lcudart




## Paths Internal to Source Code
FLASH2_INCLUDE_DIR = src
CUTLASS_CUTE_INCLUDE_DIR = ../include


ALL_INCLUDES = -I${FLASH2_INCLUDE_DIR} -I${CUTLASS_CUTE_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}

ALL_LIBS = ${WRAPPER_LIB_DEPENDS} -L${CUDA_RUNTIME_LIB_DIR} ${CUDA_RUNTIME_LIB} -Wl,--enable-new-dtags -Wl,-rpath,${CUDA_RUNTIME_LIB_DIR}



## for each ARCH VERSION DO THIS
CU_KERNEL_SOURCES = ${wildcard src/instantiations/*.cu}

## create same base filename in different directory
CU_KERNEL_OBJS_SM120 = ${patsubst src/instantiations/%.cu, build/instantiations/sm120/%.o, ${CU_KERNEL_SOURCES}}

BUILD_CU_KERNEL_OBJS = ${CU_KERNEL_OBJS_SM120}


FLASH_API_SRC = src/flash_api.cpp
FLASH_API_OBJ = build/flash_api.o

WRAPPER_SHARED_LIB = lib/libflash2.so


all: create_dirs ${WRAPPER_SHARED_LIB}

create_dirs:
	mkdir -p lib && mkdir -p build && mkdir -p build/instantiations && mkdir -p build/instantiations/sm120

${WRAPPER_SHARED_LIB}: ${BUILD_CU_KERNEL_OBJS} ${FLASH_API_OBJ} 
	${CXX} ${CXX_FLAGS} -shared ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

${FLASH_API_OBJ}: ${FLASH_API_SRC}
	${CXX} ${CXX_FLAGS} ${ALL_INCLUDES} -c -o $@ $<

build/instantiations/sm120/%.o: src/instantiations/%.cu
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_120a -code=sm_120a $< -c -o $@

clean:
	rm -f ${WRAPPER_SHARED_LIB} ${FLASH_API_OBJ}

clean_kernels:
	rm -f ${BUILD_CU_KERNEL_OBJS}
