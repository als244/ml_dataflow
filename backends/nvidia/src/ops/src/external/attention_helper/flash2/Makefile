## C++ Compiler
CXX = g++
CXX_FLAGS = -std=c++17 -g -fPIC -O3 -Wall

## NVCC Compilation
## SYSTEM-DEPENDENT
## (may be in weird location if python installation of CUDA
## within some environment...)
NVCC = nvcc
NVCC_FLAGS = -Xcompiler=-fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__  -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --generate-line-info -Xptxas=-v 

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
CU_KERNEL_SOURCES_SM80 = ${wildcard src/instantiations/sm80/*.cu}
CU_KERNEL_SOURCES_SM90 = ${wildcard src/instantiations/sm90/*.cu}
CU_KERNEL_SOURCES_SM100 = ${wildcard src/instantiations/sm100/*.cu}
CU_KERNEL_SOURCES_SM120 = ${wildcard src/instantiations/sm120/*.cu}



## create same base filename in different directory
CU_KERNEL_OBJS_SM80 = ${patsubst src/instantiations/sm80/%.cu, build/instantiations/sm80/%.o, ${CU_KERNEL_SOURCES_SM80}}
CU_KERNEL_OBJS_SM90 = ${patsubst src/instantiations/sm90/%.cu, build/instantiations/sm90/%.o, ${CU_KERNEL_SOURCES_SM90}}
CU_KERNEL_OBJS_SM100 = ${patsubst src/instantiations/sm100/%.cu, build/instantiations/sm100/%.o, ${CU_KERNEL_SOURCES_SM100}}
CU_KERNEL_OBJS_SM120 = ${patsubst src/instantiations/sm120/%.cu, build/instantiations/sm120/%.o, ${CU_KERNEL_SOURCES_SM120}}

BUILD_CU_KERNEL_OBJS = ${CU_KERNEL_OBJS_SM80} ${CU_KERNEL_OBJS_SM90} ${CU_KERNEL_OBJS_SM100} ${CU_KERNEL_OBJS_SM120}


FLASH_API_SRC = src/flash_api.cpp
FLASH_API_OBJ = build/flash_api.o

WRAPPER_SHARED_LIB = lib/libflash2.so


all: create_dirs ${WRAPPER_SHARED_LIB}

create_dirs:
	mkdir -p lib && mkdir -p build && mkdir -p build/instantiations && mkdir -p build/instantiations/sm80 && mkdir -p build/instantiations/sm90 && mkdir -p build/instantiations/sm100 && mkdir -p build/instantiations/sm120

${WRAPPER_SHARED_LIB}: ${BUILD_CU_KERNEL_OBJS} ${FLASH_API_OBJ} 
	${CXX} ${CXX_FLAGS} -shared ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

${FLASH_API_OBJ}: ${FLASH_API_SRC}
	${CXX} ${CXX_FLAGS} ${ALL_INCLUDES} -c -o $@ $<


build/instantiations/sm80/%.o: src/instantiations/sm80/%.cu
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_80 -code=sm_80 $< -c -o $@

build/instantiations/sm90/%.o: src/instantiations/sm90/%.cu
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_90a -code=sm_90a $< -c -o $@

build/instantiations/sm100/%.o: src/instantiations/sm100/%.cu
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_100a -code=sm_100a $< -c -o $@

build/instantiations/sm120/%.o: src/instantiations/sm120/%.cu
	${NVCC} ${NVCC_FLAGS} ${ALL_INCLUDES} -arch=compute_120a -code=sm_120a $< -c -o $@

clean:
	rm -f ${WRAPPER_SHARED_LIB} ${FLASH_API_OBJ}

clean_kernels:
	rm -f ${BUILD_CU_KERNEL_OBJS}
