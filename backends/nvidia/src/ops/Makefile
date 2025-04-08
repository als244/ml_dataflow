CC = gcc
CFLAGS = -g -fPIC


CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64/stubs
CUDA_LIB_DEPENDS = -lcuda


## Native OP

NATIVE_OP_PATH = src/native

## setting up variables for native op kernels

NVCC = nvcc
DEV_NVCC_FLAGS = -g -G
BUILD_NVCC_FLAGS = -O4 --use_fast_math
PROF_NVCC_FLAGS = -O4 --use_fast_math --generate-line-info -Xptxas=-v
NVCC_FLAGS = ${PROF_NVCC_FLAGS}


## for now just makign for one arch, but in reality can compile down to all archs and 
## select at runtime...

## or do jit compilation durin init...

## just for now only build for speciic, arch but in reality we can generate
## many cubins (or ptx) and then dynamically select...
NATIVE_OP_CUDA_ARCH_NUM = 90a

NATIVE_OP_KERNELS_PTX = ${NATIVE_OP_PATH}/build/cuda_kernels.ptx
NATIVE_OP_KERNELS_CUBIN = cuda_kernels.cubin

NATIVE_OP_KERNELS_ALL_INCLUDES = -I${CUDA_INCLUDE_DIR} -I${NATIVE_OP_PATH}/include

NATIVE_OP_KERNELS_SRC = ${NATIVE_OP_PATH}/src/cuda_kernels.cu


## Setting variablNative Functions & Launch Config Lib

DATAFLOW_LIB_INCLUDE_DIR = ../../../../libdataflow/include
CUDA_DATAFLOW_HANDLE_INCLUDE_DIR = ../../include

NATIVE_OP_CONFIG_LIB_SRC = ${NATIVE_OP_PATH}/src/preprocess/embedding_table_config.c ${NATIVE_OP_PATH}/src/norm/rms_norm_config.c ${NATIVE_OP_PATH}/src/attention_misc/attn_misc_config.c ${NATIVE_OP_PATH}/src/moe/moe_config.c ${NATIVE_OP_PATH}/src/activations/swiglu_config.c ${NATIVE_OP_PATH}/src/loss_misc/loss_misc_config.c

NATIVE_OP_CONFIG_LIB_ALL_INCLUDES = -I${NATIVE_OP_PATH}/include -I${CUDA_DATAFLOW_HANDLE_INCLUDE_DIR} -I${DATAFLOW_LIB_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR}

NATIVE_OP_CONFIG_LIB = cuda_kernels_config.so



## External Ops

EXTERNAL_OP_PATH = ${CURDIR}/src/external

## Attention Helper Library

## 0.) create flash3 lib as helper lib

FLASH3_DIR = ${EXTERNAL_OP_PATH}/attention_helper/flash3
### flash3 lib dir
FLASH3_LIB_DIR = ${FLASH3_DIR}/lib
### flash3 include dir
FLASH3_INCLUDE_DIR = ${FLASH3_DIR}/include
### flash3 lib
FLASH3_LIB = -lflash3

## ensure we make flash3 lib


## 1.) set up variables for attention helper lib

EXTERNAL_OP_ATTENTION_LIB_OBJS = ${EXTERNAL_OP_PATH}/attention_helper/objs/attention_helper.o
EXTERNAL_OP_ATTENTION_LIB_ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${CUDA_DATAFLOW_HANDLE_INCLUDE_DIR} -I${CUDA_INCLUDE_DIR} -I${EXTERNAL_OP_PATH}/attention_helper/include -I${FLASH3_INCLUDE_DIR}
EXTERNAL_OP_ATTENTION_LIB_OTHER_LIB_DEPENDS = -L${FLASH3_LIB_DIR} ${FLASH3_LIB} -Wl,-rpath,${FLASH3_LIB_DIR}


EXTERNAL_OP_ATTENTION_LIB = libattentionwrapper.so


## 2.) set up variables for matmul helper lib

CUBLAS_LIB_DIR = /usr/local/cuda/lib64

EXTERNAL_OP_MATMUL_LIB_OBJS = ${EXTERNAL_OP_PATH}/matmul_helper/objs/matmul_helper.o

EXTERNAL_OP_MATMUL_LIB_ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${CUDA_DATAFLOW_HANDLE_INCLUDE_DIR} -I${EXTERNAL_OP_PATH}/matmul_helper/include -I${CUDA_INCLUDE_DIR}

EXTERNAL_OP_MATMUL_LIB_OTHER_LIB_DEPENDS = -L${CUDA_DATAFLOW_HANDLE_LIB_DIR} ${CUDA_DATAFLOW_HANDLE_LIB} -Wl,-rpath,${CUDA_DATAFLOW_HANDLE_LIB_DIR} -L${CUBLAS_LIB_DIR} -lcublasLt -L${CUDA_LIB_DIR} -lcuda -Wl,-rpath,${CUBLAS_LIB_DIR} -Wl,-rpath,${CUDA_LIB_DIR} -Wl,--enable-new-dtags

EXTERNAL_OP_MATMUL_LIB = libmatmulwrapper.so


## 3.) set up variables for register ops
SET_OP_SKELETONS_INCLUDE_DIR = ../../../../libdataflowops/include

REGISTER_OPS_INCLUDE_DIR = ../../include

REGISTER_OPS_ALL_INCLUDES = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${SET_OP_SKELETONS_INCLUDE_DIR} -I${REGISTER_OPS_INCLUDE_DIR}

REGISTER_OPS_LIB = ../../lib/libcuda_register_default_ops.so


all: create_dirs gen_native_ptx gen_native_cubin lib/native/${NATIVE_OP_CONFIG_LIB} lib/external/${EXTERNAL_OP_ATTENTION_LIB} lib/external/${EXTERNAL_OP_MATMUL_LIB} ${REGISTER_OPS_LIB}

create_dirs:
	mkdir -p lib && mkdir -p lib/native && mkdir -p lib/external && mkdir -p ${NATIVE_OP_PATH}/build && mkdir -p ${EXTERNAL_OP_PATH}/matmul_helper/objs && mkdir -p ${EXTERNAL_OP_PATH}/attention_helper/objs

## native op kernels
gen_native_ptx: ${NATIVE_OP_KERNELS_SRC} 
	nvcc ${NVCC_FLAGS} ${NATIVE_OP_KERNELS_ALL_INCLUDES} $^ -o ${NATIVE_OP_KERNELS_PTX} -ptx -arch=compute_${NATIVE_OP_CUDA_ARCH_NUM} -code=sm_${NATIVE_OP_CUDA_ARCH_NUM}

gen_native_cubin: ${NATIVE_OP_KERNELS_SRC}
	nvcc ${NVCC_FLAGS} ${NATIVE_OP_KERNELS_ALL_INCLUDES} $^ -o lib/native/${NATIVE_OP_KERNELS_CUBIN} -cubin -arch=compute_${NATIVE_OP_CUDA_ARCH_NUM} -code=sm_${NATIVE_OP_CUDA_ARCH_NUM}

## native op config lib

lib/native/${NATIVE_OP_CONFIG_LIB}: ${NATIVE_OP_CONFIG_LIB_SRC}
	${CC} ${CFLAGS} ${NATIVE_OP_CONFIG_LIB_ALL_INCLUDES} -shared $^ -o $@ ${NATIVE_OP_CONFIG_LIB_OTHER_LIB_DEPENDS}



## External Attention Lib
lib/external/${EXTERNAL_OP_ATTENTION_LIB}: build_flash3_lib ${EXTERNAL_OP_ATTENTION_LIB_OBJS}
	${CC} ${CFLAGS} ${EXTERNAL_OP_ATTENTION_LIB_ALL_INCLUDES} -shared ${EXTERNAL_OP_ATTENTION_LIB_OBJS} -o $@ ${EXTERNAL_OP_ATTENTION_LIB_OTHER_LIB_DEPENDS}

build_flash3_lib:
	make -C ${FLASH3_DIR}

${EXTERNAL_OP_PATH}/attention_helper/objs/attention_helper.o: ${EXTERNAL_OP_PATH}/attention_helper/src/attention_helper.c
	${CC} ${CFLAGS} ${EXTERNAL_OP_ATTENTION_LIB_ALL_INCLUDES} -c $< -o $@


## External Matmul Lib
lib/external/${EXTERNAL_OP_MATMUL_LIB}: ${EXTERNAL_OP_MATMUL_LIB_OBJS}
	${CC} ${CFLAGS} ${EXTERNAL_OP_MATMUL_LIB_ALL_INCLUDES} -shared $^ -o $@ ${EXTERNAL_OP_MATMUL_LIB_OTHER_LIB_DEPENDS}

${EXTERNAL_OP_PATH}/matmul_helper/objs/matmul_helper.o: ${EXTERNAL_OP_PATH}/matmul_helper/src/matmul_helper.c
	${CC} ${CFLAGS} ${EXTERNAL_OP_MATMUL_LIB_ALL_INCLUDES} -c $< -o $@


## Register Ops Lib
${REGISTER_OPS_LIB}: src/register_ops/register_ops.c
	${CC} ${CFLAGS} -shared -DOPS_ROOT_DIR=\"${CURDIR}\" ${REGISTER_OPS_ALL_INCLUDES} $^ -o $@


## don't wipe flash3 lib as this takes a while to build...
## can go into that directory and run clean if wanted...
clean:
	rm -f ${NATIVE_OP_KERNELS_PTX} lib/native/${NATIVE_OP_KERNELS_CUBIN} lib/native/${NATIVE_OP_CONFIG_LIB} lib/external/${EXTERNAL_OP_ATTENTION_LIB} lib/external/${EXTERNAL_OP_MATMUL_LIB} ${EXTERNAL_OP_PATH}/matmul_helper/objs/*.o ${EXTERNAL_OP_PATH}/attention_helper/objs/*.o ${REGISTER_OPS_LIB}

