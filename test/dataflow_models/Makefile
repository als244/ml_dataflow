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

#EXECS = naiveTransformer transformerHostAccumOpt transformerHostOpt  transformerRecompute transformer
EXECS = transformer transformerMoE 

all: create_data_dirs ${EXECS}

create_data_dirs:
	i=0; while [ $$i -le 31 ]; do \
        	mkdir -p test_transformer_data/layers_fwd/$$i; \
		mkdir -p test_transformer_data/layers_bwd/$$i; \
		mkdir -p test_transformer_data/model_grads/layers/$$i; \
		mkdir -p test_transformer_data/optimizer_states/layers/$$i; \
		mkdir -p correct_transformer_data/layers_fwd/$$i; \
		mkdir -p correct_transformer_data/layers_bwd/$$i; \
		i=$$(($$i + 1)); \
    	done; \
	mkdir -p test_transformer_data/head_fwd; \
	mkdir -p test_transformer_data/head_bwd; \
	mkdir -p test_transformer_data/embedding_fwd; \
	mkdir -p test_transformer_data/embedding_bwd; \
	mkdir -p test_transformer_data/model_grads/embedding; \
	mkdir -p test_transformer_data/model_grads/head; \
	mkdir -p test_transformer_data/optimizer_states/embedding; \
	mkdir -p test_transformer_data/optimizer_states/head; \
	mkdir -p correct_transformer_data/head_fwd; \
	mkdir -p correct_transformer_data/head_bwd; \
	mkdir -p correct_transformer_data/model_grads; \
	mkdir -p correct_transformer_data/optimizer_states; \
	mkdir -p profiling


## version where activations can exceed host memory capacity and we selectively decide which acts to save, and if not saved then recompute before doing bwds...
transformer: transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

transformerMoE: transformer_moe.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

## version where host just stores temporary data and all ops are on device
transformerNoRecompute: transformer_no_recompute.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

## version where the host runs adam optimizer (makes sense in distributed case)
transformerHostOpt: transformer_host_opt.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}

## version where the host accumulates gradients and run adam optimizer (doesn't make sense, but here for reference)
transformerHostAccumOpt: transformer_host_accum_opt.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}


## no data transfers at all, everything is on device
naiveTransformer: naive_transformer.c
	${CC} ${CFLAGS} ${ALL_INCLUDES} $^ -o $@ ${ALL_LIBS}



clean:
	rm -f ${EXECS}

clean_data:
	rm -rf test_transformer_data/ && rm -rf correct_transformer_data && make create_data_dirs

clean_test_data:
	rm -rf test_transformer_data/ && make create_data_dirs

clean_correct_data:
	rm -rf correct_transformer_data/ && make create_data_dirs

