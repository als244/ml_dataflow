CC = gcc
CFLAGS = -fPIC -g -std=c99

DATAFLOW_LIB_INCLUDE_DIR = ../libdataflow/include

SELF_INCLUDE_DIR = include

ALL_INCLUDE_DIRS = -I${DATAFLOW_LIB_INCLUDE_DIR} -I${SELF_INCLUDE_DIR}

DATAFLOW_SET_OPS_OBJS = objs/set_op_skeletons.o

DATAFLOW_SUBMIT_OPS_OBJS = objs/attention_op.o objs/attn_misc_ops.o objs/matmul_op.o objs/mlp_misc_ops.o objs/moe_ops.o objs/norm_ops.o objs/preprocess_ops.o objs/loss_misc_ops.o

DATAFLOW_OP_OBJS = ${DATAFLOW_SET_OPS_OBJS} ${DATAFLOW_SUBMIT_OPS_OBJS}

DATAFLOW_OP_LIB = libdataflowops.so

all: mkdirs lib/${DATAFLOW_OP_LIB}

mkdirs:
	mkdir -p objs && mkdir -p lib

lib/${DATAFLOW_OP_LIB}: ${DATAFLOW_OP_OBJS}
	${CC} ${CFLAGS} -shared ${ALL_INCLUDE_DIRS} $^ -o $@ 


## submit ops objs

objs/attention_op.o: src/submit_ops/attention_op.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/attn_misc_ops.o: src/submit_ops/attn_misc_ops.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/matmul_op.o: src/submit_ops/matmul_op.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/mlp_misc_ops.o: src/submit_ops/mlp_misc_ops.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/moe_ops.o: src/submit_ops/moe_ops.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/norm_ops.o: src/submit_ops/norm_ops.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/preprocess_ops.o: src/submit_ops/preprocess_ops.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

objs/loss_misc_ops.o: src/submit_ops/loss_misc_ops.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@	


## set ops objs

objs/set_op_skeletons.o: src/set_ops/set_op_skeletons.c
	${CC} ${CFLAGS} ${ALL_INCLUDE_DIRS} -c $< -o $@

clean:
	rm -f lib/${DATAFLOW_OP_LIB} objs/*.o












