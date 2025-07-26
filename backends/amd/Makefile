
## Ensuring correct dirs are available and calling make from subdirectors

all: create_dirs create_handle_lib create_register_ops_lib

create_dirs:
	mkdir -p lib && mkdir -p src/handle/objs && mkdir -p src/ops/objs

create_handle_lib:
	${MAKE} -C src/handle

create_register_ops_lib:
	${MAKE} -C src/ops


clean:
	${MAKE} clean -C src/handle && ${MAKE} clean -C src/ops

