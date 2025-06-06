
backend_name=nvidia

# Declare targets that are not files as .PHONY.
# This prevents make from getting confused by directories with the same name
# and ensures the recipes are always run in the correct order.
.PHONY: all libdataflow backends libdataflowops libdataflowmodels dataflow_models_test copy_artifacts_to_root

all: libdataflow backends libdataflowops libdataflowmodels dataflow_models_test

libdataflow:
	${MAKE} -C libdataflow

backends: libdataflow
	${MAKE} -C backends/host && ${MAKE} -C backends/${backend_name}

libdataflowops: libdataflow
	${MAKE} -C libdataflowops

libdataflowmodels: libdataflowops
	${MAKE} -C libdataflowmodels

dataflow_models_test: libdataflowmodels backends
	${MAKE} -C test/dataflow_models


copy_artificacts_to_root: dataflow_models_test:
	cp test/dataflow_models/transformer . && cp test/dataflow_models/do_transformer_profile.sh .
