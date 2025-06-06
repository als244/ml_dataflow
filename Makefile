
backend_name=nvidia

all: backends libdataflow libdataflowops libdataflowmodels dataflow_models_test

backends:
	${MAKE} -C backends/host && ${MAKE} -C backends/${backend_name}

libdataflow:
	${MAKE} -C libdataflow

libdataflowops:
	${MAKE} -C libdataflowops

libdataflowmodels:
	${MAKE} -C libdataflowmodels

dataflow_models_test:
	${MAKE} -C test/dataflow_models
