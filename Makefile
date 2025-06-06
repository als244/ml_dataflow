
backend_name=nvidia

all: backends libdataflow libdataflowops libdataflowmodels dataflow_models_test

backends:
	${MAKE} -c backends/host && ${MAKE} -c backends/${backend_name}

libdataflow:
	${MAKE} -c libdataflow

libdataflowops:
	${MAKE} -c libdataflowops

libdataflowmodels:
	${MAKE} -c libdataflowmodels

dataflow_models_test:
	${MAKE} -c test/dataflow_models
