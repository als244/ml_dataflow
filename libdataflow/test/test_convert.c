#include "dataflow.h"

#define NUM_TO_PRINT 5

int main(int argc, char * argv[]){

	long size = 100000;
	int num_threads = 1;

	float orig_fp32_val = 3.0;



	float * fp32_arr = malloc(size * sizeof(float));

	for (long i = 0; i < size; i++){
		fp32_arr[i] = (float) i;
	}

	uint16_t * fp16_arr = calloc(size, sizeof(uint16_t));
	uint16_t * bf16_arr = calloc(size, sizeof(uint16_t));

	int ret;

	ret = dataflow_convert_datatype(fp16_arr, fp32_arr, DATAFLOW_FP16, DATAFLOW_FP32, size, num_threads);
	if (ret){
		fprintf(stderr, "Error: unable to convert data type from fp32 to fp16...\n");
		return -1;
	}

	ret = dataflow_convert_datatype(bf16_arr, fp32_arr, DATAFLOW_BF16, DATAFLOW_FP32, size, num_threads);
	if (ret){
		fprintf(stderr, "Error: unable to convert data type from fp32 to bf16...\n");
		return -1;
	}

	for (long i = 0; i < NUM_TO_PRINT; i++){
		printf("fp16_arr[%ld] = 0x%04X\n", (long) i, fp16_arr[i]);
	}

	printf("\n\n\n");

	for (long i = 0; i < NUM_TO_PRINT; i++){
		printf("bf16_arr[%ld] = 0x%04X\n", (long) i, bf16_arr[i]);
	}

	printf("\n\n\n");

	return 0;
}