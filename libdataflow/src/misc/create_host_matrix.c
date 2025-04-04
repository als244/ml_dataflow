#include "dataflow.h"

float rand_normal(float mean, float std) {

	if (std == 0){
		return mean;
	}

    static float spare;
    static int has_spare = 0;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    } else {
        float u, v, s;
        do {
            u = (rand() / (float)RAND_MAX) * 2.0 - 1.0;
            v = (rand() / (float)RAND_MAX) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        s = sqrtf(-2.0 * logf(s) / s);
        spare = v * s;
        has_spare = 1;
        return mean + std * u * s;
    }
}


void * create_zero_host_matrix(uint64_t M, uint64_t N, DataflowDatatype dt, void * opt_dest) {

	int ret;
	
	uint64_t num_els = M * N;
	uint64_t dtype_size = dataflow_sizeof_element(dt);

	void * zero_matrix;
	if (!opt_dest){
		zero_matrix = malloc(num_els * dtype_size);
		if (!zero_matrix){
			fprintf(stderr, "Error: could not allocate zero matrix on host of size %lu...\n", num_els * dtype_size);
			return NULL;
		}
	}
	else{
		zero_matrix = opt_dest;
	}

	memset(zero_matrix, 0, num_els * dtype_size);

	return zero_matrix;
}

void * create_rand_host_matrix(uint64_t M, uint64_t N, float mean, float std, DataflowDatatype dt, void * opt_dest) {

	

	uint64_t num_els = M * N;

	float * rand_float_matrix = malloc(num_els * sizeof(float));

	if (!rand_float_matrix){
		fprintf(stderr, "Error: could not allocate temp random float matrix\n\tM: %lu\n\tN: %lu\n\n", M, N);
		return NULL;
	}

	for (uint64_t i = 0; i < num_els; i++){
		rand_float_matrix[i] = rand_normal(mean, std);
	}

	size_t el_size = dataflow_sizeof_element(dt);

	void * dest_matrix;

	if (opt_dest){
		dest_matrix = opt_dest;
	}
	else{
		dest_matrix = malloc(num_els * el_size);
		if (!dest_matrix){
			fprintf(stderr, "Error: failed to alloc dest host matrix\n\tM: %lu\n\tN: %lu\n\n", M, N);
			free(rand_float_matrix);
			return NULL;
		}
	}

	int num_threads = 8;

	int ret = dataflow_convert_datatype(dest_matrix, rand_float_matrix, dt, DATAFLOW_FP32, num_els, num_threads);
	
	free(rand_float_matrix);
	
	if (ret){
		fprintf(stderr, "Error: failure in conversion from random float matrix to %s matrix during create rand...\n", dataflow_datatype_as_string(dt));
		if (!opt_dest){
			free(dest_matrix);
		}
		return NULL;
	}

	return dest_matrix;
}

void * create_identity_host_matrix(uint64_t N, DataflowDatatype dt, void * opt_dest) {

	

	uint64_t num_els = N * N;

	float * identity_float_matrix = malloc(num_els * sizeof(float));

	if (!identity_float_matrix){
		fprintf(stderr, "Error: could not allocate temp identity float matrix\n\tM: %lu\n\tN: %lu\n\n", N, N);
		return NULL;
	}

	for (uint64_t i = 0; i < N; i++){
		identity_float_matrix[i + i * N] = 1.0f;
	}

	size_t el_size = dataflow_sizeof_element(dt);

	void * dest_matrix;

	if (opt_dest){
		dest_matrix = opt_dest;
	}
	else{
		dest_matrix = malloc(num_els * el_size);
		if (!dest_matrix){
			fprintf(stderr, "Error: failed to alloc dest host matrix\n\tM: %lu\n\tN: %lu\n\n", N, N);
			free(identity_float_matrix);
			return NULL;
		}
	}

	int num_threads = 8;

	int ret = dataflow_convert_datatype(dest_matrix, identity_float_matrix, dt, DATAFLOW_FP32, num_els, num_threads);
	
	free(identity_float_matrix);
	
	if (ret){
		fprintf(stderr, "Error: failure in conversion from identity float matrix to %s matrix during create identity...\n", dataflow_datatype_as_string(dt));
		if (!opt_dest){
			free(dest_matrix);
		}
		return NULL;
	}

	return dest_matrix;
}

void * create_index_identity_host_matrix(uint64_t M, uint64_t N, DataflowDatatype dt, void * opt_dest) {

	

	uint64_t num_els = M * N;

	float * index_identity_float_matrix = malloc(num_els * sizeof(float));

	if (!index_identity_float_matrix){
		fprintf(stderr, "Error: could not allocate temp index identity float matrix\n\tM: %lu\n\tN: %lu\n\n", M, N);
		return NULL;
	}

	for (uint64_t i = 0; i < num_els; i++){
		index_identity_float_matrix[i] = (float) i;
	}

	size_t el_size = dataflow_sizeof_element(dt);

	void * dest_matrix;

	if (opt_dest){
		dest_matrix = opt_dest;
	}
	else{
		dest_matrix = malloc(num_els * el_size);
		if (!dest_matrix){
			fprintf(stderr, "Error: failed to alloc dest host matrix\n\tM: %lu\n\tN: %lu\n\n", M, N);
			free(index_identity_float_matrix);
			return NULL;
		}
	}

	int num_threads = 8;

	int ret = dataflow_convert_datatype(dest_matrix, index_identity_float_matrix, dt, DATAFLOW_FP32, num_els, num_threads);
	
	free(index_identity_float_matrix);
	
	if (ret){
		fprintf(stderr, "Error: failure in conversion from index identity float matrix to %s matrix during create index identity...\n", dataflow_datatype_as_string(dt));
		if (!opt_dest){
			free(dest_matrix);
		}
		return NULL;
	}

	return dest_matrix;
}



void * load_host_matrix_from_file(char * filepath, uint64_t M, uint64_t N, DataflowDatatype orig_dt, DataflowDatatype new_dt, void * opt_dest) {

	FILE * fp = fopen(filepath, "rb");
	if (!fp){
		fprintf(stderr, "Error: could not load matrix from file: %s\n", filepath);
		return NULL;
	}

	uint64_t num_els = M * N;
	uint64_t orig_dtype_size = dataflow_sizeof_element(orig_dt);

	void * orig_matrix = malloc(num_els * orig_dtype_size);
	if (!orig_matrix){
		fprintf(stderr, "Error: malloc failed to allocate memory for orig matrix (M = %lu, N = %lu)\n", M, N);
		fclose(fp);
		return NULL;
	}

	size_t nread = fread(orig_matrix, orig_dtype_size, num_els, fp);
	if (nread != num_els){
		fprintf(stderr, "Error: couldn't read expected number of elements from matrix file: %s. (Expected %lu, read: %lu)\n", filepath, num_els, nread);
		free(orig_matrix);
		fclose(fp);
		return NULL;
	}

	fclose(fp);

	
	uint64_t new_dtype_size = dataflow_sizeof_element(new_dt);

	void * dest_matrix;

	if (opt_dest){
		dest_matrix = opt_dest;
	}
	else{
		dest_matrix = malloc(num_els * new_dtype_size);
		if (!dest_matrix){
			fprintf(stderr, "Error: failed to create destination matrix when loading matrix from file (M = %lu, N = %lu)\n", M, N);
			free(orig_matrix);
			return NULL;
		}
	}


	int num_threads = 8;

	int ret = dataflow_convert_datatype(dest_matrix, orig_matrix, new_dt, orig_dt, num_els, num_threads);
	
	free(orig_matrix);
	
	if (ret){
		fprintf(stderr, "Error: failure in conversion from orig matrix from file of type %s to new matrix of type %s...\n", dataflow_datatype_as_string(orig_dt), dataflow_datatype_as_string(new_dt));
		if (!opt_dest){
			free(dest_matrix);
		}
		return NULL;
	}

	return dest_matrix;
}

int save_host_matrix(char * filename, void * mat, uint64_t M, uint64_t N, DataflowDatatype dt){

	FILE * fp = fopen(filename, "wb");
	if (!fp){
		fprintf(stderr, "Error: could not open file with path: %s\n", filename);
		return -1;
	}

	size_t el_size = dataflow_sizeof_element(dt);
	size_t num_els = M * N;

	size_t num_written = fwrite(mat, el_size, num_els, fp);
	if (num_written != num_els){
		fprintf(stderr, "Error: write failed, expected to write %lu els, but only wrote %lu...\n", num_els, num_written);
		fclose(fp);
		return -1;
	}

	fclose(fp);

	return 0;
}