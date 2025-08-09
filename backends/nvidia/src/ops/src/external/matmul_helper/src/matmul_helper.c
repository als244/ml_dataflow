#include "matmul_helper.h"

#define MAX_ALGO_SEARCH 3

#define MATMUL_ALGO_KEY_FINGERPRINT_TYPE 0
#define MATMUL_ALGO_KEY_FINGERPRINT_NUM_BYTES 32

#define MATMUL_ALGO_TABLE_MIN_SIZE 1UL << 10
#define MATMUL_ALGO_TABLE_MAX_SIZE 1UL << 26

#define MATMUL_DIMS_ROUND_UP_MULTIPLE 64

#define MATMUL_DIM_ROUND_UP(dim) (((dim) + (MATMUL_DIMS_ROUND_UP_MULTIPLE) - 1) & ~((MATMUL_DIMS_ROUND_UP_MULTIPLE) - 1))

#define MATMUL_WORKSPACE_ROUND_DOWN_MULTIPLE 1UL << 21

#define MATMUL_WORKSPACE_ROUND_DOWN(workspace_bytes) (((workspace_bytes) / (MATMUL_WORKSPACE_ROUND_DOWN_MULTIPLE)) * (MATMUL_WORKSPACE_ROUND_DOWN_MULTIPLE))

typedef struct matmul_runtime_params {
	int M;
	int N;
	int K;
	uint64_t workspaceBytes;
	uint16_t alpha_h;
	float alpha_f;
	double alpha_d;
	void * A;
	void * B;
	uint16_t beta_h;
	float beta_f;
	double beta_d;
	void * C;
	void * D;
	void * workspace;
} Matmul_Runtime_Params;

typedef struct matmul_algo_key {
	int num_sms;
	int rounded_M;
	int rounded_N;
	int rounded_K;
	DataflowDatatype a_dt;
	DataflowDatatype b_dt;
	DataflowDatatype c_dt;
	DataflowDatatype d_dt;
	DataflowDatatype compute_dt;
	int to_trans_a;
	int to_trans_b;
	uint64_t rounded_workspaceBytes;
} Matmul_Algo_Key;

typedef struct cublas_matrix_layouts {
	cublasLtMatrixLayout_t Adesc;
	cublasLtMatrixLayout_t Bdesc;
	cublasLtMatrixLayout_t Cdesc;
	cublasLtMatrixLayout_t Ddesc;
} Cublas_Matrix_Layouts;

typedef struct cublas_matmul_algo_value {
	cublasLtMatmulDesc_t computeDesc;
	cudaDataType scale_cuda_dt;
	cublasLtMatmulAlgo_t algo;
} Cublas_Matmul_Algo_Value;

typedef struct cublas_matmul_params {
	// THESE ARE INPUTS
	Matmul_Runtime_Params * runtime_params;
	Matmul_Algo_Key * algo_key;
	// THESE GET POPULATED BY THE SET PARAMS 

	// this is just used temporarily
	Cublas_Matrix_Layouts * matrix_layouts;
	// this is copied into the table
	Cublas_Matmul_Algo_Value * cublas_matmul_algo_value;
} Cublas_Matmul_Params;



static uint64_t cublas_fingerprint_to_least_sig64(uint8_t * matmul_algo_key_fingerprint, int fingerprint_num_bytes){
	uint8_t * least_sig_start = matmul_algo_key_fingerprint + fingerprint_num_bytes - sizeof(uint64_t);
	uint64_t result = 0;
    for(int i = 0; i < 8; i++){
        result <<= 8;
        result |= (uint64_t)least_sig_start[i];
    }
    return result;
}

uint64_t matmul_algo_table_hash_func(void * matmul_algo_key_fingerprint, uint64_t table_size) {
	uint64_t least_sig_64bits = cublas_fingerprint_to_least_sig64((void *) matmul_algo_key_fingerprint, MATMUL_ALGO_KEY_FINGERPRINT_NUM_BYTES);
	return least_sig_64bits % table_size;
}

static int dtype_to_cuda_dtype(DataflowDatatype dtype, cudaDataType * ret_dtype){

	switch(dtype){
		case DATAFLOW_FP32:
			*ret_dtype = CUDA_R_32F;
			break;
		case DATAFLOW_FP16:
			*ret_dtype = CUDA_R_16F;
			break;
		case DATAFLOW_BF16:
			*ret_dtype = CUDA_R_16BF;
			break;
		case DATAFLOW_FP8E4M3:
			*ret_dtype = CUDA_R_8F_E4M3;
			break;
		case DATAFLOW_FP8E5M2:
			*ret_dtype = CUDA_R_8F_E5M2;
			break;
		default:
			printf("Error: unsupported dtype to convert to cuda\n");
			return -1;
	}

	return 0;
}

static int set_cuda_dtypes(Cublas_Matmul_Params * matmul_params, 
							cudaDataType * a_cuda_dt, cudaDataType * b_cuda_dt, cudaDataType * c_cuda_dt, cudaDataType * d_cuda_dt) {

	int ret;

	Matmul_Algo_Key * algo_key = matmul_params -> algo_key;
	Matmul_Runtime_Params * runtime_params = matmul_params -> runtime_params;
	
	ret = dtype_to_cuda_dtype(algo_key -> a_dt, a_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unsupported A dtype of %s\n", dataflow_datatype_as_string(algo_key -> a_dt));
		return -1;
	}


	ret = dtype_to_cuda_dtype(algo_key -> b_dt, b_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unsupported B dtype of %s\n", dataflow_datatype_as_string(algo_key -> b_dt));
		return -1;
	}

	

	ret = dtype_to_cuda_dtype(algo_key -> d_dt, d_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unsupported D dtype of %s\n", dataflow_datatype_as_string(algo_key -> d_dt));
		return -1;
	}

	if ((algo_key -> c_dt == DATAFLOW_NONE) || (!runtime_params -> C) || (runtime_params -> beta_f == 0)) {
		if ((algo_key -> a_dt == DATAFLOW_FP8E4M3) || (algo_key -> a_dt == DATAFLOW_FP8E5M2)){
			*c_cuda_dt = CUDA_R_16F;
		}
		else {
			*c_cuda_dt = *d_cuda_dt;
		}
	}
	else{
		ret = dtype_to_cuda_dtype(algo_key -> c_dt, c_cuda_dt);
		if (ret){
			fprintf(stderr, "Error: unsupported C dtype of %s\n", dataflow_datatype_as_string(algo_key -> c_dt));
			return -1;
		}
	}

	return 0;
}

static int set_cublas_compute_scale_types(cublasComputeType_t * cublas_compute_type, cudaDataType * scale_cuda_dt, DataflowDatatype compute_dt, cudaDataType a_cuda_dt, cudaDataType b_cuda_dt, cudaDataType c_cuda_dt, cudaDataType d_cuda_dt){

	switch (compute_dt){
		case DATAFLOW_FP32:
			// ERROR CHECKING
			switch (a_cuda_dt){
				case CUDA_R_32F:
					if (b_cuda_dt != CUDA_R_32F){
						fprintf(stderr, "Error: when A dt = FP32, B must also be FP32...\n");
						return -1;
					}
					if (c_cuda_dt != CUDA_R_32F){
						fprintf(stderr, "Error: when A dt = FP32, C must also be FP32...\n");
						return -1;
					}
					break;
				case CUDA_R_16F:
					if (b_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: when A dt = FP16, B must also be FP16...\n");
						return -1;
					}
					if ((c_cuda_dt != CUDA_R_32F) && (c_cuda_dt != CUDA_R_16F)){
						fprintf(stderr, "Error: when A dt = FP16, C must either be FP16 or FP32...\n");
						return -1;
					}
					break;
				case CUDA_R_16BF:
					if (b_cuda_dt != CUDA_R_16BF){
						fprintf(stderr, "Error: when A dt = BF16, B must also be BF16...\n");
						return -1;
					}
					if ((c_cuda_dt != CUDA_R_32F) && (c_cuda_dt != CUDA_R_16BF)){
						fprintf(stderr, "Error: when A dt = BF16, C must either be BF16 or FP32...\n");
						return -1;
					}
					break;
				case CUDA_R_8F_E4M3:
					switch (b_cuda_dt){
						case CUDA_R_8F_E4M3:
							switch (c_cuda_dt){
								case CUDA_R_16F:
									if ((d_cuda_dt != CUDA_R_16F) && (d_cuda_dt != CUDA_R_8F_E4M3)){
										fprintf(stderr, "Error: when A dt = B dt FP8E4M3 and C = FP16, D must be either FP16 or FP8E4M3...\n");
										return -1;
									}
									break;
								case CUDA_R_16BF:
									if ((d_cuda_dt != CUDA_R_16BF) && (d_cuda_dt != CUDA_R_8F_E4M3)){
										fprintf(stderr, "Error: when A dt = B dt FP8E4M3 and C = BF16, D must be either BF16 or FP8E4M3...\n");
										return -1;
									}
									break;
								case CUDA_R_32F:
									if (d_cuda_dt != CUDA_R_32F){
										fprintf(stderr, "Error: when A dt = B dt FP8E4M3 and C = FP32, D must be FP32...\n");
										return -1;
									}
									break;
								default:
									fprintf(stderr, "Error: when A dt = B dt FP8E4M3, C dt must be FP16, BF16, or FP32...\n");
									return -1;
							}
							break;
						case CUDA_R_8F_E5M2:
							switch (c_cuda_dt){
								case CUDA_R_16F:
									if ((d_cuda_dt != CUDA_R_16F) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
										fprintf(stderr, "Error: when A dt = FP8E4M3, B dt = FP8E5M2 and C = FP16, D must be either FP16, FP8E4M3, or FP8E5M2...\n");
										return -1;
									}
									break;
								case CUDA_R_16BF:
									if ((d_cuda_dt != CUDA_R_16BF) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
										fprintf(stderr, "Error: when A dt = FP8E4M3, B dt = FP8E5M2 and C = BF16, D must be either BF16 or FP8E4M3, or FP8E5M2...\n");
										return -1;
									}
									break;
								case CUDA_R_32F:
									if (d_cuda_dt != CUDA_R_32F){
										fprintf(stderr, "Error: when A dt = FP8E4M3, B dt = FP8E5M2 and C = FP32, D must be FP32...\n");
										return -1;
									}
									break;
								default:
									fprintf(stderr, "Error: when when A dt = FP8E4M3, B dt = FP8E5M2, C dt must be FP16, BF16, or FP32...\n");
									return -1;
							}
							break;
						default:
							fprintf(stderr, "Error: when A dt = FP8E4M3, B dt must be FP8E4M3 or FP8E5M2...\n");
							return -1;
					}
					break;
				case CUDA_R_8F_E5M2:
					if (b_cuda_dt != CUDA_R_8F_E4M3){
						fprintf(stderr, "Error: when A = FP8E5M2, B dt must be FP8E4M3...\n");
						return -1;
					}
					switch (c_cuda_dt){
						case CUDA_R_16F:
							if ((d_cuda_dt != CUDA_R_16F) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
								fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3 and C = FP16, D must be either FP16, FP8E4M3, or FP8E5M2...\n");
								return -1;
							}
							break;
						case CUDA_R_16BF:
							if ((d_cuda_dt != CUDA_R_16BF) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
								fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3 and C = BF16, D must be either BF16 or FP8E4M3, or FP8E5M2...\n");
								return -1;
							}
							break;
						case CUDA_R_32F:
							if (d_cuda_dt != CUDA_R_32F){
								fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3 and C = FP32, D must be FP32...\n");
								return -1;
							}
							break;
						default:
							fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3, C dt must be FP16, BF16, or FP32...\n");
							return -1;
					}
					break;
				default:
					fprintf(stderr, "Error: unsupported A dt of type: %d\n", a_cuda_dt);
					return -1;
			}
			*cublas_compute_type = CUBLAS_COMPUTE_32F;
			*scale_cuda_dt = CUDA_R_32F;
			return 0;
		case DATAFLOW_FP16:
			switch (a_cuda_dt){
				case CUDA_R_16F:
					if (b_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: if compute type is FP16 and A dt = FP16, B dt must also be FP16...\n");
						return -1;
					}
					if (c_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: if compute type is FP16 and A dt = B dt = FP16, C dt must also be FP16...\n");
						return -1;
					}
					if (d_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: if compute type is FP16 and A dt = B dt = C dt = FP16, D dt must also be FP16...\n");
						return -1;
					}
					*cublas_compute_type = CUBLAS_COMPUTE_16F;
					*scale_cuda_dt = CUDA_R_16F;
					return 0;
				case CUDA_R_32F:
					if ((b_cuda_dt != CUDA_R_32F) || (c_cuda_dt != CUDA_R_32F) || (d_cuda_dt != CUDA_R_32F)) {
						fprintf(stderr, "Error: if compute type is FP16 and A dt = FP32, B dt, C dt, and D dt must also be FP32...\n");
						return -1;
					}
					*cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
					*scale_cuda_dt = CUDA_R_32F;
					return 0;
				default:
					fprintf(stderr, "Error: if compute dtpe is FP16, A dt must be either FP16 or FP32...\n");
					return -1;
			}
			break;
		case DATAFLOW_BF16:
			if ((a_cuda_dt != CUDA_R_32F) || (b_cuda_dt != CUDA_R_32F) || (c_cuda_dt != CUDA_R_32F) || (d_cuda_dt != CUDA_R_32F)) {
				fprintf(stderr, "Error: if compute type is BF16 and A dt, B dt, C dt, and D dt must all be FP32...\n");
				return -1;
			}
			*cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
			*scale_cuda_dt = CUDA_R_32F;
			return 0;
		default:
			fprintf(stderr, "Error: compute type must be either FP32, FP16, or BF16...\n");
			return -1;
	}

	// won't get here
	return -1;
}

static int set_cublas_matrix_layouts(Cublas_Matmul_Params * matmul_params,
											cudaDataType a_cuda_dt, cudaDataType b_cuda_dt, cudaDataType c_cuda_dt, cudaDataType d_cuda_dt,
											Cublas_Matrix_Layouts * matrix_layouts) {

	int ret;

	cublasStatus_t status;

	Matmul_Algo_Key * algo_key = matmul_params -> algo_key;
	Matmul_Runtime_Params * runtime_params = matmul_params -> runtime_params;

	int M = runtime_params -> M;
	int K = runtime_params -> K;
	int N = runtime_params -> N;

	if (algo_key -> to_trans_a){
		status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Adesc), a_cuda_dt, K, M, K);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: Adesc matmul layout could not be created\n");
			return -1;
		}
	}
	else{
		status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Adesc), a_cuda_dt, M, K, M);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: Adesc matmul layout could not be created\n");
			return -1;
		}
	}

	if (algo_key -> to_trans_b){
		// Now Bdesc actually referes the A matrix. b_cuda_dt has already been set appropriately
		status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Bdesc), b_cuda_dt, N, K, N);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: Bdesc matmul layout could not be created\n");
			return -1;
		}
	}
	else{
		// Now Bdesc actually referes the A matrix. b_cuda_dt has already been set appropriately
		status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Bdesc), b_cuda_dt, K, N, K);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: Bdesc matmul layout could not be created\n");
			return -1;
		}
	}
	

	status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Ddesc), d_cuda_dt, M, N, M);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: Cdesc matmul layout could not be created\n");
		return -1;
	}

	if ((algo_key -> c_dt == DATAFLOW_NONE) || (!runtime_params -> C) || (runtime_params -> beta_f == 0)){
		if ((algo_key -> a_dt == DATAFLOW_FP8E4M3) || (algo_key -> a_dt == DATAFLOW_FP8E5M2)){
			status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Cdesc), c_cuda_dt, M, N, M);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "Error: Ddesc matmul layout could not be created\n");
                return -1;
            }
		}
		else{
			// Cdesc is same as Ddesc
			status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Cdesc), d_cuda_dt, M, N, M);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "Error: Cdesc matmul layout could not be created\n");
				return -1;
			}
		}
	}
	else{
		status = cublasLtMatrixLayoutCreate(&(matrix_layouts -> Cdesc), c_cuda_dt, M, N, M);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: Ddesc matmul layout could not be created\n");
			return -1;
		}
	}

	return 0;
}

static int destroy_matrix_layouts(Cublas_Matrix_Layouts * matrix_layouts){

	int ret;

	cublasStatus_t status;

	status = cublasLtMatrixLayoutDestroy(matrix_layouts -> Adesc);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: could not destroy Adesc...\n");
		return -1;
	}

	status = cublasLtMatrixLayoutDestroy(matrix_layouts -> Bdesc);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: could not destroy Adesc...\n");
		return -1;
	}

	status = cublasLtMatrixLayoutDestroy(matrix_layouts -> Cdesc);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: could not destroy Ddesc...\n");
		return -1;
	}

	status = cublasLtMatrixLayoutDestroy(matrix_layouts -> Ddesc);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: could not destroy Adesc...\n");
		return -1;
	}


	return 0;
}

static int destroy_matmul_descriptor(Cublas_Matmul_Algo_Value * cublas_matmul_algo_value){

	int ret;

	cublasStatus_t status;

	status = cublasLtMatmulDescDestroy(cublas_matmul_algo_value -> computeDesc);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: could not destroy computeDesc...\n");
		return -1;
	}

	return 0;
}





// ASSUMING UNDERLYING STORAGE IS ROW-MAJOR for A, C, D
// AND COL-MAJOR for B!
static int set_cublas_matmul_params(Cublas_Matmul_Params * matmul_params, Op * op, cublasLtHandle_t cublas_handle){

	int ret;

	cublasStatus_t status;

	cudaDataType a_cuda_dt;
	cudaDataType b_cuda_dt;
	cudaDataType c_cuda_dt;
	cudaDataType d_cuda_dt;

	void ** op_args = op -> op_args;

	// INPUTS
	Matmul_Runtime_Params * runtime_params = matmul_params -> runtime_params;
	Matmul_Algo_Key * algo_key = matmul_params -> algo_key;

	// OUTPUTS
	Cublas_Matrix_Layouts * matrix_layouts = matmul_params -> matrix_layouts;
	Cublas_Matmul_Algo_Value * cublas_matmul_algo_value = matmul_params -> cublas_matmul_algo_value;
	

	int num_sms = algo_key -> num_sms;



	DataflowDatatype a_dt = algo_key -> a_dt;
	DataflowDatatype b_dt = algo_key -> b_dt;
	DataflowDatatype c_dt = algo_key -> c_dt;
	DataflowDatatype d_dt = algo_key -> d_dt;

	void * A = runtime_params -> A;
	void * B = runtime_params -> B;
	void * C = runtime_params -> C;
	void * D = runtime_params -> D;


	ret = set_cuda_dtypes(matmul_params, &a_cuda_dt, &b_cuda_dt, &c_cuda_dt, &d_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unable to set cuda dtypes...\n");
		return -1;
	}

	DataflowDatatype compute_dt = algo_key -> compute_dt;

	cudaDataType scale_cuda_dt;
	cublasComputeType_t cublas_compute_type;

	ret = set_cublas_compute_scale_types(&cublas_compute_type, &scale_cuda_dt, compute_dt, a_cuda_dt, b_cuda_dt, c_cuda_dt, d_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unable to get cublas compute type and scale type...\n");
		return -1;
	}

	memcpy(&(cublas_matmul_algo_value -> scale_cuda_dt), &scale_cuda_dt, sizeof(cudaDataType));

	status = cublasLtMatmulDescCreate(&(cublas_matmul_algo_value -> computeDesc), cublas_compute_type, scale_cuda_dt);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: cublaslt matmul desc could not be created...\n");
		return -1;
	}

	if (num_sms > 0){
		status = cublasLtMatmulDescSetAttribute(cublas_matmul_algo_value -> computeDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &num_sms, sizeof(num_sms));
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: matmul desc sm count attribute could not be set\n");
			return -1;
		}
	}

	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;

	int to_trans_a = algo_key -> to_trans_a;
	int to_trans_b = algo_key -> to_trans_b;

	if (to_trans_a){
		transa = CUBLAS_OP_T;
	}

	if (to_trans_b){
		transb = CUBLAS_OP_T;
	}

	status = cublasLtMatmulDescSetAttribute(cublas_matmul_algo_value -> computeDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: matmul desc attribute transa could not be set\n");
		return -1;
	}
	status = cublasLtMatmulDescSetAttribute(cublas_matmul_algo_value -> computeDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: matmul desc attribute transb could not be set\n");
		return -1;
	}


	int M = runtime_params -> M;
	int K = runtime_params -> K;
	int N = runtime_params -> N;


	ret = set_cublas_matrix_layouts(matmul_params, a_cuda_dt, b_cuda_dt, c_cuda_dt, d_cuda_dt,
											matrix_layouts);
	if (ret){
		fprintf(stderr, "Error: could not set cublas matmul desc params...\n");
		return -1;
	}

	

	uint64_t workspaceBytes = algo_key -> rounded_workspaceBytes;

	cublasLtMatmulPreference_t matmul_algo_pref;

	status = cublasLtMatmulPreferenceCreate(&(matmul_algo_pref));
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: matmul pref could not be created\n");
		return -1;
	}
	// Allowing just a small amount of workspace mem (2 MB) makes a big difference
	status = cublasLtMatmulPreferenceSetAttribute(matmul_algo_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: matmul pref attribute could not be set\n");
		return -1;
	}
		
	int algoCount = MAX_ALGO_SEARCH;
	int retAlgoCount = 0;

	cublasLtMatmulHeuristicResult_t heuristicResultsArray[MAX_ALGO_SEARCH];

	status = cublasLtMatmulAlgoGetHeuristic(cublas_handle, cublas_matmul_algo_value -> computeDesc, matrix_layouts -> Adesc, matrix_layouts -> Bdesc, matrix_layouts -> Cdesc, matrix_layouts -> Ddesc, matmul_algo_pref, algoCount, heuristicResultsArray, &retAlgoCount);
	
	if ((status != CUBLAS_STATUS_SUCCESS) || (retAlgoCount == 0)) {
		fprintf(stderr, "Error: could not get matmul algo heuristic: %s\n", cublasLtGetStatusString(status));
		status = cublasLtMatmulPreferenceDestroy(matmul_algo_pref);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy pref...\n");
		}
		return -1;
	}

	status = cublasLtMatmulPreferenceDestroy(matmul_algo_pref);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: could not destroy pref...\n");
		return -1;
	}

	// copy the best algo into the cublas matmul algo value
	memcpy(&(cublas_matmul_algo_value -> algo), &(heuristicResultsArray[0].algo), sizeof(cublasLtMatmulAlgo_t));

	return 0;
}





int cublas_matmul_init(Dataflow_Handle * dataflow_handle, void * op_table_value) {
	
	Cuda_Function * cuda_function = (Cuda_Function *) op_table_value;

	// allocate space for op extra
	Cublas_Matmul_Op_Extra * op_extra = malloc(sizeof(Cublas_Matmul_Op_Extra));
	if (!op_extra){
		fprintf(stderr, "Error: malloc failed to alloc space for matmul op extra...\n");
		return -1;
	}

	op_extra -> num_algos_saved = 0;
	op_extra -> num_matmuls_called = 0;
	op_extra -> num_algo_hits = 0;

	// initialize handle
	cublasStatus_t status = cublasLtCreate(&(op_extra -> cublas_handle));
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: cublaslt create failed within matmul init function...\n");
		return -1;
	}

	// initialize cublas matmul algo table
	Hash_Func hash_func = &matmul_algo_table_hash_func;
	uint64_t key_size_bytes = MATMUL_ALGO_KEY_FINGERPRINT_NUM_BYTES;
	uint64_t value_size_bytes = sizeof(Cublas_Matmul_Algo_Value);

	uint64_t min_table_size = MATMUL_ALGO_TABLE_MIN_SIZE;
	uint64_t max_table_size = MATMUL_ALGO_TABLE_MAX_SIZE;

	float load_factor = 0.25f;
	float shrink_factor = 0.1f;

	int ret = dataflow_init_table(&(op_extra -> cublas_matmul_algo_table), hash_func, key_size_bytes, value_size_bytes, min_table_size, max_table_size, load_factor, shrink_factor);
	if (ret){
		fprintf(stderr, "Error: could not initialize cublas matmul algo table...\n");
		return -1;
	}

	// set the op extra field
	cuda_function -> op_extra = (void *) op_extra;

	return 0;
}


void populate_matmul_algo_key(Matmul_Algo_Key * matmul_algo_key, Op * op){

	void ** op_args = op -> op_args;

	int num_sms = *((int *) op_args[0]);

	DataflowDatatype a_dt = *((DataflowDatatype *) op_args[1]);
	DataflowDatatype b_dt = *((DataflowDatatype *) op_args[2]);
	DataflowDatatype c_dt = *((DataflowDatatype *) op_args[3]);
	DataflowDatatype d_dt = *((DataflowDatatype *) op_args[4]);

	DataflowDatatype compute_dt = *((DataflowDatatype *) op_args[5]);

	int to_trans_a = *((int *) op_args[6]);
	int to_trans_b = *((int *) op_args[7]);

	int M = *((int *) op_args[8]);
	int K = *((int *) op_args[9]);
	int N = *((int *) op_args[10]);

	uint64_t workspaceBytes = *((uint64_t *) op_args[17]);

	matmul_algo_key -> num_sms = num_sms;
	matmul_algo_key -> rounded_M = MATMUL_DIM_ROUND_UP(M);
	matmul_algo_key -> rounded_N = MATMUL_DIM_ROUND_UP(N);
	matmul_algo_key -> rounded_K = MATMUL_DIM_ROUND_UP(K);
	matmul_algo_key -> a_dt = a_dt;
	matmul_algo_key -> b_dt = b_dt;
	matmul_algo_key -> c_dt = c_dt;
	matmul_algo_key -> d_dt = d_dt;
	matmul_algo_key -> compute_dt = compute_dt;
	matmul_algo_key -> to_trans_a = to_trans_a;
	matmul_algo_key -> to_trans_b = to_trans_b;
	matmul_algo_key -> rounded_workspaceBytes = MATMUL_WORKSPACE_ROUND_DOWN(workspaceBytes);
	
	return;
}

void populate_matmul_runtime_params(Matmul_Runtime_Params * matmul_runtime_params, Op * op){

	void ** op_args = op -> op_args;

	matmul_runtime_params -> M = *((int *) op_args[8]);
	matmul_runtime_params -> K = *((int *) op_args[9]);
	matmul_runtime_params -> N = *((int *) op_args[10]);
	matmul_runtime_params -> workspaceBytes = *((uint64_t *) op_args[17]);

	matmul_runtime_params -> alpha_f = *((float *) op_args[11]);
	matmul_runtime_params -> beta_f = *((float *) op_args[12]);

	matmul_runtime_params -> alpha_h = solo_fp32_to_fp16(matmul_runtime_params -> alpha_f);
	matmul_runtime_params -> beta_h = solo_fp32_to_fp16(matmul_runtime_params -> beta_f);

	matmul_runtime_params -> alpha_d = (double) matmul_runtime_params -> alpha_f;
	matmul_runtime_params -> beta_d = (double) matmul_runtime_params -> beta_f;
	
	matmul_runtime_params -> A = (void *) *((uint64_t *) op_args[13]);
	matmul_runtime_params -> B = (void *) *((uint64_t *) op_args[14]);
	matmul_runtime_params -> C = (void *) *((uint64_t *) op_args[15]);
	matmul_runtime_params -> D = (void *) *((uint64_t *) op_args[16]);

	matmul_runtime_params -> workspace = (void *) *((uint64_t *) op_args[18]);

	return;
}

int set_scale_refs(Matmul_Runtime_Params * matmul_runtime_params, cudaDataType scale_cuda_dt, void ** alpha, void ** beta){

	if (scale_cuda_dt == CUDA_R_16F){
		*alpha = &(matmul_runtime_params -> alpha_h);
		*beta = &(matmul_runtime_params -> beta_h);
	}
	else if (scale_cuda_dt == CUDA_R_32F){
		*alpha = &(matmul_runtime_params -> alpha_f);
		*beta = &(matmul_runtime_params -> beta_f);
	}
	else if (scale_cuda_dt == CUDA_R_64F){
		*alpha = &(matmul_runtime_params -> alpha_d);
		*beta = &(matmul_runtime_params -> beta_d);
	}
	else{
		fprintf(stderr, "Error: unsupported scale dtype of %d\n", scale_cuda_dt);
		return -1;
	}

	return 0;
}

int cublas_matmul(Dataflow_Handle * dataflow_handle, int stream_id, Op * op, void * op_extra){

	int ret;

	cublasStatus_t status;

	// cast the op extra and retrieve cublas handle
	Cublas_Matmul_Op_Extra * matmul_op_extra = (Cublas_Matmul_Op_Extra *) op_extra;
	
	if (!matmul_op_extra){
		fprintf(stderr, "Error: matmul op extra is NULL, but need to obtain cublas handle...\n");
		return -1;
	}

	if (stream_id >= dataflow_handle -> num_streams){
		fprintf(stderr, "Error: trying to dispatch matmul on stream id %d, but only have %d streams...\n", stream_id, dataflow_handle -> num_streams);
		return -1;
	}

	CUstream * streams = (CUstream *) (dataflow_handle -> streams);

	CUstream stream = streams[stream_id];

	cublasLtHandle_t cublas_handle = matmul_op_extra -> cublas_handle;

	Dataflow_Table * cublas_matmul_algo_table = &(matmul_op_extra -> cublas_matmul_algo_table);

	Matmul_Runtime_Params matmul_runtime_params;
	populate_matmul_runtime_params(&matmul_runtime_params, op);

	Matmul_Algo_Key matmul_algo_key;
	populate_matmul_algo_key(&matmul_algo_key, op);

	// First check if we have an existing cublas matmul algo key
	uint8_t fingerprint[MATMUL_ALGO_KEY_FINGERPRINT_NUM_BYTES];
	dataflow_do_fingerprinting(&matmul_algo_key, sizeof(Matmul_Algo_Key), fingerprint);

	// check if we can skip building descriptors and importantly skip the heuristic search...
	Cublas_Matmul_Algo_Value * cublas_matmul_algo_value = NULL;

	// Will populate these every time 
	// (can optimize later, but rather keep table smaller and the creation/delettion of these is quick...)
	Cublas_Matrix_Layouts matrix_layouts;

	long table_ind = dataflow_find_table(cublas_matmul_algo_table, fingerprint, false, (void **) &cublas_matmul_algo_value);

	// Either it is already in the table or we will insert it...
	// only sets to 0 if insertion fails (table is full)
	int in_table = 1;

	int algo_cache_hit = 0;

	// we didn't find a match, so now we need to search and populate the cublas matmul algo value
	if (!cublas_matmul_algo_value || (table_ind == -1)){

		// can use temporary memory here because we do a copy into the table...
		Cublas_Matmul_Algo_Value new_cublas_matmul_algo_value;

		Cublas_Matmul_Params matmul_params;

		matmul_params.runtime_params = &matmul_runtime_params;
		matmul_params.algo_key = &matmul_algo_key;
		matmul_params.cublas_matmul_algo_value = &new_cublas_matmul_algo_value;
		matmul_params.matrix_layouts = &matrix_layouts;
		
		ret = set_cublas_matmul_params(&matmul_params, op, cublas_handle);
		if (ret){
			fprintf(stderr, "Error: unable to set cublas matmul params...\n");
			return -1;
		}
		

		// now insert the value into the table...
		ret = dataflow_insert_table(cublas_matmul_algo_table, &fingerprint, &new_cublas_matmul_algo_value);
		if (ret){
			fprintf(stderr, "Error: failed to insert cublas matmul algo value to cublas matmul algo table...\n");
			in_table = 0;
		}

		matmul_op_extra -> num_algos_saved += 1;
		// reset the cublas matmul algo value pointer
		cublas_matmul_algo_value = &new_cublas_matmul_algo_value;
	}
	// if we didn't search for an algo, 
	// then we need to create new layout descriptors but can use the same algo...
	else {
		algo_cache_hit = 1;
		Cublas_Matmul_Params matmul_params;

		matmul_params.runtime_params = &matmul_runtime_params;
		matmul_params.algo_key = &matmul_algo_key;

		cudaDataType a_cuda_dt;
		cudaDataType b_cuda_dt;
		cudaDataType c_cuda_dt;
		cudaDataType d_cuda_dt;

		ret = set_cuda_dtypes(&matmul_params, &a_cuda_dt, &b_cuda_dt, &c_cuda_dt, &d_cuda_dt);
		if (ret){
			fprintf(stderr, "Error: unable to set cuda dtypes...\n");
			return -1;
		}

		ret = set_cublas_matrix_layouts(&matmul_params, a_cuda_dt, b_cuda_dt, c_cuda_dt, d_cuda_dt,
											&matrix_layouts);
		if (ret){
			fprintf(stderr, "Error: unable to set cublas matrix layouts...\n");
			return -1;
		}
	}

	void * alpha;
	void * beta;

	ret = set_scale_refs(&matmul_runtime_params, cublas_matmul_algo_value -> scale_cuda_dt, &alpha, &beta);
	if (ret){
		fprintf(stderr, "Error: unable to set scale refs...\n");
		return -1;
	}


	status = cublasLtMatmul(cublas_handle,
							cublas_matmul_algo_value -> computeDesc,
							alpha,
							matmul_runtime_params.A,
							matrix_layouts.Adesc,
							matmul_runtime_params.B,
							matrix_layouts.Bdesc,
							beta,
							matmul_runtime_params.C,
							matrix_layouts.Cdesc,
							matmul_runtime_params.D,
							matrix_layouts.Ddesc,
							&(cublas_matmul_algo_value -> algo),
							matmul_runtime_params.workspace,
							matmul_algo_key.rounded_workspaceBytes,
							stream);

	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: cublasLtMatmul failed...\n");
		
		// remove this key from the table and destroy the cublas matmul algo value
		if (in_table){
			ret = dataflow_remove_table(cublas_matmul_algo_table, fingerprint, NULL);
			if (ret){
				fprintf(stderr, "Error: failed to remove cublas matmul algo value from cublas matmul algo table...\n");
			}
		}
		
		ret = destroy_matmul_descriptor(cublas_matmul_algo_value);
		if (ret){
			fprintf(stderr, "Error: failed to destroy cublas matmul algo value...\n");
		}

		// destroy the matrix layouts
		ret = destroy_matrix_layouts(&matrix_layouts);
		if (ret){
			fprintf(stderr, "Error: failed to destroy matrix layouts...\n");
		}

		return -1;
	}

	// destroy the matrix layouts
	ret = destroy_matrix_layouts(&matrix_layouts);
	if (ret){
		fprintf(stderr, "Error: failed to destroy matrix layouts...\n");
	}

	matmul_op_extra -> num_matmuls_called += 1;
	matmul_op_extra -> num_algo_hits += algo_cache_hit;
	return 0;
}
