#include "register_ops.h"

// Define OPS_ROOT_DIR if it's not already defined
#ifndef OPS_ROOT_DIR
#define OPS_ROOT_DIR "."
#endif

int dataflow_register_external_ops(Dataflow_Handle * dataflow_handle) {

    int added_funcs;

	Op_Skeleton matmul_skeletons[1];
	dataflow_set_matmul_skeleton(&matmul_skeletons[0]);

    char matmul_lib[PATH_MAX];

	sprintf(matmul_lib, "%s/lib/external/libmatmulwrapper.so", (const char *) OPS_ROOT_DIR);
	
	char * matmul_symbols[1] = {"cublas_matmul"};

	char * matmul_init_symbols[1] = {"cublas_matmul_init"};

    int total_added_funcs = 0;

	added_funcs = (dataflow_handle -> register_external_code)(dataflow_handle, matmul_lib, 1, matmul_skeletons, matmul_symbols, matmul_init_symbols);
	
    if (added_funcs != 1){
		fprintf(stderr, "Error: failed to register matmul op, expected 1 functions, got %d...\n", added_funcs);
		return -1;
	}

    total_added_funcs += added_funcs;

	// Register flash attention op

	Op_Skeleton flash_attention_skeletons[2];
	dataflow_set_flash3_attention_fwd_skeleton(&flash_attention_skeletons[0]);
	dataflow_set_flash3_attention_bwd_skeleton(&flash_attention_skeletons[1]);

	char flash_attention_lib[PATH_MAX];

	sprintf(flash_attention_lib, "%s/lib/external/libattentionwrapper.so", (const char *) OPS_ROOT_DIR);
	
	char * flash_attention_symbols[2] = {"flash3_attention_fwd", "flash3_attention_bwd"};

	char * flash_attention_init_symbols[2] = {NULL, NULL};

	added_funcs = (dataflow_handle -> register_external_code)(dataflow_handle, flash_attention_lib, 2, flash_attention_skeletons, flash_attention_symbols, flash_attention_init_symbols);
	if (added_funcs != 2){
		fprintf(stderr, "Error: failed to register flash attention op, expected 2 functions, got %d...\n", added_funcs);
		return -1;
	}

    total_added_funcs += added_funcs;

    return total_added_funcs;
}


// This function should just be called during the registration of the op
// Not performance senesitive so the strcmps are fine...
int dataflow_register_native_ops(Dataflow_Handle * dataflow_handle) {

    char native_function_code_filename[PATH_MAX];
    char native_function_config_filename[PATH_MAX];

    sprintf(native_function_code_filename, "%s/lib/native/cuda_kernels.cubin", (const char *) OPS_ROOT_DIR);
    sprintf(native_function_config_filename, "%s/lib/native/cuda_kernels_config.so", (const char *) OPS_ROOT_DIR);

	int num_fwd_datatypes = 5;
	int num_bwd_datatypes = 3;

	DataflowDatatype fwd_datatypes[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2};
	DataflowDatatype bwd_datatypes[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16};

	int num_base_ops = 7;
	char * op_base_names[7] = {"embedding_table", "rms_norm", "rms_norm_noscale", "rope", "swiglu", "softmax", "cross_entropy_loss"};

	char * op_init_symbols[7] = {NULL, "rms_norm_set_attribute_config", "rms_norm_set_attribute_config", NULL, NULL, NULL, NULL};
	
	
	// cross entropy loss doesn't have function for fp8 yet...
	bool num_fwd_ops[7] = {5, 5, 5, 5, 5, 7, 3};
	bool num_bwd_ops[7] = {0, 14, 7, 3, 7, 0, 0};

	int num_funcs = 66;

	bool has_bwd_x[7] = {false, true, true, true, true, false, false};
	bool has_bwd_w[7] = {false, true, false, false, false, false, false};

	int bwd_combos = 7;

	char * fwd_strs[] = {"fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2"};
	char * bwd_strs[] = {"fp32", "fp16", "bf16"};
	char * bwd_combo_strs[] = {"fp32_fp32", "fp16_fp16", "bf16_bf16", \
								"fp8e4m3_fp16", "fp8e4m3_bf16", "fp8e5m2_fp16", "fp8e5m2_bf16"};


	DataflowDatatype bwd_combo_fwd_dts[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, 
											DATAFLOW_FP8E4M3, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2, DATAFLOW_FP8E5M2};
	DataflowDatatype bwd_combo_bwd_dts[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, 
											DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP16, DATAFLOW_BF16};

	char * suffix = "kernel";

	Op_Skeleton * native_op_skeletons = (Op_Skeleton *) malloc(num_funcs * sizeof(Op_Skeleton));

	char ** native_func_symbols = (char **) malloc(num_funcs * sizeof(char *));
	char ** native_func_launch_symbols = (char **) malloc(num_funcs * sizeof(char *));
	for (int i = 0; i < num_funcs; i++){
		// must 
		native_func_symbols[i] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
		native_func_launch_symbols[i] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
	}
	char ** native_func_init_symbols = (char **) malloc(num_funcs * sizeof(char *));
	for (int i = 0; i < num_funcs; i++){
		native_func_init_symbols[i] = NULL;
	}
	
	int cur_func = 0;

	char op_base_bwd_extented[FUNC_SYMBOL_MAX_LEN];

	for (int i = 0; i < num_base_ops; i++){
		// all ops have fwd

		

		if ((strcmp(op_base_names[i], "softmax") != 0) && (strcmp(op_base_names[i], "cross_entropy_loss") != 0)) {
			for (int s = 0; s < num_fwd_datatypes; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_names[i], fwd_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_names[i]);

				dataflow_set_op_skeleton(&native_op_skeletons[cur_func], op_base_names[i], fwd_datatypes[s], DATAFLOW_NONE);
				cur_func++;
			}
		}
		// special cases for softmax and cross entropy loss

		// softmax transitions from fwd dt to bwd and takes both (all combos)
		if (strcmp(op_base_names[i], "softmax") == 0) {
			for (int s = 0; s < bwd_combos; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_names[i], bwd_combo_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_names[i]);
				dataflow_set_op_skeleton(&native_op_skeletons[cur_func], op_base_names[i], bwd_combo_fwd_dts[s], bwd_combo_bwd_dts[s]);
				cur_func++;
			}
		}

		// only do cross entropy loss for fp32, fp16, bf16
		// only the bwd_dt is used for cross entropy loss
		if (strcmp(op_base_names[i], "cross_entropy_loss") == 0) {
			for (int s = 0; s < num_bwd_datatypes; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_names[i], bwd_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_names[i]);

				dataflow_set_op_skeleton(&native_op_skeletons[cur_func], op_base_names[i], DATAFLOW_NONE, bwd_datatypes[s]);
				cur_func++;
			}
		}

		if (has_bwd_x[i]){
			sprintf(op_base_bwd_extented, "%s_bwd_x", op_base_names[i]);

			// rope_bwd_x only takes in the bwd_dt...
			if (strcmp(op_base_bwd_extented, "rope_bwd_x") != 0){

				for (int s = 0; s < bwd_combos; s++){
					if (op_init_symbols[i]){
						native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
						sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
					}
					sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_bwd_extented, bwd_combo_strs[s], suffix);
					sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_bwd_extented);
					dataflow_set_op_skeleton(&native_op_skeletons[cur_func], op_base_bwd_extented, bwd_combo_fwd_dts[s], bwd_combo_bwd_dts[s]);
					cur_func++;
				}
			}
			else {
				for (int s = 0; s < num_bwd_datatypes; s++){
					if (op_init_symbols[i]){
						native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
						sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
					}
					sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_bwd_extented, bwd_strs[s], suffix);
					sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_bwd_extented);
					dataflow_set_op_skeleton(&native_op_skeletons[cur_func], op_base_bwd_extented, DATAFLOW_NONE, bwd_datatypes[s]);
					cur_func++;
				}
			}
		}
		

		if (has_bwd_w[i]){
			sprintf(op_base_bwd_extented, "%s_bwd_w", op_base_names[i]);
			for (int s = 0; s < bwd_combos; s++){
				if (op_init_symbols[i]){
					native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
					sprintf(native_func_init_symbols[cur_func], "%s", op_init_symbols[i]);
				}
				sprintf(native_func_symbols[cur_func], "%s_%s_%s", op_base_bwd_extented, bwd_combo_strs[s], suffix);
				sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", op_base_bwd_extented);
				dataflow_set_op_skeleton(&native_op_skeletons[cur_func], op_base_bwd_extented, bwd_combo_fwd_dts[s], bwd_combo_bwd_dts[s]);
				cur_func++;
			}
		}
	}

	// Now finally register all of the functions...

	int added_funcs = (dataflow_handle -> register_native_code)(dataflow_handle, native_function_code_filename, native_function_config_filename, 
																num_funcs, native_op_skeletons, 
																native_func_symbols, native_func_launch_symbols, native_func_init_symbols);
	if (added_funcs != num_funcs){
		fprintf(stderr, "WARNING: failed to register all, native ops, expected %d functions, got %d...\n", num_funcs, added_funcs);
	}

	// Now can free the metadata used to register...

	for (int i = 0; i < num_funcs; i++){
		free(native_func_symbols[i]);
		free(native_func_launch_symbols[i]);
		if (native_func_init_symbols[i]){
			free(native_func_init_symbols[i]);
		}
	}

	free(native_func_symbols);
	free(native_func_launch_symbols);
	free(native_func_init_symbols);
	free(native_op_skeletons);

	return added_funcs;
}


int dataflow_register_default_ops(Dataflow_Handle * dataflow_handle) {

    int total_added_funcs = 0;

    total_added_funcs += dataflow_register_external_ops(dataflow_handle);

    total_added_funcs += dataflow_register_native_ops(dataflow_handle);

    return total_added_funcs;
}