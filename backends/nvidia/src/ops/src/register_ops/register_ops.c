#include "register_ops.h"

// Define OPS_ROOT_DIR if it's not already defined
#ifndef OPS_ROOT_DIR
#define OPS_ROOT_DIR "."
#endif

typedef void (*opt_base_register_skeleton_func)(Op_Skeleton * skeleton, DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt);

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

	Op_Skeleton flash_attention_skeletons[3];
	dataflow_set_flash_attention_fwd_skeleton(&flash_attention_skeletons[0]);
	dataflow_set_flash_attention_bwd_skeleton(&flash_attention_skeletons[1]);
	dataflow_set_flash_attention_get_workspace_size_skeleton(&flash_attention_skeletons[2]);

	char flash_attention_lib[PATH_MAX];

	sprintf(flash_attention_lib, "%s/lib/external/libattentionwrapper.so", (const char *) OPS_ROOT_DIR);
	
	char * flash_attention_symbols[3] = {"flash_attention_fwd", "flash_attention_bwd", "flash_attention_get_workspace_size"};

	char * flash_attention_init_symbols[3] = {NULL, NULL, NULL};

	added_funcs = (dataflow_handle -> register_external_code)(dataflow_handle, flash_attention_lib, 3, flash_attention_skeletons, flash_attention_symbols, flash_attention_init_symbols);
	if (added_funcs != 3){
		fprintf(stderr, "Error: failed to register flash attention op, expected 3 functions, got %d...\n", added_funcs);
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

    Cuda_Device_Info * dev_info = (Cuda_Device_Info *) dataflow_handle -> device_info;

    int arch_num = dev_info -> arch_num;

    sprintf(native_function_code_filename, "%s/lib/native/cuda_kernels_%d.cubin", (const char *) OPS_ROOT_DIR, arch_num);
    sprintf(native_function_config_filename, "%s/lib/native/cuda_kernels_config.so", (const char *) OPS_ROOT_DIR);

	int num_fwd_datatypes = 5;
	int num_bwd_datatypes = 3;

	DataflowDatatype fwd_datatypes[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16, DATAFLOW_FP8E4M3, DATAFLOW_FP8E5M2};
	DataflowDatatype bwd_datatypes[] = {DATAFLOW_FP32, DATAFLOW_FP16, DATAFLOW_BF16};

	int num_base_ops = 9;
	char * op_base_names[9] = {"default_embedding_table", "default_rms_norm", "default_rms_norm_recompute", "default_rms_norm_noscale", "default_rope", "default_select_experts", "default_swiglu", "default_softmax", "default_cross_entropy_loss"};

	char * op_init_symbols[9] = {"default_embedding_table_set_attribute_config", "default_rms_norm_set_attribute_config", "default_rms_norm_set_attribute_config", "default_rms_norm_set_attribute_config", NULL, "default_select_experts_set_attribute_config", NULL, NULL, NULL};
	
	
	// cross entropy loss doesn't have function for fp8 yet...
	bool num_fwd_ops[9] = {5, 5, 5, 5, 5, 5, 5, 7, 3};
	bool num_bwd_ops[9] = {3, 17, 0, 7, 3, 0, 7, 0, 0};

	int num_base_funcs = 82;

	bool has_bwd_x[9] = {false, true, false, true, true, false, true, false, false};
	bool has_bwd_w[9] = {true, true, false, false, false, false, false, false, false};

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

	int num_misc_funcs = 1;
	char * misc_func_names[] = {"default_set_average_loss"};
	char * misc_func_init_symbols[] = {NULL};


	int num_opt_base_funcs = 1;
	char * opt_base_names[] = {"default_adamw_step"};
	opt_base_register_skeleton_func opt_base_register_skeleton_funcs[] = {dataflow_set_default_adamw_step_skeleton};
	char * opt_init_symbols[] = {NULL};

	int num_opt_funcs_per_base[] = {1};

	// [base_funcs][opt_funcs_per_base[i]][4]
	DataflowDatatype opt_dt_combos[1][1][4] = {{{DATAFLOW_BF16, DATAFLOW_BF16, DATAFLOW_BF16, DATAFLOW_BF16}}};

	// [base_funcs][opt_funcs_per_base[i]]
	char * opt_dt_str_combos[1][1] = {{"bf16_bf16_bf16_bf16"}};



	

	int total_opt_funcs = 0;

	for (int i = 0; i < num_opt_base_funcs; i++){
		total_opt_funcs += num_opt_funcs_per_base[i];
	}

	char * opt_suffix = "kernel";



	int num_funcs = num_base_funcs + num_misc_funcs + total_opt_funcs;





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

		if ((strcmp(op_base_names[i], "default_softmax") != 0) && (strcmp(op_base_names[i], "default_cross_entropy_loss") != 0)) {
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
		if (strcmp(op_base_names[i], "default_softmax") == 0) {
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
		if (strcmp(op_base_names[i], "default_cross_entropy_loss") == 0) {
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
			if (strcmp(op_base_bwd_extented, "default_rope_bwd_x") != 0){

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
			if (strcmp(op_base_bwd_extented, "default_embedding_table_bwd_w") != 0){
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

			// rms_norm_bwd_w_combine only takes in the bwd_dt...
			if (strcmp(op_base_bwd_extented, "default_rms_norm_bwd_w") == 0){
				char new_op_base_bwd_extented[PATH_MAX];
				sprintf(new_op_base_bwd_extented, "%s_combine", op_base_bwd_extented);
				for (int s = 0; s < num_bwd_datatypes; s++){
					sprintf(native_func_symbols[cur_func], "%s_%s_%s", new_op_base_bwd_extented, bwd_strs[s], suffix);
					sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", new_op_base_bwd_extented);	
					dataflow_set_op_skeleton(&native_op_skeletons[cur_func], new_op_base_bwd_extented, DATAFLOW_NONE, bwd_datatypes[s]);
					cur_func++;
				}
			}
		}
	}

	for (int i = 0; i < num_misc_funcs; i++){
		sprintf(native_func_symbols[cur_func], "%s_%s", misc_func_names[i], suffix);
		sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", misc_func_names[i]);
		dataflow_set_op_skeleton(&native_op_skeletons[cur_func], misc_func_names[i], DATAFLOW_NONE, DATAFLOW_NONE);
		cur_func++;
	}

	for (int i = 0; i < num_opt_base_funcs; i++){
		
		for (int j = 0; j < num_opt_funcs_per_base[i]; j++){
			if (opt_init_symbols[i]){
				native_func_init_symbols[cur_func] = calloc(FUNC_SYMBOL_MAX_LEN, sizeof(char));
				sprintf(native_func_init_symbols[cur_func], "%s", opt_init_symbols[i]);
			}
			sprintf(native_func_symbols[cur_func], "%s_%s_%s", opt_base_names[i], opt_dt_str_combos[i][j], opt_suffix);
			sprintf(native_func_launch_symbols[cur_func], "%s_set_launch_config", opt_base_names[i]);
			opt_base_register_skeleton_funcs[i](&native_op_skeletons[cur_func], opt_dt_combos[i][j][0], opt_dt_combos[i][j][1], opt_dt_combos[i][j][2], opt_dt_combos[i][j][3]);
			cur_func++;
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
