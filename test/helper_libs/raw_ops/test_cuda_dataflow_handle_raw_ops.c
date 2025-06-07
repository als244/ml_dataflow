#include "cuda_dataflow_handle.h"
#include "create_host_matrix.h"
#include "dataflow_ops.h"

int main(int argc, char * argv[]){
	
	int ret;

	Dataflow_Handle cuda_dataflow_handle;
	
	ComputeType compute_type = COMPUTE_CUDA;
	int device_id = 0;

	// In case we want to create multiple contexts per device, 
	// higher level can create multiple instances of dataflow handles...
	int ctx_id = 0;
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;

	int num_streams = 8;
	int opt_stream_prios[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	char * opt_stream_names[8] = {"Inbound (a)", "Compute (a)", "Outbound (a)", "Peer (a)", "Inbound (b)", "Compute (b)", "Outbound (b)", "Peer (b)"};
	
	// char * all_function_meta_filename = "../../../ops/nvidia/lib/cuda_all_functions_meta.dat";
	// char * native_function_config_filename = "../../../ops/nvidia/lib/cuda_kernels_config.so";
	// char * native_function_lib_filename = "../../../ops/nvidia/lib/cuda_kernels.cubin";

	// ret = init_cuda_dataflow_handle(&cuda_dataflow_handle, compute_type, device_id, 
	// 		ctx_id, ctx_flags, 
	// 		num_streams, opt_stream_prios, opt_stream_names, 
	// 		all_function_meta_filename, native_function_config_filename, native_function_lib_filename); 

	ret = init_cuda_dataflow_handle(&cuda_dataflow_handle, compute_type, device_id, 
			ctx_id, ctx_flags, 
			num_streams, opt_stream_prios, opt_stream_names); 
	
	if (ret){
		fprintf(stderr, "Error: failed to init cuda dataflow handle...\n");
		return -1;
	}

	int alignment = 4096;

	void * host_mem;

	// 8 GB...
	size_t host_size_bytes = 1UL << 33;

	printf("Allocating host memory of size: %lu...\n", host_size_bytes);

	ret = posix_memalign(&host_mem, alignment, host_size_bytes);
	if (ret){
		fprintf(stderr, "Error: posix memalign failed...\n");
		return -1;
	}


	printf("Registering host memory...\n");

	ret = cuda_dataflow_handle.enable_access_to_host_mem(&cuda_dataflow_handle, host_mem, host_size_bytes, 0);
	if (ret){
		fprintf(stderr, "Registration of host memory failed...\n");
		return -1;
	}

	

	// Seed the random number generator with a constant value
    srand(42);

	uint64_t M = 16384;
	uint64_t N = 8192;


	int iM = (int) M;
	int iN = (int) N;

	float mean = 0.0;
	float std = 0.006;


	float eps = 1e-5;

	DataflowDatatype fwd_dt = DATAFLOW_FP16;
	DataflowDatatype bwd_dt = DATAFLOW_FP16;

	size_t el_size = dataflow_sizeof_element(fwd_dt);
	size_t bwd_el_size = dataflow_sizeof_element(fwd_dt);
	uint64_t mat_size = M * N * el_size;
	uint64_t bwd_mat_size = M * N * bwd_el_size;

	void * orig_matrix = host_mem;
	void * out_matrix = orig_matrix + mat_size;
	void * rms_weight = out_matrix + mat_size;
	void * weighted_sums = rms_weight + (el_size * N);
	void * rms_vals = weighted_sums + (sizeof(float) * M);
	void * upstream_dX = rms_vals + (sizeof(float) * M);
	void * dX = upstream_dX + bwd_mat_size;
	void * dW = dX + bwd_mat_size;


	printf("Creating random host matrix (M: %lu, N; %lu, dt: %s)...\n", M, N, dataflow_datatype_as_string(fwd_dt));

	void * res = create_rand_host_matrix(M, N, mean, std, fwd_dt, orig_matrix);
	if (!res){
		fprintf(stderr, "Error: creating random host memory matrix failed...\n");
		return -1;
	}

	res = create_rand_host_matrix(N, 1, mean, std, fwd_dt, rms_weight);
	if (!res){
		fprintf(stderr, "Error: creating random host memory matrix failed...\n");
		return -1;
	}

	res = create_rand_host_matrix(M, N, 1.0, 0, bwd_dt, upstream_dX);
	if (!res){
		fprintf(stderr, "Error: creating random host memory matrix failed...\n");
		return -1;
	}

	printf("Saving orig matrix...\n");

	char * orig_matrix_filename = "test_rms/orig_matrix.dat";
	char * rms_weight_filename = "test_rms/weights.dat";
	char * upstream_dX_filename = "test_rms/upstream_dX.dat";

	ret = save_host_matrix(orig_matrix_filename, orig_matrix, M, N, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save original matrix...\n");
		return -1;
	}

	ret = save_host_matrix(rms_weight_filename, rms_weight, N, 1, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save original matrix...\n");
		return -1;
	}

	ret = save_host_matrix(upstream_dX_filename, upstream_dX, M, N, bwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save original matrix...\n");
		return -1;
	}

	// 8 GB...	
	size_t dev_size_bytes = 1UL << 33;

	printf("Allocating device memory of size: %lu...\n", dev_size_bytes);


	void * dev_mem = cuda_dataflow_handle.alloc_mem(&cuda_dataflow_handle, dev_size_bytes);
	if (!dev_mem){
		fprintf(stderr, "Error: device memory allocation failed...\n");
		return -1;
	}

	void * d_orig_matrix = dev_mem;
	void * d_out_matrix = d_orig_matrix + mat_size;
	void * d_rms_weight = d_out_matrix + mat_size;
	void * d_weighted_sums = d_rms_weight + (el_size * N);
	void * d_rms_vals = d_weighted_sums + (sizeof(float) * M);
	void * d_upstream_dX = d_rms_vals + (sizeof(float) * M);
	void * d_dX = d_upstream_dX + bwd_mat_size;
	void * d_dW = d_dX + bwd_mat_size;


	printf("Transferring matrix on host to device of size: %lu...\n", mat_size);

	int inbound_stream_id_a = 0;
	int compute_stream_id_a = 1;
	int outbound_stream_id_a = 2;
	int peer_stream_id_a = 3;
	int inbound_stream_id_b = 4;
	int compute_stream_id_b = 5;
	int outbound_stream_id_b = 6;
	int peer_stream_id_b = 7;

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_orig_matrix, orig_matrix, mat_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_rms_weight, rms_weight, el_size * N);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_inbound_transfer(&cuda_dataflow_handle, inbound_stream_id_a, d_upstream_dX, upstream_dX, bwd_mat_size);
	if (ret){
		fprintf(stderr, "Error: host to device transfer failed...\n");
		return -1;
	}

	printf("Syncing with device after transfer...\n");

	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, inbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer...\n");
		return -1;
	}


	printf("Submitting RMS norm op...!\n");
	
	/*
	ret = submit_rms_norm(handle, stream_id, fwd_dt, n_rows, n_cols, eps, rms_weight, d_orig_matrix, d_out_matrix, d_weighted_sums, d_rms_vals);
	if (ret){
		fprintf(stderr, "Error: could not submit rms nrom...\n");
		return -1;
	}
	*/

	Op rms_norm_op;

	set_native_rms_norm_skeleton(&rms_norm_op.op_skeleton, fwd_dt);

	void ** fwd_op_args = rms_norm_op.op_args;

	fwd_op_args[0] = &iM;
	fwd_op_args[1] = &iN;
	fwd_op_args[2] = &eps;
	fwd_op_args[3] = &d_rms_weight;
	fwd_op_args[4] = &d_orig_matrix;
	fwd_op_args[5] = &d_out_matrix;
	fwd_op_args[6] = &d_weighted_sums;
	fwd_op_args[7] = &d_rms_vals;


	ret = cuda_dataflow_handle.submit_op(&cuda_dataflow_handle, &rms_norm_op, compute_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to submit op...\n");
		return -1;
	}


	printf("Submitting dependency for outbound transfer...\n");

	void * compute_stream_state = cuda_dataflow_handle.get_stream_state(&cuda_dataflow_handle, compute_stream_id_a);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_dependency(&cuda_dataflow_handle, outbound_stream_id_a, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	printf("Submitting outbound transfer...\n");

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, out_matrix, d_out_matrix, mat_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, weighted_sums, d_weighted_sums, M * el_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, rms_vals, d_rms_vals, M * el_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	printf("Syncing with outbound transfer...\n");


	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}


	printf("Saving transformed matrix...\n");

	char * out_matrix_filename = "test_rms/fwd_out_matrix.dat";
	char * weighted_sums_filename = "test_rms/weighted_sums.dat";
	char * rms_vals_filename = "test_rms/rms_vals.dat";

	ret = save_host_matrix(out_matrix_filename, out_matrix, M, N, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save output matrix...\n");
		return -1;
	}

	printf("Saving extra data returned from op...\n");

	ret = save_host_matrix(weighted_sums_filename, weighted_sums, M, 1, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save sq sums matrix...\n");
		return -1;
	}

	ret = save_host_matrix(rms_vals_filename, rms_vals, M, 1, DATAFLOW_FP32);
	if (ret){
		fprintf(stderr, "Error: failed to save sq sums matrix...\n");
		return -1;
	}


	printf("Doing BWD X op...\n");
	Op rms_norm_bwd_x_op;

	set_native_rms_norm_bwd_x_skeleton(&rms_norm_bwd_x_op.op_skeleton, fwd_dt, bwd_dt);

	void ** bwd_x_op_args = rms_norm_bwd_x_op.op_args;

	bwd_x_op_args[0] = &iM;
	bwd_x_op_args[1] = &iN;
	bwd_x_op_args[2] = &eps;
	bwd_x_op_args[3] = &d_weighted_sums;
	bwd_x_op_args[4] = &d_rms_vals;
	bwd_x_op_args[5] = &d_rms_weight;
	bwd_x_op_args[6] = &d_orig_matrix;
	bwd_x_op_args[7] = &d_upstream_dX;
	bwd_x_op_args[8] = &d_dX;


	ret = cuda_dataflow_handle.submit_op(&cuda_dataflow_handle, &rms_norm_bwd_x_op, compute_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to submit op...\n");
		return -1;
	}

	printf("Submitting dependency for outbound BWD X transfer...\n");

	compute_stream_state = cuda_dataflow_handle.get_stream_state(&cuda_dataflow_handle, compute_stream_id_a);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_dependency(&cuda_dataflow_handle, outbound_stream_id_a, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	printf("Submitting outbound BWD X transfer...\n");

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, dX, d_dX, bwd_mat_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}

	printf("Syncing outbound stream and saving dX...\n");

	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}

	char * dX_filename = "test_rms/dX_matrix.dat";

	ret = save_host_matrix(dX_filename, dX, M, N, bwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save output matrix...\n");
		return -1;
	}


	printf("Doing BWD W op...\n");

	Op rms_norm_bwd_w_op;

	set_native_rms_norm_bwd_w_skeleton(&rms_norm_bwd_w_op.op_skeleton, fwd_dt, bwd_dt);

	void ** bwd_w_op_args = rms_norm_bwd_w_op.op_args;

	bwd_w_op_args[0] = &iM;
	bwd_w_op_args[1] = &iN;
	bwd_w_op_args[2] = &eps;
	bwd_w_op_args[3] = &d_rms_vals;
	bwd_w_op_args[4] = &d_orig_matrix;
	bwd_w_op_args[5] = &d_upstream_dX;
	bwd_w_op_args[6] = &d_dW;


	ret = cuda_dataflow_handle.submit_op(&cuda_dataflow_handle, &rms_norm_bwd_w_op, compute_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to submit op...\n");
		return -1;
	}

	printf("Submitting dependency for outbound BWD W transfer...\n");

	compute_stream_state = cuda_dataflow_handle.get_stream_state(&cuda_dataflow_handle, compute_stream_id_a);
	if (!compute_stream_state){
		fprintf(stderr, "Error: failed to get stream state...\n");
		return -1;
	}

	ret = cuda_dataflow_handle.submit_dependency(&cuda_dataflow_handle, outbound_stream_id_a, compute_stream_state);
	if (ret){
		fprintf(stderr, "Error: failed to submit dependency...\n");
		return -1;
	}


	printf("Submitting outbound BWD W transfer...\n");	

	ret = cuda_dataflow_handle.submit_outbound_transfer(&cuda_dataflow_handle, outbound_stream_id_a, dW, d_dW, N * el_size);
	if (ret){
		fprintf(stderr, "Error: could not submit outbound transfer...\n");
		return -1;
	}


	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}


	printf("Syncing outbound stream and saving dW...\n");
	
	ret = cuda_dataflow_handle.sync_stream(&cuda_dataflow_handle, outbound_stream_id_a);
	if (ret){
		fprintf(stderr, "Error: failed to sync stream after transfer back to host...\n");
		return -1;
	}

	char * dW_filename = "test_rms/dWeights.dat";

	ret = save_host_matrix(dW_filename, dW, N, 1, fwd_dt);
	if (ret){
		fprintf(stderr, "Error: failed to save sq sums matrix...\n");
		return -1;
	}


	printf("\n\n\nSuccessfully Performed Op...!!!\n");

	return 0;
}
