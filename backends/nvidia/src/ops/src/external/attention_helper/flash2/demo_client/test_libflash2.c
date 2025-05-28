#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>

#include <cuda.h>

#include "flash2_wrapper.h"

#define ROUND_UP_TO_128(x) (((x) + 127) & ~127)

// NEEDS TO AGREE WITH FLASH 3 & DataflowDatatype ORDERING AND VALUES!
typedef enum {
    DATAFLOW_NONE,
    DATAFLOW_VOID,
    DATAFLOW_FP64,
    DATAFLOW_FP32,
    DATAFLOW_FP16,
    DATAFLOW_BF16,
    DATAFLOW_FP8E4M3,
    DATAFLOW_FP8E5M2,
    DATAFLOW_UINT64,
    DATAFLOW_UINT32,
    DATAFLOW_UINT16,
    DATAFLOW_UINT8,
    DATAFLOW_LONG,
    DATAFLOW_INT,
    DATAFLOW_BOOL,
    DATAFLOW_FP64_SCALAR,
    DATAFLOW_FP32_SCALAR,
    DATAFLOW_FP16_SCALAR,
    DATAFLOW_BF16_SCALAR,
    DATAFLOW_FP8E4M3_SCALAR,
    DATAFLOW_FP8E5M2_SCALAR,
    DATAFLOW_UINT64_SCALAR,
    DATAFLOW_UINT32_SCALAR,
    DATAFLOW_UINT16_SCALAR,
    DATAFLOW_UINT8_SCALAR,
    DATAFLOW_LONG_SCALAR,
    DATAFLOW_INT_SCALAR,
    DATAFLOW_BOOL_SCALAR
} DataflowDatatype;


int initialize_driv(){

	CUresult result;
	const char * err;

	unsigned long flags = 0;
	result = cuInit(flags);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not init driver: %s\n", err);
    	return -1;
	}
	return 0;
}

int initialize_ctx(CUcontext * ctx, int device_id){

	CUresult result;
	const char * err;



	CUdevice dev;
	result = cuDeviceGet(&dev, device_id);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not get device: %s\n", err);
    	return -1;
	}


	// Set the host thread to spin waiting for completetion from GPU
	unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;
	
	result = cuCtxCreate(ctx, ctx_flags, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not create context: %s\n", err);
    	return -1;
	}

	result = cuCtxPushCurrent(*ctx);
        if (result != CUDA_SUCCESS){
                fprintf(stderr, "Error: could not set context...\n");
                return -1;
        }

	// SUCCESS!
	return 0;
}


int initialize_stream(CUstream * stream, int prio){

	CUresult result;
	const char * err;

	
	result = cuStreamCreateWithPriority(stream, CU_STREAM_NON_BLOCKING, prio);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to create cuda stream\n");
		return -1;
	}

	return 0;
}

int get_dev_info(int device_id, int * arch, int * num_sms){

	CUresult result;
	const char * err;


	CUdevice dev;
	result = cuDeviceGet(&dev, device_id);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    		fprintf(stderr, "Error: Could not get device: %s\n", err);
    		return -1;
	}

	int major_arch_num;
	int minor_arch_num;

	result = cuDeviceGetAttribute(&major_arch_num, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get device major arch number: %s\n", err);
		return -1;
	}

	result = cuDeviceGetAttribute(&minor_arch_num, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get device minor arch number: %s\n", err);
		return -1;
	}

	int arch_num = 10 * major_arch_num + minor_arch_num;

	

	int sm_count;

	result = cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get sm count: %s\n", err);
		return -1;
	}

	*arch = arch_num;
	*num_sms = sm_count;

	return 0;

}


int reserve_dev_memory(int total_q, int total_k, int num_seqs, size_t x_dt_bytes, 
				int num_q_heads, int num_kv_heads, int head_dim, 
				void ** q_seq_offsets, void ** q_seq_lens, int max_seqlen_q,
				void ** k_seq_offsets, void ** k_seq_lens, int max_seqlen_k,
				void ** x_q, void  ** x_k, void ** x_v, 
				void ** x_attn_out, void ** softmax_lse,
				void ** attn_workspace, uint64_t * ret_attn_workspace_size,
				void ** dx_out,
				void ** dx_q, void ** dx_k, void ** dx_v,
				void ** attn_bwd_workspace, uint64_t * ret_attn_bwd_workspace_size){

	CUresult result;
	const char * err;

	int offsets_size = (num_seqs + 1) * sizeof(int);
	int lens_size = (num_seqs) * sizeof(int);

	result = cuMemAlloc((CUdeviceptr *) q_seq_offsets, offsets_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate q_seq_offsets on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) q_seq_lens, lens_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate q_seq_lens on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) k_seq_offsets, offsets_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate k_seq_offsets on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) k_seq_lens, lens_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate k_seq_lens on device: %s\n", err);
    	return -1;
	}

	int q_size = total_q * num_q_heads * head_dim * x_dt_bytes;
	int out_size = q_size;
	// if FP8, then out must be BF16...
	if (x_dt_bytes == 1){
		out_size *= 2;
	}

	result = cuMemAlloc((CUdeviceptr *) x_q, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_q on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) dx_q, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate dx_q on device: %s\n", err);
    	return -1;
	}

	result = cuMemsetD8(*((CUdeviceptr *) dx_q), 0, q_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not set dx_q workspace to 0 on device: %s\n", err);
        	return -1;
	}


	int kv_size = total_k * num_kv_heads * head_dim * x_dt_bytes;

	result = cuMemAlloc((CUdeviceptr *) x_k, kv_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_k on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) dx_k, kv_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate dx_k on device: %s\n", err);
    	return -1;
	}

	result = cuMemsetD8(*((CUdeviceptr *) dx_k), 0, kv_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not set dx_k workspace to 0 on device: %s\n", err);
        	return -1;
	}


	result = cuMemAlloc((CUdeviceptr *) x_v, kv_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_v on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) dx_v, kv_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate dx_v on device: %s\n", err);
    	return -1;
	}

	result = cuMemsetD8(*((CUdeviceptr *) dx_v), 0, kv_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not set dx_v workspace to 0 on device: %s\n", err);
        	return -1;
	}


	result = cuMemAlloc((CUdeviceptr *) x_attn_out, out_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate x_attn_out on device: %s\n", err);
    	return -1;
	}

	result = cuMemAlloc((CUdeviceptr *) dx_out, out_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate dx_out on device: %s\n", err);
    	return -1;
	}



	int softmax_lse_size = num_q_heads * total_q * sizeof(float);

	
	result = cuMemAlloc((CUdeviceptr *) softmax_lse, softmax_lse_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not allocate softmax_lse on device: %s\n", err);
    	return -1;
	}


	// To compute required size of attn_workspace:

	// attn_workspace_size = 0

	// Occum and LSE accum:
	// If num_splits > 1:
	//      attn_workspace_size += num_splits * sizeof(float) * num_q_heads * total_q * (1 + head_dim)

	// Tile count sem: 
	// If arch >= 90 || num_splits > 1:
	//      attn_workspace_size += sizeof(int)

	// Dynamic split ptr for each seq:
	// If num_seqs <= 992:
	//      attn_workspace_size += num_seqs * sizeof(int)

	// just get enough...

	int max_num_splits = 256;


	int attn_workspace_size = 0;

	// cover oaccum and lse accum
	attn_workspace_size += max_num_splits * sizeof(float) * num_q_heads * total_q * (1 + head_dim);
	
	// cover potential tile count sem
	attn_workspace_size += sizeof(int);

	// covert potential dynamic split
	attn_workspace_size += num_seqs * sizeof(int);

	*ret_attn_workspace_size = attn_workspace_size;

	result = cuMemAlloc((CUdeviceptr *) attn_workspace, attn_workspace_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not allocate attn workspace on device: %s\n", err);
        	return -1;
	}

	result = cuMemsetD8(*((CUdeviceptr *) attn_workspace), 0, attn_workspace_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not set attn workspace to 0 on device: %s\n", err);
        	return -1;
	}


	int attn_bwd_workspace_size = 0;

	int max_total_q_rounded = ROUND_UP_TO_128(total_q + num_seqs * 128);
	int max_total_k_rounded = ROUND_UP_TO_128(total_k + num_seqs * 128);

	int softmax_workspace =  2 * num_q_heads * max_total_q_rounded * sizeof(float);
	int dq_accum_workspace = num_q_heads * max_total_q_rounded * head_dim * sizeof(float);
	int dq_sem_workspace =  ((max_seqlen_q + 127) / (128)) * num_seqs * num_q_heads * sizeof(int);

	int dkv_accum_workspace = 0;
	int dkv_sem_workspace = 0;
	if (num_q_heads != num_kv_heads){
		dkv_accum_workspace = 2 * num_kv_heads * max_total_k_rounded * head_dim * sizeof(float);
		dkv_sem_workspace = 2 * ((max_seqlen_k + 127) / 128) * num_seqs * num_kv_heads * sizeof(int);
	}

	attn_bwd_workspace_size = 10 * (softmax_workspace + dq_accum_workspace + dq_sem_workspace + dkv_accum_workspace + dkv_sem_workspace); 
	

	*ret_attn_bwd_workspace_size = attn_bwd_workspace_size;

	result = cuMemAlloc((CUdeviceptr *) attn_bwd_workspace, attn_bwd_workspace_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not allocate attn bwd workspace on device: %s\n", err);
        	return -1;
	}


	result = cuMemsetD8(*((CUdeviceptr *) attn_bwd_workspace), 0, attn_bwd_workspace_size);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not set attn bwd workspace to 0 on device: %s\n", err);
        	return -1;
	}

	return 0;
}


int load_from_file_to_dev(char * filepath, size_t size_bytes, void * dev_ptr){

	CUresult result;
	const char * err;

	FILE * fp = fopen(filepath, "rb");
	if (!fp){
		fprintf(stderr, "Error: could not open file: %s", filepath);
		return -1;
	}

	void * sys_temp = malloc(size_bytes);

	if (!sys_temp){
		fprintf(stderr, "Error: could not alloc temp memory...\n");
		return -1;
	}

	size_t nread = fread(sys_temp, 1, size_bytes, fp);
	if (nread != size_bytes){
		fprintf(stderr, "Error: could not ready expected # of bytes from file %s. Expected %lu, read %lu...\n", filepath, size_bytes, nread);
		return -1;
	}

	fclose(fp);


	result = cuMemcpyHtoD((CUdeviceptr) dev_ptr, sys_temp, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not memcpy contents from file %s (loaded in sys mem) of size %lu to device: %s\n", filepath, size_bytes, err);
    	return -1;
	}

	free(sys_temp);
}


int load_and_copy_sample_inputs(char * data_dir, int num_seqs, int total_q, int total_k, int model_dim, int kv_dim, int dtype_size, 
					void * q_seq_offsets, void * q_seq_lens, void * k_seq_offsets, void * k_seq_lens, 
					void * x_q, void  * x_k, void * x_v, 
					void * dx_out, 
					int chunk_num){

	// -100 to not get warnings
	char exts[8][PATH_MAX - 100];

	// don't have any prefix
	if (chunk_num == -1){
		strcpy(exts[0], "q_seq_offsets.dat");
		strcpy(exts[1], "q_seq_lens.dat");
		strcpy(exts[2], "k_seq_offsets.dat");
		strcpy(exts[3], "k_seq_lens.dat");
	}
	else{
		sprintf(exts[0], "%d_%s", chunk_num, "q_seq_offsets.dat");
       		sprintf(exts[1], "%d_%s", chunk_num, "q_seq_lens.dat");
       		sprintf(exts[2], "%d_%s", chunk_num, "k_seq_offsets.dat");
       		sprintf(exts[3], "%d_%s", chunk_num, "k_seq_lens.dat");
	}
       	
       	strcpy(exts[4], "x_q.dat");
	strcpy(exts[5], "x_k.dat");
	strcpy(exts[6], "x_v.dat");
	strcpy(exts[7], "dx_out.dat");

	size_t sizes[8];
	sizes[0] = (num_seqs + 1) * sizeof(int);
	sizes[1] = (num_seqs) * sizeof(int);
	sizes[2] = (num_seqs + 1) * sizeof(int);
	sizes[3] = (num_seqs) * sizeof(int);
	sizes[4] = (total_q * model_dim * dtype_size);
	sizes[5] = (total_k * kv_dim * dtype_size);
	sizes[6] = (total_k * kv_dim * dtype_size);
	sizes[7] = (total_q * model_dim * dtype_size);

	void * dev_ptrs[8];
	dev_ptrs[0] = q_seq_offsets;
	dev_ptrs[1] = q_seq_lens;
	dev_ptrs[2] = k_seq_offsets;
	dev_ptrs[3] = k_seq_lens;
	dev_ptrs[4] = x_q;
	dev_ptrs[5] = x_k;
	dev_ptrs[6] = x_v;
	dev_ptrs[7] = dx_out;

	int ret;

	char filepath[PATH_MAX];

	int upper_load = 8;
	// only need to load in new metadata...
	if (chunk_num >= 0){
		upper_load = 4;
	}
	for (int i = 0; i < 8; i++){
		
		sprintf(filepath, "%s/%s", data_dir, exts[i]);

		ret = load_from_file_to_dev(filepath, sizes[i], dev_ptrs[i]);
		if (ret){
			fprintf(stderr, "Error: could not load file: %s to device...\n", filepath);
			return -1;
		}
	}

	return 0;




}


int save_file_from_dev(char * filepath, size_t size_bytes, void * dev_ptr){

	CUresult result;
        const char * err;

	FILE * fp = fopen(filepath, "wb");
	if (!fp){
		fprintf(stderr, "Error: could not open filepath: %s to write...\n", filepath);
	}

	void * sys_temp = malloc(size_bytes);

	result = cuMemcpyDtoH(sys_temp, (CUdeviceptr) dev_ptr, size_bytes);
        if (result != CUDA_SUCCESS){
                cuGetErrorString(result, &err);
        	fprintf(stderr, "Error: Could not memcpy contents to file %s (loaded in sys mem) from device: %s\n", filepath, err);
        	return -1;
        }

	size_t nwritten = fwrite(sys_temp, 1, size_bytes, fp);
	if (nwritten != size_bytes){
		fprintf(stderr, "Error: could not write  expected # of bytes from file %s. Expected %lu, read %lu...\n", filepath, size_bytes, nwritten);
                return -1;
	}

	fclose(fp);

	return 0;

}


int save_flash_lib_out(char * data_dir, int total_q, int model_dim, int num_q_heads, size_t x_dt_bytes, void * x_attn_out, void * softmax_lse){
	
	char * exts[2];
	exts[0] = "x_out.dat";
	exts[1] = "softmax_lse.dat";

	size_t sizes[2];

	// if FP8 input, then BF16 output...
	if (x_dt_bytes == 1){
		x_dt_bytes == 2;
	}
	sizes[0] = total_q * model_dim * x_dt_bytes;
	sizes[1] = total_q * num_q_heads * sizeof(float);

	void * dev_ptrs[2];
	dev_ptrs[0] = x_attn_out;
	dev_ptrs[1] = softmax_lse;

	int ret;

        char filepath[PATH_MAX];

        for (int i = 0; i < 2; i++){

                sprintf(filepath, "%s/%s", data_dir, exts[i]);

                ret = save_file_from_dev(filepath, sizes[i], dev_ptrs[i]);
                if (ret){
                        fprintf(stderr, "Error: could not load file: %s to device...\n", filepath);
                        return -1;
                }
        }

	return 0;

}

int save_flash_lib_bwd_out(char * data_dir, int total_q, int model_dim, int total_k, int kv_dim, size_t x_dt_bytes, void * dx_q, void * dx_k, void * dx_v){
	
	char * exts[3];
	exts[0] = "dx_q.dat";
	exts[1] = "dx_k.dat";
	exts[2] = "dx_v.dat";

	size_t sizes[3];

	sizes[0] = total_q * model_dim * x_dt_bytes;
	sizes[1] = total_k * kv_dim * x_dt_bytes;
	sizes[2] = total_k * kv_dim * x_dt_bytes;

	void * dev_ptrs[3];
	dev_ptrs[0] = dx_q;
	dev_ptrs[1] = dx_k;
	dev_ptrs[2] = dx_v;

	int ret;

        char filepath[PATH_MAX];

        for (int i = 0; i < 3; i++){

                sprintf(filepath, "%s/%s", data_dir, exts[i]);

                ret = save_file_from_dev(filepath, sizes[i], dev_ptrs[i]);
                if (ret){
                        fprintf(stderr, "Error: could not load file: %s to device...\n", filepath);
                        return -1;
                }
        }

	return 0;

}

int main (int argc, char * argv[]){
	
	int ret;

	if (argc != 3){
		fprintf(stderr, "Error Usage: ./test_libflash3_bwd <num_seqs> <seq_len>\n");
		return -1;
	}	

	int num_seqs = atoi(argv[1]);
	int seq_len = atoi(argv[2]);

	char * cwd;
	char buffer[PATH_MAX];

	cwd = getcwd(buffer, PATH_MAX);
	if (!cwd){
		fprintf(stderr, "Error: getcwd() error...\n");
		return -1;
	}

	char data_dir[PATH_MAX];

	sprintf(data_dir, "%s/data/%dx%d", cwd, num_seqs, seq_len);

	ret = initialize_driv();
	if (ret){
		fprintf(stderr, "Error: could not initialize cuda driver...\n");
		return -1;
	}

	CUcontext ctx;

	int device_id = 0;

	ret = initialize_ctx(&ctx, device_id);
	if (ret){
		fprintf(stderr, "Error: could not initialize context...\n");
		return -1;
	}


	CUstream stream;

	ret = initialize_stream(&stream, 0);
	if (ret){
		fprintf(stderr, "Error: could not initialize stream...\n");
		return -1;
	}



	DataflowDatatype flash_dtype = DATAFLOW_BF16;


	// For now setting total q == total k
	// and also every sequence of equal length
	// (but still using varlen API)

	// With cache, total k might be greater!
	//int total_q = num_seqs * seq_len;
	
	// testing with last sequence only having last query
	int total_q = num_seqs * seq_len;

	int total_k = num_seqs * seq_len;



	int max_seqlen_q = seq_len;
	int max_seqlen_k = seq_len;

	size_t x_dt_bytes = 2;
	
	
	int num_q_heads = 32;
	int num_kv_heads = 8;
	int head_dim = 128;	

	void * q_seq_offsets;
	void * q_seq_lens;
	void * k_seq_offsets;
	void * k_seq_lens;
	void * x_q;
	void * x_k;
	void * x_v;
	void * x_attn_out;
	void * softmax_lse;
	void * attn_workspace;

	void * dx_out;
	void * dx_q;
	void * dx_k;
	void * dx_v;

	void * attn_bwd_workspace;

	uint64_t attn_workspace_size;
	uint64_t attn_bwd_workspace_size;

	ret = reserve_dev_memory(total_q, total_k, num_seqs, x_dt_bytes, 
					num_q_heads, num_kv_heads, head_dim, 
					&q_seq_offsets, &q_seq_lens, max_seqlen_q,
					&k_seq_offsets, &k_seq_lens, max_seqlen_k,
					&x_q, &x_k, &x_v, 
					&x_attn_out, &softmax_lse, 
					&attn_workspace, &attn_workspace_size,
					&dx_out,
					&dx_q, &dx_k, &dx_v,
					&attn_bwd_workspace, &attn_bwd_workspace_size);
	if (ret){
		fprintf(stderr, "Error: could not reserve device memory...\n");
		return -1;
	}


	int model_dim = head_dim * num_q_heads;
	int kv_dim = head_dim * num_kv_heads;


	ret = load_and_copy_sample_inputs(data_dir, num_seqs, total_q, total_k, model_dim, kv_dim, x_dt_bytes, 
						q_seq_offsets, q_seq_lens, k_seq_offsets, k_seq_lens, 
						x_q, x_k, x_v, dx_out, -1);
	if (ret){
		fprintf(stderr, "Error: could not load and copy sample inputs...\n");
		return -1;
	}


	printf("CALLING FLASH ATTENTION...!\n\n");

	int arch;
	int num_sms;

	ret = get_dev_info(device_id, &arch, &num_sms);
	if (ret){
		fprintf(stderr, "Error: failed to get arch and sm count for device id: %d...\n", device_id);
		return -1;
	}


	int is_causal = 1;


	printf("Submitting forwards pass...\n");
	ret = flash2_fwd_wrapper(stream, arch, num_sms,
					(int) flash_dtype, 
					num_seqs, total_q, total_k,
					q_seq_offsets, q_seq_lens, max_seqlen_q,
					k_seq_offsets, k_seq_lens, max_seqlen_k,
					num_q_heads, num_kv_heads, head_dim,
					x_q, x_k, x_v, 
					x_attn_out, softmax_lse,
					is_causal,
					attn_workspace_size, attn_workspace);
	if (ret){
		fprintf(stderr, "Error: submitting flash3 fwd failed...\n");
		return -1;
	}
	
	printf("Waiting for stream sync...!\n");
	cuStreamSynchronize(stream);

	printf("Saving forward results...\n");
	save_flash_lib_out(data_dir, total_q, model_dim, num_q_heads, x_dt_bytes, x_attn_out, softmax_lse);


	ret = flash2_bwd_wrapper(stream, arch, num_sms,
					(int) flash_dtype, 
					num_seqs, total_q, total_k,
					q_seq_offsets, q_seq_lens, max_seqlen_q,
					k_seq_offsets, k_seq_lens, max_seqlen_k,
					num_q_heads, num_kv_heads, head_dim,
					x_q, x_k, x_v, 
					x_attn_out, softmax_lse,  
					dx_out,
					dx_q, dx_k, dx_v,
					is_causal,
					attn_bwd_workspace_size, attn_bwd_workspace);

	printf("Submitting backward pass...\n");
	if (ret){
		fprintf(stderr, "Error: submitting flash3 bwd failed...\n");
		return -1;
	}

	/* CHUNKED VERSION...

	printf("Loading metadata for backpass chunk 0...\n");
	ret = load_and_copy_sample_inputs(data_dir, num_seqs, 0, 0, 0, 0, 0, 
						q_seq_offsets, q_seq_lens, k_seq_offsets, k_seq_lens, 
						NULL, NULL, NULL, NULL, 0);


	printf("Submitting backward pass for chunk 0...\n");
	ret = flash3_bwd_wrapper(stream, arch, num_sms,
					(int) flash_dtype, 
					num_seqs, total_q, total_k,
					q_seq_offsets, q_seq_lens, max_seqlen_q,
					k_seq_offsets, k_seq_lens, max_seqlen_k,
					num_q_heads, num_kv_heads, head_dim,
					x_q, x_k, x_v, 
					x_attn_out, softmax_lse,  
					dx_out,
					dx_q, dx_k, dx_v,
					attn_bwd_workspace);
	if (ret){
		fprintf(stderr, "Error: submitting flash3 bwd failed...\n");
		return -1;
	}


	printf("Loading metadata for backpass chunk 1...\n");
	ret = load_and_copy_sample_inputs(data_dir, num_seqs, 0, 0, 0, 0, 0, 
						q_seq_offsets, q_seq_lens, k_seq_offsets, k_seq_lens, 
						NULL, NULL, NULL, NULL, 1);


	printf("Submitting backward pass...\n");
	ret = flash3_bwd_wrapper(stream, arch, num_sms,
					(int) flash_dtype, 
					num_seqs, total_q, total_k,
					q_seq_offsets, q_seq_lens, max_seqlen_q,
					k_seq_offsets, k_seq_lens, max_seqlen_k,
					num_q_heads, num_kv_heads, head_dim,
					x_q, x_k, x_v, 
					x_attn_out, softmax_lse,  
					dx_out,
					dx_q, dx_k, dx_v,
					attn_bwd_workspace);
	if (ret){
		fprintf(stderr, "Error: submitting flash3 bwd failed...\n");
		return -1;
	}
	*/

	printf("Waiting for stream sync...!\n");
	cuStreamSynchronize(stream);

	printf("Saving bwd results...!\n");
	save_flash_lib_bwd_out(data_dir, total_q, model_dim, total_k, kv_dim, x_dt_bytes, dx_q, dx_k, dx_v);

	printf("Flash3 FWD + BWD complete!\n\n");

	return 0;
}
