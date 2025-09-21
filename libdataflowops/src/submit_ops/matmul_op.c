#include "dataflow_ops.h"


// If A, C, D are all stored in Row-Major
// And B is stored in Col-Major. If so, it compute:
// D = alpha * AB + beta * C

// If B is stored in Row-Major that implies it computes:
// D = alpha * AB^T + beta * C
int dataflow_submit_matmul(Dataflow_Handle * handle, int stream_id, 
					DataflowDatatype a_dt, DataflowDatatype b_dt, DataflowDatatype c_dt, DataflowDatatype d_dt,
					DataflowDatatype compute_dt,
					int to_trans_a, int to_trans_b,
					int M, int K, int N,
					float alpha, float beta,
					void * A, void * B, void * C, void * D,
					uint64_t workspaceBytes, void * workspace) {


	int ret;

	Op matmul_op;

	dataflow_set_matmul_skeleton(&matmul_op.op_skeleton);

	void ** op_args = matmul_op.op_args;


	// need to have a function poitner to
	// query handle's number of procs!
	// for now just setting to 0 (all procs)

	int num_procs = (handle -> get_num_procs)(handle);
	op_args[0] = &num_procs; 

	op_args[1] = &a_dt;
	op_args[2] = &b_dt;
	op_args[3] = &c_dt;
	op_args[4] = &d_dt;
	op_args[5] = &compute_dt;
	op_args[6] = &to_trans_a;
	op_args[7] = &to_trans_b;
	op_args[8] = &M;
	op_args[9] = &K;
	op_args[10] = &N;
	op_args[11] = &alpha;
	op_args[12] = &beta;
	op_args[13] = &A;
	op_args[14] = &B;
	op_args[15] = &C;
	op_args[16] = &D;
	op_args[17] = &workspaceBytes;
	op_args[18] = &workspace;

	ret = (handle -> submit_op)(handle, &matmul_op, stream_id);
	if (ret){
		fprintf(stderr, "Error: failed to submit matmul_op...\n");
		return -1;
	}

	return 0;


}