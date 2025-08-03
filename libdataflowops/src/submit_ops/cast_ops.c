#include "dataflow_ops.h"

int dataflow_submit_cast(Dataflow_Handle *handle, int stream_id, DataflowDatatype src_dt, DataflowDatatype dst_dt, uint64_t num_elements, void *src, void *dst){

    Op cast_op;
    dataflow_set_default_cast_skeleton(&cast_op.op_skeleton, src_dt, dst_dt, DATAFLOW_NONE);

    void ** op_args = cast_op.op_args;

    op_args[0] = &num_elements;
    op_args[1] = &src;
    op_args[2] = &dst;

    int ret = (handle -> submit_op)(handle, &cast_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit cast op...\n");
        return -1;
    }

    return 0;
}

int dataflow_submit_cast_and_add(Dataflow_Handle * handle, int stream_id,
		DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
		uint64_t num_els,
		float alpha, void * A, float beta, void * B, void * C){

    Op cast_and_add_op;

    dataflow_set_default_cast_and_add_skeleton(&cast_and_add_op.op_skeleton, A_dt, B_dt, C_dt);

    void ** op_args = cast_and_add_op.op_args;

    op_args[0] = &num_els;
    op_args[1] = &alpha;
    op_args[2] = A;
    op_args[3] = &beta;
    op_args[4] = B;
    op_args[5] = C;

    int ret = (handle -> submit_op)(handle, &cast_and_add_op, stream_id);
    if (ret){
        fprintf(stderr, "Error: failed to submit cast and add op...\n");
        return -1;
    }

    return 0;
}