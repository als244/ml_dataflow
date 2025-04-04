#ifndef DATAFLOW_OP_STRUCTS_H
#define DATAFLOW_OP_STRUCTS_H

#include "dataflow_common.h"

#define MAX_OP_ARGS 32

#define MAX_OP_NICKNAME_SIZE 255

typedef struct op_skeleton_header {
	// can specify different variants here
	char op_nickname[MAX_OP_NICKNAME_SIZE + 1];
	int num_args;
	DataflowDatatype arg_dtypes[MAX_OP_ARGS];
} Op_Skeleton_Header;

typedef struct op_identifier {
	uint8_t fingerprint[DATAFLOW_OP_IDENTIFIER_FINGERPRINT_NUM_BYTES];
} Op_Identifier;

typedef struct Op_Skeleton {
	Op_Skeleton_Header header;
	// identifier is tied to the hash of the header
	Op_Identifier identifier;
} Op_Skeleton;

typedef struct op {
	Op_Skeleton op_skeleton;
	void * op_args[MAX_OP_ARGS];
} Op;

#endif