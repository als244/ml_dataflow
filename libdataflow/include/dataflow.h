#ifndef DATAFLOW_H
#define DATAFLOW_H

#include "dataflow_common.h"
#include "dataflow_op_structs.h"
#include "dataflow_table.h"
#include "dataflow_handle.h"
#include "dataflow_utils.h"

// For backends to implement...
int dataflow_init_handle(Dataflow_Handle * dataflow_handle, ComputeType compute_type, int device_id, int ctx_id, unsigned int ctx_flags, int num_streams, int * opt_stream_prios, char ** opt_stream_names);

// E.g. returns the datatype size corresponding to the elements within the array
// not the 64-bits corresponding to pointer iteself

// may need to eventually reutrn double to express 4-bit values...
size_t dataflow_sizeof_element(DataflowDatatype arr_dtype);

char * dataflow_datatype_as_string(DataflowDatatype dtype);

int dataflow_convert_datatype(void * to, void * from, DataflowDatatype to_dt, DataflowDatatype from_dt, long n, int num_threads);

int dataflow_do_fingerprinting(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint);


// Assumes memory has already been allocated for fast_table container
int dataflow_init_table(Dataflow_Table * table, Hash_Func hash_func, uint64_t key_size_bytes, uint64_t value_size_bytes, 
						uint64_t min_table_size, uint64_t max_table_size, float load_factor, float shrink_factor);


// all it does is free fast_table -> items
void dataflow_destroy_table(Dataflow_Table * table);



// returns 0 on success, -1 on error

// does memcopiess of key and value into the table array
// assumes the content of the key cannot be 0 of size key_size_bytes
int dataflow_insert_table(Dataflow_Table * table, void * key, void * value);



// Returns the index at which item was found on success, fast_table -> max_size on not found
//	- returning the index makes remove easy (assuming single threaded)

// A copy of the value assoicated with key in the table
// Assumes that memory of value_sized_bytes as already been allocated to ret_val
// And so a memory copy will succeed

// If to_copy_value is set the copy back the the item. If no item exists and this flag is set, ret_value is set to NULL
long dataflow_find_table(Dataflow_Table * table, void * key, bool to_copy_value, void ** ret_value);

// returns 0 upon success, -1 upon error
// The value will copied into ret_value
int dataflow_remove_table(Dataflow_Table * table, void * key, void * ret_value);


int dataflow_set_op_skeleton(Op_Skeleton * skeleton, char * op_name, DataflowDatatype fwd_dt, DataflowDatatype bwd_dt);

#endif
