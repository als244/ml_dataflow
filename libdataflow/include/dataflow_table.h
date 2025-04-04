#ifndef DATAFLOW_TABLE_H
#define DATAFLOW_TABLE_H

#include "dataflow_common.h"

// NOTE: Assumes SINGLE-THREADED and that this table will be responsible for the memory
//			of key's and values inserted. It memcopies the arguments passed in because
//			we will be creating values on the stack and passing those references into
//			these functions. ** Special case: For very large sized keys/values (in bytes), then we would
//			be careful to not have stack-overflow and would dynamically allocate and then
//			copy again and free the original.


// For more details on role of load/shrink factor and relationship with min/max
// sett config.h for example.

// Note that load_factor = 0.25


// This stores the static values of the 
// the table and can be shared

// Within fast tree we can make a TON
// of fast tables so need to conserve memory
// by having trees at each level point to 
// this struct

typedef struct table_config {
	uint64_t min_size;
	uint64_t max_size;
	// LOAD FACTOR:
	// if ((new insertion) && ((size < max_size) && (new_cnt > size * load_factor)):
	//	- new_size = min(size * (1 / load_factor), max_size)
	//		- resize_table with new allocation and freeing of old
	float load_factor;
	// SHRINK FACTOR:
	// if ((new removal) && ((size > min_size) & (new_cnt < size * shrink_factor):
	//	- new size = max(size * (1 - shrink_factor), min_size)
	float shrink_factor;
	Hash_Func hash_func;
	// will be used to advance in the hash table
	uint64_t key_size_bytes;
	// to know how much room to allocate
	uint64_t value_size_bytes;
} Table_Config;


typedef struct table {
	uint64_t cnt;
	uint64_t size;
	Table_Config config;
	// a bit vector of capacity size >> 6 uint64_t's
	// upon an insert an item's current index is checked
	// against this vector to be inserted

	// initialized to all ones. when something get's inserted
	// it flips bit to 0. 

	// will use __builtin_ffsll() to get bit position of least-significant
	// 1 in order to determine the next empty slot
	uint64_t * is_empty_bit_vector;
	// array that is sized
	// (key_size_bytes + value_size_bytes * size
	// the indicies are implied by the total size
	// Assumes all items inserted have the first
	// key_size_bytes of the entry representing
	// the key for fast comparisons
	void * items;
} Dataflow_Table;


#endif