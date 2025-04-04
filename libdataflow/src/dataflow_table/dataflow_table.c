#include "dataflow_table.h"


// if memory error on creating table returns -1
// otherwise 0
int dataflow_init_table(Dataflow_Table * table, Hash_Func hash_func, uint64_t key_size_bytes, uint64_t value_size_bytes, 
						uint64_t min_table_size, uint64_t max_table_size, float load_factor, float shrink_factor) {

	(table -> config).hash_func = hash_func;
	(table -> config).key_size_bytes = key_size_bytes;
	(table -> config).value_size_bytes = value_size_bytes;
	(table -> config).min_size = min_table_size;
	(table -> config).max_size = max_table_size;
	(table -> config).load_factor = load_factor;
	(table -> config).shrink_factor = shrink_factor;


	uint64_t min_size = min_table_size;

	table -> cnt = 0;

	table -> size = min_size;


	// also uses an extra byte per entry to indicate if 
	// there is an item in that slot (otherwise wouldn't be able
	// to distinguish the key/value should be 0's or if they are empty)

	// this will be a bit vector where each item is a uint64_t
	// it is of size ceil(table -> size / 64) == size shifted by 

	// the index into the table is the high order bits 56 bits of the bucket index within
	// table -> items

	// and the bit position within each vector is the low order 6 bits
	
	int bit_vector_els = MY_CEIL(min_size, 64);
	table -> is_empty_bit_vector = (uint64_t *) malloc(bit_vector_els * sizeof(uint64_t));
	if (unlikely(!table -> is_empty_bit_vector)){
		return -1;
	}

	// initialize everything to empty up to table size...
	// we know that bit vector els is the minimum number of 
	// elements to span the table size, but the last one might be
	// partially full
	for (int i = 0; i < bit_vector_els - 1; i++){
		(table -> is_empty_bit_vector)[i] = 0xFFFFFFFFFFFFFFFF;
	}

	int last_vec_els = min_size & 0x3F;

	if (last_vec_els == 0){
		(table -> is_empty_bit_vector)[bit_vector_els - 1] = 0xFFFFFFFFFFFFFFFF;
	}
	else{
		(table -> is_empty_bit_vector)[bit_vector_els - 1] = (1ULL << last_vec_els) - 1;
	}


	table -> items = calloc(min_size, key_size_bytes + value_size_bytes);
	if (unlikely(!table -> items)){
		return -1;
	}
	return 0;
}


void dataflow_destroy_table(Dataflow_Table * table) {
	free(table -> is_empty_bit_vector);
	free(table -> items);
	table -> items = NULL;
	table -> is_empty_bit_vector = NULL;
}

// returns size upon failure to find slot. should never happen because checked if null
// beforehand

// leaving the option to set to_get_empty to true, meaning that we should search for the next non-null slot
// in the table

// This is valuable during resizing
static long get_next_ind_table(uint64_t * is_empty_bit_vector, uint64_t table_size, uint64_t start_ind){

	// need to pass in a valid starting index
	if (unlikely(start_ind >= table_size)){
		return -1;
	}

	// assert (is_empty_bit_vector) == (table_size >> 6) + 1

	// Determine the next largest empty index in the table (wraping around at the end)
	// The low order 6 bits of hash_ind refer to the bit position within each element of the
	// the bit vector.  The high-order 56 bits refer the index within the actual vector

	uint64_t bit_vector_size = MY_CEIL(table_size, 64);
	// higher order bits the hash index
	uint64_t start_vec_ind = start_ind >> 6;
	// low order 6 bits
	uint8_t start_bit_ind = start_ind & 0x3F; 
		
	// before we start we need to clear out the bits strictly less than bit ind
	// if we don't find a slot looking through all the other elements in the bit 
	// vector we will wrap around to these value
	uint64_t search_vector = is_empty_bit_vector[start_vec_ind] & (0xFFFFFFFFFFFFFFFF << start_bit_ind);

	// need to ensure search vector bits

	// 64 because each element in bit-vector contains 64 possible hash buckets that could be full
	// Will add the returned next closest bit position to this value to obtain the next empty slot
	// With a good hash function and load factor hopefully this vector contains the value or at least
	// in the next few searches and won't need more than 1 search vector
		
	
	uint64_t cur_vec_ind = start_vec_ind;
	uint8_t least_sig_one_bit_ind;
	uint64_t insert_ind;

	// less than or equal because we might need the low order bits that we originally masked out

	// With good hash function and load factor this should only be 1 iteration when doing inserts
	// However for resizing we may find a long 

	// can be equal if we wrap around to the same starting index and
	// look at low order bits
	while(cur_vec_ind <= start_vec_ind + bit_vector_size){

		if (search_vector == 0){
			cur_vec_ind++;
			// if the cur_vec_ind would be wrapped around we don't
			// need to do any masking because we just care about the low
			// order bits which weren't seen the first go-around
			search_vector = is_empty_bit_vector[cur_vec_ind % bit_vector_size];
			continue;
		}

		// returns index of the least significant 1-bit
		least_sig_one_bit_ind = __builtin_ctzll(search_vector);
		
		insert_ind = 64 * (cur_vec_ind % bit_vector_size) + least_sig_one_bit_ind;
		return insert_ind;
	}

	// indicate that we couldn't find an empty slot (or full slot if to_flip_empty flag is true)
	return -1;

}


static int resize_table(Dataflow_Table * table, uint64_t new_size){
	
	uint64_t old_size = table -> size;
	uint64_t cnt = table -> cnt;


	// 1.) Allocate new memory for the table

	// create new table that will replace old one
	// ensure new table is initialized to all null

	uint64_t key_size_bytes = (table -> config).key_size_bytes;
	uint64_t value_size_bytes = (table -> config).value_size_bytes;

	int bit_vector_els = MY_CEIL(new_size, 64);
	uint64_t * new_is_empty_bit_vector = (uint64_t *) malloc(bit_vector_els * sizeof(uint64_t));
	if (unlikely(!new_is_empty_bit_vector)){
		fprintf(stderr, "Error: trying to resize table from %lu to %lu failed.\n", old_size, new_size);
		return -1;
	}

	// initialize everything to empty up to table size...
	// we know that bit vector els is the minimum number of 
	// elements to span the table size, but the last one might be
	// partially full
	for (int i = 0; i < bit_vector_els - 1; i++){
		new_is_empty_bit_vector[i] = 0xFFFFFFFFFFFFFFFF;
	}

	int last_vec_els = new_size & 0x3F;

	if (last_vec_els == 0){
		new_is_empty_bit_vector[bit_vector_els - 1] = 0xFFFFFFFFFFFFFFFF;
	}
	else{
		new_is_empty_bit_vector[bit_vector_els - 1] = (1ULL << last_vec_els) - 1;
	}
	
	void * new_items = (void *) calloc(new_size, key_size_bytes + value_size_bytes);
	if (unlikely(!new_items)){
		fprintf(stderr, "Error: trying to resize table from %lu to %lu failed.\n", old_size, new_size);
		return -1;
	}

	void * old_items = table -> items;
	uint64_t * old_is_empty_bit_vector = table -> is_empty_bit_vector;

	// we know how many items we need to re-insert

	uint64_t seen_cnt = 0;

	void * old_key;
	void * old_value;
	void * new_key_pos;
	void * new_value_pos;


	uint64_t new_hash_ind;
	long new_insert_ind;


	// because we know the count we don't need to error check for next index returning the table size.
	// they are guaranteed to succeed
	uint64_t cur_bit_vec;
	uint64_t cur_pos_in_vec;
	for (uint64_t old_ind = 0; old_ind < old_size; old_ind++){
		
		cur_bit_vec = old_is_empty_bit_vector[old_ind >> 6];
		cur_pos_in_vec = (old_ind & 0x3F);

		// if this position was empty then continue
		if (cur_bit_vec & (1ULL << cur_pos_in_vec)){
			continue;
		}

		// Now we need to re-hash this item into new table
		old_key = (void *) (((uint64_t) old_items) + (old_ind * (key_size_bytes + value_size_bytes)));
		old_value = (void *) (((uint64_t) old_key) + key_size_bytes);

		new_hash_ind = ((table -> config).hash_func)(old_key, new_size);

		// now we can get the insert index for the new table. now we are searching for next free slot
		// and we will use the new bit vector
		new_insert_ind = get_next_ind_table(new_is_empty_bit_vector, new_size, new_hash_ind);

		if (unlikely(new_insert_ind == -1)){
			fprintf(stderr, "Error: when trying to resize table from %lu to %lu failed.\n", old_size, new_size);
			return -1;
		}

		// now setting the pointer to be within new_items
		new_key_pos = (void *) (((uint64_t) new_items) + ((uint64_t) new_insert_ind * (key_size_bytes + value_size_bytes)));
		new_value_pos = (void *) (((uint64_t) new_key_pos) + key_size_bytes);

		// Actually copy into the new table place in table
		memcpy(new_key_pos, old_key, key_size_bytes);
		memcpy(new_value_pos, old_value, value_size_bytes);

		// Ensure to update the new bit vector that we inserted at new_insert_ind 
		// (because there may be collisions within re-inserting and we need the bit vector to track these)

		// No point in updating old bit vector because we started from zero and are only increasing updwards 
		// until we see all cnt elements

		// clearing the entry for this insert_ind in the bit vector

		// the bucket's upper bits represent index into the bit vector elements
		// and the low order 6 bits represent offset into element. Set to 0 

		// needs to be 1ULL otherwise will default to 1 byte
		new_is_empty_bit_vector[new_insert_ind >> 6] &= ~(1ULL << (new_insert_ind & 0x3F));
		seen_cnt += 1;

		if (seen_cnt == cnt){
			break;
		}
	}


	// now reset the table size, items, and bit vector 
	// and free the old memory

	free(old_is_empty_bit_vector);
	free(old_items);

	table -> size = new_size;
	table -> is_empty_bit_vector = new_is_empty_bit_vector;
	table -> items = new_items;

	return 0;
}







// Returns the index in the table and returns
// the index at which it was found
// Returns -1

// A copy of the value assoicated with key in the table
// Assumes that memory of value_sized_bytes as already been allocated to ret_val
// And so a memory copy will succeed
long dataflow_find_table(Dataflow_Table * table, void * key, bool to_copy_value, void ** ret_value){

	uint64_t value_size_bytes = (table -> config).value_size_bytes;

	// Assume we aren't finding the element...
	if (ret_value){
		if (to_copy_value){
			memset((void *) ret_value, 0, value_size_bytes);
		}
		else{
			*ret_value = NULL;
		}
	}

	if (table -> items == NULL){
		return -1;
	}

	uint64_t size = table -> size;
	uint64_t key_size_bytes = (table -> config).key_size_bytes;
	uint64_t hash_ind = ((table -> config).hash_func)(key, size);


	// get the next null value and search up to that point
	// because we are using linear probing if we haven't found the key
	// by this point then we can terminate and return that we didn't find anything

	// we could just walk along and check the bit vector as we go, but this is easily
	// (at albeit potential performance hit if table is very full and we do wasted work)
	uint64_t * is_empty_bit_vector = table -> is_empty_bit_vector;
	
	long next_empty = get_next_ind_table(is_empty_bit_vector, size, hash_ind);
	
	if (unlikely(next_empty == -1)){
		fprintf(stderr, "Error: when trying to find key in table.\n");
		return -1;
	}

	uint64_t cur_ind = hash_ind;


	void * cur_table_key = (void *) (((uint64_t) table -> items) + (cur_ind * (key_size_bytes + value_size_bytes)));

	uint64_t items_to_check;
	if (cur_ind <= (uint64_t) next_empty){
		items_to_check = next_empty - cur_ind;
	}
	// the next empty slot needs to be
	// wrapped around
	else{
		items_to_check = (size - cur_ind) + next_empty + 1;
	}


	int key_cmp;
	uint64_t i = 0;
	while (i < items_to_check) {

		// compare the key
		key_cmp = memcmp(key, cur_table_key, key_size_bytes);
		// if we found the key
		if (key_cmp == 0){

			// if we want the key, we want the value immediately after, so we add key_size_bytes
			// to the current key
			void * table_value = (void *) (((uint64_t) cur_table_key) + key_size_bytes);

			if (ret_value){
				// now we want to copy the value and then can return
				// if we are copying then we assume the return value is just
				// directly the pointer to copy to
				if (to_copy_value) {
					memcpy((void *) ret_value, table_value, value_size_bytes);
				}
				else{
					*ret_value = table_value;
				}
			}

			return cur_ind;
		}

		// update the next key position which will be just 1 element higher so we can add the size of 1 item

		

		// next empty might have a returned a value that wrapped around
		// if the whole table
		cur_ind = (cur_ind + 1) % size;

		cur_table_key = (void *) (((uint64_t) table -> items) + (cur_ind * (key_size_bytes + value_size_bytes)));

		i += 1;
	}
	
	// We didn't find the element
	return -1;
}

// returns 0 on success, -1 on error

// does memcopiess of key and value into the table array
// assumes the content of the key cannot be 0 of size key_size_bytes
int dataflow_insert_table(Dataflow_Table * table, void * key, void * value) {

	uint64_t size = table -> size;
	uint64_t cnt = table -> cnt;

	// should only happen when cnt = max_size
	// otherwise we would have grown the table after the 
	// prior triggering insertion
	if (unlikely(cnt == size)){
		return -1;
	}

	long ret = dataflow_find_table(table, key, false, NULL);
	if (ret != -1){
		fprintf(stderr, "Error: key already exists in table. Cannot insert...\n");
		return -1;
	}

	// 1.) Lookup where to place this item in the table

	// acutally compute the hash index
	uint64_t hash_ind = ((table -> config).hash_func)(key, table -> size);

	// we already saw cnt != size so we are guaranteed for this to succeed
	long insert_ind = get_next_ind_table(table -> is_empty_bit_vector, table -> size, hash_ind);
	if (unlikely(insert_ind == -1)){
		fprintf(stderr, "Error: when trying to insert key into table.\n");
		return -1;
	}
	
	uint64_t key_size_bytes = (table -> config).key_size_bytes;
	uint64_t value_size_bytes = (table -> config).value_size_bytes;


	// Now we want to insert into the table by copying key and value 
	// into the appropriate insert ind and then setting the 
	// is_empty bit to 0 within the bit vector


	// 2.) Copy the key and value into the table 
	//		(memory has already been allocated for them within the table)

	void * items = table -> items;

	// setting the position for the key in the table
	// this is based on the insert_index that was returned to us
	void * key_pos = (void *) (((uint64_t) items) + (insert_ind * (key_size_bytes + value_size_bytes)));
	// advance passed the key we will insert
	void * value_pos = (void *) (((uint64_t) key_pos) + key_size_bytes);

	// Actually place in table
	memcpy(key_pos, key, key_size_bytes);
	memcpy(value_pos, value, value_size_bytes);


	// 3.) Update bookkeeping values

	cnt += 1;
	table -> cnt = cnt;


	// clearing the entry for this insert_ind in the bit vector

	// the bucket's upper bits represent index into the bit vector elements
	// and the low order 6 bits represent offset into element. Set to 0 

	// needs to be 1ULL otherwise will default to 1 byte
	(table -> is_empty_bit_vector)[insert_ind >> 6] &= ~(1ULL << (insert_ind & 0x3F));


	// 4.) Potentially resize

	// check if we exceed load and are below max cap
	// if so, grow
	float load_factor = (table -> config).load_factor;
	uint64_t max_size = (table -> config).max_size;
	// make sure types are correct when multiplying uint64_t by float
	uint64_t load_cap = (uint64_t) (size * load_factor);
	if ((size < max_size) && (cnt > load_cap)){
		// casting from float to uint64 is fine
		size = (uint64_t) (size * (1.0f / load_factor));
		if (size > max_size){
			size = max_size;
		}
		int ret = resize_table(table, size);
		// check if there was an error growing table
		// indicate that there was an error by being able to insert

		// might want a different error message here because this is fatal
		if (unlikely(ret == -1)){
			return -1;
		}
	}

	return 0;


}


// returns 0 upon successfully removing, -1 on error finding. 

// Note: Might want to have different return value
// from function to indicate a fatal error that could have occurred within resized (in the gap of freeing larger
// table and allocating new, smaller one)

// if copy_val is set to true then copy back the item
int dataflow_remove_table(Dataflow_Table * table, void * key, void * ret_value) {


	// remove is equivalent to find, except we need to also:
	//	a.) Confirm that other positions can still be found (by replacing as needed)
	//	b.) mark the empty bit/decrease count
	//	c.) potentially shrink


	// 1.) Search for item!

	// if the item existed this will handle copying
	// because we are removing from the table we need to copy the value
	long empty_ind = dataflow_find_table(table, key, true, (void **) ret_value);

	// item didn't exist so we immediately return
	if (empty_ind == -1){
		return -1;
	}


	// 2.) Ensure that we will still be able to find other items that have collided
	// 		with a hash that is >= to the index we removed
	uint64_t size = table -> size;
	uint64_t key_size_bytes = (table -> config).key_size_bytes;
	uint64_t value_size_bytes = (table -> config).value_size_bytes;
	uint64_t * is_empty_bit_vector = table -> is_empty_bit_vector;
	long next_empty = get_next_ind_table(is_empty_bit_vector, size, empty_ind);

	long items_to_check;


	// Now we are only checking AFTER the "removed" index

	// we know next_empty != empty_ind because we already checked this item existed
	if (empty_ind < next_empty){
		items_to_check = next_empty - empty_ind - 1;
	}
	// the next empty slot needs to be
	// wrapped around
	else{
		// -1 because we start after the empty ind
		items_to_check = (size - empty_ind - 1) + next_empty;
	}

	long i = 0;
	uint64_t hash_ind;
	void * cur_table_key;
	long cur_ind = (empty_ind + 1) % size;
	void * empty_table_key = (void *) (((uint64_t) table -> items) + (empty_ind * (key_size_bytes + value_size_bytes)));
	
	while (i < items_to_check){ 

		// assset (is_empty_bit_vector[cur_ind >> 6] & (cur_ind & 0x3F)) == 1

		cur_table_key = (void *) (((uint64_t) table -> items) + (cur_ind * (key_size_bytes + value_size_bytes)));

		// get the hash index for the entry in the table to see if it could still be found
		hash_ind = ((table -> config).hash_func)(cur_table_key, size);

		// Ref: https://stackoverflow.com/questions/9127207/hash-table-why-deletion-is-difficult-in-open-addressing-scheme

		// If cur_table_key wouldn't be able to be found again we need to move it to the 
		// empty_ind position to ensure that it could be
		if (((cur_ind > empty_ind) && ((long) hash_ind <= empty_ind || (long) hash_ind > cur_ind))
			|| ((cur_ind < empty_ind) && ((long) hash_ind <= empty_ind && (long) hash_ind > cur_ind))){
			
			// perform the replacement
			memcpy(empty_table_key, cur_table_key, key_size_bytes + value_size_bytes);

			// now reset the next time we might need to replace
			empty_ind = cur_ind;
			empty_table_key = cur_table_key;
		}

		i += 1;

		// update position of cur table key and index
		cur_ind = (cur_ind + 1) % size;
		

	}

	// 3.) Do proper bookkeeping. Mark the last empty slot (after re-shuffling) as empty now

	// otherwise we need to update
	table -> cnt -= 1;

	// clearing the entry for this insert_ind in the bit vector

	// remove element from table
	memset(empty_table_key, 0, key_size_bytes + value_size_bytes);

	// the bucket's upper bits represent index into the bit vector elements
	// and the low order 6 bits represent offset into element. 

	// Set to 1 to indicate this bucket is now free 
	(table -> is_empty_bit_vector)[empty_ind >> 6] |= (1ULL << (empty_ind & 0x3F));


	// 3.) Check if this removal triggered 


	// check if we should shrink

	// now this is updated afer we decremented
	uint64_t cnt = table -> cnt;


	float shrink_factor = (table -> config).shrink_factor;
	uint64_t min_size = (table -> config).min_size;
	// make sure types are correct when multiplying uint64_t by float
	uint64_t shrink_cap = (uint64_t) (size * shrink_factor);
	if ((size > min_size) && (cnt < shrink_cap)) {
		size = (uint64_t) (size * (1 - shrink_factor));
		if (size < min_size){
			size = min_size;
		}
		int ret = resize_table(table, size);
		// check if there was an error growing table
		// fatal error here
		if (unlikely(ret == -1)){
			return -1;
		}
	}

	return 0;

}
