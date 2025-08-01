#include "nvidia_ops.h"

extern "C" __global__ void default_select_experts_fp32_kernel(int total_tokens, int n_experts, int top_k_experts,  float * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum) {
   return;
}



extern "C" __global__ void default_select_experts_fp16_kernel(int total_tokens, int n_experts, int top_k_experts,  __half * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum) {
    return;
}

extern "C" __global__ void default_select_experts_bf16_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_bfloat16 * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * host_expert_counts) {

    // DETERMINING HOW MANY TOKENS THIS THREADBLOCK IS RESPONSIBLE FOR

    int num_blocks = gridDim.x;

    // each warp will update the number of tokens it assigns to each expert
    // and at the end a single thread-block leader will aggregate across warps
    // and then atomically update across blocks

    // Having each threadblock process configurable number of rows
    // in order to use specfied number of SMs and also better reductions
    // (happening within a threadblock) when tracking expert sizes
    int row_base = blockIdx.x;

    if (row_base >= total_tokens){
        // mark this block as completed...
        if (threadIdx.x == 0){
            atomicAdd(&(expert_counts[n_experts]), 1);
        }
        return;
    }

    int rows_per_block = total_tokens / num_blocks;
    
    int rows_remain = total_tokens % num_blocks;
    int row_offset;
    if (row_base < rows_remain){
        // this block will need to do an extra row
        rows_per_block += 1;
        // all prior blocks also had an extra row
        row_offset = row_base * rows_per_block;
    }
    else{
        row_offset = row_base * rows_per_block + rows_remain;
    }

    int thread_id = threadIdx.x;

    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;

    if (row_offset + warp_id >= row_offset + rows_per_block){
        return;
    }

    // each warp is responsible for a row and every lane
    // will update it's current value for leader to scan through
    // the top k values in iteration and see if any of them 
    // will become part of new top k
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ int block_completion_num;

    // this gets dynamically allocated the size of model_dim
    extern __shared__ uint8_t sdata[];

    // could make this __half is smem is bottleneck
    float * warp_cur_vals = (float *) sdata;


    float * warp_top_k_expert_vals = warp_cur_vals + (num_warps * WARP_SIZE);

    uint16_t * warp_top_k_expert_inds = (uint16_t *) (warp_top_k_expert_vals + (num_warps * top_k_experts));

    // smem counts with atomicAdd before doing global sync in order to copy token row correctly into
    int * block_expert_counts = (int *) (warp_top_k_expert_inds + (num_warps * top_k_experts));


    uint64_t orig_token_row;
    uint16_t expert_id;

    // running cutoff to determine if needing to search
    // through
    float min_top_k_val;
    uint16_t min_top_k_ind;

    float top_k_sum;
    float routed_val;
    float max_val_top_k;


    for (int e = thread_id; e < n_experts; e+=blockDim.x){
        block_expert_counts[e] = 0;
    }

    __syncthreads();

    for (int i = lane_id; i < top_k_experts; i+=WARP_SIZE) {
        warp_top_k_expert_vals[warp_id * top_k_experts + i] = CONST_DEV_FLOAT_NEG_INF;
        warp_top_k_expert_inds[warp_id * top_k_experts + i] = (uint16_t) i;
    }

    __syncwarp();

    // every warp is responsible for a row...
    for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){
        
        orig_token_row = (uint64_t) (row_id);

        top_k_sum = 0;
        max_val_top_k = CONST_DEV_FLOAT_NEG_INF;
        // reset per-token init values...
        min_top_k_val = CONST_DEV_FLOAT_NEG_INF;
        min_top_k_ind = 0;
        
        // reset the top_k_expert_vals
        if (lane_id == 0){
            for (int i = 0; i < top_k_experts; i++){
                warp_top_k_expert_vals[warp_id * top_k_experts + i] = CONST_DEV_FLOAT_NEG_INF;
                warp_top_k_expert_inds[warp_id * top_k_experts + i] = 0xFFFF;
            }
        }

        __syncwarp();

        // every lane within warp will fetch a value
        // n_experts typically low (10's - 100's)
        for (int i = 0; i < n_experts; i+=WARP_SIZE){

            if ((i + lane_id) < n_experts){
                routed_val = __bfloat162float(X_routed[(orig_token_row * (uint64_t) n_experts) + (i + lane_id)]);

                warp_cur_vals[warp_id * WARP_SIZE + lane_id] = routed_val;
            }
            else{
                warp_cur_vals[warp_id * WARP_SIZE + lane_id] = CONST_DEV_FLOAT_NEG_INF;
            }

            __syncwarp();

            // kinda wasteful implementaion with excessive looping
            // (because a top-k heap should be used)
            // but typically small number of experts and top_k experts
            // so the redundant "search fo next smallest" iterations are fine and simpler...
            if (lane_id == 0){

                for (int j = 0; j < WARP_SIZE; j++){
                    // see if this expert within the top_k
                    // most of the time this will be false, so 
                    // wont need to do more work
                    if ((warp_cur_vals[warp_id * WARP_SIZE + j] > min_top_k_val) && ((i + j) < n_experts)) {

                        if (warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind] != CONST_DEV_FLOAT_NEG_INF){
                            top_k_sum -= warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind];
                        }

                        if (warp_cur_vals[warp_id * WARP_SIZE + j] > max_val_top_k){
                            max_val_top_k = warp_cur_vals[warp_id * WARP_SIZE + j];
                        }

                        warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind] = warp_cur_vals[warp_id * WARP_SIZE + j];
                        warp_top_k_expert_inds[warp_id * top_k_experts + min_top_k_ind] = (uint16_t) (i + j);

                        top_k_sum += warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind];

                        // reset min top k val
                        min_top_k_val = CONST_DEV_FLOAT_INF;
                        for (int k = 0; k < top_k_experts; k++){
                            if (warp_top_k_expert_vals[warp_id * top_k_experts + k] < min_top_k_val){
                                min_top_k_val = warp_top_k_expert_vals[warp_id * top_k_experts + k];
                                min_top_k_ind = (uint16_t) k;
                            }
                        }
                    }
                }
            }

            __syncwarp();
        }


        // get softmax denominator of top k experts
        // max_val_top_k is the max
        if (lane_id == 0){

            float top_k_sum_exp = 0;

            // now we can update this tokens weights...
            // can easy parallelize across lanes...
            for (int k = 0; k < top_k_experts; k++){;
                top_k_sum_exp += expf(warp_top_k_expert_vals[warp_id * top_k_experts + k] - max_val_top_k);
            }

            for (int k = 0; k < top_k_experts; k++){

                expert_id = warp_top_k_expert_inds[warp_id * top_k_experts + k];

                // set the gate value
                token_expert_weights[(orig_token_row * (uint64_t) top_k_experts) + k] = expf(warp_top_k_expert_vals[warp_id * top_k_experts + k]- max_val_top_k) / top_k_sum_exp;

                // add this expert to array in order for it's token to be copied
                // into holding zone for "expert_id"
                chosen_experts[(orig_token_row * (uint64_t) top_k_experts) + k] = expert_id;

                // assert expert_id < n_experts
                atomicAdd(&(block_expert_counts[expert_id]), 1);
            }
        }
        
        __syncwarp();
    }

    // finished processing all tokens in this block...
    __syncthreads();

    // do atomic adds for all experts
    for (int e = thread_id; e < n_experts; e+=blockDim.x){
        atomicAdd(&(expert_counts[e]), block_expert_counts[e]);
    }

    int num_blocks_completed;
    if (thread_id == 0){
        num_blocks_completed = atomicAdd(&(expert_counts[n_experts]), 1);

        block_completion_num = num_blocks_completed;
        
    }

    __syncthreads();

    // atomicAdd returns old value.
    // the last block to finish will will see num_blocks - 1 as its completion num
    if (block_completion_num < (gridDim.x - 1)){
        return;
    }

    // just the final block is left...

    // everyone except lead warp is done
    if (warp_id != 0){
        return;
    }

    // just block 0 warp 0 is left

    // Initialize the cumulative sum for this warp's chunks.
    int warp_total_sum = 0;

    // Iterate over the expert_counts array in chunks of WARP_SIZE (32).
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        // Calculate the global index 'e' for the current thread.
        int e = i + lane_id;

        // 1. Load the count for the current expert.
        // If the index is out of bounds, use 0 to avoid affecting the sum.
        int lane_val = 0;
        if (e < n_experts) {
            lane_val = expert_counts[e];
        }

        // 2. Perform a warp-wide INCLUSIVE prefix sum (scan).
        // Each lane calculates the sum of 'lane_val' from all lanes with a lower or equal ID.
        int inclusive_sum = lane_val;
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            // Fetch value from a lane 'offset' positions lower.
            int tmp = __shfl_up_sync(0xFFFFFFFF, inclusive_sum, offset);
            // Add it to the current lane's sum if it's a valid contributor.
            if (lane_id >= offset) {
                inclusive_sum += tmp;
            }
        }

        // 3. Convert the inclusive sum to an EXCLUSIVE sum.
        // The exclusive sum for lane 'j' is the inclusive sum from lane 'j-1'.
        int exclusive_sum = __shfl_up_sync(0xFFFFFFFF, inclusive_sum, 1);
        // Lane 0's exclusive sum is always 0 within its local chunk.
        if (lane_id == 0) {
            exclusive_sum = 0;
        }

        // 4. Write the final cumulative sum to the output array.
        // This is the sum of all previous chunks ('warp_total_sum') plus the
        // local exclusive sum within this chunk.
        if (e < n_experts) {
            expert_counts_cumsum[e] = warp_total_sum + exclusive_sum;
        }

        // 5. Update the total sum for the next iteration.
        // Get the total sum of the current chunk by reading the inclusive_sum from the last lane.
        int chunk_total = __shfl_sync(0xFFFFFFFF, inclusive_sum, WARP_SIZE - 1);
        // Add this chunk's total to the running total for the next loop iteration.
        warp_total_sum += chunk_total;
    }

    // COPY THE EXPERT COUNTS TO HOST
    // NEED TO DO IT WITHIN KERNEL, OTHERWISE MIGHT BE BEHIND in DMA QUEUE!

    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        int e = i + lane_id;

        if (e < n_experts) {
            host_expert_counts[e] = expert_counts[e];
        }
    }
}


extern "C" __global__ void default_select_experts_sigmoid_bf16_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_bfloat16 * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * host_expert_counts) {

    // DETERMINING HOW MANY TOKENS THIS THREADBLOCK IS RESPONSIBLE FOR

    int num_blocks = gridDim.x;

    // each warp will update the number of tokens it assigns to each expert
    // and at the end a single thread-block leader will aggregate across warps
    // and then atomically update across blocks

    // Having each threadblock process configurable number of rows
    // in order to use specfied number of SMs and also better reductions
    // (happening within a threadblock) when tracking expert sizes
    int row_base = blockIdx.x;

    if (row_base >= total_tokens){
        // mark this block as completed...
        if (threadIdx.x == 0){
            atomicAdd(&(expert_counts[n_experts]), 1);
        }
        return;
    }

    int rows_per_block = total_tokens / num_blocks;
    
    int rows_remain = total_tokens % num_blocks;
    int row_offset;
    if (row_base < rows_remain){
        // this block will need to do an extra row
        rows_per_block += 1;
        // all prior blocks also had an extra row
        row_offset = row_base * rows_per_block;
    }
    else{
        row_offset = row_base * rows_per_block + rows_remain;
    }

    int thread_id = threadIdx.x;

    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;

    if (row_offset + warp_id >= row_offset + rows_per_block){
        return;
    }

    // each warp is responsible for a row and every lane
    // will update it's current value for leader to scan through
    // the top k values in iteration and see if any of them 
    // will become part of new top k
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ int block_completion_num;

    // this gets dynamically allocated the size of model_dim
    extern __shared__ uint8_t sdata[];

    // could make this __half is smem is bottleneck
    float * warp_cur_vals = (float *) sdata;


    float * warp_top_k_expert_vals = warp_cur_vals + (num_warps * WARP_SIZE);

    uint16_t * warp_top_k_expert_inds = (uint16_t *) (warp_top_k_expert_vals + (num_warps * top_k_experts));

    // smem counts with atomicAdd before doing global sync in order to copy token row correctly into
    int * block_expert_counts = (int *) (warp_top_k_expert_inds + (num_warps * top_k_experts));


    uint64_t orig_token_row;
    uint16_t expert_id;

    // running cutoff to determine if needing to search
    // through
    float min_top_k_val;
    uint16_t min_top_k_ind;

    float top_k_sum;
    float routed_val;


    for (int e = thread_id; e < n_experts; e+=blockDim.x){
        block_expert_counts[e] = 0;
    }

    __syncthreads();

    for (int i = lane_id; i < top_k_experts; i+=WARP_SIZE) {
        warp_top_k_expert_vals[warp_id * top_k_experts + i] = CONST_DEV_FLOAT_NEG_INF;
        warp_top_k_expert_inds[warp_id * top_k_experts + i] = (uint16_t) i;
    }

    __syncwarp();

    // every warp is responsible for a row...
    for (int row_id = row_offset + warp_id; row_id < row_offset + rows_per_block; row_id+=num_warps){
        
        orig_token_row = (uint64_t) (row_id);

        top_k_sum = 0;

        // reset per-token init values...
        min_top_k_val = CONST_DEV_FLOAT_NEG_INF;
        min_top_k_ind = 0;
        
        // reset the top_k_expert_vals
        if (lane_id == 0){
            for (int i = 0; i < top_k_experts; i++){
                warp_top_k_expert_vals[warp_id * top_k_experts + i] = CONST_DEV_FLOAT_NEG_INF;
                warp_top_k_expert_inds[warp_id * top_k_experts + i] = 0xFFFF;
            }
        }

        __syncwarp();

        // every lane within warp will fetch a value
        // n_experts typically low (10's - 100's)
        for (int i = 0; i < n_experts; i+=WARP_SIZE){

            if ((i + lane_id) < n_experts){
                routed_val = __bfloat162float(X_routed[(orig_token_row * (uint64_t) n_experts) + (i + lane_id)]);

                warp_cur_vals[warp_id * WARP_SIZE + lane_id] = 1.0 / (1 + expf(-1 * routed_val));
            }
            else{
                warp_cur_vals[warp_id * WARP_SIZE + lane_id] = CONST_DEV_FLOAT_NEG_INF;
            }

            __syncwarp();

            // kinda wasteful implementaion with excessive looping
            // (because a top-k heap should be used)
            // but typically small number of experts and top_k experts
            // so the redundant "search fo next smallest" iterations are fine and simpler...
            if (lane_id == 0){

                for (int j = 0; j < WARP_SIZE; j++){
                    // see if this expert within the top_k
                    // most of the time this will be false, so 
                    // wont need to do more work
                    if ((warp_cur_vals[warp_id * WARP_SIZE + j] > min_top_k_val) && ((i + j) < n_experts)) {

                        if (warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind] != CONST_DEV_FLOAT_NEG_INF){
                            top_k_sum -= warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind];
                        }

                        warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind] = warp_cur_vals[warp_id * WARP_SIZE + j];
                        warp_top_k_expert_inds[warp_id * top_k_experts + min_top_k_ind] = (uint16_t) (i + j);

                        top_k_sum += warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind];

                        // reset min top k val
                        min_top_k_val = CONST_DEV_FLOAT_INF;
                        for (int k = 0; k < top_k_experts; k++){
                            if (warp_top_k_expert_vals[warp_id * top_k_experts + k] < min_top_k_val){
                                min_top_k_val = warp_top_k_expert_vals[warp_id * top_k_experts + k];
                                min_top_k_ind = (uint16_t) k;
                            }
                        }
                    }
                }
            }

            __syncwarp();
        }


        if (lane_id == 0){

            // now we can update this tokens weights...
            // can easy parallelize across lanes...
            for (int k = 0; k < top_k_experts; k++){
            
                expert_id = warp_top_k_expert_inds[warp_id * top_k_experts + k];

                // set the gate value
                token_expert_weights[(orig_token_row * (uint64_t) top_k_experts) + k] = warp_top_k_expert_vals[warp_id * top_k_experts + k] / top_k_sum;

                // add this expert to array in order for it's token to be copied
                // into holding zone for "expert_id"
                chosen_experts[(orig_token_row * (uint64_t) top_k_experts) + k] = expert_id;

                // assert expert_id < n_experts
                atomicAdd(&(block_expert_counts[expert_id]), 1);
            }
        }
        
        __syncwarp();
    }

    // finished processing all tokens in this block...
    __syncthreads();

    // do atomic adds for all experts
    for (int e = thread_id; e < n_experts; e+=blockDim.x){
        atomicAdd(&(expert_counts[e]), block_expert_counts[e]);
    }

    int num_blocks_completed;
    if (thread_id == 0){
        num_blocks_completed = atomicAdd(&(expert_counts[n_experts]), 1);

        block_completion_num = num_blocks_completed;
        
    }

    __syncthreads();

    // atomicAdd returns old value.
    // the last block to finish will will see num_blocks - 1 as its completion num
    if (block_completion_num < (gridDim.x - 1)){
        return;
    }

    // just the final block is left...

    // everyone except lead warp is done
    if (warp_id != 0){
        return;
    }

    // just block 0 warp 0 is left

    // Initialize the cumulative sum for this warp's chunks.
    int warp_total_sum = 0;

    // Iterate over the expert_counts array in chunks of WARP_SIZE (32).
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        // Calculate the global index 'e' for the current thread.
        int e = i + lane_id;

        // 1. Load the count for the current expert.
        // If the index is out of bounds, use 0 to avoid affecting the sum.
        int lane_val = 0;
        if (e < n_experts) {
            lane_val = expert_counts[e];
        }

        // 2. Perform a warp-wide INCLUSIVE prefix sum (scan).
        // Each lane calculates the sum of 'lane_val' from all lanes with a lower or equal ID.
        int inclusive_sum = lane_val;
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            // Fetch value from a lane 'offset' positions lower.
            int tmp = __shfl_up_sync(0xFFFFFFFF, inclusive_sum, offset);
            // Add it to the current lane's sum if it's a valid contributor.
            if (lane_id >= offset) {
                inclusive_sum += tmp;
            }
        }

        // 3. Convert the inclusive sum to an EXCLUSIVE sum.
        // The exclusive sum for lane 'j' is the inclusive sum from lane 'j-1'.
        int exclusive_sum = __shfl_up_sync(0xFFFFFFFF, inclusive_sum, 1);
        // Lane 0's exclusive sum is always 0 within its local chunk.
        if (lane_id == 0) {
            exclusive_sum = 0;
        }

        // 4. Write the final cumulative sum to the output array.
        // This is the sum of all previous chunks ('warp_total_sum') plus the
        // local exclusive sum within this chunk.
        if (e < n_experts) {
            expert_counts_cumsum[e] = warp_total_sum + exclusive_sum;
        }

        // 5. Update the total sum for the next iteration.
        // Get the total sum of the current chunk by reading the inclusive_sum from the last lane.
        int chunk_total = __shfl_sync(0xFFFFFFFF, inclusive_sum, WARP_SIZE - 1);
        // Add this chunk's total to the running total for the next loop iteration.
        warp_total_sum += chunk_total;
    }

    // COPY THE EXPERT COUNTS TO HOST
    // NEED TO DO IT WITHIN KERNEL, OTHERWISE MIGHT BE BEHIND in DMA QUEUE!

    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        int e = i + lane_id;

        if (e < n_experts) {
            host_expert_counts[e] = expert_counts[e];
        }
    }
}



extern "C" __global__ void default_select_experts_fp8e4m3_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_fp8_e4m3 * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum) {
    return;
}


extern "C" __global__ void default_select_experts_fp8e5m2_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_fp8_e5m2 * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum) {
    return;
}