#include "nvidia_ops.h"

extern "C" __global__ void default_select_experts_fp32_kernel(int total_tokens, int n_experts, int top_k_experts,  float * X_routed, float * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert) {

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
                routed_val = X_routed[(orig_token_row * (uint64_t) n_experts) + (i + lane_id)];

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

            // Also only 1 thread out of warp is working, but 
            // if top_k is a large number then this could be problematic...
            if (lane_id == 0){

                for (int j = 0; j < WARP_SIZE; j++){
                    // see if this expert within the top_k
                    if ((warp_cur_vals[warp_id * WARP_SIZE + j] > min_top_k_val) && ((i + j) < n_experts)) {

                        if (warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind] != CONST_DEV_FLOAT_NEG_INF){
                            top_k_sum -= warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind];
                        }

                        warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind] = warp_cur_vals[warp_id * WARP_SIZE + j];
                        warp_top_k_expert_inds[warp_id * top_k_experts + min_top_k_ind] = (uint16_t) (i + j);

                        top_k_sum += warp_top_k_expert_vals[warp_id * top_k_experts + min_top_k_ind];

                        // NOTE: this is the wasteful loop
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

        // main block will be responsible for doing prefix sum 
        // (only a small number of experts so easier this way..)
        if (blockIdx.x == 0){
            while (num_blocks_completed < num_blocks){
                num_blocks_completed = atomicCAS(&(expert_counts[n_experts]), num_blocks, 0);
            }
        }
    }

    // everyone except lead warp is done
    if (blockIdx.x != 0 || warp_id != 0){
        return;
    }

    __syncwarp();

    // just block 0 warp 0 is left

    int prev_cumsum = 0;
    int tmp;
    int cur_expert_cnt;

    for (int e = lane_id; e < n_experts; e+=WARP_SIZE){

        cur_expert_cnt = expert_counts[e];

        // now do local prefix sum
        for (int offset = 1; offset < 32; offset <<= 1){
            // assumning n_experts will be a multiple of 32...
            tmp = __shfl_up_sync(0xFFFFFFFF, cur_expert_cnt, offset);
            if (lane_id >= offset){
                cur_expert_cnt += tmp;
            }
        }

         // add results with prev_cumsum;
        expert_counts_cumsum[e] = prev_cumsum + cur_expert_cnt;

        prev_cumsum = __shfl_sync(0xFFFFFFFF, prev_cumsum + cur_expert_cnt, 31);

        // reset this array before the call to route_experts
        num_routed_by_expert[e] = 0;
    }    
}



extern "C" __global__ void default_select_experts_fp16_kernel(int total_tokens, int n_experts, int top_k_experts,  __half * X_routed, __half * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert) {

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
                routed_val = __half2float(X_routed[(orig_token_row * (uint64_t) n_experts) + (i + lane_id)]);

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
                token_expert_weights[(orig_token_row * (uint64_t) top_k_experts) + k] = __float2half(warp_top_k_expert_vals[warp_id * top_k_experts + k] / top_k_sum);

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

        // main block will be responsible for doing prefix sum 
        // (only a small number of experts so easier this way..)
        if (blockIdx.x == 0){
            while (num_blocks_completed < num_blocks){
                num_blocks_completed = atomicCAS(&(expert_counts[n_experts]), num_blocks, 0);
            }
        }
    }

    // everyone except lead warp is done
    if (blockIdx.x != 0 || warp_id != 0){
        return;
    }

    __syncwarp();

    // just block 0 warp 0 is left

    int prev_cumsum = 0;
    int tmp;
    int cur_expert_cnt;

    for (int e = lane_id; e < n_experts; e+=WARP_SIZE){

        cur_expert_cnt = expert_counts[e];

        // now do local prefix sum
        for (int offset = 1; offset < 32; offset <<= 1){
            // assumning n_experts will be a multiple of 32...
            tmp = __shfl_up_sync(0xFFFFFFFF, cur_expert_cnt, offset);
            if (lane_id >= offset){
                cur_expert_cnt += tmp;
            }
        }

         // add results with prev_cumsum;
        expert_counts_cumsum[e] = prev_cumsum + cur_expert_cnt;

        prev_cumsum = __shfl_sync(0xFFFFFFFF, prev_cumsum + cur_expert_cnt, 31);

        // reset this array before the call to route_experts
        num_routed_by_expert[e] = 0;
    }    
}


extern "C" __global__ void default_select_experts_bf16_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_bfloat16 * X_routed, __nv_bfloat16 * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert) {

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
                token_expert_weights[(orig_token_row * (uint64_t) top_k_experts) + k] = __float2bfloat16(warp_top_k_expert_vals[warp_id * top_k_experts + k] / top_k_sum);

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

        // main block will be responsible for doing prefix sum 
        // (only a small number of experts so easier this way..)
        if (blockIdx.x == 0){
            while (num_blocks_completed < num_blocks){
                num_blocks_completed = atomicCAS(&(expert_counts[n_experts]), num_blocks, 0);
            }
        }
    }

    // everyone except lead warp is done
    if (blockIdx.x != 0 || warp_id != 0){
        return;
    }

    __syncwarp();

    // just block 0 warp 0 is left

    int prev_cumsum = 0;
    int tmp;
    int cur_expert_cnt;

    for (int e = lane_id; e < n_experts; e+=WARP_SIZE){

        cur_expert_cnt = expert_counts[e];

        // now do local prefix sum
        for (int offset = 1; offset < 32; offset <<= 1){
            // assumning n_experts will be a multiple of 32...
            tmp = __shfl_up_sync(0xFFFFFFFF, cur_expert_cnt, offset);
            if (lane_id >= offset){
                cur_expert_cnt += tmp;
            }
        }

         // add results with prev_cumsum;
        expert_counts_cumsum[e] = prev_cumsum + cur_expert_cnt;

        prev_cumsum = __shfl_sync(0xFFFFFFFF, prev_cumsum + cur_expert_cnt, 31);

        // reset this array before the call to route_experts
        num_routed_by_expert[e] = 0;
    }    
}



extern "C" __global__ void default_select_experts_fp8e4m3_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_fp8_e4m3 * X_routed, __nv_fp8_e4m3 * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert) {

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
                routed_val = float(X_routed[(orig_token_row * (uint64_t) n_experts) + (i + lane_id)]);

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
                token_expert_weights[(orig_token_row * (uint64_t) top_k_experts) + k] = __nv_fp8_e4m3(warp_top_k_expert_vals[warp_id * top_k_experts + k] / top_k_sum);

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

        // main block will be responsible for doing prefix sum 
        // (only a small number of experts so easier this way..)
        if (blockIdx.x == 0){
            while (num_blocks_completed < num_blocks){
                num_blocks_completed = atomicCAS(&(expert_counts[n_experts]), num_blocks, 0);
            }
        }
    }

    // everyone except lead warp is done
    if (blockIdx.x != 0 || warp_id != 0){
        return;
    }

    __syncwarp();

    // just block 0 warp 0 is left

    int prev_cumsum = 0;
    int tmp;
    int cur_expert_cnt;

    for (int e = lane_id; e < n_experts; e+=WARP_SIZE){

        cur_expert_cnt = expert_counts[e];

        // now do local prefix sum
        for (int offset = 1; offset < 32; offset <<= 1){
            // assumning n_experts will be a multiple of 32...
            tmp = __shfl_up_sync(0xFFFFFFFF, cur_expert_cnt, offset);
            if (lane_id >= offset){
                cur_expert_cnt += tmp;
            }
        }

         // add results with prev_cumsum;
        expert_counts_cumsum[e] = prev_cumsum + cur_expert_cnt;

        prev_cumsum = __shfl_sync(0xFFFFFFFF, prev_cumsum + cur_expert_cnt, 31);

        // reset this array before the call to route_experts
        num_routed_by_expert[e] = 0;
    }    
}


extern "C" __global__ void default_select_experts_fp8e5m2_kernel(int total_tokens, int n_experts, int top_k_experts,  __nv_fp8_e5m2 * X_routed, __nv_fp8_e5m2 * token_expert_weights, uint16_t * chosen_experts, int * expert_counts, int * expert_counts_cumsum, int * num_routed_by_expert) {

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
                routed_val = float(X_routed[(orig_token_row * (uint64_t) n_experts) + (i + lane_id)]);

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
                token_expert_weights[(orig_token_row * (uint64_t) top_k_experts) + k] = __nv_fp8_e5m2(warp_top_k_expert_vals[warp_id * top_k_experts + k] / top_k_sum);

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

        // main block will be responsible for doing prefix sum 
        // (only a small number of experts so easier this way..)
        if (blockIdx.x == 0){
            while (num_blocks_completed < num_blocks){
                num_blocks_completed = atomicCAS(&(expert_counts[n_experts]), num_blocks, 0);
            }
        }
    }

    // everyone except lead warp is done
    if (blockIdx.x != 0 || warp_id != 0){
        return;
    }

    __syncwarp();

    // just block 0 warp 0 is left

    int prev_cumsum = 0;
    int tmp;
    int cur_expert_cnt;

    for (int e = lane_id; e < n_experts; e+=WARP_SIZE){

        cur_expert_cnt = expert_counts[e];

        // now do local prefix sum
        for (int offset = 1; offset < 32; offset <<= 1){
            // assumning n_experts will be a multiple of 32...
            tmp = __shfl_up_sync(0xFFFFFFFF, cur_expert_cnt, offset);
            if (lane_id >= offset){
                cur_expert_cnt += tmp;
            }
        }

         // add results with prev_cumsum;
        expert_counts_cumsum[e] = prev_cumsum + cur_expert_cnt;

        prev_cumsum = __shfl_sync(0xFFFFFFFF, prev_cumsum + cur_expert_cnt, 31);

        // reset this array before the call to route_experts
        num_routed_by_expert[e] = 0;
    }    
}