#include "nvidia_ops.h"

// this is 32
#define BWD_W_COMBINE_WARPS_PER_BLOCK WARP_SIZE

// each warp loads in 1 slices and also aggregates 1 slices per round
#define BWD_W_ORIG_COLS_PER_BLOCK 32


extern "C" __global__ void default_rms_norm_bwd_w_combine_fp32_kernel(int * num_orig_blocks_launched, int model_dim, float * dW_workspace, float * dW){

    int dim_offset = BWD_W_ORIG_COLS_PER_BLOCK * blockIdx.x;

    if (dim_offset >= model_dim){
        return;
    }

    int num_to_combine = *num_orig_blocks_launched;

    __shared__ float dW_slice[BWD_W_ORIG_COLS_PER_BLOCK];

    // add +1 to avoid bank conflicts
    // we will have BWD_W_COMBINE_THREADS_PER_BLOCK rows and BWD_W_ORIG_COLS_PER_BLOCK columns
    __shared__ float dW_workspace_block[BWD_W_COMBINE_WARPS_PER_BLOCK][BWD_W_ORIG_COLS_PER_BLOCK + 1];


    int num_dims = BWD_W_ORIG_COLS_PER_BLOCK;
    if (dim_offset + BWD_W_ORIG_COLS_PER_BLOCK > model_dim){
        num_dims = model_dim - dim_offset;
    }

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW_slice[i] = dW[dim_offset + i];
    }

    __syncthreads();

    int num_rounds = ceilf((float) num_to_combine / (float)BWD_W_COMBINE_WARPS_PER_BLOCK);

    int warp_agg_rounds = ceilf((float)BWD_W_ORIG_COLS_PER_BLOCK / (float)WARP_SIZE);

    int cur_warp_agg_round;
    int cur_orig_dim;;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float val;

    unsigned int warp_mask = 0xffffffff;

    for (int r = 0; r < num_rounds; r++){
        int b = r * BWD_W_COMBINE_WARPS_PER_BLOCK + warp_id;

        // load in values for this warp
        if (b < num_to_combine){
            // now the section of dW_workspace will be cached nicely...
            // so dims > d will be near this cache line...
            for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
                if (d < num_dims){
                    val = dW_workspace[b * model_dim + dim_offset + d];
                }
                else{
                    val = 0;
                }

                dW_workspace_block[warp_id][d] = val;
                
            }
        }
        else{
            for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
                dW_workspace_block[warp_id][d] = 0;
            }
        }

        __syncthreads();

        // now we want to aggregate in vertically
        // have each warp work on a specific dimension

        cur_warp_agg_round = 0;
        cur_orig_dim = warp_id;

        while ((cur_warp_agg_round < warp_agg_rounds) && (cur_orig_dim < num_dims)){

            val = dW_workspace_block[lane_id][cur_orig_dim];
            
             // now combine the values within the warp and update dimension
            for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
                val += __shfl_down_sync(warp_mask, val, warp_offset);
            }

            if (lane_id == 0){
                dW_slice[cur_orig_dim] += val;
            }

            cur_orig_dim += BWD_W_COMBINE_WARPS_PER_BLOCK;
            cur_warp_agg_round++;
        }

        __syncthreads();
    }

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW[dim_offset + i] = dW_slice[i];
    }
}

extern "C" __global__ void default_rms_norm_bwd_w_combine_fp16_kernel(int * num_orig_blocks_launched, int model_dim, __half * dW_workspace, __half * dW){

    int dim_offset = BWD_W_ORIG_COLS_PER_BLOCK * blockIdx.x;

    if (dim_offset >= model_dim){
        return;
    }

    int num_to_combine = *num_orig_blocks_launched;

    __shared__ float dW_slice[BWD_W_ORIG_COLS_PER_BLOCK];

    // add +1 to avoid bank conflicts
    // we will have BWD_W_COMBINE_THREADS_PER_BLOCK rows and BWD_W_ORIG_COLS_PER_BLOCK columns
    __shared__ float dW_workspace_block[BWD_W_COMBINE_WARPS_PER_BLOCK][BWD_W_ORIG_COLS_PER_BLOCK + 1];


    int num_dims = BWD_W_ORIG_COLS_PER_BLOCK;
    if (dim_offset + BWD_W_ORIG_COLS_PER_BLOCK > model_dim){
        num_dims = model_dim - dim_offset;
    }

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW_slice[i] = dW[dim_offset + i];
    }

    __syncthreads();

    int num_rounds = ceilf((float) num_to_combine / (float)BWD_W_COMBINE_WARPS_PER_BLOCK);

    int warp_agg_rounds = ceilf((float)BWD_W_ORIG_COLS_PER_BLOCK / (float)WARP_SIZE);

    int cur_warp_agg_round;
    int cur_orig_dim;;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float val;

    unsigned int warp_mask = 0xffffffff;

    for (int r = 0; r < num_rounds; r++){
        int b = r * BWD_W_COMBINE_WARPS_PER_BLOCK + warp_id;

        // load in values for this warp
        if (b < num_to_combine){
            // now the section of dW_workspace will be cached nicely...
            // so dims > d will be near this cache line...
            for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
                if (d < num_dims){
                    val = __half2float(dW_workspace[b * model_dim + dim_offset + d]);
                }
                else{
                    val = 0;
                }

                dW_workspace_block[warp_id][d] = val;
                
            }
        }
        else{
            for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
                dW_workspace_block[warp_id][d] = 0;
            }
        }

        __syncthreads();

        // now we want to aggregate in vertically
        // have each warp work on a specific dimension

        cur_warp_agg_round = 0;
        cur_orig_dim = warp_id;

        while ((cur_warp_agg_round < warp_agg_rounds) && (cur_orig_dim < num_dims)){

            val = dW_workspace_block[lane_id][cur_orig_dim];
            
             // now combine the values within the warp and update dimension
            for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
                val += __shfl_down_sync(warp_mask, val, warp_offset);
            }

            if (lane_id == 0){
                dW_slice[cur_orig_dim] += val;
            }

            cur_orig_dim += BWD_W_COMBINE_WARPS_PER_BLOCK;
            cur_warp_agg_round++;
        }

        __syncthreads();
    }

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW[dim_offset + i] = __float2half(dW_slice[i]);
    }

}

extern "C" __global__ void default_rms_norm_bwd_w_combine_bf16_kernel(int * num_orig_blocks_launched, int model_dim, __nv_bfloat16 * dW_workspace, __nv_bfloat16 * dW){
    
    int dim_offset = BWD_W_ORIG_COLS_PER_BLOCK * blockIdx.x;

    if (dim_offset >= model_dim){
        return;
    }

    int num_to_combine = *num_orig_blocks_launched;

    __shared__ float dW_slice[BWD_W_ORIG_COLS_PER_BLOCK];

    // add +1 to avoid bank conflicts
    // we will have BWD_W_COMBINE_THREADS_PER_BLOCK rows and BWD_W_ORIG_COLS_PER_BLOCK columns
    __shared__ float dW_workspace_block[BWD_W_COMBINE_WARPS_PER_BLOCK][BWD_W_ORIG_COLS_PER_BLOCK + 1];


    int num_dims = BWD_W_ORIG_COLS_PER_BLOCK;
    if (dim_offset + BWD_W_ORIG_COLS_PER_BLOCK > model_dim){
        num_dims = model_dim - dim_offset;
    }

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW_slice[i] = dW[dim_offset + i];
    }

    __syncthreads();

    int num_rounds = ceilf((float) num_to_combine / (float)BWD_W_COMBINE_WARPS_PER_BLOCK);

    int warp_agg_rounds = ceilf((float)BWD_W_ORIG_COLS_PER_BLOCK / (float)WARP_SIZE);

    int cur_warp_agg_round;
    int cur_orig_dim;;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float val;

    unsigned int warp_mask = 0xffffffff;

    for (int r = 0; r < num_rounds; r++){
        int b = r * BWD_W_COMBINE_WARPS_PER_BLOCK + warp_id;

        // load in values for this warp
        if (b < num_to_combine){
            // now the section of dW_workspace will be cached nicely...
            // so dims > d will be near this cache line...
            for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
                if (d < num_dims){
                    val = __bfloat162float(dW_workspace[b * model_dim + dim_offset + d]);
                }
                else{
                    val = 0;
                }

                dW_workspace_block[warp_id][d] = val;
                
            }
        }
        else{
            for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
                dW_workspace_block[warp_id][d] = 0;
            }
        }

        __syncthreads();

        // now we want to aggregate in vertically
        // have each warp work on a specific dimension

        cur_warp_agg_round = 0;
        cur_orig_dim = warp_id;

        while ((cur_warp_agg_round < warp_agg_rounds) && (cur_orig_dim < num_dims)){

            val = dW_workspace_block[lane_id][cur_orig_dim];
            
             // now combine the values within the warp and update dimension
            for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
                val += __shfl_down_sync(warp_mask, val, warp_offset);
            }

            if (lane_id == 0){
                dW_slice[cur_orig_dim] += val;
            }

            cur_orig_dim += BWD_W_COMBINE_WARPS_PER_BLOCK;
            cur_warp_agg_round++;
        }

        __syncthreads();
    }

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW[dim_offset + i] = __float2bfloat16(dW_slice[i]);
    }
}
