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

    __shared__ float warp_agg_block[BWD_W_COMBINE_WARPS_PER_BLOCK][BWD_W_ORIG_COLS_PER_BLOCK + 1];

    int num_dims = BWD_W_ORIG_COLS_PER_BLOCK;
    if (dim_offset + BWD_W_ORIG_COLS_PER_BLOCK > model_dim){
        num_dims = model_dim - dim_offset;
    }

    for (int i = threadIdx.x; i < BWD_W_ORIG_COLS_PER_BLOCK; i+=blockDim.x){
        if (i < num_dims){
            dW_slice[i] = dW[dim_offset + i];
        }
        else{
            dW_slice[i] = 0;
        }
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int num_warps = blockDim.x / 32;

    //assert(num_warps == BWD_W_COMBINE_WARPS_PER_BLOCK);

    for (int i = lane_id; i < BWD_W_COMBINE_WARPS_PER_BLOCK; i+=WARP_SIZE){
        warp_agg_block[warp_id][i] = 0;
    }

    __syncthreads();


    float val;

    unsigned int warp_mask = 0xffffffff;

    int cur_row = warp_id;

    while (cur_row < num_to_combine){

        // load in values for this warp
        for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
            if (d < num_dims){
                val = dW_workspace[cur_row * model_dim + dim_offset + d];
                warp_agg_block[warp_id][d] += val;
            }
        }

        cur_row += num_warps;
    }

    __syncthreads();

    // now combine each warps values for a given dimension...

    int warp_dim = warp_id;

    while (warp_dim < num_dims){

        if (lane_id < BWD_W_COMBINE_WARPS_PER_BLOCK){
            val = warp_agg_block[lane_id][warp_dim];
        }
        else{
            val = 0;
        }

        for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
            val += __shfl_down_sync(warp_mask, val, warp_offset);
        }
        
        if (lane_id == 0){
            dW_slice[warp_dim] += val;
        }

        warp_dim += num_warps;

    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW[dim_offset + i] = dW_slice[i];
    }
}

extern "C" __global__ void default_rms_norm_bwd_w_combine_fp16_kernel(int * num_orig_blocks_launched, int model_dim, float * dW_workspace, __half * dW){

     int dim_offset = BWD_W_ORIG_COLS_PER_BLOCK * blockIdx.x;

    if (dim_offset >= model_dim){
        return;
    }

    int num_to_combine = *num_orig_blocks_launched;

    __shared__ float dW_slice[BWD_W_ORIG_COLS_PER_BLOCK];

    __shared__ float warp_agg_block[BWD_W_COMBINE_WARPS_PER_BLOCK][BWD_W_ORIG_COLS_PER_BLOCK + 1];

    int num_dims = BWD_W_ORIG_COLS_PER_BLOCK;
    if (dim_offset + BWD_W_ORIG_COLS_PER_BLOCK > model_dim){
        num_dims = model_dim - dim_offset;
    }

    for (int i = threadIdx.x; i < BWD_W_ORIG_COLS_PER_BLOCK; i+=blockDim.x){
        if (i < num_dims){
            dW_slice[i] = dW[dim_offset + i];
        }
        else{
            dW_slice[i] = 0;
        }
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int num_warps = blockDim.x / 32;

    //assert(num_warps == BWD_W_COMBINE_WARPS_PER_BLOCK);

    for (int i = lane_id; i < BWD_W_COMBINE_WARPS_PER_BLOCK; i+=WARP_SIZE){
        warp_agg_block[warp_id][i] = 0;
    }

    __syncthreads();

    float val;

    unsigned int warp_mask = 0xffffffff;

    int cur_row = warp_id;

    while (cur_row < num_to_combine){

        // load in values for this warp
        for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
            if (d < num_dims){
                val = dW_workspace[cur_row * model_dim + dim_offset + d];
                warp_agg_block[warp_id][d] += val;
            }
        }

        cur_row += num_warps;
    }

    __syncthreads();

    // now combine each warps values for a given dimension...

    int warp_dim = warp_id;

    while (warp_dim < num_dims){

        if (lane_id < BWD_W_COMBINE_WARPS_PER_BLOCK){
            val = warp_agg_block[lane_id][warp_dim];
        }
        else{
            val = 0;
        }

        for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
            val += __shfl_down_sync(warp_mask, val, warp_offset);
        }
        
        if (lane_id == 0){
            dW_slice[warp_dim] += val;
        }

        warp_dim += num_warps;

    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW[dim_offset + i] = __float2half(dW_slice[i]);
    }

}

extern "C" __global__ void default_rms_norm_bwd_w_combine_bf16_kernel(int * num_orig_blocks_launched, int model_dim, float * dW_workspace, __nv_bfloat16 * dW){
    
     int dim_offset = BWD_W_ORIG_COLS_PER_BLOCK * blockIdx.x;

    if (dim_offset >= model_dim){
        return;
    }

    int num_to_combine = *num_orig_blocks_launched;

    __shared__ float dW_slice[BWD_W_ORIG_COLS_PER_BLOCK];

    __shared__ float warp_agg_block[BWD_W_COMBINE_WARPS_PER_BLOCK][BWD_W_ORIG_COLS_PER_BLOCK + 1];

    int num_dims = BWD_W_ORIG_COLS_PER_BLOCK;
    if (dim_offset + BWD_W_ORIG_COLS_PER_BLOCK > model_dim){
        num_dims = model_dim - dim_offset;
    }

    for (int i = threadIdx.x; i < BWD_W_ORIG_COLS_PER_BLOCK; i+=blockDim.x){
        if (i < num_dims){
            dW_slice[i] = dW[dim_offset + i];
        }
        else{
            dW_slice[i] = 0;
        }
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int num_warps = blockDim.x / 32;

    //assert(num_warps == BWD_W_COMBINE_WARPS_PER_BLOCK);

    for (int i = lane_id; i < BWD_W_COMBINE_WARPS_PER_BLOCK; i+=WARP_SIZE){
        warp_agg_block[warp_id][i] = 0;
    }

    __syncthreads();

  

    float val;

    unsigned int warp_mask = 0xffffffff;

    int cur_row = warp_id;

    while (cur_row < num_to_combine){

        // load in values for this warp
        for (int d = lane_id; d < BWD_W_ORIG_COLS_PER_BLOCK; d+=WARP_SIZE){
            if (d < num_dims){
                val = dW_workspace[cur_row * model_dim + dim_offset + d];
                warp_agg_block[warp_id][d] += val;
            }
        }

        cur_row += num_warps;
    }

    __syncthreads();

    // now combine each warps values for a given dimension...

    int warp_dim = warp_id;

    while (warp_dim < num_dims){

        if (lane_id < BWD_W_COMBINE_WARPS_PER_BLOCK){
            val = warp_agg_block[lane_id][warp_dim];
        }
        else{
            val = 0;
        }

        for (int warp_offset = 16; warp_offset > 0; warp_offset >>= 1){
            val += __shfl_down_sync(warp_mask, val, warp_offset);
        }
        
        if (lane_id == 0){
            dW_slice[warp_dim] += val;
        }

        warp_dim += num_warps;

    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_dims; i+=blockDim.x){
        dW[dim_offset + i] = __float2bfloat16(dW_slice[i]);
    }
}
