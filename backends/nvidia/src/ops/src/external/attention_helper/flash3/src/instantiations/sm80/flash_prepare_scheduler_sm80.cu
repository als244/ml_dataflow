// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_prepare_scheduler_launch_template.h"

template<>
void run_prepare_varlen_num_blocks_<80>(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl){
    prepare_varlen_num_blocks<80>(params, stream, packgqa, blockM, blockN, enable_pdl);
}
