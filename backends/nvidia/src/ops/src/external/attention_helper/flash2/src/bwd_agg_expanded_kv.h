
template<typename T>
__global__ void bwd_agg_expanded_kv(int num_seqs, int * k_seq_offsets, int * k_seq_lens,
                                    int head_dim, int n_q_heads, int n_kv_heads,
                                    const T* __restrict__ new_dk_expanded, const T* __restrict__ new_dv_expanded,
                                    T* __restrict__ orig_dk, T* __restrict__ orig_dv) { 

    int seq_ind = blockIdx.y;

    if (seq_ind >= num_seqs) {
        return;
    }

    int seq_len = k_seq_lens[seq_ind];

    int seq_pos = blockIdx.x;

    if (seq_pos >= seq_len) {
        return;
    }

    int seq_start = k_seq_offsets[seq_ind];

    int cur_row = seq_start + seq_pos;

    int model_dim = n_q_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // Should be enough space to store row of orig_dk and orig_dv....
    extern __shared__ T shared_mem[];

    float * dk_row = shared_mem;
    float * dv_row = dk_row + kv_dim;
    T * new_dk_row = orig_dv_row + kv_dim;
    T * new_dv_row = new_dk_row + model_dim;


    uint64_t orig_el_start = cur_row * kv_dim;
    uint64_t new_el_start = cur_row * model_dim;


    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        dk_row[i] = float(orig_dk[orig_el_start + i]);
        dv_row[i] = float(orig_dv[orig_el_start + i]);
    }

    for (int i = threadIdx.x; i < model_dim; i += blockDim.x) {
        new_dk_row[i] = new_dk_expanded[new_el_start + i];
        new_dv_row[i] = new_dv_expanded[new_el_start + i];
    }

    __syncthreads();


    int head_dim_ratio = n_kv_heads / n_q_heads;

    for (int kv_head = 0; kv_head < n_kv_heads; kv_head++) {
        for (int q_head_ind = kv_head * head_dim_ratio; q_head_ind < (kv_head + 1) * head_dim_ratio; q_head_ind++) {
            for (int head_dim_ind = threadIdx.x; head_dim_ind < head_dim; head_dim_ind += blockDim.x) {
                dk_row[kv_head * head_dim + head_dim_ind] += float(new_dk_row[q_head_ind * head_dim + head_dim_ind]);
                dv_row[kv_head * head_dim + head_dim_ind] += float(new_dv_row[q_head_ind * head_dim + head_dim_ind]);
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        orig_dk[orig_el_start + i] = T(dk_row[i]);
        orig_dv[orig_el_start + i] = T(dv_row[i]);
    }
    
    return;
    
}