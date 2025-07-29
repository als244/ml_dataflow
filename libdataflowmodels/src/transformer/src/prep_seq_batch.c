#include "dataflow_seq_batch.h"

/**
 * @brief Reads an entire file into a dynamically allocated string.
 *
 * @param filename The path to the file.
 * @return A dynamically allocated string with the file contents.
 * The caller is responsible for freeing this memory with free().
 * Returns NULL on error.
 */
 char* read_file_into_string(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // 1. Get the size of the file
    fseek(file, 0, SEEK_END);      // Go to the end of the file
    long file_size = ftell(file);  // Get the current position (the size)
    rewind(file);                  // Go back to the beginning

    // 2. Allocate a buffer to hold the contents + null terminator
    char* buffer = (char*)malloc(file_size + 1);
    if (buffer == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for file buffer.\n");
        fclose(file);
        return NULL;
    }

    // 3. Read the entire file into the buffer
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Error reading file.\n");
        fclose(file);
        free(buffer);
        return NULL;
    }

    // 4. Null-terminate the string
    buffer[file_size] = '\0';

    // 5. Close the file and return the buffer
    fclose(file);
    return buffer;
}

Transformer_Model_Config * parse_config(char * config_path) {

    FILE * file = fopen(config_path, "r");
    if (!file) {
        fprintf(stderr, "Error: failed to open config file: %s...\n", config_path);
        return NULL;
    }

    char * config_string = read_file_into_string(config_path);
    if (!config_string) {
        fprintf(stderr, "Error: failed to read config file: %s...\n", config_path);
        return NULL;
    }

    Transformer_Model_Config * config = malloc(sizeof(Transformer_Model_Config));
    if (!config) {
        fprintf(stderr, "Error: failed to allocate config...\n");
        free(config_string);
        return NULL;
    }

    
    // sscanf reads data from a string and parses it according to the format.
    // We provide the expected static text directly in the format string.
    // - "%s" reads a sequence of non-whitespace characters (a string).
    // - "%d" reads a decimal integer.
    // - "%f" reads a floating-point number (handles decimal and scientific e-notation).
    // - The " " (space) before each format specifier handles leading whitespace,
    //   including newlines, making the parsing robust to different line endings.
    //
    // The function returns the number of items successfully matched.
    // We expect to match 18 values in total.
    int items_scanned = sscanf(config_string,
        "Embed Dtype: %15s\n"
        "Attn Dtype: %15s\n"
        "Expert Dtype: %15s\n"
        "Head Dtype: %15s\n"
        "Vocab Size: %d\n"
        "Num Layers: %d\n"
        "Model Dim: %d\n"
        "Num Q Heads: %d\n"
        "Num KV Heads: %d\n"
        "QK Norm Type: %15s\n"
        "QK Norm Weight Type: %15s\n"
        "Num Shared Experts: %d\n"
        "Num Routed Experts: %d\n"
        "Top K Routed Experts: %d\n"
        "Expert Dim: %d\n"
        "Expert MLP Type: %15s\n"
        "Rope Theta: %d\n"
        "RMS Norm Epsilon: %f\n",
        config->embed_dtype,
        config->attn_dtype,
        config->expert_dtype,
        config->head_dtype,
        &(config->vocab_size),
        &(config->num_layers),
        &(config->model_dim),
        &(config->num_q_heads),
        &(config->num_kv_heads),
        config->qk_norm_type,
        config->qk_norm_weight_type,
        &(config->num_shared_experts),
        &(config->num_routed_experts),
        &(config->top_k_routed_experts),
        &(config->expert_dim),
        config->expert_mlp_type,
        &(config->rope_theta),
        &(config->rms_norm_epsilon));

    // Check if all 18 items were successfully scanned.
    if (items_scanned != 18) {
        fprintf(stderr, "Error: Configuration string does not match the expected format.\n");
        fprintf(stderr, "Expected 18 items, but sscanf only matched %d.\n", items_scanned);
        free(config_string);
        free(config);
        return NULL; // Return an error code
    }

    return config;
}

void init_seq_batch_metadata_offsets(Seq_Batch_Metadata_Offsets * metadata_offsets, int total_tokens, int num_seqs) {

    uint64_t cur_offset = 0;
    
    // embedding offsets
    metadata_offsets -> token_ids = cur_offset;
    cur_offset += total_tokens * sizeof(uint32_t);

    metadata_offsets -> sorted_token_ids = cur_offset;
    cur_offset += total_tokens * sizeof(uint32_t);

    metadata_offsets -> sorted_token_mapping = cur_offset;
    cur_offset += total_tokens * sizeof(uint32_t);

    metadata_offsets -> unique_token_sorted_inds_start = cur_offset;
    // worst case each token is unique
    cur_offset += (total_tokens + 1) * sizeof(uint32_t);

    // attention offsets
    metadata_offsets -> seq_positions = cur_offset;
    cur_offset += total_tokens * sizeof(int);

    metadata_offsets -> q_seq_offsets = cur_offset;
    cur_offset += (num_seqs + 1) * sizeof(int);

    metadata_offsets -> q_seq_lens = cur_offset;
    cur_offset += num_seqs * sizeof(int);

    metadata_offsets -> k_seq_offsets = cur_offset;
    cur_offset += (num_seqs + 1) * sizeof(int);

    metadata_offsets -> k_seq_lens = cur_offset;
    cur_offset += num_seqs * sizeof(int);

    // loss offsets
    metadata_offsets -> labels = cur_offset;
    cur_offset += total_tokens * sizeof(uint32_t);

    metadata_offsets -> loss_vec = cur_offset;
    cur_offset += (total_tokens + 1) * sizeof(float);

    metadata_offsets -> total_size = cur_offset;

    return;
}

void init_seq_batch_saved_activations_offsets(Seq_Batch_Saved_Activations_Offsets * saved_activations_offsets, int total_tokens, Transformer_Block_Config * block_config) {

    uint64_t cur_offset = 0;

    DataflowDatatype block_dt = block_config -> block_dt;

    size_t dt_size = dataflow_sizeof_element(block_dt);

    uint64_t model_dim = (uint64_t) (block_config -> model_dim);
    uint64_t kv_dim = (uint64_t) (block_config -> kv_dim);
    uint64_t ffn_dim = (uint64_t) (block_config -> ffn_dim);

    saved_activations_offsets -> x_inp = cur_offset;
    cur_offset += total_tokens * model_dim * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> attn_norm_rms_vals = cur_offset;
    cur_offset += total_tokens * sizeof(float);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> ffn_norm_rms_vals = cur_offset;
    cur_offset += total_tokens * sizeof(float);


    // CUTOFF FOR SAVING INP ONLY...


    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> x_k_local = cur_offset;
    cur_offset += total_tokens * kv_dim * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> x_v_local = cur_offset;
    cur_offset += total_tokens * kv_dim * dt_size;


    saved_activations_offsets -> inp_only_cutoff = cur_offset;


    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    int num_q_heads = (block_config -> num_q_heads);

    saved_activations_offsets -> softmax_lse = cur_offset;
    cur_offset += total_tokens * num_q_heads * sizeof(float);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> x_attn_out = cur_offset;
    cur_offset += total_tokens * model_dim * dt_size;
    
    saved_activations_offsets -> inp_attn_only_cutoff = cur_offset;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> x_q = cur_offset;
    cur_offset += total_tokens * model_dim * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> x_o = cur_offset;
    cur_offset += total_tokens * model_dim * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    int num_local_experts = saved_activations_offsets -> num_local_experts;
        // need to allocate space for saved activations offsets...
    int num_shared_experts = (block_config -> moe_config).num_shared_experts;
    int num_global_routed_experts = (block_config -> moe_config).num_global_routed_experts;
    int top_k_active = (block_config -> moe_config).top_k_experts;

    // special case for dense moe where there is no combining and w2 output
    // is exactly the same as layer output...
    if ((num_shared_experts == 1) && (num_local_experts == 1) && (num_global_routed_experts == 0)){

        (saved_activations_offsets -> x_1)[0] = cur_offset;
        cur_offset += total_tokens * ffn_dim * dt_size;

        // Align offset to 256 bytes
        cur_offset = (cur_offset + 255) & ~255UL;

        (saved_activations_offsets -> x_3)[0] = cur_offset;
        cur_offset += total_tokens * ffn_dim * dt_size;

        // Align offset to 256 bytes
        cur_offset = (cur_offset + 255) & ~255UL;

        saved_activations_offsets -> total_size = cur_offset;
        return;
    }

    saved_activations_offsets -> x_routed = cur_offset;
    cur_offset += total_tokens * num_global_routed_experts * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> chosen_experts = cur_offset;
    cur_offset += total_tokens * top_k_active * sizeof(uint16_t);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> token_expert_weights = cur_offset;
    cur_offset += total_tokens * top_k_active * sizeof(float);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;


    // if MoE, need to allocate space for extra metadata...
    saved_activations_offsets -> expert_counts = cur_offset;
    cur_offset += (num_local_experts + 1) * sizeof(int);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> expert_counts_cumsum = cur_offset;
    cur_offset += num_local_experts * sizeof(int);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> expert_mapping = cur_offset;
    cur_offset += total_tokens * top_k_active * sizeof(int);

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;


    // Assign x_1[0] as the start of teh workspace, we know
    // what the total size will be but not how it is partitioned
    // until dynamically selected by the router...
    for (int i = 0; i < num_local_experts; i++){
        (saved_activations_offsets -> x_1)[i] = cur_offset;
    }
    cur_offset += total_tokens * top_k_active * ffn_dim * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    // Assign x_3[0] as the start of the workspace, we know
    // what the total size will be but not how it is partitioned
    // until dynamically selected by the router...

    for (int i = 0; i < num_local_experts; i++){
        (saved_activations_offsets -> x_3)[i] = cur_offset;
    }
    cur_offset += total_tokens * top_k_active * ffn_dim * dt_size;

    // Align offset to 256 bytes
    cur_offset = (cur_offset + 255) & ~255UL;

    saved_activations_offsets -> total_size = cur_offset;
    
    return;
}

int set_moe_saved_activations_offsets(Seq_Batch_Saved_Activations * sys_saved_activations, Seq_Batch_Saved_Activations * working_activations) {

    // TODO...

    // PROBABLY NEED TO CHANGE THE FUNCTION INTERFACE, AND THINK ABOUT CRITICAL PATH INTERACTIONS WITH DATATRANSFERS AND WORKSPACE PARTITIONING...
}
    

void init_seq_batch_recomputed_activations_offsets(Seq_Batch_Recomputed_Activations_Offsets * recomputed_activations_offsets, int total_tokens, Transformer_Block_Config * block_config) {

    uint64_t cur_offset = 0;

    DataflowDatatype block_dt = block_config -> block_dt;

    uint64_t dt_size = dataflow_sizeof_element(block_dt);
    uint64_t model_dim = (uint64_t) (block_config -> model_dim);
    
    recomputed_activations_offsets -> recomputed_attn_norm = cur_offset;
    cur_offset += total_tokens * model_dim * dt_size;

    recomputed_activations_offsets -> recomputed_ffn_norm = cur_offset;
    cur_offset += total_tokens * model_dim * dt_size;

    recomputed_activations_offsets -> total_size = cur_offset;

    return;
}

int init_seq_batch_offsets(Seq_Batch * seq_batch, int total_tokens, int num_seqs, Transformer_Block_Config * block_config, int max_total_local_expert_tokens) {

    seq_batch -> total_tokens = total_tokens;
    seq_batch -> num_seqs = num_seqs;

    init_seq_batch_metadata_offsets(&(seq_batch -> metadata_offsets), total_tokens, num_seqs);


    // need to allocate space for saved activations offsets...
    int num_local_experts = (block_config -> moe_config).num_local_experts;
    int top_k = (block_config -> moe_config).top_k_experts;
    
    Seq_Batch_Saved_Activations_Offsets * saved_activations_offsets = &(seq_batch -> saved_activations_offsets);
    saved_activations_offsets -> num_local_experts = num_local_experts;
    saved_activations_offsets -> top_k = top_k;
    saved_activations_offsets -> max_total_local_expert_tokens = max_total_local_expert_tokens;

    saved_activations_offsets -> x_1 = malloc(num_local_experts * sizeof(void *));
    if (!saved_activations_offsets -> x_1){
        fprintf(stderr, "Error: failed to allocate x_1 for saved activations offsets...\n");
        return -1;
    }

    saved_activations_offsets -> x_3 = malloc(num_local_experts * sizeof(void *));
    if (!saved_activations_offsets -> x_3){
        fprintf(stderr, "Error: failed to allocate x_3 for saved activations offsets...\n");
        return -1;
    }

    // This will only work with non-MoE for now...
    init_seq_batch_saved_activations_offsets(&(seq_batch -> saved_activations_offsets), total_tokens, block_config);
    init_seq_batch_recomputed_activations_offsets(&(seq_batch -> recomputed_activations_offsets), total_tokens, block_config);

    return 0; 
}

uint64_t get_seq_batch_metadata_buffer_size(int num_seqs, int total_tokens) {

    uint64_t metadata_buffer_size = 0;

    metadata_buffer_size += 4 * (total_tokens) * sizeof(uint32_t);
    metadata_buffer_size += (total_tokens + 1) * sizeof(uint32_t);
    metadata_buffer_size += (total_tokens) * sizeof(int);
    metadata_buffer_size += 2 * (num_seqs + 1) * sizeof(int);
    metadata_buffer_size += 2 * num_seqs * sizeof(int);
    metadata_buffer_size += (total_tokens + 1) * sizeof(float);

    return metadata_buffer_size;
}

int bind_seq_batch_metadata_buffer(Seq_Batch * seq_batch, void * metadata_buffer, uint64_t metadata_buffer_size) {

    Seq_Batch_Metadata_Offsets * metadata_offsets = &(seq_batch -> metadata_offsets);
    if (metadata_buffer_size < metadata_offsets -> total_size){
        fprintf(stderr, "Error: metadata buffer size is less than the required size. Tried to bind with size %lu, but required size is %lu...\n", metadata_buffer_size, metadata_offsets -> total_size);
        return -1;
    }

    seq_batch -> devMetadataBuffer = metadata_buffer;
    seq_batch -> devMetadataBufferBytes = metadata_buffer_size;

    // Seq Batch Embedding Config
    Seq_Batch_Embedding_Config * embedding_config = &(seq_batch -> embedding_config);
    embedding_config -> token_ids = (uint32_t *) (metadata_buffer + metadata_offsets -> token_ids);
    embedding_config -> sorted_token_ids = (uint32_t *) (metadata_buffer + metadata_offsets -> sorted_token_ids);
    embedding_config -> sorted_token_mapping = (uint32_t *) (metadata_buffer + metadata_offsets -> sorted_token_mapping);
    embedding_config -> unique_token_sorted_inds_start = (uint32_t *) (metadata_buffer + metadata_offsets -> unique_token_sorted_inds_start);

    // Seq Batch Attention Config
    Seq_Batch_Attention_Config * attention_config = &(seq_batch -> attention_config);
    attention_config -> seq_positions = (int *) (metadata_buffer + metadata_offsets -> seq_positions);
    attention_config -> q_seq_offsets = (int *) (metadata_buffer + metadata_offsets -> q_seq_offsets);
    attention_config -> q_seq_lens = (int *) (metadata_buffer + metadata_offsets -> q_seq_lens);
    attention_config -> k_seq_offsets = (int *) (metadata_buffer + metadata_offsets -> k_seq_offsets);
    attention_config -> k_seq_lens = (int *) (metadata_buffer + metadata_offsets -> k_seq_lens);

    // Seq Batch Loss Config
    Seq_Batch_Loss_Config * loss_config = &(seq_batch -> loss_config);
    loss_config -> labels = (uint32_t *) (metadata_buffer + metadata_offsets -> labels);
    loss_config -> loss_vec = (float *) (metadata_buffer + metadata_offsets -> loss_vec);

    return 0;
}

// struct to hold token and original index
struct token_index_pair {
    uint32_t token;
    uint32_t index;
};

// comparison function for qsort
static int compare_pairs(const void *a, const void *b) {
    struct token_index_pair *pair_a = (struct token_index_pair *)a;
    struct token_index_pair *pair_b = (struct token_index_pair *)b;
    return pair_a -> token - pair_b -> token;
}

int populate_seq_batch_metadata_buffer(Dataflow_Handle * dataflow_handle, int inbound_stream_id, 
                                        Seq_Batch * seq_batch,
                                        void * sys_registered_metadata_buffer, uint64_t sys_registered_metadata_buffer_size,
                                        int seq_id, int chunk_id, int total_tokens, int num_seqs,
                                        uint32_t * sys_token_ids, uint32_t * sys_labels,
                                        int * sys_seq_positions, 
                                        int * sys_q_seq_offsets, int * sys_q_seq_lens,
                                        int * sys_k_seq_offsets, int * sys_k_seq_lens) {

        
        int ret;

        seq_batch -> seq_id = seq_id;
        seq_batch -> chunk_id = chunk_id;
        seq_batch -> total_tokens = total_tokens;
        seq_batch -> num_seqs = num_seqs;

        Seq_Batch_Metadata_Offsets * metadata_offsets = &(seq_batch -> metadata_offsets);
        if (sys_registered_metadata_buffer_size < metadata_offsets -> total_size){
            fprintf(stderr, "Error: sys_registered_metadata_buffer_size is less than the required size. Tried to bind with size %lu, but required size is %lu...\n", sys_registered_metadata_buffer_size, metadata_offsets -> total_size);
            return -1;
        }

        Seq_Batch sys_seq_batch;
        init_seq_batch_metadata_offsets(&(sys_seq_batch.metadata_offsets), total_tokens, num_seqs);
        ret = bind_seq_batch_metadata_buffer(&sys_seq_batch, sys_registered_metadata_buffer, sys_registered_metadata_buffer_size);
        // should have already been checked above...
        if (ret){
            fprintf(stderr, "Error: failed to bind seq_batch metadata buffer...\n");
            return -1;
        }
        
        // copy the metadata buffer to the system memory buffer...

        // embedding config...
        memcpy(sys_seq_batch.embedding_config.token_ids, sys_token_ids, total_tokens * sizeof(uint32_t));

        // create array of pairs
        struct token_index_pair *pairs = malloc(total_tokens * sizeof(struct token_index_pair));
        if (!pairs){
            fprintf(stderr, "Error: failed to allocate pairs (size %lu) for sorted token ids...\n", total_tokens * sizeof(struct token_index_pair));
            return -1;
        }

        // populate pairs with token ids and original indices
        for (int i = 0; i < total_tokens; i++) {
            pairs[i].token = sys_token_ids[i];
            pairs[i].index = i;
        }


        qsort(pairs, total_tokens, sizeof(struct token_index_pair), compare_pairs);

        // copy back sorted results and count unique tokens
        sys_seq_batch.embedding_config.unique_token_sorted_inds_start[0] = 0;
        int unique_count = 1;
        uint32_t current_token = pairs[0].token;
        
        for (int i = 0; i < total_tokens; i++) {
            sys_seq_batch.embedding_config.sorted_token_ids[i] = pairs[i].token;
            sys_seq_batch.embedding_config.sorted_token_mapping[i] = pairs[i].index;
            
            if (i > 0 && pairs[i].token != current_token) {
                sys_seq_batch.embedding_config.unique_token_sorted_inds_start[unique_count] = i;
                unique_count++;
                current_token = pairs[i].token;
            }
        }

        // Add final boundary
        sys_seq_batch.embedding_config.unique_token_sorted_inds_start[unique_count] = total_tokens;

        // can free the temp pairs buffer now
        free(pairs);
        
        // Set the number of unique tokens in the seq_batch that we want to populate...
        (seq_batch->embedding_config).num_unique_tokens = unique_count;

       
        // attention config... 

        // iterate over the total q lens to make sure it equals the total tokens...
        int total_q = 0;
        int max_seqlen_q = 0;
        for (int i = 0; i < num_seqs; i++){
            total_q += sys_q_seq_lens[i];
            if (sys_q_seq_lens[i] > max_seqlen_q){
                max_seqlen_q = sys_q_seq_lens[i];
            }
        }

        if (total_q != total_tokens){
            fprintf(stderr, "Error: total q (%d) is not equal to total tokens (%d)...\n", total_q, total_tokens);
            return -1;
        }

        // get total k lens and max k seqlen...
        int total_k = 0;
        int max_seqlen_k = 0;
        for (int i = 0; i < num_seqs; i++){
            total_k += sys_k_seq_lens[i];
            if (sys_k_seq_lens[i] > max_seqlen_k){
                max_seqlen_k = sys_k_seq_lens[i];
            }
        }

        // set num seqs, total q, total k, max q, max k seqlens...
        (seq_batch -> attention_config).num_seqs = num_seqs;
        (seq_batch -> attention_config).total_q = total_q;
        (seq_batch -> attention_config).total_k = total_k;
        (seq_batch -> attention_config).max_seqlen_q = max_seqlen_q;
        (seq_batch -> attention_config).max_seqlen_k = max_seqlen_k;
        
        // now copy the user provided arrays to sys seq batch using registered metadata buffer before transferring to device...
        memcpy(sys_seq_batch.attention_config.seq_positions, sys_seq_positions, total_tokens * sizeof(int));
        memcpy(sys_seq_batch.attention_config.q_seq_offsets, sys_q_seq_offsets, (num_seqs + 1) * sizeof(int));
        memcpy(sys_seq_batch.attention_config.q_seq_lens, sys_q_seq_lens, num_seqs * sizeof(int));

        // need to populate the k_seq_offsets and k_seq_lens buffers...
        memcpy(sys_seq_batch.attention_config.k_seq_offsets, sys_k_seq_offsets, (num_seqs + 1) * sizeof(int));
        memcpy(sys_seq_batch.attention_config.k_seq_lens, sys_k_seq_lens, num_seqs * sizeof(int));

        // loss config...
    
        (seq_batch -> loss_config).num_tokens_to_predict = 0;

        
        if (sys_labels != NULL){
            (seq_batch -> loss_config).num_tokens_to_predict = total_tokens;
            memcpy(sys_seq_batch.loss_config.labels, sys_labels, total_tokens * sizeof(uint32_t));
        }

        // now can copy the sys_seq_batch metadata buffer to the device...
        ret = (dataflow_handle -> submit_inbound_transfer)(dataflow_handle, inbound_stream_id, 
                                                            seq_batch -> devMetadataBuffer, sys_seq_batch.devMetadataBuffer, 
                                                            sys_seq_batch.metadata_offsets.total_size);
        if (ret){
            fprintf(stderr, "Error: failed to submit inbound transfer for seq_batch metadata buffer of size %lu...\n", sys_seq_batch.metadata_offsets.total_size);
            return -1;
        }


        // save the host pointers into the seq_batch struct...
        seq_batch -> sys_token_ids = sys_token_ids;
        seq_batch -> sys_labels = sys_labels;
        seq_batch -> sys_seq_positions = sys_seq_positions;

        return 0;
}


uint64_t get_seq_batch_saved_activations_buffer_size(Seq_Batch * seq_batch, SavedActivationLevel saved_activation_level) {

    if (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ONLY){
        return seq_batch -> saved_activations_offsets.inp_only_cutoff;
    }
    else if (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY){
        return seq_batch -> saved_activations_offsets.inp_attn_only_cutoff;
    }
    else {
        return seq_batch -> saved_activations_offsets.total_size;
    }
}

int bind_seq_batch_saved_activations_buffer(Seq_Batch * seq_batch, Seq_Batch_Saved_Activations * saved_activations, void * saved_activations_buffer, SavedActivationLevel saved_activation_level, uint64_t saved_activations_buffer_size,
                                            int layer_id) {
    
    Seq_Batch_Saved_Activations_Offsets * saved_activations_offsets = &(seq_batch -> saved_activations_offsets);

    if (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ONLY && saved_activations_buffer_size < saved_activations_offsets -> inp_only_cutoff){
        fprintf(stderr, "Error: saved activations buffer size is less than the required size. Tried to bind with size %lu, but required size is %lu...\n", saved_activations_buffer_size, saved_activations_offsets -> inp_only_cutoff);
        return -1;
    }
    else if (saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY && saved_activations_buffer_size < saved_activations_offsets -> inp_attn_only_cutoff){
        fprintf(stderr, "Error: saved activations buffer size is less than the required size. Tried to bind with size %lu, but required size is %lu...\n", saved_activations_buffer_size, saved_activations_offsets -> inp_attn_only_cutoff);
        return -1;
    }
    else if (saved_activation_level == SAVED_ACTIVATION_LEVEL_FULL && saved_activations_buffer_size < saved_activations_offsets -> total_size){
        fprintf(stderr, "Error: saved activations buffer size is less than the required size. Tried to bind with size %lu, but required size is %lu...\n", saved_activations_buffer_size, saved_activations_offsets -> total_size);
        return -1;
    }
    

    // SAVE THESE INPS NO MATTER WHAT...!

    saved_activations -> seq_batch = seq_batch;
    saved_activations -> layer_id = layer_id;

    saved_activations -> savedActivationsBuffer = saved_activations_buffer;
    saved_activations -> savedActivationsBufferBytes = saved_activations_buffer_size;

    
    saved_activations -> x_inp = (void *) (saved_activations_buffer + saved_activations_offsets -> x_inp);
    saved_activations -> attn_norm_rms_vals = (float *) (saved_activations_buffer + saved_activations_offsets -> attn_norm_rms_vals);
    saved_activations -> ffn_norm_rms_vals = (float *) (saved_activations_buffer + saved_activations_offsets -> ffn_norm_rms_vals);
    saved_activations -> x_k_local = (void *) (saved_activations_buffer + saved_activations_offsets -> x_k_local);
    saved_activations -> x_v_local = (void *) (saved_activations_buffer + saved_activations_offsets -> x_v_local);
    
    // Only save if higher saved level...
    if ((saved_activation_level == SAVED_ACTIVATION_LEVEL_INP_ATTN_ONLY) || (saved_activation_level == SAVED_ACTIVATION_LEVEL_FULL)){
        // Cutoff for inp only...
        saved_activations -> softmax_lse = (float *) (saved_activations_buffer + saved_activations_offsets -> softmax_lse);
        saved_activations -> x_attn_out = (void *) (saved_activations_buffer + saved_activations_offsets -> x_attn_out);
    }
    else{
        saved_activations -> softmax_lse = NULL;
        saved_activations -> x_attn_out = NULL;
    }
    
    // only save if full...
    if (saved_activation_level == SAVED_ACTIVATION_LEVEL_FULL){
        saved_activations -> x_q = (void *) (saved_activations_buffer + saved_activations_offsets -> x_q);
        saved_activations -> x_o = (void *) (saved_activations_buffer + saved_activations_offsets -> x_o);
    }
    else{
        saved_activations -> x_q = NULL;
        saved_activations -> x_o = NULL;
    }
    


    // for non-MoE, num_local_experts should be 1 
    // and the initial init_offsets should have set the saved_activations_offsets -> x_1, x_2, x_3 to the correct offsets...
    int num_local_experts = saved_activations_offsets -> num_local_experts;

    saved_activations -> x_1 = malloc(num_local_experts * sizeof(void *));
    if (!saved_activations -> x_1){
        fprintf(stderr, "Error: failed to allocate x_1 for saved activations...\n");
        return -1;
    }

    saved_activations -> x_3 = malloc(num_local_experts * sizeof(void *));
    if (!saved_activations -> x_3){
        fprintf(stderr, "Error: failed to allocate x_3 for saved activations...\n");
        return -1;
    }

    saved_activations -> x_routed = NULL;
    saved_activations -> chosen_experts = NULL;
    saved_activations -> token_expert_weights = NULL;
    saved_activations -> expert_counts = NULL;
    saved_activations -> expert_counts_cumsum = NULL;
    saved_activations -> expert_mapping = NULL;

    if (saved_activations_offsets -> num_local_experts > 1){
        
        saved_activations -> x_routed = (void *) (saved_activations_buffer + saved_activations_offsets -> x_routed);
        saved_activations -> chosen_experts = (uint16_t *) (saved_activations_buffer + saved_activations_offsets -> chosen_experts);
        saved_activations -> token_expert_weights = (float *) (saved_activations_buffer + saved_activations_offsets -> token_expert_weights);

        // for non-MoE, these are not needed...
        saved_activations -> expert_counts = (void *) (saved_activations_buffer + saved_activations_offsets -> expert_counts);
        saved_activations -> expert_counts_cumsum = (void *) (saved_activations_buffer + saved_activations_offsets -> expert_counts_cumsum);
        saved_activations -> expert_mapping = (void *) (saved_activations_buffer + saved_activations_offsets -> expert_mapping);
    }

    // only save if full...
    if (saved_activation_level == SAVED_ACTIVATION_LEVEL_FULL) {
        for (int i = 0; i < num_local_experts; i++){
            (saved_activations -> x_1)[i] = (void *) (saved_activations_buffer + (saved_activations_offsets -> x_1)[i]);
        }

        for (int i = 0; i < num_local_experts; i++){
            (saved_activations -> x_3)[i] = (void *) (saved_activations_buffer + (saved_activations_offsets -> x_3)[i]);
        }
    }
    else{
        for (int i = 0; i < num_local_experts; i++){
            (saved_activations -> x_1)[i] = NULL;
        }

        for (int i = 0; i < num_local_experts; i++){
            (saved_activations -> x_3)[i] = NULL;
        }
    }

    return 0;
}

uint64_t get_seq_batch_recomputed_activations_buffer_size(Seq_Batch * seq_batch) {
    return seq_batch -> recomputed_activations_offsets.total_size;
}

int bind_seq_batch_recomputed_activations_buffer(Seq_Batch_Recomputed_Activations_Offsets * recomputed_activations_offsets, Seq_Batch_Recomputed_Activations * recomputed_activations, void * recomputed_activations_buffer, uint64_t recomputed_activations_buffer_size) {

    if (recomputed_activations_buffer_size < recomputed_activations_offsets -> total_size){
        fprintf(stderr, "Error: recomputed activations buffer size is less than the required size. Tried to bind with size %lu, but required size is %lu...\n", recomputed_activations_buffer_size, recomputed_activations_offsets -> total_size);
        return -1;
    }

    recomputed_activations -> recomputedActivationsBuffer = recomputed_activations_buffer;
    recomputed_activations -> recomputedActivationsBufferBytes = recomputed_activations_buffer_size;

    recomputed_activations -> recomputed_attn_norm = (void *) (recomputed_activations_buffer + recomputed_activations_offsets -> recomputed_attn_norm);
    recomputed_activations -> recomputed_ffn_norm = (void *) (recomputed_activations_buffer + recomputed_activations_offsets -> recomputed_ffn_norm);

    return 0;
}

uint64_t get_seq_batch_activation_workspace_buffer_size(Seq_Batch * seq_batch, Transformer_Block_Config * block_config){

    

    uint64_t total_tokens = seq_batch -> total_tokens;

    uint64_t dtype_size = dataflow_sizeof_element(block_config -> block_dt);

    uint64_t activation_workspace_size = 0;
    activation_workspace_size += total_tokens * (uint64_t) block_config -> model_dim * (uint64_t) dtype_size;
    activation_workspace_size += total_tokens * (uint64_t) block_config -> ffn_dim * (uint64_t) dtype_size;

    return activation_workspace_size;
}

