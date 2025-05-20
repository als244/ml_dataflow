#define _GNU_SOURCE
#include "add.h"

// Scalar conversion functions from your AVX512 example
// Ensure these are included or defined in your "add.h" or current file.
static inline float scalar_u16_to_fp32_fp16(uint16_t val_fp16_bits) {
    _Float16 f16_val; // Assumes compiler support for _Float16
    memcpy(&f16_val, &val_fp16_bits, sizeof(f16_val));
    return (float)f16_val;
}

static inline uint16_t scalar_fp32_to_u16_fp16(float val_fp32) {
    _Float16 f16_val = (_Float16)val_fp32; // Assumes compiler support for _Float16
    uint16_t u16_bits;
    memcpy(&u16_bits, &f16_val, sizeof(u16_bits));
    return u16_bits;
}

static inline float scalar_u16_to_fp32_bf16(uint16_t val_bf16_bits) {
    uint32_t val_fp32_bits = ((uint32_t)val_bf16_bits) << 16;
    float fp32_val;
    memcpy(&fp32_val, &val_fp32_bits, sizeof(fp32_val));
    return fp32_val;
}

static inline uint16_t scalar_fp32_to_u16_bf16(float val_fp32) {
    uint32_t u32_in;
    memcpy(&u32_in, &val_fp32, sizeof(u32_in));

    if (isnan(val_fp32)) {
        return ((u32_in & 0x80000000) >> 16) | 0x7FC0; // Preserve sign, return a QNaN pattern
    }
    if (isinf(val_fp32)) {
        return ((u32_in & 0x80000000) >> 16) | 0x7F80; // Preserve sign, Inf
    }

    uint32_t sign = u32_in & 0x80000000;
    uint32_t u32_abs = u32_in & 0x7FFFFFFF; 

    if (u32_abs == 0) {
        return (uint16_t)(sign >> 16);
    }
    
    uint16_t truncated_val_abs = (uint16_t)(u32_abs >> 16);
    uint32_t remainder = u32_abs & 0xFFFF;

    if (remainder > 0x8000) { 
        truncated_val_abs++;
    } else if (remainder == 0x8000) { 
        if ((truncated_val_abs & 1) != 0) { 
            truncated_val_abs++;
        }
    }

    if ((truncated_val_abs & 0x7F80) == 0x7F80) {
        if (!isinf(val_fp32)) { 
            return (uint16_t)(sign >> 16) | 0x7F80; 
        }
    }
    return (uint16_t)(sign >> 16) | (truncated_val_abs & 0x7FFF);
}

static void * thread_func_add_fp32(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    float * A = add_worker_args->A;
    float * B = add_worker_args->B;
    float * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    for (size_t i = start_ind; i < start_ind + num_els; i++){
        C[i] = A[i] + B[i];
    }

    return NULL;
}

static void * thread_func_add_bf16(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    uint16_t * A = add_worker_args->A;
    uint16_t * B = add_worker_args->B;
    uint16_t * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    float val_a_fp32;
    float val_b_fp32;
    float result_fp32;

    for (size_t i = start_ind; i < start_ind + num_els; i++){
        val_a_fp32 = scalar_u16_to_fp32_bf16(A[i]);
        val_b_fp32 = scalar_u16_to_fp32_bf16(B[i]);
        result_fp32 = alpha * val_a_fp32 + beta * val_b_fp32;
        C[i] = scalar_fp32_to_u16_bf16(result_fp32);
    }

    return NULL;
}


static void * thread_func_add_fp16(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    uint16_t * A = add_worker_args->A;
    uint16_t * B = add_worker_args->B;
    uint16_t * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    float val_a_fp32;
    float val_b_fp32;
    float result_fp32;

    for (size_t i = start_ind; i < start_ind + num_els; i++){
        val_a_fp32 = scalar_u16_to_fp32_fp16(A[i]);
        val_b_fp32 = scalar_u16_to_fp32_fp16(B[i]);
        result_fp32 = alpha * val_a_fp32 + beta * val_b_fp32;
        C[i] = scalar_fp32_to_u16_fp16(result_fp32);
    }

    return NULL;
}



static void * thread_func_add_fp8e4m3(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    uint8_t * A = add_worker_args->A;
    uint8_t * B = add_worker_args->B;
    uint8_t * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    fprintf(stderr, "Error: add_fp8e4m3 add not supported yet...\n");
    return NULL;

}

static void * thread_func_add_fp8e5m2(void * _add_worker_args){

    Add_Worker_Args * add_worker_args = (Add_Worker_Args *) _add_worker_args;

    uint8_t * A = add_worker_args->A;
    uint8_t * B = add_worker_args->B;
    uint8_t * C = add_worker_args->C;

    float alpha = add_worker_args->alpha;
    float beta = add_worker_args->beta;

    size_t start_ind = add_worker_args->start_ind;
    size_t num_els = add_worker_args->num_els;

    fprintf(stderr, "Error: add_fp8e5m2 add not supported yet...\n");
    return NULL;

}





int do_add_host(DataflowDatatype A_dt, DataflowDatatype B_dt, DataflowDatatype C_dt,
                int num_threads, size_t num_els, void * A, void * B, void * C,
                float alpha, float beta){
    

    // for now only support when all same dtype..
    if (A_dt != B_dt || A_dt != C_dt){
        fprintf(stderr, "Error: all dtypes must be the same for avx512 add...\n");
        return -1;
    }

    void * (*add_func)(void * _add_worker_args);

    if (A_dt == DATAFLOW_FP32){
        add_func = thread_func_add_fp32;
    }
    else if (A_dt == DATAFLOW_BF16){
        add_func = thread_func_add_bf16;
    }
    else if (A_dt == DATAFLOW_FP16){
        add_func = thread_func_add_fp16;
    }
    /* Not ready for fp8 yet
    else if (A_dt == DATAFLOW_FP8E4M3){
        add_func = thread_func_add_avx512_fp8e4m3;
    }
    else if (A_dt == DATAFLOW_FP8E5M2){
        add_func = thread_func_add_avx512_fp8e5m2;
    }
    */
    else{
        fprintf(stderr, "Error: unsupported dtype for avx512 add...\n");
        return -1;
    }

    if (!add_func){
        fprintf(stderr, "Error: failed to get thread function for avx512 add...\n");
        return -1;
    }

    if (num_threads <= 1){
        Add_Worker_Args add_worker_args;
        add_worker_args.A = A;
        add_worker_args.B = B;
        add_worker_args.C = C;
        add_worker_args.start_ind = 0;
        add_worker_args.num_els = num_els;
        add_worker_args.alpha = alpha;
        add_worker_args.beta = beta;

        add_func(&add_worker_args);
    }


    // Ensure NUMA affinity...

    pthread_attr_t attr;
    int ret_attr_init = pthread_attr_init(&attr);
    if (ret_attr_init != 0) {
        perror("pthread_attr_init failed");
        // Decide if this is a fatal error or if you want to proceed without affinity
        // For now, let's proceed but affinity won't be set.
    }

    cpu_set_t *cpu_mask_for_node = NULL;
    int affinity_set_successfully = 0;
    int max_cpus = 0; // Will store the number of possible CPUs

    if (numa_available()) {
 
        int current_cpu = sched_getcpu();
        if (current_cpu < 0) {
            perror("Warning: sched_getcpu failed");
        } else {
            int target_numa_node = numa_node_of_cpu(current_cpu);
            if (target_numa_node < 0) {
                perror("Warning: numa_node_of_cpu failed");
            } else {
                printf("Calling thread is on CPU %d, NUMA node %d. Attempting to pin worker threads to this node.\n", current_cpu, target_numa_node);
                
                max_cpus = numa_num_possible_cpus();
                if (max_cpus <= 0) {
                     fprintf(stderr, "Warning: numa_num_possible_cpus() returned invalid value %d\n", max_cpus);
                } else {
                    cpu_mask_for_node = CPU_ALLOC(max_cpus);
                    if (cpu_mask_for_node == NULL) {
                        perror("Warning: CPU_ALLOC failed");
                    } else {
                        size_t cpuset_size = CPU_ALLOC_SIZE(max_cpus);
                        CPU_ZERO_S(cpuset_size, cpu_mask_for_node);

                        struct bitmask *numa_cpumask = numa_allocate_cpumask();
                        if (numa_cpumask == NULL) {
                            perror("Warning: numa_allocate_cpumask failed");
                            CPU_FREE(cpu_mask_for_node);
                            cpu_mask_for_node = NULL;
                        } else {
                            if (numa_node_to_cpus(target_numa_node, numa_cpumask) < 0) {
                                perror("Warning: numa_node_to_cpus failed");
                            } else {
                                for (int k = 0; k < max_cpus; ++k) {
                                    if (numa_bitmask_isbitset(numa_cpumask, k)) {
                                        CPU_SET_S(k, cpuset_size, cpu_mask_for_node);
                                    }
                                }
                                affinity_set_successfully = 1; // Mark that we have a valid mask
                            }
                            numa_free_cpumask(numa_cpumask);
                        }
                    }
                }
            }
        }
    }

    if (affinity_set_successfully && ret_attr_init == 0 && cpu_mask_for_node != NULL) {
        size_t cpuset_size = CPU_ALLOC_SIZE(max_cpus);
        int ret_set_affinity = pthread_attr_setaffinity_np(&attr, cpuset_size, cpu_mask_for_node);
        if (ret_set_affinity != 0) {
            perror("Warning: pthread_attr_setaffinity_np failed. Threads will use default affinity.");
            affinity_set_successfully = 0; // Revert status
        } else {
            printf("Successfully prepared to set worker thread affinity to NUMA node.\n");
        }
    } else {
        affinity_set_successfully = 0; // Ensure it's clear affinity isn't being applied
    }

    pthread_t * add_workers = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    if (!add_workers){
        fprintf(stderr, "Error: failed to allocate add workers...\n");
        if (ret_attr_init == 0) {
            pthread_attr_destroy(&attr);
        }
        if (cpu_mask_for_node) {
            CPU_FREE(cpu_mask_for_node);
        }
        return -1;
    }

    Add_Worker_Args * add_worker_args_array = (Add_Worker_Args *) malloc(num_threads * sizeof(Add_Worker_Args));
    if (!add_worker_args_array){
        fprintf(stderr, "Error: failed to allocate add worker args...\n");
        free(add_workers);
        if (ret_attr_init == 0) {
            pthread_attr_destroy(&attr);
        }
        if (cpu_mask_for_node) {
            CPU_FREE(cpu_mask_for_node);
        }
        return -1;
    }

    // Same arguments for all workers
    for (int i = 0; i < num_threads; i++){
        add_worker_args_array[i].A = A;
        add_worker_args_array[i].B = B;
        add_worker_args_array[i].C = C;
        add_worker_args_array[i].alpha = alpha;
        add_worker_args_array[i].beta = beta;
    }
    
    // Give slices to each worker...

    uint64_t base_chunk_size = num_els / num_threads;
    uint64_t remainder_elements = num_els % num_threads;
    uint64_t current_start_index = 0;
    uint64_t elements_for_this_thread;

    for (int i = 0; i < num_threads; i++){
        
        add_worker_args_array[i].start_ind = current_start_index;
        elements_for_this_thread = base_chunk_size + (i < remainder_elements ? 1 : 0);
        add_worker_args_array[i].num_els = elements_for_this_thread;
        current_start_index += elements_for_this_thread;

        // Start worker
        int ret_create;
        // Only use attr if successfully initialized and affinity mask is ready
        if (ret_attr_init == 0 && affinity_set_successfully) { 
            ret_create = pthread_create(&add_workers[i], &attr, add_func, &add_worker_args_array[i]);
        } else {
            ret_create = pthread_create(&add_workers[i], NULL, add_func, &add_worker_args_array[i]);
        }
        if (ret_create != 0) {
            fprintf(stderr, "[Add] Error creating thread %d: %s\n", i, strerror(ret_create));
            // Handle thread creation error: cleanup already created threads, allocated memory etc.
            // For simplicity here, we'll just print and continue, but in robust code, you'd manage this.
            // For now, just free and exit might be simpler than partial cleanup.
            for(int k=0; k<i; ++k) {
                pthread_cancel(add_workers[k]);
            }
            for(int k=0; k<i; ++k) {
                pthread_join(add_workers[k], NULL);
            }
            free(add_workers);
            free(add_worker_args_array);
            if (ret_attr_init == 0) {
                pthread_attr_destroy(&attr);
            }
            if (cpu_mask_for_node) {
                CPU_FREE(cpu_mask_for_node);
            }
            return -1;
        }
    }
    
    // Clean up pthread attributes if they were initialized
    if (ret_attr_init == 0) {
        pthread_attr_destroy(&attr);
    }
    // Free the cpu_set_t if it was allocated
    if (cpu_mask_for_node) {
        CPU_FREE(cpu_mask_for_node);
    }

    // Wait for all workers to finish
    for (int i = 0; i < num_threads; i++){
        pthread_join(add_workers[i], NULL);
    }

    // Free allocated memory
    free(add_workers);
    free(add_worker_args_array);
    

    return 0;
}