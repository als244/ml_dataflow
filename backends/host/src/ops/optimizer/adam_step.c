#define _GNU_SOURCE
#include "adam_step.h"

// Helper union for BF16 <-> FP32 conversion in scalar part
typedef union {
    float f;
    uint32_t u;
    uint16_t bf16[2]; // bf16[1] would be the upper 16 bits
} fp32_bf16_caster;

typedef union {
    float f;
    uint32_t u;
} float_uint32_caster;

// Convert float to bfloat16 (scalar, round-to-nearest-even style)
static inline uint16_t float_to_bfloat16_scalar(float f_val) {
    fp32_bf16_caster caster;
    caster.f = f_val;
    // Basic rounding: add 0x7FFF and then truncate.
    // Check for NaN: if (isnan(f_val)) return 0x7FC0; // Example NaN representation
    uint32_t u_val = caster.u;
    if (~u_val & 0x7F800000) { // Check if not Inf or NaN
        // Add 0x8000 for rounding to nearest, then shift.
        // This specific rounding adds half of the LSB of the truncated value.
        u_val += 0x7FFF + ((u_val >> 16) & 1);
    } else if (u_val & 0x007FE000) { // Signaling NaN, convert to QNaN
        u_val |= 0x00400000;
    }
    return (uint16_t)(u_val >> 16);
}

// Convert bfloat16 to float (scalar)
static inline float bfloat16_to_float_scalar(uint16_t bf16_val) {
    fp32_bf16_caster caster;
    caster.u = ((uint32_t)bf16_val) << 16;
    return caster.f;
}

// Simplified FP16 to FP32 scalar conversion
// Handles normals, zero, Inf, NaN. Subnormals are mostly flushed to zero or handled imperfectly.
static inline float float16_to_float32_scalar(uint16_t fp16_val) {
    float_uint32_caster caster;
    const uint32_t sign_mask_fp16 = 0x8000;
    const uint32_t exp_mask_fp16 = 0x7C00;
    const uint32_t man_mask_fp16 = 0x03FF;
    const uint32_t fp16_exp_bias = 15;
    const uint32_t fp32_exp_bias = 127;

    uint32_t sign_fp32 = ((uint32_t)(fp16_val & sign_mask_fp16)) << 16; // Shift sign to FP32 position
    uint32_t exp_fp16 = (fp16_val & exp_mask_fp16) >> 10;
    uint32_t man_fp16 = fp16_val & man_mask_fp16;

    if (exp_fp16 == 0x1F) { // FP16 Inf or NaN
        caster.u = sign_fp32 | 0x7F800000 | (man_fp16 ? 0x00400000 : 0); // Propagate NaN payload crudely
        return caster.f;
    }
    if (exp_fp16 == 0) { // FP16 Zero or Subnormal
        if (man_fp16 == 0) { // Zero
            caster.u = sign_fp32;
            return caster.f;
        } else { // Subnormal FP16 -> treat as very small, effectively flush to zero for simplicity or try to convert
                 // A more proper conversion would be: val = sign * (man_fp16 / 1024.0f) * powf(2, 1 - fp16_exp_bias)
                 // For simplicity in this fallback, let's try a basic conversion attempt which might lose precision
                 // or become zero if not representable as FP32 normal.
            int current_exp = 1 - fp16_exp_bias; // actual exponent
            uint32_t current_man = man_fp16;
            while (!(current_man & 0x0400) && current_man != 0) { // Normalize (0x0400 is implicit 1 for fp16)
                current_man <<= 1;
                current_exp--;
            }
            if (current_man == 0) { // Became zero
                caster.u = sign_fp32; return caster.f;
            }
            uint32_t exp_fp32_val = current_exp + fp32_exp_bias;
            if (exp_fp32_val <= 0) { // Still underflow for FP32 normal
                 caster.u = sign_fp32; return caster.f; // flush to zero
            }
            if (exp_fp32_val >= 0xFF) { // Overflow
                caster.u = sign_fp32 | 0x7F800000; return caster.f; // to Inf
            }
            uint32_t man_fp32_val = (current_man & man_mask_fp16) << (23 - 10);
            caster.u = sign_fp32 | (exp_fp32_val << 23) | man_fp32_val;
            return caster.f;
        }
    }

    // Normal FP16 number
    uint32_t exp_fp32 = (exp_fp16 - fp16_exp_bias + fp32_exp_bias);
    uint32_t man_fp32 = man_fp16 << (23 - 10); // 23 - 10 = 13
    caster.u = sign_fp32 | (exp_fp32 << 23) | man_fp32;
    return caster.f;
}

// Simplified FP32 to FP16 scalar conversion (basic rounding)
// Handles normals, zero, Inf, NaN. Subnormals flushed to zero or handled imperfectly.
static inline uint16_t float32_to_float16_scalar(float f_val) {
    float_uint32_caster caster_in;
    caster_in.f = f_val;
    uint32_t u = caster_in.u;

    const uint16_t sign_fp16 = (uint16_t)((u >> 31) & 0x1);
    const int32_t fp32_exp_bias = 127;
    const int32_t fp16_exp_bias = 15;
    const uint32_t fp32_exp_all_ones = 0xFF;

    // Handle NaN and Inf for FP32 input
    if (((u >> 23) & fp32_exp_all_ones) == fp32_exp_all_ones) {
        if (u & 0x007FFFFF) { // FP32 NaN
            return (sign_fp16 << 15) | 0x7DFF; // Return a quiet NaN in FP16 (e.g. S_11111_0111111111)
        } else { // FP32 Inf
            return (sign_fp16 << 15) | 0x7C00; // FP16 Inf (S_11111_0000000000)
        }
    }
    // Handle Zero for FP32 input
    if ((u & 0x7FFFFFFF) == 0) { // Positive or negative zero
        return (sign_fp16 << 15);
    }

    int32_t exp_fp32_unbiased = ((u >> 23) & fp32_exp_all_ones) - fp32_exp_bias;
    uint32_t man_fp32 = u & 0x007FFFFF; // 23 mantissa bits

    int32_t exp_fp16_biased = exp_fp32_unbiased + fp16_exp_bias;

    uint16_t man_fp16;

    if (exp_fp16_biased >= 0x1F) { // Overflow to Inf
        return (sign_fp16 << 15) | 0x7C00;
    }

    if (exp_fp16_biased <= 0) { // Underflow to zero or subnormal for FP16
        // This simplified version will flush small numbers to zero.
        // A proper conversion would handle FP16 subnormals.
        // If exp_fp16_biased is between, say, -9 and 0, it might be an FP16 subnormal.
        // e.g. if exp_fp16_biased = 0, value is (1.man_fp32) * 2^(exp_fp32_unbiased)
        // We need (0.man_fp16_sub) * 2^(1-fp16_exp_bias).
        // Shift amount = (1-fp16_exp_bias) - exp_fp32_unbiased = 1 - exp_fp16_biased
        if (exp_fp16_biased > -10) { // Potential for subnormal if we were to implement it
            uint32_t full_man_fp32 = (1 << 23) | man_fp32; // Add implicit 1
            int shift = 1 - exp_fp16_biased + (23 - 10); // total shift for fp16 subnormal mantissa
            if (shift < 32) { // Avoid excessive shift
                 man_fp16 = (full_man_fp32 + ((1 << (shift -1)) -1) ) >> shift; // Basic rounding during shift
                 return (sign_fp16 << 15) | (man_fp16 & 0x03FF); // exp=0 for subnormal
            }
        }
        return (sign_fp16 << 15); // Flush to zero
    }

    // Normal numbers: Add implicit 1 to mantissa, round, then shift
    man_fp32 = (1 << 23) | man_fp32; // Add implicit leading 1 to FP32 mantissa

    // Rounding: add 0.5 of the LSB of the part being truncated.
    // We are dropping (23 - 10) = 13 bits. The rounding bit is at position 12.
    uint32_t rounding_add = (1 << 12);
    man_fp32 += rounding_add;

    // Check if rounding caused mantissa to overflow (carrying to exponent)
    if (man_fp32 >= (1 << 24)) { // If mantissa (with implicit 1) overflowed
        man_fp32 >>= 1; // It became 10.000... so after shift is 1.000...
        exp_fp16_biased++;
        if (exp_fp16_biased >= 0x1F) { // Exponent overflowed after mantissa rounding
            return (sign_fp16 << 15) | 0x7C00; // FP16 Inf
        }
    }
    
    man_fp16 = (man_fp32 >> (23 - 10)) & 0x03FF; // Shift and mask to get 10 explicit mantissa bits

    return (sign_fp16 << 15) | (uint16_t)(exp_fp16_biased << 10) | man_fp16;
}








static void * thread_func_adam_step_fp32(void * _adam_worker_args){
    Adam_Worker_Args * adam_worker_args = (Adam_Worker_Args *) _adam_worker_args;

    float * restrict base_param_ptr = (float *) adam_worker_args->param;
    float * restrict base_grad_ptr  = (float *) adam_worker_args->grad;
    float * restrict base_mean_ptr  = (float *) adam_worker_args->mean;
    float * restrict base_var_ptr   = (float *) adam_worker_args->var;

    uint64_t start_index_for_thread = adam_worker_args->start_ind;
    uint64_t num_elements_for_thread = adam_worker_args->num_els;

    float lr_val = adam_worker_args->lr;
    float beta1_val = adam_worker_args->beta1;
    float beta2_val = adam_worker_args->beta2;
    float weight_decay_val = adam_worker_args->weight_decay;
    float epsilon_val = adam_worker_args->epsilon;
    uint64_t current_step = adam_worker_args->step_num; // <-- Get current step

    float one_minus_beta1_val = 1.0f - beta1_val;
    float one_minus_beta2_val = 1.0f - beta2_val;

    // Calculate bias correction factors once
    float bias_correction1_factor = 1.0f;
    float bias_correction2_factor = 1.0f;
    if (current_step > 0) {
        float beta1_pow_t = powf(beta1_val, (float)current_step);
        float beta2_pow_t = powf(beta2_val, (float)current_step);
        // Assuming beta < 1.0 and step_num >= 1, 1.0 - beta_pow_t > 0
        bias_correction1_factor = 1.0f / (1.0f - beta1_pow_t);
        bias_correction2_factor = 1.0f / (1.0f - beta2_pow_t);
    }

    for (uint64_t i_local = 0; i_local < num_elements_for_thread; ++i_local) {
        uint64_t current_global_idx = start_index_for_thread + i_local;

        float p_val = base_param_ptr[current_global_idx];
        float g_val = base_grad_ptr[current_global_idx];
        float m_val = base_mean_ptr[current_global_idx];
        float v_val = base_var_ptr[current_global_idx];

        if (weight_decay_val != 0.0f) {
            g_val += weight_decay_val * p_val;
        }

        m_val = beta1_val * m_val + one_minus_beta1_val * g_val;
        v_val = beta2_val * v_val + one_minus_beta2_val * g_val * g_val;

        // Apply bias correction
        float m_hat_val = m_val * bias_correction1_factor;
        float v_hat_val = v_val * bias_correction2_factor;
        
        float update_val = lr_val * m_hat_val / (sqrtf(v_hat_val) + epsilon_val);
        p_val -= update_val;

        base_param_ptr[current_global_idx] = p_val;
        base_mean_ptr[current_global_idx]  = m_val; // Store original biased mean
        base_var_ptr[current_global_idx]   = v_val; // Store original biased var
    }
    
    return NULL;
}

static void * thread_func_adam_step_bf16(void * _adam_worker_args){
    Adam_Worker_Args * adam_worker_args = (Adam_Worker_Args *) _adam_worker_args;

    uint16_t * restrict base_param_ptr = (uint16_t *) adam_worker_args->param;
    uint16_t * restrict base_grad_ptr  = (uint16_t *) adam_worker_args->grad;
    uint16_t * restrict base_mean_ptr  = (uint16_t *) adam_worker_args->mean;
    uint16_t * restrict base_var_ptr   = (uint16_t *) adam_worker_args->var;

    uint64_t start_index_for_thread = adam_worker_args->start_ind;
    uint64_t num_elements_for_thread = adam_worker_args->num_els;

    float lr_val = adam_worker_args->lr;
    float beta1_val = adam_worker_args->beta1;
    float beta2_val = adam_worker_args->beta2;
    float weight_decay_val = adam_worker_args->weight_decay;
    float epsilon_val = adam_worker_args->epsilon;
    uint64_t current_step = adam_worker_args->step_num; // <-- Get current step

    float one_minus_beta1_val = 1.0f - beta1_val;
    float one_minus_beta2_val = 1.0f - beta2_val;

    float bias_correction1_factor = 1.0f;
    float bias_correction2_factor = 1.0f;
    if (current_step > 0) {
        float beta1_pow_t = powf(beta1_val, (float)current_step);
        float beta2_pow_t = powf(beta2_val, (float)current_step);
        bias_correction1_factor = 1.0f / (1.0f - beta1_pow_t);
        bias_correction2_factor = 1.0f / (1.0f - beta2_pow_t);
    }

    for (uint64_t i_local = 0; i_local < num_elements_for_thread; ++i_local) {
        uint64_t current_global_idx = start_index_for_thread + i_local;

        float p_val_fp32 = bfloat16_to_float_scalar(base_param_ptr[current_global_idx]);
        float g_val_fp32 = bfloat16_to_float_scalar(base_grad_ptr[current_global_idx]);
        float m_val_fp32 = bfloat16_to_float_scalar(base_mean_ptr[current_global_idx]);
        float v_val_fp32 = bfloat16_to_float_scalar(base_var_ptr[current_global_idx]);

        if (weight_decay_val != 0.0f) {
            g_val_fp32 += weight_decay_val * p_val_fp32;
        }
        m_val_fp32 = beta1_val * m_val_fp32 + one_minus_beta1_val * g_val_fp32;
        v_val_fp32 = beta2_val * v_val_fp32 + one_minus_beta2_val * g_val_fp32 * g_val_fp32;

        float m_hat_val_fp32 = m_val_fp32 * bias_correction1_factor;
        float v_hat_val_fp32 = v_val_fp32 * bias_correction2_factor;

        float update_val_fp32 = lr_val * m_hat_val_fp32 / (sqrtf(v_hat_val_fp32) + epsilon_val);
        p_val_fp32 -= update_val_fp32;

        base_param_ptr[current_global_idx] = float_to_bfloat16_scalar(p_val_fp32);
        base_mean_ptr[current_global_idx]  = float_to_bfloat16_scalar(m_val_fp32); // Store original biased mean
        base_var_ptr[current_global_idx]   = float_to_bfloat16_scalar(v_val_fp32); // Store original biased var
    }
    
    return NULL;
}

static void * thread_func_adam_step_fp16(void * _adam_worker_args){
    Adam_Worker_Args * adam_worker_args = (Adam_Worker_Args *) _adam_worker_args;

    uint16_t * restrict base_param_ptr = (uint16_t *) adam_worker_args->param;
    uint16_t * restrict base_grad_ptr  = (uint16_t *) adam_worker_args->grad;
    uint16_t * restrict base_mean_ptr  = (uint16_t *) adam_worker_args->mean;
    uint16_t * restrict base_var_ptr   = (uint16_t *) adam_worker_args->var;

    uint64_t start_index_for_thread = adam_worker_args->start_ind;
    uint64_t num_elements_for_thread = adam_worker_args->num_els;

    float lr_val = adam_worker_args->lr;
    float beta1_val = adam_worker_args->beta1;
    float beta2_val = adam_worker_args->beta2;
    float weight_decay_val = adam_worker_args->weight_decay;
    float epsilon_val = adam_worker_args->epsilon;
    uint64_t current_step = adam_worker_args->step_num; // <-- Get current step

    float one_minus_beta1_val = 1.0f - beta1_val;
    float one_minus_beta2_val = 1.0f - beta2_val;

    float bias_correction1_factor = 1.0f;
    float bias_correction2_factor = 1.0f;
    if (current_step > 0) {
        float beta1_pow_t = powf(beta1_val, (float)current_step);
        float beta2_pow_t = powf(beta2_val, (float)current_step);
        bias_correction1_factor = 1.0f / (1.0f - beta1_pow_t);
        bias_correction2_factor = 1.0f / (1.0f - beta2_pow_t);
    }

    for (uint64_t i_local = 0; i_local < num_elements_for_thread; ++i_local) {
        uint64_t current_global_idx = start_index_for_thread + i_local;

        float p_val_fp32 = float16_to_float32_scalar(base_param_ptr[current_global_idx]);
        float g_val_fp32 = float16_to_float32_scalar(base_grad_ptr[current_global_idx]);
        float m_val_fp32 = float16_to_float32_scalar(base_mean_ptr[current_global_idx]);
        float v_val_fp32 = float16_to_float32_scalar(base_var_ptr[current_global_idx]);

        if (weight_decay_val != 0.0f) {
            g_val_fp32 += weight_decay_val * p_val_fp32;
        }
        m_val_fp32 = beta1_val * m_val_fp32 + one_minus_beta1_val * g_val_fp32;
        v_val_fp32 = beta2_val * v_val_fp32 + one_minus_beta2_val * g_val_fp32 * g_val_fp32;

        float m_hat_val_fp32 = m_val_fp32 * bias_correction1_factor;
        float v_hat_val_fp32 = v_val_fp32 * bias_correction2_factor;

        float update_val_fp32 = lr_val * m_hat_val_fp32 / (sqrtf(v_hat_val_fp32) + epsilon_val);
        p_val_fp32 -= update_val_fp32;

        base_param_ptr[current_global_idx] = float32_to_float16_scalar(p_val_fp32);
        base_mean_ptr[current_global_idx]  = float32_to_float16_scalar(m_val_fp32); // Store original biased mean
        base_var_ptr[current_global_idx]   = float32_to_float16_scalar(v_val_fp32); // Store original biased var
    }
    
    return NULL;
}

static void * thread_func_adam_step_fp8e4m3(void * _adam_worker_args) {

    Adam_Worker_Args * adam_worker_args = (Adam_Worker_Args *) _adam_worker_args;

    uint8_t * param = (uint8_t *) adam_worker_args->param;
    uint8_t * grad = (uint8_t *) adam_worker_args->grad;
    uint8_t * mean = (uint8_t *) adam_worker_args->mean;
    uint8_t * var = (uint8_t *) adam_worker_args->var;

    return NULL;
}

static void * thread_func_adam_step_fp8e5m2(void * _adam_worker_args){

    Adam_Worker_Args * adam_worker_args = (Adam_Worker_Args *) _adam_worker_args;

    uint8_t * param = (uint8_t *) adam_worker_args->param;
    uint8_t * grad = (uint8_t *) adam_worker_args->grad;
    uint8_t * mean = (uint8_t *) adam_worker_args->mean;
    uint8_t * var = (uint8_t *) adam_worker_args->var;


    return NULL;
}




int do_adam_step_host(DataflowDatatype param_dt, DataflowDatatype grad_dt, DataflowDatatype mean_dt, DataflowDatatype var_dt,
                             int num_threads,
                             uint64_t num_els, uint64_t step_num, float lr, float beta1, float beta2, float weight_decay, float epsilon,
                             void * param, void * grad, void * mean, void * var){

    // for now only support when all same dtype..
    if (param_dt != grad_dt || param_dt != mean_dt || param_dt != var_dt){
        fprintf(stderr, "Error: all dtypes must be the same for adam step...\n");
        return -1;
    }

    void * (*adam_step_func)(void * _adam_worker_args);

    if (param_dt == DATAFLOW_FP32){
        adam_step_func = thread_func_adam_step_fp32;
    }
    else if (param_dt == DATAFLOW_BF16){
        adam_step_func = thread_func_adam_step_bf16;
    }
    else if (param_dt == DATAFLOW_FP16){
        adam_step_func = thread_func_adam_step_fp16;
    }
    /* Not ready for fp8 yet
    else if (param_dt == DATAFLOW_FP8E4M3){
        adam_step_func = thread_func_adam_step_fp8e4m3;
    }
    else if (param_dt == DATAFLOW_FP8E5M2){
        adam_step_func = thread_func_adam_step_fp8e5m2;
    }
    */
    else{
        fprintf(stderr, "Error: unsupported dtype for adam step...\n");
        return -1;
    }

    if (!adam_step_func){
        fprintf(stderr, "Error: failed to get thread function for adam step...\n");
        return -1;
    }

    if (num_threads <= 1){
        Adam_Worker_Args adam_worker_args;
        adam_worker_args.param = param;
        adam_worker_args.grad = grad;
        adam_worker_args.mean = mean;
        adam_worker_args.var = var;
        adam_worker_args.start_ind = 0;
        adam_worker_args.num_els = num_els;
        adam_worker_args.lr = lr;
        adam_worker_args.beta1 = beta1;
        adam_worker_args.beta2 = beta2;
        adam_worker_args.weight_decay = weight_decay;
        adam_worker_args.epsilon = epsilon;
        adam_worker_args.step_num = step_num;

        adam_step_func(&adam_worker_args);
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

    pthread_t * adam_workers = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    if (!adam_workers){
        fprintf(stderr, "Error: failed to allocate adam workers...\n");
        if (ret_attr_init == 0) {
            pthread_attr_destroy(&attr);
        }
        if (cpu_mask_for_node) {
            CPU_FREE(cpu_mask_for_node);
        }
        return -1;
    }

    Adam_Worker_Args * adam_worker_args_array = (Adam_Worker_Args *) malloc(num_threads * sizeof(Adam_Worker_Args));
    if (!adam_worker_args_array){
        fprintf(stderr, "Error: failed to allocate adam worker args...\n");
        free(adam_workers);
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
        adam_worker_args_array[i].param = param;
        adam_worker_args_array[i].grad = grad;
        adam_worker_args_array[i].mean = mean;
        adam_worker_args_array[i].var = var;
        adam_worker_args_array[i].lr = lr;
        adam_worker_args_array[i].beta1 = beta1;
        adam_worker_args_array[i].beta2 = beta2;
        adam_worker_args_array[i].weight_decay = weight_decay;
        adam_worker_args_array[i].epsilon = epsilon;
        adam_worker_args_array[i].step_num = step_num;
    }
    
    // Give slices to each worker...

    uint64_t base_chunk_size = num_els / num_threads;
    uint64_t remainder_elements = num_els % num_threads;
    uint64_t current_start_index = 0;
    uint64_t elements_for_this_thread;

    for (int i = 0; i < num_threads; i++){
        
        adam_worker_args_array[i].start_ind = current_start_index;
        elements_for_this_thread = base_chunk_size + (i < remainder_elements ? 1 : 0);
        adam_worker_args_array[i].num_els = elements_for_this_thread;
        current_start_index += elements_for_this_thread;

        // Start worker
        int ret_create;
        // Only use attr if successfully initialized and affinity mask is ready
        if (ret_attr_init == 0 && affinity_set_successfully) { 
            ret_create = pthread_create(&adam_workers[i], &attr, adam_step_func, &adam_worker_args_array[i]);
        } else {
            ret_create = pthread_create(&adam_workers[i], NULL, adam_step_func, &adam_worker_args_array[i]);
        }
        if (ret_create != 0) {
            fprintf(stderr, "[Adam] Error creating thread %d: %s\n", i, strerror(ret_create));
            // Handle thread creation error: cleanup already created threads, allocated memory etc.
            // For simplicity here, we'll just print and continue, but in robust code, you'd manage this.
            // For now, just free and exit might be simpler than partial cleanup.
            for(int k=0; k<i; ++k) {
                pthread_cancel(adam_workers[k]);
            }
            for(int k=0; k<i; ++k) {
                pthread_join(adam_workers[k], NULL);
            }
            free(adam_workers);
            free(adam_worker_args_array);
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
        pthread_join(adam_workers[i], NULL);
    }

    // Free allocated memory
    free(adam_workers);
    free(adam_worker_args_array);
    

    return 0;
}

