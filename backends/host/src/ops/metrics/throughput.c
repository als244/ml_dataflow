#include "host_ops.h"

int get_human_readable_time(struct timespec * ts, char * time_buf){

	// Convert seconds to struct tm (local time)
    struct tm tminfo;
    if (localtime_r(&(ts -> tv_sec), &tminfo) == NULL) {
		fprintf(stderr, "localtime_r returned NULL\n");
        return 1;
    }

    // Format the time into a string
    // %Y: Year with century
    // %m: Month as a decimal number (01-12)
    // %d: Day of the month as a decimal number (01-31)
    // %H: Hour (24-hour clock) as a decimal number (00-23)
    // %M: Minute as a decimal number (00-59)
    // %S: Second as a decimal number (00-60)
    if (strftime(time_buf, 100, "%Y-%m-%d %H:%M:%S", &tminfo) == 0) {
        fprintf(stderr, "strftime returned 0\n");
        return -1;
    }

	return 0;
}

float get_seq_flops(int seq_len, int vocab_size, int model_dim, int kv_dim, int num_shared_experts, int num_total_routed_experts, int num_active_routed_experts, int expert_dim, int num_layers, 
							float * ret_seq_flops_fwd, float * ret_seq_flops_head, float * ret_seq_flops_bwd_x, float * ret_seq_flops_bwd_w){

	float seq_flops_fwd = 0;
	float seq_flops_head = 0;
	float seq_flops_bwd_x = 0;
	float seq_flops_bwd_w = 0;

	// need to cast to float otherwise results in integer overflow...
	float seq_len_f = (float) seq_len;
	float model_dim_f = (float) model_dim;
	float kv_dim_f = (float) kv_dim;
	float expert_dim_f = (float) expert_dim;
	float vocab_size_f = (float) vocab_size;


	// 1.) FORWARD FLOPS

	// attention inp projection
	// q
	seq_flops_fwd += 2 * seq_len_f * model_dim_f * model_dim_f;
	// for k, v
	seq_flops_fwd += 2 * (2 * seq_len_f * model_dim_f * kv_dim_f);

	// attention flops

	// assume causal attention, otherwise should be 4 * instead of 2 *
	seq_flops_fwd += 2 * seq_len_f * seq_len_f * model_dim_f;

	// attention out projection
	seq_flops_fwd += 2 * seq_len_f * model_dim_f * model_dim_f;

	// router flops (if num_routed_experts > 0)
	if (num_total_routed_experts > 0){
		seq_flops_fwd += 2 * seq_len_f * model_dim_f * (float)num_total_routed_experts;
	}

	// Assuming SwiGLU gate with 3 matrices per expert...

	// shared experts flops
	seq_flops_fwd += num_shared_experts * 3 * (2 * seq_len_f * model_dim_f * expert_dim_f);

	// routed experts flops
	seq_flops_fwd += num_active_routed_experts * 3 * (2 * seq_len_f * model_dim_f * expert_dim_f);

	// 2.) HEAD FLOPS

	// forwards head
	seq_flops_head += 2 * seq_len_f * model_dim_f * vocab_size_f;

	// backwards head
	seq_flops_head += 2 * seq_len_f * model_dim_f * vocab_size_f;
	
	// 3.) BACKWARD FLOPS

	// a.) BWD X (same as forward, but 2.5 for attention)
	seq_flops_bwd_x = seq_flops_fwd;
	seq_flops_bwd_x -= (2 * seq_len_f * seq_len_f * model_dim_f);
	seq_flops_bwd_x += (2.5 * seq_len_f * seq_len_f * model_dim_f);


	// b.) BWD W

	// no attention computation needed for bwd w, otherwise same as forward
	seq_flops_bwd_w = seq_flops_fwd;
	seq_flops_bwd_w -= (2 * seq_len_f * seq_len_f * model_dim_f);


	// Multiply by num_layers (besides head)
	seq_flops_fwd *= num_layers;
	seq_flops_bwd_x *= num_layers;
	seq_flops_bwd_w *= num_layers;


	float total_seq_flops = seq_flops_fwd + seq_flops_head + seq_flops_bwd_x + seq_flops_bwd_w;

	if (ret_seq_flops_fwd){
		*ret_seq_flops_fwd = seq_flops_fwd;
	}
	if (ret_seq_flops_head){
		*ret_seq_flops_head = seq_flops_head;
	}
	if (ret_seq_flops_bwd_x){
		*ret_seq_flops_bwd_x = seq_flops_bwd_x;
	}
	if (ret_seq_flops_bwd_w){
		*ret_seq_flops_bwd_w = seq_flops_bwd_w;
	}
	
	return total_seq_flops;
}

int start_step_metrics(void * _step_throughput_op_args){

	Step_Throughput_Host_Op_Args * step_throughput_op_args = (Step_Throughput_Host_Op_Args *) _step_throughput_op_args;

	clock_gettime(CLOCK_REALTIME, &(step_throughput_op_args->start_time));

	int num_seqs = step_throughput_op_args->num_seqs;
	int * seqlens = step_throughput_op_args->seqlens;

	int seq_len;
	float seq_flops_fwd;
	float seq_flops_head;
	float seq_flops_bwd_x;
	float seq_flops_bwd_w;

	for (int i = 0; i < num_seqs; i++){
		seq_len = seqlens[i];

		float total_seq_flops = get_seq_flops(seq_len, step_throughput_op_args->vocab_size, step_throughput_op_args->model_dim, step_throughput_op_args->kv_dim, 
														step_throughput_op_args->num_shared_experts, step_throughput_op_args->num_total_routed_experts, step_throughput_op_args->num_active_routed_experts, step_throughput_op_args->expert_dim, step_throughput_op_args->num_layers, 
														&seq_flops_fwd, &seq_flops_head, &seq_flops_bwd_x, &seq_flops_bwd_w);

		step_throughput_op_args->total_tokens += seq_len;
		step_throughput_op_args->total_fwd_flops += seq_flops_fwd;
		step_throughput_op_args->total_head_flops += seq_flops_head;
		step_throughput_op_args->total_bwd_x_flops += seq_flops_bwd_x;
		step_throughput_op_args->total_bwd_w_flops += seq_flops_bwd_w;
		step_throughput_op_args->total_computation_flops += total_seq_flops;
	}

	// set start time	
	clock_gettime(CLOCK_REALTIME, &(step_throughput_op_args->start_time));

	return 0;
}


int end_step_metrics(void * _step_throughput_op_args){

	int ret;

    Step_Throughput_Host_Op_Args * step_throughput_op_args = (Step_Throughput_Host_Op_Args *) _step_throughput_op_args;

    // populate the end time
    clock_gettime(CLOCK_REALTIME, &(step_throughput_op_args->end_time));

	uint64_t start_time_ns = step_throughput_op_args->start_time.tv_sec * 1e9 + step_throughput_op_args->start_time.tv_nsec;
	uint64_t end_time_ns = step_throughput_op_args->end_time.tv_sec * 1e9 + step_throughput_op_args->end_time.tv_nsec;

	uint64_t duration_ns = end_time_ns - start_time_ns;

	step_throughput_op_args->duration_ns = duration_ns;

	float duration_s = (float) duration_ns / 1e9;
	step_throughput_op_args->duration_s = duration_s;
	


	float achieved_flop_rate = step_throughput_op_args->total_computation_flops / duration_s;
	step_throughput_op_args->achieved_flop_rate = achieved_flop_rate;


	float mfu = achieved_flop_rate / step_throughput_op_args->peak_hardware_flop_rate;
	step_throughput_op_args->mfu = mfu;

	float tokens_per_second = step_throughput_op_args->total_tokens / duration_s;
	step_throughput_op_args->tokens_per_second = tokens_per_second;

	if (step_throughput_op_args->to_print_metrics){
		printf("\n\n[THROUGHPUT: Completed Step %d. Num Seqs: %d, Total Tokens: %d]: %.2f TFLOPS\n\n\tTokens/S: %.2f\n\tMFU: %.2f%%\n\n", step_throughput_op_args->step_num, step_throughput_op_args->num_seqs, step_throughput_op_args->total_tokens, (achieved_flop_rate / 1e12), tokens_per_second, mfu * 100);
	}
	
	char start_time_buf[100];
	ret = get_human_readable_time(&(step_throughput_op_args->start_time), start_time_buf);
	if (ret != 0){
		fprintf(stderr, "get_human_readable_time returned %d\n", ret);
		return ret;
	}
	char end_time_buf[100];
	ret = get_human_readable_time(&(step_throughput_op_args->end_time), end_time_buf);
	if (ret != 0){
		fprintf(stderr, "get_human_readable_time returned %d\n", ret);
		return ret;
	}
	if ((step_throughput_op_args->to_print_metrics) && (step_throughput_op_args->to_print_verbose)){
		printf("\tTime Info:\n\t\tStart Time: %s\n\t\tEnd Time: %s\n\t\tDuration: %.4f seconds\n", start_time_buf, end_time_buf, duration_s);

		float fwd_pct = (step_throughput_op_args->total_fwd_flops / step_throughput_op_args->total_computation_flops) * 100;
		float head_pct = (step_throughput_op_args->total_head_flops / step_throughput_op_args->total_computation_flops) * 100;
		float bwd_x_pct = (step_throughput_op_args->total_bwd_x_flops / step_throughput_op_args->total_computation_flops) * 100;
		float bwd_w_pct = (step_throughput_op_args->total_bwd_w_flops / step_throughput_op_args->total_computation_flops) * 100;

		printf("\tFLOP Info:\n\t\tTotal FLOPS: %.3e\n\t\t\tFwd Flops: %.2f%%\n\t\t\tHead Flops: %.2f%%\n\t\t\tBwd X Flops: %.2f%%\n\t\t\tBwd W Flops: %.2f%%\n\n\n", step_throughput_op_args->total_computation_flops, fwd_pct, head_pct, bwd_x_pct, bwd_w_pct);
	}

	return 0;
}