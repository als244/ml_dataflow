import sys
import os
import time
import json

def get_throughput(host_mem_size, device_mem_size, seq_len, model_size, total_steps, warmup_steps, output_filepath):    
    return os.system(f"./transformerRecordThroughput {host_mem_size} {device_mem_size} {seq_len} {model_size} {total_steps} {warmup_steps} {output_filepath}")
    


def run_sweep(sweep_config_filepath, experiment_name, output_filepath):
    with open(sweep_config_filepath, "r") as f:
        sweep_config = json.load(f)

    experiment_config = sweep_config[experiment_name]    
    
    host_mem_sizes = experiment_config["host_mem_gb"]
    device_mem_sizes = experiment_config["device_mem_gb"]
    seq_lens = experiment_config["seq_len"]
    model_sizes = experiment_config["model_size"]

    all_runs = []
    for host_mem_size in host_mem_sizes:
        for device_mem_size in device_mem_sizes:
            for seq_len in seq_lens:
                for model_size in model_sizes:
                    all_runs.append((host_mem_size, device_mem_size, seq_len, model_size))

    start_run_ind = experiment_config["start_run_ind"]
    total_runs = len(host_mem_sizes) * len(device_mem_sizes) * len(seq_lens) * len(model_sizes)

    total_steps = int(experiment_config["total_steps"])
    warmup_steps = int(experiment_config["warmup_steps"])
    

    for cur_run_num in range(start_run_ind, total_runs):
        run = all_runs[cur_run_num]
        host_mem_size, device_mem_size, seq_len, model_size = run
        print(f"{experiment_name}: ({cur_run_num + 1}/{total_runs}): {host_mem_size} {device_mem_size} {seq_len} {model_size}", flush=True)
        exit_status = get_throughput(host_mem_size, device_mem_size, seq_len, model_size, total_steps, warmup_steps, output_filepath)
        if exit_status != 0:
            print(f"\tFailed to run...", flush=True)
            out_file = open(output_filepath, "a")
            # used_host_mem_gb, used_dev_mem_gb, chunk_size, total_home_acts, num_inp_only_saved, num_inp_attn_saved, num_full_saved, total_dev_acts, num_rounds_per_step, seqs per step, recompute pct, attn flop pct,avg. step time, tok/sec, TFLOPS, MFU, HFU
            out_file.write(f"{host_mem_size},{device_mem_size},{seq_len},{model_size},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
            out_file.close()
        ## wait two seconds for gpu to cool down
        time.sleep(2)
        cur_run_num += 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python sweep_training_environments.py <sweep config filepath> <       e> <output filepath>")
        sys.exit(1)
    sweep_config_filepath = sys.argv[1]
    experiment_name = sys.argv[2]
    output_filepath = sys.argv[3]
    run_sweep(sweep_config_filepath, experiment_name, output_filepath)
