import sys
import os
import time
import json

def get_throughput(host_mem_size, device_mem_size, seq_len, model_size, output_filepath):    
    return os.system(f"./transformerRecordThroughput {host_mem_size} {device_mem_size} {seq_len} {model_size} {output_filepath}")
    


def run_sweep(sweep_config_filepath, output_filepath):
    with open(sweep_config_filepath, "r") as f:
        sweep_config = json.load(f)
    
    host_mem_sizes = sweep_config["host_mem_gb"]
    device_mem_sizes = sweep_config["device_mem_gb"]
    seq_lens = sweep_config["seq_len"]
    model_sizes = sweep_config["model_size"]

    cur_run_num = 1
    total_runs = len(host_mem_sizes) * len(device_mem_sizes) * len(seq_lens) * len(model_sizes)
    
    for host_mem_size in host_mem_sizes:
        for device_mem_size in device_mem_sizes:
            for seq_len in seq_lens:
                for model_size in model_sizes:
                    print(f"Running ({cur_run_num}/{total_runs}): {host_mem_size} {device_mem_size} {seq_len} {model_size}")
                    exit_status = get_throughput(host_mem_size, device_mem_size, seq_len, model_size, output_filepath)
                    if exit_status != 0:
                        print(f"\tFailed to run...")
                        out_file = open(output_filepath, "a")
                        # chunk_size, total_home_acts, num_inp_only_saved, num_inp_attn_saved, num_full_saved, total_dev_acts, seqs per step, avg. step time, tok/sec, TFLOPS, MFU, HFU
                        out_file.write(f"{host_mem_size},{device_mem_size},{seq_len},{model_size},0,0,0,0,0,0,0,0,0,0,0,0\n")
                        out_file.close()
                    ## wait two seconds for gpu to cool down
                    time.sleep(2)
                    cur_run_num += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sweep_training_environments.py <sweep config filepath> <output filepath>")
        sys.exit(1)
    sweep_config_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    run_sweep(sweep_config_filepath, output_filepath)
