import os


def sweep_step_batching_combinations(seqs_per_step, seq_len, max_micro_tokens, min_batch_size=1, max_batch_size=None):
    """
    Generate parameter sweep with token constraint
    
    Args:
        seqs_per_step (int): Target product value
        seq_len (int): Sequence length
        max_micro_tokens (int): Maximum tokens per batch (seq_len * seqs_per_batch must be < this)
        min_batch_size (int): Minimum seqs_per_batch value
        max_batch_size (int): Maximum seqs_per_batch value (None for no limit)
        
    Returns:
        List of tuples: [(seqs_per_batch, grad_accum_steps), ...]
    """
    combinations = []
    
    # Calculate max allowed seqs_per_batch based on token constraint
    max_seqs_from_tokens = (max_micro_tokens - 1) // seq_len  # -1 to ensure strict inequality
    
    # Apply all constraints
    max_batch = max_batch_size if max_batch_size else seqs_per_step
    effective_max = min(max_batch, max_seqs_from_tokens, seqs_per_step)
    
    for seqs_per_batch in range(min_batch_size, effective_max + 1):
        if seqs_per_step % seqs_per_batch == 0:
            grad_accum_steps = seqs_per_step // seqs_per_batch
            
            # Double-check the token constraint
            batch_tokens = seq_len * seqs_per_batch
            if batch_tokens < max_micro_tokens:
                combinations.append((seqs_per_batch, grad_accum_steps))
    
    return combinations


def run_experiment(log_dir, experiment_config):
    """Run a single experiment using the launcher script"""
    experiment_name = experiment_config["experiment_name"]
    model_name = experiment_config["model_name"]
    model_config = experiment_config["model_config"]
    seq_len = experiment_config["seq_len"]
    seqs_per_batch = experiment_config["seqs_per_batch"]
    grad_accum_steps = experiment_config["grad_accum_steps"]
    num_steps = experiment_config["num_steps"]
    zero_stage = experiment_config["zero_stage"]
    save_act_layer_frac = experiment_config["save_act_layer_frac"]
    
    # Ensure log directories exist
    os.makedirs(log_dir, exist_ok=True)
    model_log_dir = os.path.join(log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    
    # Build the command string
    cmd = "./launch_train.sh"
    
    # Add optional arguments
    if zero_stage is not None:
        cmd += f" --zero_stage {zero_stage}"
    
    if save_act_layer_frac != 0:  # Only add if not default value
        cmd += f" --save_act_layer_frac {save_act_layer_frac}"
    
    # Add positional arguments
    cmd += f" {model_config} {seq_len} {seqs_per_batch} {grad_accum_steps} {num_steps}"
    
    # Redirect output to separate log files
    log_file = os.path.join(model_log_dir, f"{experiment_name}.log")
    err_file = os.path.join(model_log_dir, f"{experiment_name}.err")
    cmd += f" > {log_file} 2> {err_file}"
    
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {cmd}")
    print(f"Stdout logging to: {log_file}")
    print(f"Stderr logging to: {err_file}")
    
    # Run the command using os.system
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Experiment completed successfully")
        return True
    else:
        print(f"Experiment failed with exit code: {exit_code}")
        return False






#model_configs = {"dense8B": "model_configs/8b_config.json", "dense15B": "model_configs/15b_config.json", "dense32B": "model_configs/32b_config.json"}
model_config_options = {"dense8B": "model_configs/8b_config.json"}
seq_lens_options = {"dense8B": {"H100": [8192], "RTX5090": [8192]}}
seqs_per_step_options = {"dense8B": {"H100": [72], "RTX5090": [24]}}
zero_stages_options = [0, 1, 2, 3]
save_act_layer_fracs_options = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
max_micro_tokens = 65536
num_steps = 3



def generate_experiment_configs(device_name):


    experiment_configs = {}


    for model_name, config_path in model_config_options.items():
        for seq_len in seq_lens_options[model_name][device_name]:
            for seqs_per_step in seqs_per_step_options[model_name][device_name]:
                step_batching_combinations = sweep_step_batching_combinations(seqs_per_step, seq_len, max_micro_tokens)

                for seqs_per_batch, grad_accum_steps in step_batching_combinations:

                    for zero_stage in zero_stages_options:

                        for save_act_layer_frac in save_act_layer_fracs_options:

                            experiment_name = f"{model_name}_{seq_len}_{seqs_per_step}_{seqs_per_batch}_{grad_accum_steps}_{zero_stage}_{save_act_layer_frac}"

                            experiment_config = {}
                            experiment_config["experiment_name"] = experiment_name
                            experiment_config["model_name"] = model_name
                            experiment_config["model_config"] = config_path
                            experiment_config["seq_len"] = seq_len
                            experiment_config["seqs_per_step"] = seqs_per_step
                            experiment_config["seqs_per_batch"] = seqs_per_batch
                            experiment_config["grad_accum_steps"] = grad_accum_steps
                            experiment_config["num_steps"] = num_steps
                            experiment_config["zero_stage"] = zero_stage
                            experiment_config["save_act_layer_frac"] = save_act_layer_frac

                            experiment_configs[experiment_name] = experiment_config


    return experiment_configs

def run_all_experiments(log_dir, experiment_configs):
    """Run all generated experiments"""
    total_experiments = len(experiment_configs)
    print(f"Generated {total_experiments} experiment configurations")
    print(f"Logs will be saved to: {log_dir}")
    
    successful = 0
    failed = 0
    
    for i, (experiment_name, experiment_config) in enumerate(experiment_configs.items(), 1):
        print(f"\n{'='*60}")
        print(f"Running experiment {i + 1}/{total_experiments}: {experiment_name}")
        print(f"{'='*60}")
        
        success = run_experiment(log_dir, experiment_config)
        if success:
            successful += 1
        else:
            failed += 1
            print(f"Experiment {experiment_name} failed!")
    
    print(f"\n{'='*60}")
    print(f"Experiment Summary:")
    print(f"Total: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Logs saved to: {log_dir}")
    

if __name__ == "__main__":

    LOG_DIR = "results"
    DEVICE_NAME = "H100"

    experiment_configs = generate_experiment_configs(DEVICE_NAME)

    run_all_experiments(LOG_DIR, experiment_configs)


    


            