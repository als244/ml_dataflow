import os

model_config_options = {"dense32B": "../models/32B"}
seqlen_options = [8192]

host_mem_options = [380]

#dev_mem_options = [16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 77]

dev_mem_options = [24, 28, 32, 40, 48, 56, 64, 72, 77]

results_dir = "results"

os.makedirs(results_dir, exist_ok=True)

for model_name, model_config in model_config_options.items():

    os.makedirs(f"{results_dir}/{model_name}", exist_ok=True)

    log_dir = f"{results_dir}/{model_name}"

    for seqlen in seqlen_options:
        for host_mem in host_mem_options:
            for dev_mem in dev_mem_options:

                exp_name = f"{model_name}_{seqlen}_{host_mem}_{dev_mem}"

                cmd = f"./transformer {host_mem} {dev_mem} {seqlen} {model_config} > {log_dir}/{exp_name}.log 2> {log_dir}/{exp_name}.err"

                os.system(cmd)

