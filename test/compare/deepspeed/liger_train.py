# train.py (Corrected)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import deepspeed
import argparse
import time

from liger_model import Llama3Model, ModelArgs

import ctypes

_cudart = ctypes.CDLL('libcudart.so')

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DeepSpeed Llama3 Training with Full Logging")
parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for training')
parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
# === CHANGE: Add deepspeed_config argument ===
# DeepSpeed launcher will automatically provide this argument.
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

torch.set_default_dtype(torch.bfloat16)

SEED = 42

torch.manual_seed(SEED)

# --- Model & Training Configuration ---
model_args = ModelArgs(
    dtype=torch.bfloat16,
    dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256,
    intermediate_size=14336, norm_eps=1e-5, max_seq_len=args.seq_len
)

epochs = 1
learning_rate = 1e-5
grad_accum_steps = 19
total_steps = 3 * grad_accum_steps

# --- DeepSpeed Configuration with Logging ---
# The ds_config can be loaded from a JSON file specified by --deepspeed_config
# For simplicity, we define it here.
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": grad_accum_steps,
    "wall_clock_breakdown": True,
    "optimizer": { "type": "AdamW", "params": { "lr": learning_rate, "betas": [0.9, 0.95], "fp32_optimizer_states": False} },
    "bf16": { "enabled": True},
    "steps_per_print": 1,
    "activation_checkpointing": { "cpu_checkpointing": True },

    # === ADDED: A safe baseline configuration for initialization ===
    # This will be used to successfully load the model before tuning begins.
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        #"offload_param": {
        #    "device": "cpu",
        #    "pin_memory": True
        #},
    },

    
    
    # --- AUTOTUNING CONFIG (No Changes Here) ---
    # The tuner will use this section to experiment *after* initializing
    # successfully with the baseline config above.
    #"autotuning": {
    #    "enabled": True,
    #    "fast": False, 
    #    "results_dir": "autotuning_logs",
    #    "log_level": "info",
    #    "zero_config": {
    #        "stage": [2, 3], # You can even ask it to test multiple stages
    #        "offload_optimizer": {
    #            "device": "cpu",
    #            "pin_memory": True
    #        },
    #        "offload_param": {
    #            "device": "cpu",
    #            "pin_memory": True
    #        }
    #    },
    #},

    #"flops_profiler": {
    #    "enabled": True,
    #    "module_depth": -1,
    #    "detailed": True,
    #    "output_file": "flops_profile.txt"
    #},
    
    # --- Logging ---
    "tensorboard": { "enabled": True, "output_path": "tensorboard_logs/" },
    "csv_monitor": { "enabled": True, "output_path": "csv_logs/" }
}

# === CHANGE: Function now returns the Dataset, not the DataLoader ===
def get_dummy_dataset(seq_length=512, total_tokens=2**24):
    """Returns a TensorDataset."""
    print("Creating dummy dataset...")
    num_samples = total_tokens // seq_length
    source_data = torch.randint(0, model_args.vocab_size, (num_samples, seq_length))
    target_data = torch.roll(source_data, shifts=-1, dims=1)
    target_data[:, -1] = -100
    dataset = TensorDataset(source_data, target_data)
    print("Dummy dataset created.")
    return dataset

# --- Initialization ---
print("Initializing model and DeepSpeed...")
model = Llama3Model(model_args)

# === CHANGE: Call the new function to get the dataset ===
dummy_dataset = get_dummy_dataset(seq_length=args.seq_len)

print("Initializing DeepSpeed...")
# === CHANGE: Pass the dataset to training_data ===
model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
    args=args, # Pass the full args object
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    training_data=dummy_dataset # Pass the Dataset here
)


torch.cuda.empty_cache()

# --- Training Loop with Throughput Calculation ---
print(f"Starting training with sequence length: {args.seq_len}...")

ret = _cudart.cudaProfilerStart()

start_time = time.time()
total_tokens = 0

num_steps = 0

for epoch in range(epochs):
    # The training_dataloader is now the one created by DeepSpeed
    for i, (inputs, labels) in enumerate(training_dataloader):
        inputs = inputs.to(model_engine.device)
        labels = labels.to(model_engine.device)
        
        loss = model_engine(inputs, labels)
        #loss = criterion(outputs.view(-1, model_args.vocab_size), labels.view(-1))

        model_engine.backward(loss)
        model_engine.step()

        num_steps += 1

        actual_bs = model_engine.train_micro_batch_size_per_gpu()
        total_tokens += actual_bs * args.seq_len
        
        if (i + 1) % grad_accum_steps == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                tokens_per_sec = total_tokens / elapsed_time
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"Epoch: {epoch+1}, Step: {i+1}, BS: {actual_bs}, Loss: {loss.item():.4f} | "
                    f"Step total_tokens: {total_tokens}, Step total_time = {elapsed_time}, Tok/sec: {tokens_per_sec:.2f} | VRAM: {mem_alloc:.2f}GB"
                )
                start_time = time.time()
                total_tokens = 0


        if num_steps == total_steps:
            break

ret = _cudart.cudaProfilerStop()

peak_mem_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

print(f"\nTraining complete! ✅\n\tPeak Memory Reserved: {peak_mem_reserved_gb:.2f} GB")
