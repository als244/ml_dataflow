# train.py (Corrected)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import deepspeed
import argparse
import time

from model import Llama3Model, ModelArgs

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DeepSpeed Llama3 Training with Full Logging")
parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for training')
parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
# === CHANGE: Add deepspeed_config argument ===
# DeepSpeed launcher will automatically provide this argument.
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# --- Model & Training Configuration ---
model_args = ModelArgs(
    dtype=torch.bfloat16,
    dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, vocab_size=128256,
    intermediate_size=14336, norm_eps=1e-5, max_seq_len=args.seq_len
)

epochs = 500
learning_rate = 1e-5

# --- DeepSpeed Configuration with Logging ---
# The ds_config can be loaded from a JSON file specified by --deepspeed_config
# For simplicity, we define it here.
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "wall_clock_breakdown": True,
    "optimizer": { "type": "AdamW", "params": { "lr": learning_rate, "betas": [0.9, 0.95] } },
    "fp16": { "enabled": True },
    "gradient_clipping": 1.0,
    "steps_per_print": 1,
    "activation_checkpointing": { "partition_activations": True, "cpu_checkpointing": True },

    # === ADDED: A safe baseline configuration for initialization ===
    # This will be used to successfully load the model before tuning begins.
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },

    
    
    # --- AUTOTUNING CONFIG (No Changes Here) ---
    # The tuner will use this section to experiment *after* initializing
    # successfully with the baseline config above.
    "autotuning": {
        "enabled": True,
        "fast": False, 
        "results_dir": "autotuning_logs",
        "log_level": "info",
        "zero_config": {
            "stage": [2, 3], # You can even ask it to test multiple stages
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            }
        },
    },

    "flops_profiler": {
        "enabled": True,
        "module_depth": -1,
        "detailed": True,
        "output_file": "flops_profile.txt"
    },
    
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

# === CHANGE: Pass the dataset to training_data ===
model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
    args=args, # Pass the full args object
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    training_data=dummy_dataset # Pass the Dataset here
)

# print("Compiling model...")
# # Note: torch.compile may have interactions with DeepSpeed's optimizations.
# # If you encounter issues, you might need to test without it first.
# model_engine = torch.compile(model_engine)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

# --- Training Loop with Throughput Calculation ---
print(f"Starting training with sequence length: {args.seq_len}...")
start_time = time.time()
total_tokens = 0

for epoch in range(epochs):
    # The training_dataloader is now the one created by DeepSpeed
    for i, (inputs, labels) in enumerate(training_dataloader):
        inputs = inputs.to(model_engine.device)
        labels = labels.to(model_engine.device)
        
        outputs = model_engine(inputs)
        loss = criterion(outputs.view(-1, model_args.vocab_size), labels.view(-1))

        model_engine.backward(loss)
        model_engine.step()

        actual_bs = model_engine.train_micro_batch_size_per_gpu()
        total_tokens += actual_bs * args.seq_len
        
        if (i + 1) % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                tokens_per_sec = total_tokens / elapsed_time
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"Epoch: {epoch+1}, Step: {i+1}, BS: {actual_bs}, Loss: {loss.item():.4f} | "
                    f"Tok/sec: {tokens_per_sec:.2f} | VRAM: {mem_alloc:.2f}GB"
                )
            start_time = time.time()
            total_tokens = 0

print("\nTraining complete! ✅")