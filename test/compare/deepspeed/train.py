# train.py (Corrected)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import deepspeed
import argparse
import time

import psutil
import os

import ctypes

import json

from model import Model, ModelArgs

_cudart = ctypes.CDLL('libcudart.so')

process = psutil.Process(os.getpid())
# Initialize peak host memory usage tracker
peak_host_mem_gb = 0.0

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DeepSpeed Training")
parser.add_argument('--zero_stage', type=int, default=None)
parser.add_argument('--save_act_layer_frac', type=float, default=0, help="Fraction of layer activatons to leave on device, deafult is 0 (full layer-wise checkpointing)")
parser.add_argument('--model_config', type=str, required=True)
parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for training')
parser.add_argument('--seqs_per_batch', type=int, default=1)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=3)
parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
# === CHANGE: Add deepspeed_config argument ===
# DeepSpeed launcher will automatically provide this argument.
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

grad_accum_steps = args.grad_accum_steps
seqs_per_batch = args.seqs_per_batch
num_steps = args.num_steps
zero_stage = args.zero_stage
save_act_layer_frac = args.save_act_layer_frac


global_steps = grad_accum_steps * num_steps

def load_model_args_from_json(json_path: str) -> ModelArgs:
    """Load ModelArgs from a JSON file."""
    
    DTYPE_MAP = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "none": None
    }

    with open(json_path, 'r') as f:
        config = json.load(f)

    # Convert dtype strings to torch.dtype
    for key in ['embed_dtype', 'attn_dtype', 'expert_dtype', 'head_dtype']:
        if key in config and config[key] in DTYPE_MAP:
            dtype_value = DTYPE_MAP[config[key]]
            if dtype_value != "none":
                config[key] = dtype_value

    # router_dtype stays as string "none"

    return ModelArgs(**config)

model_args = load_model_args_from_json(args.model_config)

SEED = 42

torch.manual_seed(SEED)

# --- Model & Training Configuration ---

epochs = 1
learning_rate = 2e-5

# --- DeepSpeed Configuration with Logging ---
# The ds_config can be loaded from a JSON file specified by --deepspeed_config
# For simplicity, we define it here.
ds_config = {
    "train_micro_batch_size_per_gpu": seqs_per_batch,
    "gradient_accumulation_steps": grad_accum_steps,
    "wall_clock_breakdown": True,
    "optimizer": { "type": "AdamW", "params": { "lr": learning_rate, "betas": [0.9, 0.999], "weight_decay": 1e-4, "eps": 1e-8}},
    "bf16": { "enabled": True},
    "steps_per_print": 1,
    ## doesnt do anything for single-GPU...
    #"activation_checkpointing": { "cpu_checkpointing": True, "partition_activations": True },
}

if zero_stage and zero_stage != 0:
    ds_config['optimizer']['fp32_optimizer_states'] = False
    if zero_stage == 1:
        ds_config['zero_optimization'] = {"stage": 1, "offload_optimizer": {"device": "cpu", "pin_memory": True}}
    elif zero_stage == 2:
        ds_config['zero_optimization'] = {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True}}
    elif zero_stage == 3:
        ds_config['zero_optimization'] = {"stage": 3, "offload_optimizer": {"device": "cpu", "pin_memory": True}, "offload_param": {"device": "cpu", "pin_memory": True}}
    else:
        print(f"Error. Zero Stage must be None, 1, 2, or 3...")
        exit(1)



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
model = Model(model_args)

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

current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
# Update peak if current is higher
peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)



torch.cuda.empty_cache()

# --- Training Loop with Throughput Calculation ---
print(f"Starting training with sequence length: {args.seq_len}...")

ret = _cudart.cudaProfilerStart()

start_time = time.time()
total_tokens = 0

num_steps = 0

step_throughputs = []

for epoch in range(epochs):
    # The training_dataloader is now the one created by DeepSpeed
    for i, (inputs, labels) in enumerate(training_dataloader):
        inputs = inputs.to(model_engine.device)
        labels = labels.to(model_engine.device)
        
        loss = model_engine(inputs, labels, save_act_layer_frac=save_act_layer_frac)
        #loss = criterion(outputs.view(-1, model_args.vocab_size), labels.view(-1))

        current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
        # Update peak if current is higher
        peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)

        model_engine.backward(loss)

        current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
        # Update peak if current is higher
        peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)

        model_engine.step()

        current_host_mem_gb = process.memory_info().rss / (1024 ** 3)
        # Update peak if current is higher
        peak_host_mem_gb = max(peak_host_mem_gb, current_host_mem_gb)

        num_steps += 1

        actual_bs = model_engine.train_micro_batch_size_per_gpu()
        total_tokens += actual_bs * args.seq_len
        
        if (i + 1) % grad_accum_steps == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                tokens_per_sec = total_tokens / elapsed_time
                step_throughputs.append(tokens_per_sec)
                print(
                    f"\n\nEpoch: {epoch+1}, Step: {i+1}, BS: {actual_bs}, Loss: {loss.item():.4f} | "
                    f"Step total_tokens: {total_tokens}, Step total_time = {elapsed_time}, Tok/sec: {tokens_per_sec:.2f}\n\n"
                )
                start_time = time.time()
                total_tokens = 0


        if num_steps == global_steps:
            break

ret = _cudart.cudaProfilerStop()

peak_mem_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

print(f"\n\n\nTraining complete! ✅\n\tThroughput: {step_throughputs[-1]} Tok/sec\n\tPeak Host Memory Reserved: {peak_host_mem_gb:.2f} GB\n\tPeak Device Memory Reserved: {peak_mem_reserved_gb:.2f} GB\n\n\n")
