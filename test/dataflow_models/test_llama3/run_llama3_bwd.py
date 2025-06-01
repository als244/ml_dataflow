import torch
import torch.nn as nn
import glob
import json

from llama3_tokenizer import Tokenizer
from llama3_model import ModelArgs, Transformer, TransformerBlock, Attention, FeedForward

import torch.optim as optim

import time
import functools
import numpy as np
import random
import os



##

BWD_SAVE_DIR = "../correct_transformer_data"



SEED = 0

## python convert_llama_weights_to_hf.py --input_dir /mnt/storage/models/llama3/meta_checkpoints/8B_inst --model_size 8B --output_dir . --llama_version 3.1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MAX_SEQ_LEN = 2048
MAX_BATCH_SIZE = 1

MODEL_PATH = "./models/1B_inst/"

with open(MODEL_PATH + "params.json", "r") as f:
	model_args = json.loads(f.read())

params = ModelArgs(max_seq_len=MAX_SEQ_LEN, max_batch_size=MAX_BATCH_SIZE, **model_args)

llama_tokenizer = Tokenizer(MODEL_PATH + "tokenizer.model")

checkpoint_paths = sorted(glob.glob(MODEL_PATH + "*.pth"))
checkpoints = [torch.load(x, map_location="cpu") for x in checkpoint_paths]

assert params.vocab_size == llama_tokenizer.n_words

torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)

print(f"Initializing Model (on host)...\n\tVocab Size: {params.vocab_size}\n\tModel Dim: {params.dim}\n\t# Layers: {params.n_layers}\n\t# Heads: {params.n_heads}\n\t# KV Heads: {params.n_kv_heads}\n\tMax Batch Size: {params.max_batch_size}\n\tMax Seq Len: {params.max_seq_len}\n\n\n")

start = time.time_ns()

model = Transformer(params)

for c in checkpoints:
	for name, tensor in c.items():
		c[name] = tensor
	model.load_state_dict(c, strict=False)

stop = time.time_ns()

time_ms = (stop - start) / 1e6

print(f"Finished Initialized Model!\n\tRuntime: {time_ms} ms\n")


device = torch.device("cuda:0")

token_id_file = "2048_token_ids_uint32.dat"
token_labels_file = "2048_labels_uint32.dat"
np_inp_tokens = np.fromfile(token_id_file, dtype=np.uint32)
np_labels = np.fromfile(token_labels_file, dtype=np.uint32)

## expects [bsz, seqlen] as input....
token_ids = torch.from_numpy(np_inp_tokens).long().to(device).unsqueeze(0)
labels = torch.from_numpy(np_labels).long().to(device).unsqueeze(0)


model.to(device)






def save_gradient_hook(module, grad_input, grad_output, layer_id, module_name):
    """
    Unified hook to save gradient streams for TransformerBlock, Attention, or FeedForward.

    Args:
        module (nn.Module): The module instance (TransformerBlock, Attention, or FeedForward).
        grad_input (tuple): Tuple of gradients w.r.t module inputs.
        grad_output (tuple): Tuple of gradients w.r.t module outputs.
        layer_id (int): The ID of the parent TransformerBlock.
        module_name (str): Name identifier for the module ('block', 'attention', 'feed_forward').
    """
    # Construct the specific directory path
    # e.g., /path/to/layers_bwd/0/attention/
    # e.g., /path/to/layers_bwd/5/feed_forward/
    # e.g., /path/to/layers_bwd/10/block/  (for the whole TransformerBlock)
    module_bwd_dir = f"{BWD_SAVE_DIR}/layers_bwd/{layer_id}"
    os.makedirs(module_bwd_dir, exist_ok=True)

    # --- Save Gradient Stream INTO the module (dLoss/dOutput) ---
    # grad_output[0] contains the upstream gradient
    if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
        inp_stream_path = f"{module_bwd_dir}/{module_name}_grad_stream_inp.pt"
        print(f"  Saving grad stream input for layer {layer_id} [{module_name}] to {inp_stream_path} (Shape: {grad_output[0].shape}, Dtype: {grad_output[0].dtype})")
        try:
            torch.save(grad_output[0].detach().cpu(), inp_stream_path)
        except Exception as e:
            print(f"  ERROR saving {inp_stream_path}: {e}")
    else:
        # This might happen for the very first input tensor if it doesn't require grad, but less likely for module outputs.
        print(f"  WARNING: grad_output[0] is None or invalid for layer {layer_id} [{module_name}]")


    # --- Save Gradient Stream OUT FROM the module (dLoss/dInput) ---
    # grad_input[0] contains the gradient passed downstream (to the previous layer/module)
    # Note: Check indices if forward methods have multiple tensor inputs requiring grad.
    # For Attention and FeedForward, the first input is usually the one we care about.
    # For TransformerBlock, grad_input[0] is dLoss/dx (output stream for the whole block).
    if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
        out_stream_path = f"{module_bwd_dir}/{module_name}_grad_stream_out.pt"
        print(f"  Saving grad stream output for layer {layer_id} [{module_name}] to {out_stream_path} (Shape: {grad_input[0].shape}, Dtype: {grad_input[0].dtype})")
        try:
            torch.save(grad_input[0].detach().cpu(), out_stream_path)
        except Exception as e:
            print(f"  ERROR saving {out_stream_path}: {e}")
    else:
         # grad_input might be None for inputs that don't require gradients (like masks, freqs_cis)
         # but grad_input[0] corresponding to the main data tensor 'x' should exist.
        print(f"  WARNING: grad_input[0] is None or invalid for layer {layer_id} [{module_name}]")

def save_specific_gradient_hook(module, grad_input, grad_output, identifier):
    """
    Hook specifically designed to capture grad_input[0] (gradient w.r.t. input activation).
    """
    print(f"Hook triggered for identifier: {identifier}")
    if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
        grad = grad_input[0].detach().cpu()
        print(f"  Capturing gradient w.r.t. input for '{identifier}'. Shape: {grad.shape}, Dtype: {grad.dtype}")
        captured_gradients[identifier] = grad
        # Optional: Save directly here if preferred
        # grad_path = f"{BWD_SAVE_DIR}/{identifier}_grad_input.pt"
        # os.makedirs(os.path.dirname(grad_path), exist_ok=True)
        # try:
        #     torch.save(grad, grad_path)
        #     print(f"  Saved gradient to {grad_path}")
        # except Exception as e:
        #     print(f"  ERROR saving gradient for {identifier}: {e}")
    else:
        print(f"  WARNING: grad_input[0] is None or invalid for '{identifier}'")










# --- Register Hooks BEFORE Forward/Backward Pass ---
print("Registering backward hooks...")
hook_handles = []
total_hooks = 0
for i, layer in enumerate(model.layers):
    if isinstance(layer, TransformerBlock): # Ensure it's the correct type
        print(f"Registering hooks for Layer {i}:")
        # 1. Hook for the entire TransformerBlock
        block_hook = functools.partial(save_gradient_hook, layer_id=i, module_name="block")
        handle = layer.register_full_backward_hook(block_hook)
        hook_handles.append(handle)
        total_hooks += 1
        print(f"  Registered hook for TransformerBlock {i}")

        # 2. Hook for the Attention submodule
        if hasattr(layer, 'attention') and isinstance(layer.attention, nn.Module):
            attn_hook = functools.partial(save_gradient_hook, layer_id=i, module_name="attention")
            handle = layer.attention.register_full_backward_hook(attn_hook)
            hook_handles.append(handle)
            total_hooks += 1
            print(f"  Registered hook for Attention in layer {i}")
        else:
            print(f"  WARNING: Layer {i} has no 'attention' nn.Module attribute.")

        # 3. Hook for the FeedForward submodule
        if hasattr(layer, 'feed_forward') and isinstance(layer.feed_forward, nn.Module):
            ff_hook = functools.partial(save_gradient_hook, layer_id=i, module_name="feed_forward")
            handle = layer.feed_forward.register_full_backward_hook(ff_hook)
            hook_handles.append(handle)
            total_hooks += 1
            print(f"  Registered hook for FeedForward in layer {i}")
        else:
             print(f"  WARNING: Layer {i} has no 'feed_forward' nn.Module attribute.")

    else:
        print(f"Warning: Item at index {i} in model.layers is not a TransformerBlock, skipping hook registration.")
print(f"Total hooks registered: {total_hooks}")


captured_gradients = {}



if hasattr(model, 'output') and isinstance(model.output, nn.Module):
    print("Registering hook for final projection layer input gradient...")
    # Use functools.partial to pass the identifier
    head_input_hook = functools.partial(save_specific_gradient_hook, identifier="head_input")
    handle = model.output.register_full_backward_hook(head_input_hook)
    hook_handles.append(handle) # Add to your list to remove later
    print("  Registered hook on model.output")
else:
    print("WARNING: Could not find 'model.output' or it's not an nn.Module. Cannot register hook.")




## FORWARD PASS


print("Starting forward pass...")
start_fwd = time.time_ns()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Ensure model and inputs are on the same device
    predictions = model.forward(token_ids.to(device)) # Shape: (bsz, seqlen, vocab_size)
stop_fwd = time.time_ns()
time_fwd_ms = (stop_fwd - start_fwd) / 1e6
print(f"Forward pass completed in {time_fwd_ms:.2f} ms")
print(f"Predictions shape: {predictions.shape}")

predictions.retain_grad()

predictions_path = f"{BWD_SAVE_DIR}/head_fwd/predictions.pt"
torch.save(predictions, predictions_path)


## LOSS CALCULATION

# --- Loss Calculation ---
print("Calculating loss...")
# CrossEntropyLoss expects preds: (N, C) or (N, C, ...), labels: (N) or (N, ...)
# N = Batch * SeqLen, C = Vocab Size
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
print(f"Loss calculated: {loss.item()}") # Use .item() to get scalar value


## BACKWARD PASS


# --- Backward Pass (Triggers Hooks) ---
print("Starting backward pass (this will trigger gradient saving hooks)...")
start_bwd = time.time_ns()
model.zero_grad() # Zero out any previous gradients
loss.backward()
stop_bwd = time.time_ns()
time_bwd_ms = (stop_bwd - start_bwd) / 1e6
print(f"Backward pass completed in {time_bwd_ms:.2f} ms")


# --- 1. Save Model Parameter Gradients (Before Optimizer Step) ---
print("Saving model parameter gradients before optimizer step...")
model_gradients_before_opt = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        model_gradients_before_opt[name] = param.grad.detach().cpu().clone()
    # else:
    #     print(f"Parameter {name} has no gradient.")


param_grads_dir = os.path.join(BWD_SAVE_DIR, "model_grads")
os.makedirs(param_grads_dir, exist_ok=True)
param_grads_save_path = os.path.join(param_grads_dir, "model_parameter_gradients_before_step.pt")
torch.save(model_gradients_before_opt, param_grads_save_path)
print(f"Saved model parameter gradients to {param_grads_save_path}")

# --- 2. Initialize and Run Optimizer ---
print("Initializing Adam optimizer and performing optimization step...")

"""
optimizer = optim.Adam(
    model.parameters(), # Pass all model parameters to the optimizer
    lr=1e-4,
    betas=(0.9, 0.999), # (beta1, beta2)
    eps=1e-8,           # epsilon
    weight_decay=1e-3
)
"""

optimizer = optim.AdamW(
    model.parameters(), # Pass all model parameters to the optimizer
    lr=1e-4,
    betas=(0.9, 0.999), # (beta1, beta2)
    eps=1e-8,           # epsilon
    weight_decay=1e-3
)

optimizer.step() # Apply gradients to model parameters, updating them
print("Optimizer step performed.")

# --- 3. Save Optimizer State Values and Updated Parameters (After Optimizer Step) ---
print("Saving optimizer state values and updated model parameters after step...")
optimizer_states_after_step = {}
# The optimizer.state is a dict where keys are parameters and values are their state (e.g., exp_avg)
for param_group in optimizer.param_groups: # Iterate over parameter groups
    for p in param_group['params']:    # Iterate over parameters in each group
        if p in optimizer.state:       # Check if the parameter has state in the optimizer
            state = optimizer.state[p]
            # Find the name of the parameter p for saving
            param_name = None
            for name, param_val in model.named_parameters():
                if param_val is p:
                    param_name = name
                    break
            
            if param_name and state: # Ensure state is not empty and name was found
                # 'step' can be a tensor or an int/float. Adam typically stores it as a tensor or Python int.
                step_val = state['step']
                if isinstance(step_val, torch.Tensor):
                    step_val = step_val.cpu().item() # Convert to Python number if it's a scalar tensor
                
                optimizer_states_after_step[param_name] = {
                    'step': step_val,
                    'exp_avg': state['exp_avg'].detach().cpu().clone(),    # 1st moment estimate
                    'exp_avg_sq': state['exp_avg_sq'].detach().cpu().clone() # 2nd moment estimate
                }
            elif not state:
                 print(f"  Parameter {param_name or 'Unknown (check mapping)'} has no state in optimizer (e.g. no grad was computed).")
            elif not param_name:
                 print(f"  Warning: Optimizer state found for a parameter not in model.named_parameters() (should not happen).")


opt_states_dir = os.path.join(BWD_SAVE_DIR, "optimizer_states")
os.makedirs(opt_states_dir, exist_ok=True)
optimizer_states_save_path = os.path.join(opt_states_dir, "optimizer_states_after_step.pt")
torch.save(optimizer_states_after_step, optimizer_states_save_path)
print(f"Saved optimizer states to {optimizer_states_save_path}")

# Optionally, save the updated model parameters themselves
updated_params_save_path = os.path.join(opt_states_dir, "model_parameters_after_step.pt")
updated_params_dict = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
torch.save(updated_params_dict, updated_params_save_path)
print(f"Saved updated model parameters to {updated_params_save_path}")



## SAVE ADDITIONAL GRADIENTS

print("Saving additional gradients...")

head_input_gradient = captured_gradients.get("head_input")

head_input_gradient_path = f"{BWD_SAVE_DIR}/head_bwd/head_input_gradient.pt"
torch.save(head_input_gradient, head_input_gradient_path)


logits_grad = predictions.grad
print(f"  Logits gradient shape: {logits_grad.shape}, dtype: {logits_grad.dtype}")

logits_grad_path = f"{BWD_SAVE_DIR}/head_bwd/logits_grad.pt"
torch.save(logits_grad, logits_grad_path)


print("Removing hooks...")
for handle in hook_handles:
    handle.remove()

print("Gradient stream saving process complete.")
print(f"Check the directory '{BWD_SAVE_DIR}' for saved .pt files.")
