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


BWD_SAVE_DIR = "../correct_transformer_data"

TO_SAVE_PREDS = False
TO_SAVE_OPT_STATE = False
TO_SAVE_PARAM_GRADS = False



SEED = 0

## python convert_llama_weights_to_hf.py --input_dir /mnt/storage/models/llama3/meta_checkpoints/8B_inst --model_size 8B --output_dir . --llama_version 3.1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MAX_SEQ_LEN = 4096
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

token_id_file = "8192_token_ids_uint32.dat"
token_labels_file = "8192_labels_uint32.dat"
np_inp_tokens = np.fromfile(token_id_file, dtype=np.uint32)[:MAX_SEQ_LEN]
np_labels = np.fromfile(token_labels_file, dtype=np.uint32)[:MAX_SEQ_LEN]

## expects [bsz, seqlen] as input....



token_ids = torch.from_numpy(np_inp_tokens).long().to(device).unsqueeze(0)
labels = torch.from_numpy(np_labels).long().to(device).unsqueeze(0)


model.to(device)

 # CrossEntropyLoss expects preds: (N, C) or (N, C, ...), labels: (N) or (N, ...)
# N = Batch * SeqLen, C = Vocab Size
criterion = torch.nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(), # Pass all model parameters to the optimizer
    lr=1e-4,
    betas=(0.9, 0.999), # (beta1, beta2)
    eps=1e-8,           # epsilon
    weight_decay=1e-3
)


n_repeats = 10

for i in range(1, n_repeats + 1):


    ## FORWARD PASS
    #print(f"[Step {i}] Starting forward pass...")
    start_fwd = time.time_ns()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Ensure model and inputs are on the same device
        predictions = model.forward(token_ids) # Shape: (bsz, seqlen, vocab_size)
    stop_fwd = time.time_ns()
    time_fwd_ms = (stop_fwd - start_fwd) / 1e6

    if (TO_SAVE_PREDS):
        print(f"[Step {i}] Saving predictions...")
        predictions_path = f"{BWD_SAVE_DIR}/head_fwd/step_{i}_predictions.pt"
        torch.save(predictions, predictions_path)



    ## LOSS CALCULATION
    loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
    print(f"[Step {i}] Loss calculated: {loss.item()}\n") # Use .item() to get scalar value


    ## BACKWARD PASS


    # --- Backward Pass (Triggers Hooks) ---
    start_bwd = time.time_ns()

    #print(f"[Step {i}] Backward pass...")
    
    optimizer.zero_grad()
    loss.backward()

    stop_bwd = time.time_ns()
    time_bwd_ms = (stop_bwd - start_bwd) / 1e6

    if (TO_SAVE_PARAM_GRADS):
        print(f"[Step {i}] Saving model parameter gradients before optimizer step...")
        model_gradients_before_opt = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                model_gradients_before_opt[name] = param.grad.detach().cpu().clone()
            else:
                print(f"Parameter {name} has no gradient.")


        param_grads_dir = os.path.join(BWD_SAVE_DIR, "model_grads")
        os.makedirs(param_grads_dir, exist_ok=True)
        param_grads_save_path = os.path.join(param_grads_dir, f"model_parameter_gradients_before_step_{i}.pt")
        torch.save(model_gradients_before_opt, param_grads_save_path)
        print(f"Saved model parameter gradients to {param_grads_save_path}")


    # --- 2. Initialize and Run Optimizer ---
    #print(f"[Step {i}] Running optimizer...")
    optimizer.step() # Apply gradients to model parameters, updating them

    # --- 3. Save Optimizer State Values and Updated Parameters (After Optimizer Step) ---
    if (TO_SAVE_OPT_STATE):
        print(f"[Step {i}] Saving optimizer state values and updated model parameters after step...")
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


    predictions = predictions.detach()
    del predictions
    torch.cuda.empty_cache()


