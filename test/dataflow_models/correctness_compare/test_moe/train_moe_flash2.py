import torch
import torch.nn as nn
import glob
import json

from moe_model_flash2 import ModelArgs, MoETransformer, SeqlensInfo

import torch.optim as optim

import time
import numpy as np
import os
import pickle

BWD_SAVE_DIR = "../correct_transformer_data"

TO_SAVE_PREDS = False
TO_SAVE_OPT_STATE = False
TO_SAVE_PARAM_GRADS = False

MODEL_PATH = "./pytorch_test_moe"

TRAIN_SEQ_LEN = 8192

model_args = pickle.load(open(f"{MODEL_PATH}_config.pkl", "rb"))

model = MoETransformer(model_args)

model.load_state_dict(torch.load(f"{MODEL_PATH}.pt"))

device = torch.device("cuda:0")

token_id_file = "65536_token_ids_uint32.dat"
token_labels_file = "65536_labels_uint32.dat"
np_inp_tokens = np.fromfile(token_id_file, dtype=np.uint32)[:TRAIN_SEQ_LEN]
np_labels = np.fromfile(token_labels_file, dtype=np.uint32)[:TRAIN_SEQ_LEN]

seqlens_q_np = np.array([TRAIN_SEQ_LEN], dtype=np.int32)
seqlens_k_np = np.array([TRAIN_SEQ_LEN], dtype=np.int32)

device = torch.device("cuda:0")

seqlens_info = SeqlensInfo(seqlens_q_np, seqlens_k_np, device)
## expects [bsz, seqlen] as input....



token_ids = torch.from_numpy(np_inp_tokens).long().to(device).unsqueeze(0)
labels = torch.from_numpy(np_labels).long().to(device).unsqueeze(0)


model.to(device)

 # CrossEntropyLoss expects preds: (N, C) or (N, C, ...), labels: (N) or (N, ...)
# N = Batch * SeqLen, C = Vocab Size
criterion = torch.nn.CrossEntropyLoss()


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
    lr=2e-5,
    betas=(0.9, 0.999), # (beta1, beta2)
    eps=1e-8,           # epsilon
    weight_decay=1e-5
)


n_repeats = 500

for i in range(1, n_repeats + 1):


    ## FORWARD PASS
    #print(f"[Step {i}] Starting forward pass...")
    start_fwd = time.time_ns()
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #     # Ensure model and inputs are on the same device
    #     predictions = model.forward(token_ids, seqlens_info, i) # Shape: (bsz, seqlen, vocab_size)
    predictions = model.forward(token_ids, seqlens_info, i) # Shape: (bsz, seqlen, vocab_size)
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
    if TO_SAVE_OPT_STATE:
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
        optimizer_states_save_path = os.path.join(opt_states_dir, f"optimizer_states_after_step_{i}.pt")
        torch.save(optimizer_states_after_step, optimizer_states_save_path)
        print(f"Saved optimizer states to {optimizer_states_save_path}")

        # Optionally, save the updated model parameters themselves
        updated_params_save_path = os.path.join(opt_states_dir, f"model_parameters_after_step_{i}.pt")
        updated_params_dict = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        torch.save(updated_params_dict, updated_params_save_path)
        print(f"Saved updated model parameters to {updated_params_save_path}")


    predictions = predictions.detach()
    del predictions
    torch.cuda.empty_cache()
         


