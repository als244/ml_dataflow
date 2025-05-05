import torch
import glob
import json

from llama3_tokenizer import Tokenizer
from llama3_model import ModelArgs, Transformer
import time

import numpy as np
import random
import os
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

token_id_file = "token_ids_uint32.dat"
token_labels_file = "labels_uint32.dat"
np_inp_tokens = np.fromfile(token_id_file, dtype=np.uint32)
np_labels = np.fromfile(token_labels_file, dtype=np.uint32)

## expects [bsz, seqlen] as input....
token_ids = torch.from_numpy(np_inp_tokens).long().to(device).unsqueeze(0)
labels = torch.from_numpy(np_labels).long().to(device).unsqueeze(0)


model.to(device)
model.eval()







##

BWD_SAVE_DIR = "/mnt/storage/research/ml_dataflow/correct_transformer_data"

def save_gradient_hook(module, grad_input, grad_output):
    """
    Hook to save gradient streams for a TransformerBlock.

    Args:
        module (nn.Module): The TransformerBlock instance.
        grad_input (tuple): Tuple of gradients w.r.t module inputs.
                            grad_input[0] corresponds to the gradient of 'x' (output stream).
        grad_output (tuple): Tuple of gradients w.r.t module outputs.
                             grad_output[0] corresponds to the gradient of 'out' (input stream).
    """
    layer_id = module.layer_id
    layer_bwd_dir = f"{BWD_SAVE_DIR}/layers_bwd/{layer_id}"
    os.makedirs(layer_bwd_dir, exist_ok=True)

    # --- Save Gradient Stream INTO the block (dLoss/dOutput) ---
    # grad_output is a tuple, we usually care about the first element if the forward returns one tensor.
    if grad_output is not None and grad_output[0] is not None:
        inp_stream_path = os.path.join(layer_bwd_dir, "grad_stream_inp.pt")
        print(f"Saving grad stream input for layer {layer_id} to {inp_stream_path} (Shape: {grad_output[0].shape}, Dtype: {grad_output[0].dtype})")
        torch.save(grad_output[0].detach().cpu(), inp_stream_path) # Detach and move to CPU to avoid GPU memory buildup if saving many layers
    else:
        print(f"WARNING: grad_output[0] is None for layer {layer_id}")


    # --- Save Gradient Stream OUT FROM the block (dLoss/dInput) ---
    # grad_input is a tuple corresponding to the inputs of the forward method (x, freqs_cis, mask).
    # We only care about the gradient w.r.t. 'x', which is grad_input[0].
    if grad_input is not None and grad_input[0] is not None:
        out_stream_path = os.path.join(layer_bwd_dir, "grad_stream_out.pt")
        print(f"Saving grad stream output for layer {layer_id} to {out_stream_path} (Shape: {grad_input[0].shape}, Dtype: {grad_input[0].dtype})")
        torch.save(grad_input[0].detach().cpu(), out_stream_path) # Detach and move to CPU
    else:
        print(f"WARNING: grad_input[0] is None for layer {layer_id}") # Should not happen in typical backprop




# --- Register Hooks BEFORE Forward/Backward Pass ---
print("Registering backward hooks...")
hook_handles = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, torch.nn.Module): # Ensure it's a module
        handle = layer.register_full_backward_hook(save_gradient_hook)
        hook_handles.append(handle)
        print(f"Registered hook for TransformerBlock layer {i}")
    else:
        print(f"Warning: Item at index {i} in model.layers is not an nn.Module, skipping hook registration.")
print(f"Total hooks registered: {len(hook_handles)}")



print("Starting forward pass...")
start_fwd = time.time_ns()
# Ensure model and inputs are on the same device
predictions = model.forward(token_ids.to(device)) # Shape: (bsz, seqlen, vocab_size)
stop_fwd = time.time_ns()
time_fwd_ms = (stop_fwd - start_fwd) / 1e6
print(f"Forward pass completed in {time_fwd_ms:.2f} ms")
print(f"Predictions shape: {predictions.shape}")

# --- Loss Calculation ---
print("Calculating loss...")
# CrossEntropyLoss expects preds: (N, C) or (N, C, ...), labels: (N) or (N, ...)
# N = Batch * SeqLen, C = Vocab Size
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
print(f"Loss calculated: {loss.item()}") # Use .item() to get scalar value


# --- Backward Pass (Triggers Hooks) ---
print("Starting backward pass (this will trigger gradient saving hooks)...")
start_bwd = time.time_ns()
model.zero_grad() # Zero out any previous gradients
loss.backward()
stop_bwd = time.time_ns()
time_bwd_ms = (stop_bwd - start_bwd) / 1e6
print(f"Backward pass completed in {time_bwd_ms:.2f} ms")



print("Removing hooks...")
for handle in hook_handles:
    handle.remove()

print("Gradient stream saving process complete.")
print(f"Check the directory '{BWD_SAVE_DIR}' for saved .pt files.")
