import torch
import glob
import json

from llama3_tokenizer import Tokenizer
from llama3_model import ModelArgs, Transformer
import time

import numpy as np
import random

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
token_labels_file = "token_labels_uint32.dat"
np_inp_tokens = np.fromfile(token_id_file, dtype=np.uint32)
np_labels = np.fromfile(token_labels_file, dtype=np.uint32)

token_ids = torch.from_numpy(np_inp_tokens).long().to(device)
labels = torch.from_numpy(np_labels).long().to(device)


predictions = model.forward(token_ids)






