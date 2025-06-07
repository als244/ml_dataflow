import numpy as np
import torch
import sys

if len(sys.argv) != 5:
    print("Error. Usage: python show_test_layer_inp_out.py <model_dim> <vocab_size> <seq_id> <chunk_id>")
    sys.exit(1)

model_dim = int(sys.argv[1])
vocab_size = int(sys.argv[2])
seq_id = int(sys.argv[3])
chunk_id = int(sys.argv[4])

logits_loss_path = f"test_transformer_data/head_bwd/seq_{seq_id}_chunk_{chunk_id}_x_logits_loss.dat"
head_proj_path = f"test_transformer_data/head_bwd/seq_{seq_id}_chunk_{chunk_id}_x_head_proj_inp.dat"
norm_inp_path = f"test_transformer_data/head_bwd/seq_{seq_id}_chunk_{chunk_id}_x_head_norm_inp.dat"


np_logits_loss = np.fromfile(logits_loss_path, dtype=np.uint16).reshape(-1, model_dim)
np_head_proj_inp = np.fromfile(head_proj_path, dtype=np.uint16).reshape(-1, model_dim)
np_norm_inp = np.fromfile(norm_inp_path, dtype=np.uint16).reshape(-1, model_dim)


t_logits_loss = torch.from_numpy(np_logits_loss).view(torch.bfloat16)
t_head_proj_inp = torch.from_numpy(np_head_proj_inp).view(torch.bfloat16)
t_norm_inp = torch.from_numpy(np_norm_inp).view(torch.bfloat16)

print("HEAD BWD")
print("-----LOGITS LOSS-----")
print(t_logits_loss)
print("\n")
print("-----GRAD wrt. HEAD PROJ INP-----")
print(t_head_proj_inp)
print("\n")
print("-----GRAD STREAM OUTPUT-----")
print(t_norm_inp)
print("\n")


