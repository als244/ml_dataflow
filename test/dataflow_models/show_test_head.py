import numpy as np
import torch
import sys

if len(sys.argv) != 4:
    print("Error. Usage: python show_test_layer_inp_out.py <num_tokens> <model_dim> <vocab_size>")
    sys.exit(1)

num_tokens = int(sys.argv[1])
model_dim = int(sys.argv[2])
vocab_size = int(sys.argv[3])

head_inp_path = f"test_transformer_data/head_fwd/x_inp.dat"
head_norm_path = f"test_transformer_data/head_fwd/x_norm.dat"
head_out_path = f"test_transformer_data/head_fwd/x_head_out.dat"
logits_path = f"test_transformer_data/head_fwd/x_logits.dat"


np_inp = np.fromfile(head_inp_path, dtype=np.uint16).reshape(num_tokens, model_dim)
np_norm = np.fromfile(head_norm_path, dtype=np.uint16).reshape(num_tokens, model_dim)
np_head_out = np.fromfile(head_out_path, dtype=np.uint16).reshape(num_tokens, vocab_size)
np_logits = np.fromfile(logits_path, dtype=np.uint16).reshape(num_tokens, vocab_size)


t_inp = torch.from_numpy(np_inp).view(torch.bfloat16)
t_norm = torch.from_numpy(np_norm).view(torch.bfloat16)
t_head_out = torch.from_numpy(np_head_out).view(torch.bfloat16)
t_logits = torch.from_numpy(np_logits).view(torch.bfloat16)

print("HEAD")
print("-----INPUT-----")
print(t_inp)
print("\n")
print("-----NORM-----")
print(t_norm)
print("\n")
print("-----HEAD OUT-----")
print(t_head_out)
print("\n")
print("-----LOGITS-----")
print(t_logits)
print("\n")



