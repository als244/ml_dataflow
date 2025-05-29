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

head_inp_path = f"test_transformer_data/head_fwd/seq_{seq_id}_chunk_{chunk_id}_x_inp.dat"
head_norm_path = f"test_transformer_data/head_fwd/seq_{seq_id}_chunk_{chunk_id}_x_norm.dat"
head_out_path = f"test_transformer_data/head_fwd/seq_{seq_id}_chunk_{chunk_id}_x_head_out.dat"
logits_path = f"test_transformer_data/head_fwd/seq_{seq_id}_chunk_{chunk_id}_x_logits.dat"


np_inp = np.fromfile(head_inp_path, dtype=np.uint16).reshape(-1, model_dim)
np_norm = np.fromfile(head_norm_path, dtype=np.uint16).reshape(-1, model_dim)
np_head_out = np.fromfile(head_out_path, dtype=np.uint16).reshape(-1, vocab_size)
np_logits = np.fromfile(logits_path, dtype=np.uint16).reshape(-1, vocab_size)

num_tokens = np_logits.shape[0]

t_inp = torch.from_numpy(np_inp).view(torch.bfloat16)
t_norm = torch.from_numpy(np_norm).view(torch.bfloat16)
t_head_out = torch.from_numpy(np_head_out).view(torch.bfloat16)
t_logits = torch.from_numpy(np_logits).view(torch.bfloat16)

t_max_token_probs, t_max_token_inds = torch.max(t_logits, dim=1)

orig_tokens_path = f"test_transformer_data/chunk_{chunk_id}_token_ids_uint32.dat"
labels_path = f"test_transformer_data/chunk_{chunk_id}_labels_uint32.dat"

np_orig_tokens = np.fromfile(orig_tokens_path, dtype=np.uint32)
np_labels = np.fromfile(labels_path, dtype=np.uint32)

t_orig_tokens = torch.from_numpy(np_orig_tokens)
t_labels = torch.from_numpy(np_labels)

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
print("-----ORIGINAL TOKENS-----")
print(t_orig_tokens)
print("\n")
print("-----PREDICTIONS-----")
print("Max Next Token Probs:")
print(t_max_token_probs)
print("\n")
print("Next Token IDs:")
print(t_max_token_inds)
print("\n")
print("----TRUE NEXT TOKENS-----")
print(t_labels)
print("\n")


