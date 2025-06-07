import numpy as np
import torch
import sys


head_inp_path = f"correct_transformer_data/head_fwd/head_inp.pt"
head_norm_path = f"correct_transformer_data/head_fwd/head_norm.pt"
head_out_path = f"correct_transformer_data/head_fwd/head_out.pt"
logits_path = f"correct_transformer_data/head_fwd/logits.pt"

t_inp = torch.load(head_inp_path).squeeze(0)
t_norm = torch.load(head_norm_path).squeeze(0)
t_head_out = torch.load(head_out_path).squeeze(0)
t_logits = torch.load(logits_path).squeeze(0)

t_max_token_probs, t_max_token_inds = torch.max(t_logits, dim=1)

orig_tokens_path = f"test_transformer_data/token_ids_uint32.dat"
labels_path = f"test_transformer_data/labels_uint32.dat"

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



