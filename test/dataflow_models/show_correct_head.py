import numpy as np
import torch
import sys


head_inp_path = f"correct_transformer_data/head_fwd/head_inp.pt"
head_norm_path = f"correct_transformer_data/head_fwd/head_norm.pt"
head_out_path = f"correct_transformer_data/head_fwd/head_out.pt"


t_inp = torch.load(head_inp_path)
t_norm = torch.load(head_norm_path)
t_head_out = torch.load(head_out_path)
# t_logits = torch.from_numpy(np_logits).view(torch.bfloat16)

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

"""
print("-----LOGITS-----")
print(t_logits)
print("\n")
"""


