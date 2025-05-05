import numpy as np
import torch
import sys

if len(sys.argv) != 5:
    print("Error. Usage: python show_test_layer_bwd.py <layer_id> <model_dim> <seq_id> <chunk_id>")
    sys.exit(1)

layer_id = int(sys.argv[1])
model_dim = int(sys.argv[2])
seq_id = int(sys.argv[3])
chunk_id = int(sys.argv[4])

layer_inp_path = f"test_transformer_data/layers_bwd/{layer_id}/seq_{seq_id}_chunk_{chunk_id}_x_upstream_grad.dat"
layer_out_path = f"test_transformer_data/layers_bwd/{layer_id}/seq_{seq_id}_chunk_{chunk_id}_x_next_grad_stream.dat"


np_inp = np.fromfile(layer_inp_path, dtype=np.uint16).reshape(-1, model_dim)

np_out = np.fromfile(layer_out_path, dtype=np.uint16).reshape(-1, model_dim)


t_inp = torch.from_numpy(np_inp).view(torch.bfloat16)

t_out = torch.from_numpy(np_out).view(torch.bfloat16)


print("LAYER ID {layer_id}")
print("-----INPUT-----")
print(t_inp)
print("\n")
print("-----OUTPUT-----")
print(t_out)
print("\n")



