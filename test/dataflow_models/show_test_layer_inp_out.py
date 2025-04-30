import numpy as np
import torch
import sys

if len(sys.argv) != 4:
    print("Error. Usage: python show_test_layer_inp_out.py <layer_id> <num_tokens> <model_dim>")
    sys.exit(1)

layer_id = int(sys.argv[1])
num_tokens = int(sys.argv[2])
model_dim = int(sys.argv[3])

layer_inp_path = f"test_transformer_data/layers_fwd/{layer_id}/x_act_stream.dat"
layer_out_path = f"test_transformer_data/layers_fwd/{layer_id}/x_act_stream_out.dat"


np_inp = np.fromfile(layer_inp_path, dtype=np.uint16).reshape(num_tokens, model_dim)

np_out = np.fromfile(layer_out_path, dtype=np.uint16).reshape(num_tokens, model_dim)


t_inp = torch.from_numpy(np_inp).view(torch.bfloat16)

t_out = torch.from_numpy(np_out).view(torch.bfloat16)


print("LAYER ID {layer_id}")
print("-----INPUT-----")
print(t_inp)
print("\n")
print("-----OUTPUT-----")
print(t_out)
print("\n")



