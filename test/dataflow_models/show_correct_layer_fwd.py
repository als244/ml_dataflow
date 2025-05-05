import torch
import sys

if len(sys.argv) < 2:
    print("Error. Usage: python show_correct_layer_fwd.py <layer_id>")
    sys.exit(1)

layer_id = int(sys.argv[1])

layer_inp_path = f"correct_transformer_data/layers_fwd/{layer_id}/block_inp.pt"
layer_out_path = f"correct_transformer_data/layers_fwd/{layer_id}/block_out.pt"

t_inp = torch.load(layer_inp_path)

t_out = torch.load(layer_out_path)


print("LAYER ID {layer_id}")
print("-----INPUT-----")
print(t_inp)
print("\n")
print("-----OUTPUT-----")
print(t_out)
print("\n")



