import torch
import sys

if len(sys.argv) < 3:
    print("Error. Usage: python show_correct_layer_fwd.py <data path> <layer_id>")
    sys.exit(1)

data_path = sys.argv[1]
layer_id = int(sys.argv[2])

layer_inp_path = f"{data_path}/layers_fwd/{layer_id}/block_inp.pt"
layer_out_path = f"{data_path}/layers_fwd/{layer_id}/block_out.pt"

t_inp = torch.load(layer_inp_path)

t_out = torch.load(layer_out_path)


print(f"LAYER ID {layer_id}")
print("-----INPUT-----")
print(t_inp)
print("\n")
print("-----OUTPUT-----")
print(t_out)
print("\n")



