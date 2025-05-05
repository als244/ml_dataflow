import torch
import sys

if len(sys.argv) < 2:
    print("Error. Usage: python show_correct_layer_bwd.py <layer_id>")
    sys.exit(1)

layer_id = int(sys.argv[1])

layer_inp_path = f"correct_transformer_data/layers_bwd/{layer_id}/block_grad_stream_inp.pt"
layer_out_path = f"correct_transformer_data/layers_bwd/{layer_id}/block_grad_stream_out.pt"

t_inp = torch.load(layer_inp_path)

t_out = torch.load(layer_out_path)


print(f"LAYER ID: {layer_id}")
print("-----UPSTREAM GRADIENT (Block Input)-----")
print(t_inp)
print("\n")
print("-----NEXT GRAD STREAM (Block Output)-----")
print(t_out)
print("\n")



