import numpy as np
import torch
import sys

if len(sys.argv) != 5:
    print("Error. Usage: python show_test_layer_bwd.py <num_layers> <model_dim> <seq_id> <chunk_id>")
    sys.exit(1)

num_layers = int(sys.argv[1])
model_dim = int(sys.argv[2])
seq_id = int(sys.argv[3])
chunk_id = int(sys.argv[4])


for layer_id in range(num_layers - 1, -1, -1):

    layer_inp_path = f"test_transformer_data/layers_bwd/{layer_id}/seq_{seq_id}_chunk_{chunk_id}_x_upstream_grad.dat"
    layer_out_path = f"test_transformer_data/layers_bwd/{layer_id}/seq_{seq_id}_chunk_{chunk_id}_x_next_grad_stream.dat"


    np_inp = np.fromfile(layer_inp_path, dtype=np.uint16).reshape(-1, model_dim)

    np_out = np.fromfile(layer_out_path, dtype=np.uint16).reshape(-1, model_dim)


    t_inp = torch.from_numpy(np_inp).view(torch.bfloat16)

    t_out = torch.from_numpy(np_out).view(torch.bfloat16)

    c_out = torch.load(f"correct_transformer_data/layers_bwd/{layer_id}/block_grad_stream_out.pt")

    diffs = torch.abs(t_out - c_out)

    max_diff = torch.max(diffs)

    avg_diff = torch.mean(diffs)

    median_diff = torch.median(diffs)

    c_min, c_max = torch.min(c_out), torch.max(c_out)
    t_min, t_max = torch.min(t_out), torch.max(t_out)

    print(f"LAYER ID: {layer_id}\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}\n\n\tPyTorch Range: [{c_min}, {c_max}]\n\tMy Range: [{t_min}, {t_max}]\n\n")



