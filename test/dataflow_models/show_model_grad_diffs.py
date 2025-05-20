import numpy as np
import torch
import sys

def load_torch_bf16_from_binary(shape, path):
    np_arr = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return torch.from_numpy(np_arr).view(torch.bfloat16)

def get_diffs(my_tensor, correct_tensor):
    diffs = torch.abs(my_tensor - correct_tensor)
    max_diff = torch.max(diffs)
    avg_diff = torch.mean(diffs)
    median_diff = torch.median(diffs)
    return max_diff, avg_diff, median_diff


if len(sys.argv) != 2:
    print("Error. Usage: python show_model_grad_diffs.py <num_layers>")
    sys.exit(1)

num_layers = int(sys.argv[1])


correct_dir = "correct_transformer_data/model_grads"
test_dir = "test_transformer_data/model_grads"

correct_dw_dict = torch.load(f"{correct_dir}/model_parameter_gradients_before_step.pt")

c_dhead = correct_dw_dict["output.weight"]

vocab_size = c_dhead.shape[0]
model_dim = c_dhead.shape[1]

c_dhead_norm = correct_dw_dict["norm.weight"]




my_dhead = load_torch_bf16_from_binary((vocab_size, model_dim), f"{test_dir}/head/w_head.dat")
my_dhead_norm = load_torch_bf16_from_binary((model_dim,), f"{test_dir}/head/w_head_norm.dat")

print("-----HEAD-----\n")

max_diff, avg_diff, median_diff = get_diffs(my_dhead, c_dhead)

print(f"Head Projection:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}\n")

max_diff, avg_diff, median_diff = get_diffs(my_dhead_norm, c_dhead_norm)

print(f"Head Norm:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}\n\n")

print("-----BLOCKS-----\n")


## w2

for layer_id in range(num_layers - 1, -1, -1):

    print(f"LAYER ID: {layer_id}...\n")

    c_w2 = correct_dw_dict[f"layers.{layer_id}.feed_forward.w2.weight"]

    ffn_dim = c_w2.shape[1]

    my_w2 = load_torch_bf16_from_binary((model_dim, ffn_dim), f"{test_dir}/layers/{layer_id}/w_2.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w2, c_w2)

    print(f"W_2:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_w3 = correct_dw_dict[f"layers.{layer_id}.feed_forward.w3.weight"]

    my_w3 = load_torch_bf16_from_binary((ffn_dim, model_dim), f"{test_dir}/layers/{layer_id}/w_3.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w3, c_w3)

    print(f"W_3:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_w1 = correct_dw_dict[f"layers.{layer_id}.feed_forward.w1.weight"]

    my_w1 = load_torch_bf16_from_binary((ffn_dim, model_dim), f"{test_dir}/layers/{layer_id}/w_1.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w1, c_w1)

    print(f"W_1:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_ffn_norm = correct_dw_dict[f"layers.{layer_id}.ffn_norm.weight"]

    my_ffn_norm = load_torch_bf16_from_binary((model_dim,), f"{test_dir}/layers/{layer_id}/w_ffn_norm.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_ffn_norm, c_ffn_norm)

    print(f"FFN Norm:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_w_out = correct_dw_dict[f"layers.{layer_id}.attention.wo.weight"]

    my_w_out = load_torch_bf16_from_binary((model_dim, model_dim), f"{test_dir}/layers/{layer_id}/w_o.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w_out, c_w_out)

    print(f"W_o:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_w_v = correct_dw_dict[f"layers.{layer_id}.attention.wv.weight"]

    kv_dim = c_w_v.shape[0]

    my_w_v = load_torch_bf16_from_binary((kv_dim, model_dim), f"{test_dir}/layers/{layer_id}/w_v.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w_v, c_w_v)

    print(f"W_v:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_w_k = correct_dw_dict[f"layers.{layer_id}.attention.wk.weight"]

    my_w_k = load_torch_bf16_from_binary((kv_dim, model_dim), f"{test_dir}/layers/{layer_id}/w_k.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w_k, c_w_k)

    print(f"W_k:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_w_q = correct_dw_dict[f"layers.{layer_id}.attention.wq.weight"]

    my_w_q = load_torch_bf16_from_binary((model_dim, model_dim), f"{test_dir}/layers/{layer_id}/w_q.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_w_q, c_w_q)

    print(f"W_q:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}")

    c_attn_norm = correct_dw_dict[f"layers.{layer_id}.attention_norm.weight"]

    my_attn_norm = load_torch_bf16_from_binary((model_dim,), f"{test_dir}/layers/{layer_id}/w_attn_norm.dat")

    max_diff, avg_diff, median_diff = get_diffs(my_attn_norm, c_attn_norm)

    print(f"Attn Norm:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}\n")


print("-----EMBEDDING-----\n")

c_tok_emb = correct_dw_dict["tok_embeddings.weight"]

my_tok_emb = load_torch_bf16_from_binary((vocab_size, model_dim), f"{test_dir}/embedding/tok_embeddings.dat")

max_diff, avg_diff, median_diff = get_diffs(my_tok_emb, c_tok_emb)

print(f"Tok Emb:\n\tMax Diff: {max_diff}\n\tAvg. Diff: {avg_diff}\n\tMedian Diff: {median_diff}\n\n")