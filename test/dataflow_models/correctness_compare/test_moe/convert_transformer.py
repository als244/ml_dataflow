import torch
import numpy as np
import os
import sys
import pickle
import gc

def get_view_dtype(torch_dtype):
    if torch_dtype == torch.bfloat16:
        return torch.uint16
    elif torch_dtype == torch.float16:
        return torch.uint16
    elif torch_dtype == torch.float32:
        return torch.float32
    else:
        return None

def convert_to_numpy(torch_tensor):
    view_dtype = get_view_dtype(torch_tensor.dtype)
    torch_tensor = torch_tensor.view(view_dtype).numpy()
    return torch_tensor

def combine_layer_weights(layer_weights):
    flattened_weights = [x.reshape(-1) for x in layer_weights]
    return np.concatenate(flattened_weights)

def save_weights(torch_tensor, output_path):
    view_dtype = get_view_dtype(torch_tensor.dtype)
    torch_tensor = torch_tensor.view(view_dtype).numpy()
    torch_tensor.tofile(output_path)

if (len(sys.argv) != 4):
    print("Usage: python convert_transformer.py <orig_model> <conversion_dict> <output_dir>")
    sys.exit(1)

orig_model_path = sys.argv[1]
conversion_dict_path = sys.argv[2]
output_dir = sys.argv[3]


orig_model = torch.load(orig_model_path, weights_only=False).cpu()

orig_model_dict = orig_model.state_dict()

with open(conversion_dict_path, "rb") as f:
    conversion_dict = pickle.load(f)


## format
## three high level keys: embedding, layers, head
## I. embedding 
# - has no extra structre, just path to weights

## II. layers 
# - has subdictionaries for each layer
## within each layer there is the following:
## 1. attn_norm
## 2. q_proj, k_proj, v_proj 
## 3. (optional) q_norm, k_norm
## 4. o_proj
## 5. ffn_norm
## 6. (optional) router
## 7. "shared_experts" and "routed_experts" which are both dictionaries
##   a.) w1, w3, w2

## III. head 
# - has keys: norm, out


os.makedirs(f"{output_dir}", exist_ok=True)

## Save the config
if "config" in conversion_dict:
    print("\nSaving config...\n")
    config_dict = conversion_dict["config"]

    config_text = f"Embed Dtype: {config_dict['embed_dtype']}\n"
    config_text += f"Attn Dtype: {config_dict['attn_dtype']}\n"
    config_text += f"Expert Dtype: {config_dict['expert_dtype']}\n"
    config_text += f"Head Dtype: {config_dict['head_dtype']}\n"
    config_text += f"Vocab Size: {config_dict['vocab_size']}\n"
    config_text += f"Num Layers: {config_dict['num_layers']}\n"
    config_text += f"Model Dim: {config_dict['model_dim']}\n"
    config_text += f"Num Q Heads: {config_dict['num_q_heads']}\n"
    config_text += f"Num KV Heads: {config_dict['num_kv_heads']}\n"
    config_text += f"QK Norm Type: {config_dict['qk_norm_type']}\n"
    config_text += f"QK Norm Weight Type: {config_dict['qk_norm_weight_type']}\n"
    config_text += f"Num Shared Experts: {config_dict['num_shared_experts']}\n"
    config_text += f"Num Routed Experts: {config_dict['num_routed_experts']}\n"
    config_text += f"Top K Routed Experts: {config_dict['top_k_routed_experts']}\n"
    config_text += f"Expert Dim: {config_dict['expert_dim']}\n"
    config_text += f"Expert MLP Type: {config_dict['expert_mlp_type']}\n"
    config_text += f"Rope Theta: {config_dict['rope_theta']}\n"
    config_text += f"RMS Norm Epsilon: {config_dict['rms_norm_epsilon']}\n"
    
    with open(f"{output_dir}/config.txt", "w") as f:
        f.write(config_text)


print("Converting weights...\n")

if "embedding" in conversion_dict:
    print("\tEmbedding...")
    os.makedirs(f"{output_dir}/embed", exist_ok=True)
    embedding_key = conversion_dict["embedding"]
    vocab_size, model_dim = orig_model_dict[embedding_key].shape
    save_weights(orig_model_dict[embedding_key], f"{output_dir}/embed/tok_embeddings.weight")
    

if "layers" in conversion_dict:
    print("\tLayers...")
    os.makedirs(f"{output_dir}/layers", exist_ok=True)
    for k in range(len(conversion_dict["layers"])):
        print(f"\t\tLayer {k}...")
        os.makedirs(f"{output_dir}/layers/{k}", exist_ok=True)
        layer_dict = conversion_dict["layers"][k]

        layer_weights = []

        if "attn_norm" in layer_dict and layer_dict["attn_norm"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["attn_norm"]]))

        if "q_proj" in layer_dict and layer_dict["q_proj"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["q_proj"]]))

        if "k_proj" in layer_dict and layer_dict["k_proj"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["k_proj"]]))

        if "v_proj" in layer_dict and layer_dict["v_proj"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["v_proj"]]))

        if "q_norm" in layer_dict and layer_dict["q_norm"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["q_norm"]]))

        if "k_norm" in layer_dict and layer_dict["k_norm"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["k_norm"]]))

        if "o_proj" in layer_dict and layer_dict["o_proj"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["o_proj"]]))

        if "ffn_norm" in layer_dict and layer_dict["ffn_norm"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["ffn_norm"]]))

        if "router" in layer_dict and layer_dict["router"] is not None:
            layer_weights.append(convert_to_numpy(orig_model_dict[layer_dict["router"]]))

        if "shared_experts" in layer_dict:
            
            for i in range(len(layer_dict["shared_experts"])):
                expert_dict = layer_dict["shared_experts"][i]
                layer_weights.append(convert_to_numpy(orig_model_dict[expert_dict["w1"]]))
                layer_weights.append(convert_to_numpy(orig_model_dict[expert_dict["w3"]]))
                layer_weights.append(convert_to_numpy(orig_model_dict[expert_dict["w2"]]))
                

        if "routed_experts" in layer_dict:
            for i in range(len(layer_dict["routed_experts"])):
                expert_dict = layer_dict["routed_experts"][i]
                
                layer_weights.append(convert_to_numpy(orig_model_dict[expert_dict["w1"]]))
                layer_weights.append(convert_to_numpy(orig_model_dict[expert_dict["w3"]]))
                layer_weights.append(convert_to_numpy(orig_model_dict[expert_dict["w2"]]))

        combined_weights = combine_layer_weights(layer_weights)
        combined_weights.tofile(f"{output_dir}/layers/{k}/combined_layer.weight")

        del combined_weights
        gc.collect()

        
if "head" in conversion_dict:
    os.makedirs(f"{output_dir}/head", exist_ok=True)
    head_dict = conversion_dict["head"]

    print("\tHead...\n")

    head_weights = []

    if "norm" in head_dict:
        head_weights.append(convert_to_numpy(orig_model_dict[head_dict["norm"]]))

    if "out" in head_dict:
        head_weights.append(convert_to_numpy(orig_model_dict[head_dict["out"]]))

    combined_weights = combine_layer_weights(head_weights)
    combined_weights.tofile(f"{output_dir}/head/combined_head.weight")


print(f"Finished conversion. Model saved to: {output_dir}\n")











