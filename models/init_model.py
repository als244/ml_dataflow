import sys
import os
import json
import torch
import numpy as np
import math

def get_torch_dtype_and_view_dtype(dtype_str):
    if dtype_str == 'bf16':
        dtype = torch.bfloat16
        view_dtype = torch.uint16
    elif dtype_str == 'fp16':
        dtype = torch.float16
        view_dtype = torch.uint16
    elif dtype_str == 'fp32':
        dtype = torch.float32
        view_dtype = torch.float32
    else:
        dtype = None
        view_dtype = None
    return dtype, view_dtype

def main(config_path, model_dir_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    if "embed_dtype" not in config:
        raise ValueError(f"embed_dtype not found in config")
    if "attn_dtype" not in config:
        raise ValueError(f"attn_dtype not found in config")
    if "expert_dtype" not in config:
        raise ValueError(f"expert_dtype not found in config")
    if "head_dtype" not in config:
        raise ValueError(f"head_dtype not found in config")

    embed_dtype_str = config['embed_dtype'].lower()
    attn_dtype_str = config['attn_dtype'].lower()
    expert_dtype_str = config['expert_dtype'].lower()
    head_dtype_str = config['head_dtype'].lower()


    embed_dtype, embed_view_dtype = get_torch_dtype_and_view_dtype(embed_dtype_str)
    if embed_dtype is None or embed_view_dtype is None:
        raise ValueError(f"Invalid embed_dtype: {embed_dtype_str}")

    attn_dtype, attn_view_dtype = get_torch_dtype_and_view_dtype(attn_dtype_str)
    if attn_dtype is None or attn_view_dtype is None:
        raise ValueError(f"Invalid attn_dtype: {attn_dtype_str}")

    expert_dtype, expert_view_dtype = get_torch_dtype_and_view_dtype(expert_dtype_str)
    if expert_dtype is None or expert_view_dtype is None:
        raise ValueError(f"Invalid expert_dtype: {expert_dtype_str}")

    head_dtype, head_view_dtype = get_torch_dtype_and_view_dtype(head_dtype_str)
    if head_dtype is None or head_view_dtype is None:
        raise ValueError(f"Invalid head_dtype: {head_dtype_str}")

    
    if "vocab_size" not in config:
        raise ValueError(f"vocab_size not found in config")
    if "n_layers" not in config:
        raise ValueError(f"n_layers not found in config")
    if "model_dim" not in config:
        raise ValueError(f"model_dim not found in config")
    if "n_heads" not in config:
        raise ValueError(f"n_heads not found in config")
    if "n_kv_heads" not in config:
        raise ValueError(f"n_kv_heads not found in config")
    if "qk_norm_type" not in config:
        raise ValueError(f"qk_norm_type not found in config")
    if "qk_norm_weight_type" not in config:
        raise ValueError(f"qk_norm_weight_type not found in config")
    if "rope_theta" not in config:
        raise ValueError(f"rope_theta not found in config")
    if "rms_norm_epsilon" not in config:
        raise ValueError(f"rms_norm_epsilon not found in config")

    vocab_size = int(config['vocab_size'])
    n_layers = int(config['n_layers'])
    model_dim = int(config['model_dim'])
    n_heads = int(config['n_heads'])
    n_kv_heads = int(config['n_kv_heads'])
    head_dim = model_dim / n_heads
    kv_dim = int(n_kv_heads * head_dim)

    qk_norm_type = config['qk_norm_type'].lower()
    qk_norm_weight_type = config['qk_norm_weight_type'].lower()

    if model_dim % 128 != 0:
        raise ValueError(f"model_dim must be divisible by 128")
    
    if head_dim % 32 != 0:
        raise ValueError(f"head_dim must be divisible by 32")

    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads must be divisible by n_kv_heads")
    
    if model_dim % head_dim != 0:
        raise ValueError(f"model_dim must be divisible by head_dim")
    
    if model_dim % kv_dim != 0:
        raise ValueError(f"model_dim must be divisible by kv_dim")
    
    if qk_norm_type != "none" and qk_norm_type != "head" and qk_norm_type != "token":
        raise ValueError(f"Invalid qk_norm_type: {qk_norm_type}. Must be one of: none, head, token")
    
    if qk_norm_weight_type != "none" and qk_norm_weight_type != "head" and qk_norm_weight_type != "token":
        raise ValueError(f"Invalid qk_norm_weight_type: {qk_norm_weight_type}. Must be one of: none, head, token")
    
    if qk_norm_type == "none" and qk_norm_weight_type != "none":
        raise ValueError(f"qk_norm_weight_type must be none if qk_norm_type is none")
    
    if qk_norm_type == "head" and (qk_norm_weight_type != "none" and qk_norm_weight_type != "head"):
        raise ValueError(f"qk_norm_weight_type must be 'none' or 'head' if qk_norm_type is head")
    
    if qk_norm_type == "token" and (qk_norm_weight_type != "none" and qk_norm_weight_type != "token"):
        raise ValueError(f"qk_norm_weight_type must be 'none' or 'token' if qk_norm_type is token")

    if "num_shared_experts" not in config:
        raise ValueError(f"num_shared_experts not found in config")
    if "num_routed_experts" not in config:
        raise ValueError(f"num_routed_experts not found in config")
    if "expert_dim" not in config:
        raise ValueError(f"expert_dim not found in config")
    if "expert_mlp_type" not in config:
        raise ValueError(f"expert_mlp_type not found in config")
    if "top_k_routed_experts" not in config:
        raise ValueError(f"top_k_routed_experts not found in config")

    num_shared_experts = int(config['num_shared_experts'])
    num_routed_experts = int(config['num_routed_experts'])
    expert_dim = int(config['expert_dim'])
    expert_mlp_type = config['expert_mlp_type'].lower()
    top_k_routed_experts = int(config['top_k_routed_experts'])

    if expert_mlp_type != "swiglu":
        raise ValueError(f"Invalid expert MLP type: {expert_mlp_type}. Only swiglu is supported.")

    rope_theta = int(config['rope_theta'])
    rms_norm_epsilon = float(config['rms_norm_epsilon'])

     # create model directory
    os.makedirs(model_dir_path, exist_ok=True)

    config_text = f"Embed Dtype: {embed_dtype_str}\nAttn Dtype: {attn_dtype_str}\nExpert Dtype: {expert_dtype_str}\nHead Dtype: {head_dtype_str}\nVocab Size: {vocab_size}\nNum Layers: {n_layers}\nModel Dim: {model_dim}\nNum Q Heads: {n_heads}\nNum KV Heads: {n_kv_heads}\nQK Norm Type: {qk_norm_type}\nQK Norm Weight Type: {qk_norm_weight_type}\nNum Shared Experts: {num_shared_experts}\nNum Routed Experts: {num_routed_experts}\nTop K Routed Experts: {top_k_routed_experts}\nExpert Dim: {expert_dim}\nExpert MLP Type: {expert_mlp_type}\nRope Theta: {rope_theta}\nRMS Norm Epsilon: {rms_norm_epsilon}\n"

    with open(f"{model_dir_path}/config.txt", "w") as f:
        f.write(config_text)

    if "rand_seed" in config:
        seed = int(config['rand_seed'])
        print(f"Setting random seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
 

    # create model directory
    os.makedirs(f"{model_dir_path}/embed", exist_ok=True)
    os.makedirs(f"{model_dir_path}/head", exist_ok=True)
    os.makedirs(f"{model_dir_path}/layers", exist_ok=True)

    for i in range(n_layers):
        os.makedirs(f"{model_dir_path}/layers/{i}", exist_ok=True)



    total_parms = 0


    if (vocab_size > 0):
        ## EMBEDDINGS!

        ## save as vocab size x model dim

        print("Initializing token embeddings...")

        total_parms = vocab_size * model_dim

        token_embedding = torch.randn(vocab_size, model_dim, dtype=embed_dtype).view(embed_view_dtype).numpy()

        token_embedding.tofile(f"{model_dir_path}/embed/tok_embeddings.weight")

    recip_model_sqrt = 1 / math.sqrt(model_dim)

    for i in range(n_layers):

        print(f"Initializing layer {i}...")

        all_weights = []

        ## create layer norm
        attn_norm = torch.ones(model_dim, dtype=attn_dtype).view(attn_view_dtype).numpy()

        ## remember to save the transposed version!

        all_weights.append(attn_norm.reshape(-1))

        total_parms += model_dim

        ## q matrix
        q_proj = torch.empty((model_dim, model_dim), dtype=attn_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(attn_view_dtype).numpy()

        all_weights.append(q_proj.reshape(-1))

        total_parms += model_dim * model_dim

        ## save
        k_proj = torch.empty((kv_dim, model_dim), dtype=attn_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(attn_view_dtype).numpy()

        all_weights.append(k_proj.reshape(-1))

        total_parms += kv_dim * model_dim

        ## save
        v_proj = torch.empty((kv_dim, model_dim), dtype=attn_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(attn_view_dtype).numpy()

        all_weights.append(v_proj.reshape(-1))

        total_parms += kv_dim * model_dim


        if qk_norm_weight_type == "head":
            ## q norm
            q_norm = torch.ones(head_dim, dtype=attn_dtype).view(attn_view_dtype).numpy()
            all_weights.append(q_norm.reshape(-1))
            total_parms += head_dim
            ## k norm
            k_norm = torch.ones(head_dim, dtype=attn_dtype).view(attn_view_dtype).numpy()
            all_weights.append(k_norm.reshape(-1))
            total_parms += head_dim  
        elif qk_norm_weight_type == "token":
            ## q norm
            q_norm = torch.ones(model_dim, dtype=attn_dtype).view(attn_view_dtype).numpy()
            all_weights.append(q_norm.reshape(-1))
            total_parms += model_dim
            ## k norm
            k_norm = torch.ones(model_dim, dtype=attn_dtype).view(attn_view_dtype).numpy()
            all_weights.append(k_norm.reshape(-1))
            total_parms += model_dim
        else:
            pass

        ## save
        o_proj = torch.empty((model_dim, model_dim), dtype=attn_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(attn_view_dtype).numpy()

        all_weights.append(o_proj.reshape(-1))

        total_parms += model_dim * model_dim

        ## mlp norm
        ffn_norm = torch.ones(model_dim, dtype=attn_dtype).view(attn_view_dtype).numpy()

        all_weights.append(ffn_norm.reshape(-1))

        total_parms += model_dim

        ## touter
        if num_routed_experts > 0:
            router_proj = torch.empty((num_routed_experts, model_dim), dtype=attn_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(attn_view_dtype).numpy()

            all_weights.append(router_proj.reshape(-1))

            total_parms += num_routed_experts * model_dim

        for j in range(num_shared_experts + num_routed_experts):

            if expert_mlp_type == "swiglu":

                ## create the weights
                w_1 = torch.empty((expert_dim, model_dim), dtype=expert_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(expert_view_dtype).numpy()

                all_weights.append(w_1.reshape(-1))

                total_parms += expert_dim * model_dim

                ## save
                w_3 = torch.empty((expert_dim, model_dim), dtype=expert_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(expert_view_dtype).numpy()

                all_weights.append(w_3.reshape(-1))

                total_parms += expert_dim * model_dim

                ## save
                recip_expert_sqrt = 1 / math.sqrt(expert_dim)
                w_2 = torch.empty((model_dim, expert_dim), dtype=expert_dtype).uniform_(-recip_expert_sqrt, recip_expert_sqrt).view(expert_view_dtype).numpy()

                all_weights.append(w_2.reshape(-1))

                total_parms += model_dim * expert_dim
            else:
                raise ValueError(f"Invalid expert MLP type: {expert_mlp_type}")
        
        combined_layer = np.concatenate(all_weights, axis=0)

        combined_layer.tofile(f"{model_dir_path}/layers/{i}/combined_layer.weight")

    
    if (vocab_size > 0):
        # HEAD -- store matrices trnasposed! (e.g. vocab size x model dim, instead of model dim x vocab size)

        print("Initializing head...")
        rms_head = torch.ones(model_dim, dtype=head_dtype).view(head_view_dtype).numpy()

        total_parms += model_dim
    
        head_proj = torch.empty((vocab_size, model_dim), dtype=head_dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(head_view_dtype).numpy()

        total_parms += vocab_size * model_dim

        combined_head = np.concatenate((rms_head.reshape(-1), head_proj.reshape(-1)), axis=0)

        combined_head.tofile(f"{model_dir_path}/head/combined_head.weight")

    print(f"\nTotal parameters: {total_parms / 1e6}M")
        

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python init_model.py <config_path> <model_dir_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    model_dir_path = sys.argv[2]

    main(config_path, model_dir_path)