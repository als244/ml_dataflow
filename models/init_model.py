import sys
import os
import json
import torch
import numpy as np
import math


def main(config_path, model_dir_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    dtype_str = config['dtype'].lower()


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
        raise ValueError(f"Invalid dtype: {dtype_str}")
    
    vocab_size = config['vocab_size']
    n_layers = config['n_layers']
    model_dim = config['model_dim']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    head_dim = model_dim / n_heads
    kv_dim = int(n_kv_heads * head_dim)

    if model_dim % 256 != 0:
        raise ValueError(f"model_dim must be divisible by 256")
    
    if head_dim % 32 != 0:
        raise ValueError(f"head_dim must be divisible by 32")

    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads must be divisible by n_kv_heads")
    
    if model_dim % head_dim != 0:
        raise ValueError(f"model_dim must be divisible by head_dim")
    
    if model_dim % kv_dim != 0:
        raise ValueError(f"model_dim must be divisible by kv_dim")


    num_shared_experts = config['num_shared_experts']
    num_routed_experts = config['num_routed_experts']
    expert_dim = config['expert_dim']
    expert_mlp_type = config['expert_mlp_type'].lower()

    if expert_mlp_type != "swiglu":
        raise ValueError(f"Invalid expert MLP type: {expert_mlp_type}. Only swiglu is supported.")

    rope_theta = config['rope_theta']
    rms_norm_epsilon = config['rms_norm_epsilon']

     # create model directory
    os.makedirs(model_dir_path, exist_ok=True)

    config_text = f"Data Type: {dtype_str}\nVocab Size: {vocab_size}\nNum Layers: {n_layers}\nModel Dim: {model_dim}\nNum Q Heads: {n_heads}\nNum KV Heads: {n_kv_heads}\nNum Shared Experts: {num_shared_experts}\nNum Routed Experts: {num_routed_experts}\nExpert Dim: {expert_dim}\nExpert MLP Type: {expert_mlp_type}\nRope Theta: {rope_theta}\nRMS Norm Epsilon: {rms_norm_epsilon}\n"

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

        token_embedding = torch.randn(vocab_size, model_dim, dtype=dtype).view(view_dtype).numpy()

        token_embedding.tofile(f"{model_dir_path}/embed/tok_embeddings.weight")

    recip_model_sqrt = 1 / math.sqrt(model_dim)

    for i in range(n_layers):

        print(f"Initializing layer {i}...")

        all_weights = []

        ## create layer norm
        attn_norm = torch.ones(model_dim, dtype=dtype).view(view_dtype).numpy()

        ## remember to save the transposed version!

        all_weights.append(attn_norm.reshape(-1))

        total_parms += model_dim

        ## q matrix
        q_proj = torch.empty((model_dim, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

        all_weights.append(q_proj.reshape(-1))

        total_parms += model_dim * model_dim

        ## save
        k_proj = torch.empty((kv_dim, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

        all_weights.append(k_proj.reshape(-1))

        total_parms += kv_dim * model_dim

        ## save
        v_proj = torch.empty((kv_dim, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

        all_weights.append(v_proj.reshape(-1))

        total_parms += kv_dim * model_dim

        ## save
        o_proj = torch.empty((model_dim, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

        all_weights.append(o_proj.reshape(-1))

        total_parms += model_dim * model_dim

        ## mlp norm
        ffn_norm = torch.ones(model_dim, dtype=dtype).view(view_dtype).numpy()

        all_weights.append(ffn_norm.reshape(-1))

        total_parms += model_dim

        ## touter
        if num_routed_experts > 0:
            router_proj = torch.empty((num_routed_experts, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

            all_weights.append(router_proj.reshape(-1))

            total_parms += num_routed_experts * model_dim

        for j in range(num_shared_experts + num_routed_experts):

            if expert_mlp_type == "swiglu":

                ## create the weights
                w_1 = torch.empty((expert_dim, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

                all_weights.append(w_1.reshape(-1))

                total_parms += expert_dim * model_dim

                ## save
                w_3 = torch.empty((expert_dim, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

                all_weights.append(w_3.reshape(-1))

                total_parms += expert_dim * model_dim

                ## save
                recip_expert_sqrt = 1 / math.sqrt(expert_dim)
                w_2 = torch.empty((model_dim, expert_dim), dtype=dtype).uniform_(-recip_expert_sqrt, recip_expert_sqrt).view(view_dtype).numpy()

                all_weights.append(w_2.reshape(-1))

                total_parms += model_dim * expert_dim
            else:
                raise ValueError(f"Invalid expert MLP type: {expert_mlp_type}")
        
        combined_layer = np.concatenate(all_weights, axis=0)

        combined_layer.tofile(f"{model_dir_path}/layers/{i}/combined_layer.weight")

    
    if (vocab_size > 0):
        # HEAD -- store matrices trnasposed! (e.g. vocab size x model dim, instead of model dim x vocab size)

        print("Initializing head...")
        rms_head = torch.ones(model_dim, dtype=dtype).view(view_dtype).numpy()

        total_parms += model_dim
    
        head_proj = torch.empty((vocab_size, model_dim), dtype=dtype).uniform_(-recip_model_sqrt, recip_model_sqrt).view(view_dtype).numpy()

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