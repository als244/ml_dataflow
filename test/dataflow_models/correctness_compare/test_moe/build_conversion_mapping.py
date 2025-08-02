import torch
import torch.nn as nn
import glob
import json

from moe_model_flash2 import ModelArgs, MoETransformer, SeqlensInfo

import torch.optim as optim

import time
import numpy as np
import os

import sys
import pickle

if (len(sys.argv) != 3):
    print("Usage: python convert_moe_to_raw.py <pytorch_model_path> <out_conversion_dict>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
OUTPUT_DICT_PATH = sys.argv[2]
## MoETransformer Class
model = torch.load(MODEL_PATH, weights_only=False)

model_args = model.params

model_state_dict = model.state_dict()

# model_state_dict_keys = list(model_state_dict.keys())

# for key in model_state_dict_keys:
#     print(key)
#     print(model_state_dict[key].shape)
#     print("\n")


conversion_dict = {}
    

## embedding stuff
tok_emb_key = "tok_embeddings.weight"

conversion_dict["embedding"] = tok_emb_key

conversion_dict["layers"] = {}

## each layer stuff
for i in range(model_args.num_layers):

    layer_dict = {}

    layer_key = f"layers.{i}"

    ## attention stuff
    attn_norm_key = f"{layer_key}.attention_norm.weight"
    attn_q_proj_key = f"{layer_key}.attention.wq.weight"
    attn_k_proj_key = f"{layer_key}.attention.wk.weight"
    attn_v_proj_key = f"{layer_key}.attention.wv.weight"
    attn_out_proj_key = f"{layer_key}.attention.wo.weight"

    layer_dict["attn_norm"] = attn_norm_key
    layer_dict["q_norm"] = None
    layer_dict["k_norm"] = None
    layer_dict["q_proj"] = attn_q_proj_key
    layer_dict["k_proj"] = attn_k_proj_key
    layer_dict["v_proj"] = attn_v_proj_key
    layer_dict["o_proj"] = attn_out_proj_key

    ## ffn stuff
    ffn_norm_key = f"{layer_key}.ffn_norm.weight"

    layer_dict["ffn_norm"] = ffn_norm_key

    ## router
    router_key = f"{layer_key}.feed_forward.gate.weight"
    layer_dict["router"] = router_key

    layer_dict["shared_experts"] = {}

    routed_experts_dict = {}

    ## ffn
    for j in range(model_args.num_routed_experts):
        expert_w1_key = f"{layer_key}.feed_forward.experts.{j}.gate_proj.weight"
        expert_w3_key = f"{layer_key}.feed_forward.experts.{j}.up_proj.weight"
        expert_w2_key = f"{layer_key}.feed_forward.experts.{j}.down_proj.weight"

        routed_experts_dict[j] = {}

        routed_experts_dict[j]["w1"] = expert_w1_key
        routed_experts_dict[j]["w2"] = expert_w2_key
        routed_experts_dict[j]["w3"] = expert_w3_key
    
    layer_dict["routed_experts"] = routed_experts_dict

    conversion_dict["layers"][i] = layer_dict
    

## head stuff

conversion_dict["head"] = {}

head_norm_key = "norm.weight"
head_out_key = "output.weight"

conversion_dict["head"]["norm"] = head_norm_key
conversion_dict["head"]["out"] = head_out_key


conversion_dict["config"] = {}

config_dict = {}
config_dict["embed_dtype"] = "bf16"
config_dict["attn_dtype"] = "bf16"
config_dict["expert_dtype"] = "bf16"
config_dict["head_dtype"] = "bf16"
config_dict["vocab_size"] = model_args.vocab_size
config_dict["num_layers"] = model_args.num_layers
config_dict["model_dim"] = model_args.model_dim
config_dict["num_q_heads"] = model_args.num_q_heads
config_dict["num_kv_heads"] = model_args.num_kv_heads
config_dict["qk_norm_type"] = None
config_dict["qk_norm_weight_type"] = None
config_dict["num_shared_experts"] = 0
config_dict["num_routed_experts"] = model_args.num_routed_experts
config_dict["top_k_routed_experts"] = model_args.top_k_routed_experts
config_dict["expert_dim"] = model_args.expert_dim
config_dict["expert_mlp_type"] = model_args.expert_mlp_type
config_dict["rope_theta"] = model_args.rope_theta
config_dict["rms_norm_epsilon"] = model_args.rms_norm_epsilon

conversion_dict["config"] = config_dict



with open(OUTPUT_DICT_PATH, "wb") as f:
    pickle.dump(conversion_dict, f)

