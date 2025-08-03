import torch
import numpy as np
import random
from moe_model_flash2 import MoETransformer, ModelArgs
import sys
import pickle
if len(sys.argv) != 2:
    print("Usage: python save_moe_model.py <model_path>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]

model_args = ModelArgs()

model_args.embed_dtype = "bf16"
model_args.attn_dtype = "bf16"
model_args.router_dtype = "fp32"
model_args.expert_dtype = "bf16"
model_args.head_dtype = "bf16"
model_args.vocab_size = 128256
model_args.num_layers = 8
model_args.model_dim = 1536
model_args.num_q_heads = 24
model_args.num_kv_heads = 3
model_args.qk_norm_type = None
model_args.qk_norm_weight_type = None
model_args.num_shared_experts = 0
model_args.num_routed_experts = 64
model_args.top_k_routed_experts = 4
model_args.expert_dim = 768
model_args.expert_mlp_type = "swiglu"
model_args.rope_theta = 500000
model_args.rms_norm_epsilon = 1e-5
model_args.max_seq_len = 1048576

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

moe_model = MoETransformer(model_args)

pickle.dump(model_args, open(f"{MODEL_PATH}_config.pkl", "wb"))

torch.save(moe_model.state_dict(), f"{MODEL_PATH}.pt")