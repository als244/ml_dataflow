import torch
import numpy as np
import random
from moe_model_flash2 import MoETransformer, ModelArgs

model_args = ModelArgs()

model_args.model_dim = 1536
model_args.expert_dim = 768
model_args.n_layers = 8
model_args.n_heads = 24
model_args.n_kv_heads = 3
model_args.num_experts = 64
model_args.num_experts_per_tok = 4
model_args.vocab_size = 128256
model_args.rope_theta = 500000
model_args.norm_eps = 1e-5
model_args.max_seq_len = 1048576

moe_model = MoETransformer(model_args)

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


torch.save(moe_model, 'full_model.pth')