import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM
import numpy as np
import sys

seq_len = int(sys.argv[1])
num_seqs = int(sys.argv[2])

## Load dataset

input_ids_np = np.fromfile("../../../../data/65536_token_ids_uint32.dat", dtype=np.uint32)
labels_np = np.fromfile("../../../../data/65536_labels_uint32.dat", dtype=np.uint32)

# --- Create one long sequence of length `seq_len` with wrapping ---
# We use the modulo operator on a range of indices. This automatically
# wraps the indices around the length of the source array.
base_len = len(input_ids_np)
indices = np.arange(seq_len) % base_len

# Use the generated indices to create the single long sequence for inputs and labels
long_input_ids_seq = input_ids_np[indices]
long_labels_seq = labels_np[indices]

# --- Make `num_seqs` copies of this sequence ---
# We use np.tile to repeat the single long sequence `num_seqs` times,
# creating a 2D array of shape (num_seqs, seq_len).
final_input_ids = np.tile(long_input_ids_seq, (num_seqs, 1))
final_labels = np.tile(long_labels_seq, (num_seqs, 1))

# Create an attention mask of all 1s with the same shape, as there is no padding.
final_attention_mask = np.ones_like(final_input_ids)

# Populate the dictionary, converting the NumPy arrays to Python lists
data_dict = {
    "input_ids": final_input_ids.tolist(),
    "labels": final_labels.tolist(),
    "attention_mask": final_attention_mask.tolist(),
}

dataset = Dataset.from_dict(data_dict)


training_args = TrainingArguments(
    output_dir="./results",

    gradient_checkpointing=True,

    # Use max_steps to run for exactly 10 steps
    max_steps=10,
    
    # Set gradient_accumulation_steps to 1 (not 0)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, 
    
    # --- AdamW Optimizer Hyperparameters (Correct) ---
    learning_rate=2e-5,
    weight_decay=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    
    # --- Learning Rate Scheduler (Correct) ---
    lr_scheduler_type="constant",
    warmup_ratio=0,
    
    # Set logging_steps to 1 to see loss at each step
    logging_steps=1,
    save_steps=10,
    bf16=True,
)



# --- At this point, your dataset has 'input_ids', 'attention_mask', and 'labels' ---

model_name = "./models/llama3_8B"

# You would now load your model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# And finally, pass the prepared dataset to the Trainer
trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=dataset
)

trainer.train()