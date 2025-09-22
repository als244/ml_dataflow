#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 6 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: $0 <model_config> <zero_stage> <seq_len> <seqs_per_batch> <grad_accum_steps> <num_steps>"
    exit 1
fi

# Assign arguments to meaningful variable names (optional but recommended)
MODEL_CONFIG=$1
ZERO_STAGE=$2
SEQ_LEN=$3
SEQS_PER_BATCH=$4
GRAD_ACCUM_STEPS=$5
NUM_STEPS=$6

MASTER_PORT=$((29500 + RANDOM % 36035))

# Run the deepspeed command
deepspeed --num_gpus=1 --master_port=$MASTER_PORT train.py \
    --model_config $MODEL_CONFIG \
    --zero_stage $ZERO_STAGE \
    --seq_len $SEQ_LEN \
    --seqs_per_batch $SEQS_PER_BATCH \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --num_steps $NUM_STEPS
