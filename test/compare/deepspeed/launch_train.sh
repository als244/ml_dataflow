#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [--zero_stage <value>] [--save_act_layer_frac <value>] <model_config> <seq_len> <seqs_per_batch> <grad_accum_steps> <num_steps>"
    echo ""
    echo "Arguments:"
    echo "  --zero_stage <value>           Optional: Zero stage value for DeepSpeed"
    echo "  --save_act_layer_frac <value>  Optional: Fraction of layer activations to leave on device (default: 0)"
    echo "  <model_config>                 Model configuration"
    echo "  <seq_len>                     Sequence length"
    echo "  <seqs_per_batch>              Sequences per batch"
    echo "  <grad_accum_steps>            Gradient accumulation steps"
    echo "  <num_steps>                   Number of training steps"
    exit 1
}

# Initialize variables
ZERO_STAGE=""
ZERO_STAGE_ARG=""
SAVE_ACT_LAYER_FRAC=""
SAVE_ACT_LAYER_FRAC_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --zero_stage)
            if [[ -n $2 && $2 != --* ]]; then
                ZERO_STAGE="$2"
                ZERO_STAGE_ARG="--zero_stage $ZERO_STAGE"
                shift 2
            else
                echo "Error: --zero_stage requires a value"
                usage
            fi
            ;;
        --save_act_layer_frac)
            if [[ -n $2 && $2 != --* ]]; then
                SAVE_ACT_LAYER_FRAC="$2"
                SAVE_ACT_LAYER_FRAC_ARG="--save_act_layer_frac $SAVE_ACT_LAYER_FRAC"
                shift 2
            else
                echo "Error: --save_act_layer_frac requires a value"
                usage
            fi
            ;;
        --help|-h)
            usage
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            ;;
        *)
            # Collect positional arguments
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if we have the correct number of positional arguments
if [ ${#POSITIONAL_ARGS[@]} -ne 5 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Expected 5 positional arguments, got ${#POSITIONAL_ARGS[@]}"
    usage
fi

# Assign positional arguments to meaningful variable names
MODEL_CONFIG=${POSITIONAL_ARGS[0]}
SEQ_LEN=${POSITIONAL_ARGS[1]}
SEQS_PER_BATCH=${POSITIONAL_ARGS[2]}
GRAD_ACCUM_STEPS=${POSITIONAL_ARGS[3]}
NUM_STEPS=${POSITIONAL_ARGS[4]}

# Generate random master port
MASTER_PORT=$((29500 + RANDOM % 36035))

# Build the deepspeed command
DEEPSPEED_CMD="deepspeed --num_gpus=1 --master_port=$MASTER_PORT train.py \
    --model_config $MODEL_CONFIG \
    $ZERO_STAGE_ARG \
    $SAVE_ACT_LAYER_FRAC_ARG \
    --seq_len $SEQ_LEN \
    --seqs_per_batch $SEQS_PER_BATCH \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --num_steps $NUM_STEPS"

# Run the deepspeed command
eval $DEEPSPEED_CMD