import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import RegularPolygon, Arc # Added Arc
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.backend_tools import Cursors
import sys
import time
import textwrap
from collections import deque # Use deque for efficient queue operations
# import pdb; pdb.set_trace() # Keep commented out unless debugging

import math

def round_up_to_multiple(n, m):
  """
  Rounds an integer n up to the nearest multiple of a positive integer m.

  Args:
    n: The integer number to round up.
    m: The positive integer multiple to round up to.

  Returns:
    The smallest multiple of m that is greater than or equal to n.
  """

  return int(((n + m - 1) // m) * m)

# --- Simulation Parameters ---
## making 100 microseconds equivalent to 1 cycle
cycles_per_second = 1e4

# --- Speed/Interval Parameters ---
min_speed_level = 1
max_speed_level = 100
min_interval = 1
max_interval = 100
initial_speed_level = 50

## Number of Devices
N = 1

## Training Info
num_sequences = 1

# 30 million tokens
#seqlen_thousands = 30 * (1 << 10)

# 1 million tokens
#seqlen_thousands = 1 * (1 << 10)

# 32k tokens
seqlen_thousands = 32
seqlen = (1 << 10) * seqlen_thousands
train_token_ratio = 1 / 4
min_chunk_size = 1536




## Model Info
bitwidth = 16
dtype_bytes = bitwidth / 8
total_layers = 32
vocab_size = 128256
model_dim = 4096
kv_factor = 1 / 4
kv_dim = int(model_dim * kv_factor)
num_experts = 1
active_experts = 1
expert_dim = 14336
## THese config params not implemented yet...
attn_type = "Exact"


max_device_memory_bytes = 23 * (1 << 30)


## hardware configs

## compute configs
## H100 TFLOPS

## FP16 = 989 TFLOPS
## FP8 = 989 * 2 TFLOPS
hardware_max_flops = int((989 * (2 / dtype_bytes)) * 1e12)
## 3.35 TB/s
hardware_mem_bw_bytes_sec = 3.35 * (1 << 40)
matmul_efficiency = 0.7
attn_efficiency = 0.6


## communication configs
home_bw_gbit_sec = 240
peer_bw_gbit_sec = 100



## derived params

hw_compute_bound = hardware_max_flops / hardware_mem_bw_bytes_sec

## Dataflow Params

## FOR NOW: HARDCODING CHUNK TYPE TO "Equal Data"
chunk_type = "Equal Data"

## determine chunk size based on arithmetic intensity of expert

## avg. expert has M value of: (num_chunks * active_experts * chunk_size) / (num_experts)
## and K value of: model_dim
## and N value of: expert_dim

## matmul arithmetic intensity is then given by:
## matmul_arith = (2 * M * K * N) / (dtype_bytes * ((M * K) + (M * N) + (K * N)))

## we want matmul_arith >= hw_compute_bound
## solve for M:
## M >= (hw_compute_bound * dtype_bytes * K * N)) / (2 * K * N - hw_compute_bound * dtype_bytes * (K + N))

## Now let's solve for chunk_size:
## chunk_size >= active_experts * (hw_compute_bound * dtype_bytes * model_dim * expert_dim)) / (2 * model_dim * expert_dim - hw_compute_bound * dtype_bytes * (model_dim + expert_dim))

## Let's round up to the nearest multiple of 256

chunk_size_base = math.ceil(active_experts * (hw_compute_bound * dtype_bytes * model_dim * expert_dim)) / (2 * model_dim * expert_dim - hw_compute_bound * dtype_bytes * (model_dim + expert_dim))

print(f"Based on hw compute bound of {hw_compute_bound} FLOS/byte, along with FFN dims of {model_dim} x {expert_dim}, the base chunk size is {chunk_size_base}")
if chunk_size_base < min_chunk_size:
    print(f"Min chunk size is {min_chunk_size}, so using that instead...")
    chunk_size_base = min_chunk_size

chunk_multiple = 256

chunk_size = round_up_to_multiple(chunk_size_base, chunk_multiple)

total_chunks = math.ceil(seqlen / chunk_size)
train_chunk_freq =  math.ceil(1/ train_token_ratio)
print(f"train_chunk_freq: {train_chunk_freq}")
total_train_chunks = int(total_chunks / train_chunk_freq)
if ((total_chunks - 1) % train_chunk_freq) != 0:
    total_train_chunks += 1



## Determine acitvaiton_capacity, layer capacity, and transition capacity
## based on Model Info, Training Info, and Max Device Memory

attn_block_size_bytes = dtype_bytes * (2 * model_dim * model_dim + 4 * model_dim * kv_dim)
ffn_block_size_bytes = dtype_bytes * (3 * model_dim * expert_dim * num_experts)
layer_size_bytes = attn_block_size_bytes + ffn_block_size_bytes

## model input, query, key, value, attn output, attn ouut output
attn_activation_bytes = dtype_bytes * 4 * (model_dim * chunk_size) + 2 * (kv_dim * chunk_size)
## saving x1 and x3 in order to correctly backprop through swiglu...
ffn_activation_bytes = dtype_bytes * (2 * chunk_size * expert_dim * active_experts)
activation_size_bytes = attn_activation_bytes + ffn_activation_bytes

## Each Chunk Context Contribution....
chunk_context_size_bytes = 2 * (chunk_size * kv_dim * dtype_bytes)

## Per-Layer Activation Size:
## Only Training Chunks Need to Save Activations...

## each chunk's context activations already incldued in context size...
per_layer_activation_size = total_train_chunks * (activation_size_bytes - chunk_context_size_bytes)

## Per-Layer Total Context Size:

## Full Context Includes All Chunks...
per_layer_full_context_size = total_chunks * chunk_context_size_bytes


## 1.) determine layer capacity, and use leftovers to determine activations/transitions....

layer_transfer_time_sec = (layer_size_bytes * 8) / (home_bw_gbit_sec * 1e9)

## Determine the computation time for each layer...


## PER CHUNK MATMUL FLOPS
matmul_attn_flops_per_layer = 2 * chunk_size * (2 *(model_dim * model_dim) + (2 * model_dim * kv_dim))
matmul_ffn_flops_per_layer = 2 * (chunk_size * active_experts) * (3 * (model_dim * expert_dim))

base_flops_per_layer = matmul_attn_flops_per_layer + matmul_ffn_flops_per_layer

# FULL SEQ ATTENTION FLOPS
full_seq_attn_fwd_flops_per_layer = 2 * seqlen * seqlen * model_dim
## (fwd pass)
full_seq_matmul_fwd_flops_per_layer = total_chunks * base_flops_per_layer

full_seq_time_per_layer = (full_seq_matmul_fwd_flops_per_layer) / (hardware_max_flops * matmul_efficiency) + (full_seq_attn_fwd_flops_per_layer) / (hardware_max_flops * attn_efficiency)

### TODO:
## Finish this calculation to dynmically determine ratio of layer capacity to activations/transitions...









## FOR NOW HARDCODING LAYER CAPACITY TO 2, GRAD CAPACITY TO 2...
layer_capacity = 2
grad_layer_capacity = 2

if (N >= total_layers):
    layer_capacity = 1
    grad_layer_capacity = 1

## HARDOCIDNG CONTEXT BUFFER CAPACITY TO 1...
context_buffer_capacity = 1
grad_context_buffer_capacity = 1



orig_dev_mem = max_device_memory_bytes

base_dev_mem = 0
base_dev_mem += (layer_capacity + grad_layer_capacity) * layer_size_bytes
base_dev_mem += (context_buffer_capacity + grad_context_buffer_capacity) * per_layer_full_context_size

remain_dev_mem = orig_dev_mem - base_dev_mem

if remain_dev_mem < activation_size_bytes:
    print(f"Error: Failed first level memory check for enough memory to hold model weights. Currently only supports activation capacity >= 1, layer capacity of 2, grad layer capacity of 2, context buffer capacity of 1, and grad context buffer capacity of 1.\nThis requires {(base_dev_mem + activation_size_bytes) / (1 << 30):.2f} GB of memory, but only {orig_dev_mem / (1 << 30):.2f} GB is available.\n\nCannot run simulation with current configuration\n")
    sys.exit(1)


## Now need to determine the required transition capacity...
## This will be based on the amount of output transitions
## Going into starting device after completeting layer id #num_devices - 1
## The starting device will still be churning out new chunks, when the first
## chunks arrive back...

output_size_bytes = dtype_bytes * (model_dim * chunk_size)

## TODO:

## For now hardcoding to total_chunks - total_devices....
## This is overly conservative, but will work for now...
## Doesn't account for computation time that each chunk takes

## If this transition capacity becomes too large, then chunk sizes should
## by dynamic, where early chunks are larger in token cnt vs. later chunks
## We can adjust this for equal compute time, or even stagger the compute
## time to be decreasing, by having especially large chunks at the beginning...
## TODO: implement chunk_type = "Equal Compute", and "Decreasing Compute"

## The attn component will cause early chunks to 
transitions_capacity = N


## TODO: determine the head transition capacity by first determining the cutoff
## along with time it take for head to compute (see below within devicie intialize)
## The capacity should be total_chunks - head_cutoff + 1

## for now just being conservative and hardcoding to total_chunks....
head_transitions_capacity = total_chunks

## Update remaining device memory...
## ignoring the special case of head, and just doing typical for now....

transition_dev_mem = 2 * transitions_capacity * output_size_bytes

remain_dev_mem -= transition_dev_mem
if (remain_dev_mem < activation_size_bytes):
    print(f"Error: Failed second level memory check for enough memory to hold transitions. Currently only supports activation capacity >= 1 {activation_size_bytes}, transition capacity = total_chunks - num_devices {total_chunks - N}, layer capacity of 2, grad layer capacity of 2, context buffer capacity of 1, and grad context buffer capacity of 1.\nThis requires {(base_dev_mem + transition_dev_mem + activation_size_bytes) / (1 << 30):.2f} GB of memory, but only {orig_dev_mem / (1 << 30):.2f} GB is available.\n\nCannot run simulation with current configuration\n")
    sys.exit(1)


## Now finally we can use remaining device memory for activations...

activations_capacity = int(remain_dev_mem // activation_size_bytes)

## Don't set activations_capacity > total nubmer of activaitons per device...
max_per_home_layers_base = math.ceil(total_layers / N)
max_activations_capacity = max_per_home_layers_base * total_train_chunks

activations_capacity = int(min(activations_capacity, max_activations_capacity))



do_backward = True




## head does fwd + bwd

## this is multiplied by the chunk_size + cur seq len (number of keys)
## here chunk size is fixed number of queries...
flops_per_attn_chunk_mult = 2 * chunk_size * model_dim

head_flops = 2 * (2 * vocab_size * model_dim * chunk_size)


total_matmul_flops = 0
total_attn_flops = 0

total_fwd_flops = 0
total_bwd_flops = 0

total_head_flops = 0



head_computation_times_sec = head_flops / (hardware_max_flops * matmul_efficiency)

computation_times_sec = {}
computation_times_frames = {}

computation_times_sec_bwd = {}
computation_times_frames_bwd = {}

prev_seq_len = 0

total_flops = 0

bwd_w_flops = base_flops_per_layer
bwd_x_flops = 0

for i in range(total_chunks):
    per_layer_chunk_flops = 0
    total_chunk_flops = 0
    cur_seq_len = prev_seq_len + chunk_size
    attn_flops = flops_per_attn_chunk_mult * cur_seq_len

    total_attn_flops += total_layers * attn_flops
    total_matmul_flops += total_layers * base_flops_per_layer
    layer_flops = base_flops_per_layer + attn_flops

    total_fwd_flops += total_layers * layer_flops

    computation_times_sec[i] = base_flops_per_layer / (hardware_max_flops * matmul_efficiency) + attn_flops / (hardware_max_flops * attn_efficiency)
    computation_times_frames[i] = math.ceil(computation_times_sec[i] * cycles_per_second)

    per_layer_chunk_flops += layer_flops

    if do_backward and (((i % train_chunk_freq) == 0) or (i == total_chunks - 1)):
        ## attention layer for bwd x has double the flops...
        bwd_x_flops = layer_flops + attn_flops
        computation_times_sec_bwd[i] = base_flops_per_layer / (hardware_max_flops * matmul_efficiency) + (2 * attn_flops) / (hardware_max_flops * attn_efficiency)
        computation_times_frames_bwd[i] = math.ceil(computation_times_sec_bwd[i] * cycles_per_second)

        per_layer_chunk_flops += bwd_x_flops

        total_matmul_flops += total_layers * base_flops_per_layer
        total_attn_flops += 2 * total_layers * attn_flops

        per_layer_chunk_flops += bwd_w_flops
        total_matmul_flops += total_layers * base_flops_per_layer

        total_bwd_flops += total_layers * bwd_x_flops + total_layers * bwd_w_flops

        total_chunk_flops += head_flops

        total_head_flops += head_flops
        total_matmul_flops += head_flops

    total_chunk_flops += total_layers * per_layer_chunk_flops

    total_flops += total_chunk_flops

    prev_seq_len = cur_seq_len




head_layer_size_bytes = dtype_bytes * (vocab_size * model_dim)
head_layer_transfer_time_sec = head_layer_size_bytes / (home_bw_gbit_sec * 1e9)

activation_transfer_time_sec = (activation_size_bytes * 8) / (home_bw_gbit_sec * 1e9)

chunk_context_transfer_time_sec = chunk_context_size_bytes / (peer_bw_gbit_sec * 1e9)

transition_transfer_time_sec = (output_size_bytes * 8) / (peer_bw_gbit_sec * 1e9)

bwd_w_time_sec = bwd_w_flops / (hardware_max_flops * matmul_efficiency)



computationFrames = computation_times_frames[0] # Cycles per compute task
max_computationFrames = computation_times_frames[total_chunks-1]

headFrames = math.ceil(head_computation_times_sec * cycles_per_second)
bwdWFrames = math.ceil(bwd_w_time_sec * cycles_per_second)

contextTransferFrames = math.ceil(chunk_context_transfer_time_sec * cycles_per_second)
if chunk_context_transfer_time_sec * cycles_per_second < 1:
    print(f"Context Transfer True Cycles: {chunk_context_transfer_time_sec * cycles_per_second}\n")
    contextTransferCycleText = "< 1 Cycle"
else:
    contextTransferCycleText = str(math.ceil(chunk_context_transfer_time_sec * cycles_per_second)) + " Cycles"
layerTransferFrames = math.ceil(layer_transfer_time_sec * cycles_per_second) # Cycles to transfer weights
headTransferFrames = math.ceil(head_layer_transfer_time_sec * cycles_per_second)
savedActivationsFrames = math.ceil(activation_transfer_time_sec * cycles_per_second) # Cycles to transfer activations (save/fetch)
activationTransitionFrames = math.ceil(transition_transfer_time_sec * cycles_per_second) # Cycles to transfer activations/grads between devices






# --- Global Control Flags ---
TO_PRINT = False # ENABLE DEBUG PRINTING

# --- Layout Parameters ---
title_fontsize = 24

# --- Home Parameters ---
inner_radius = 4
inner_node_radius = 1.5
inner_node_opacity = 0.8

# --- Device Parameters ---
outer_node_radius = 2.5
device_opacity = 0.3
transferDistance = 13
total_distance = transferDistance

# --- Stall Node Parameters ---
stall_node_distance_offset = 5.25
stall_node_radius = 2.5
stall_node_opacity = 0.7
stall_node_border_width = 4
stall_node_fontsize = 6

# --- Transfer Parameters ---
edge_linewidth = 1.5
edge_label_fontsize = 7
label_offset_distance = 0.5

# --- Computation Arc Parameters ---
compute_arc_linewidth = 3
compute_arc_radius_scale = 1.3 # Arc radius relative to outer node radius

# --- Arrow/Geometry Parameters ---
arrow_offset_dist = inner_node_radius * 0.3
head_len = 1
head_wid = 0.3
mut_scale = 6

if sys.platform == 'darwin':
    matplotlib.use('MacOSX')

lower_bound = int(((total_matmul_flops / N) / (hardware_max_flops * matmul_efficiency) + (total_attn_flops / N) / (hardware_max_flops * attn_efficiency)) * cycles_per_second)
max_frames = math.ceil(lower_bound * 1.25)


def calculate_interval(speed_level, s_min, s_max, i_min, i_max):
    """Linearly maps speed level (s_min to s_max) to interval (i_max to i_min)."""
    if s_max == s_min:
        interval_range = 0
    else:
        interval_range = i_min - i_max

    if s_max == s_min:
        return i_min
    else:
        speed_range = s_max - s_min
        if speed_range == 0:
            return i_min
        else:
            speed_normalized = (speed_level - s_min) / speed_range
            interval = i_max + speed_normalized * interval_range
            # Ensure interval is at least 1
            return int(round(max(1, interval)))


initial_frame_interval = calculate_interval(
    initial_speed_level,
    min_speed_level, max_speed_level,
    min_interval, max_interval
)
if TO_PRINT:
    print(f"Initial Speed Level: {initial_speed_level}, Initial Interval: {initial_frame_interval} ms")


# --- Define Default Colors ---
COLOR_INBOUND_DEFAULT = 'gray'
COLOR_INBOUND_WEIGHT = 'olive'
COLOR_INBOUND_BWD_FETCHED_ACTIVATION = 'magenta'
COLOR_INBOUND_BWD_FETCHED_CTX = 'cyan'

COLOR_OUTBOUND_DEFAULT = 'gray'
COLOR_OUTBOUND_FWD_ACTIVATION = 'magenta'
COLOR_OUTBOUND_FWD_CTX = 'cyan'
COLOR_OUTBOUND_WGT_GRAD = 'saddlebrown'

COLOR_RING_CCW = 'indigo'
COLOR_RING_CW = 'maroon'

COLOR_COMPUTE_DEFAULT = 'gray'
COLOR_COMPUTE_FWD = 'darkgreen'
COLOR_COMPUTE_BWD_X = 'orangered'
COLOR_COMPUTE_BWD_W = 'teal'
COLOR_COMPUTE_HEAD = 'lawngreen'
COLOR_STALL_NODE_FILL = (1.0, 0.0, 0.0, stall_node_opacity)


# --- Setup ---
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_aspect('equal')
lim = total_distance + outer_node_radius + stall_node_distance_offset + stall_node_radius + 0.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axis('off')
center_pos = np.array([0, 0])

# --- Legend Text ---
wrap_width = 40

## FULL MODEL TRAINING MEMORY INFO
train_model_size = (layer_size_bytes * total_layers + head_layer_size_bytes)
## only training chunks need to save activations
train_activation_size = (total_train_chunks * activation_size_bytes * total_layers)
## all contexts need to be saved
train_context_size = ((total_chunks - total_train_chunks) * chunk_context_size_bytes * total_layers)
train_gradient_size = (train_model_size)
train_optimizer_state_size = (2 * train_model_size)

aggregate_memory_size = train_model_size + train_activation_size + train_context_size + train_gradient_size + train_optimizer_state_size


## DEVICE MEMORY INFO
chunk_workspace_size = (chunk_size * expert_dim * active_experts * dtype_bytes)

layer_allocation = (layer_capacity * layer_size_bytes)
grad_layer_allocation = (grad_layer_capacity * layer_size_bytes)
context_buffer_allocation = (context_buffer_capacity * per_layer_full_context_size)
grad_context_buffer_allocation = (grad_context_buffer_capacity * per_layer_full_context_size)
activation_allocation = (activations_capacity * activation_size_bytes)
typical_transition_allocation = (2 * transitions_capacity * output_size_bytes)
head_transition_allocation = (head_transitions_capacity + transitions_capacity) * output_size_bytes

typical_device_memory_size = layer_allocation + grad_layer_allocation + context_buffer_allocation + grad_context_buffer_allocation + activation_allocation + typical_transition_allocation
speical_device_memory_size = layer_allocation + grad_layer_allocation + context_buffer_allocation + grad_context_buffer_allocation + activation_allocation + head_transition_allocation

per_home_layers_base = total_layers // N
print(f"Per-Home Layers Base: {per_home_layers_base}")
remain_layers = total_layers % N

head_id = total_layers % N

per_home_num_blocks = []
per_home_layer_sizes = [] 

for i in range(N):
    if i < remain_layers:
        per_home_num_blocks.append(per_home_layers_base + 1)
        per_home_layer_sizes.append((per_home_layers_base + 1) * layer_size_bytes)
        extra_home_size = per_home_layer_sizes[i]
    elif i == head_id:
        per_home_num_blocks.append(per_home_layers_base)
        per_home_layer_sizes.append((per_home_layers_base) * layer_size_bytes + head_layer_size_bytes)
        head_home_size = per_home_layer_sizes[i]
    else:
        per_home_num_blocks.append(per_home_layers_base)
        per_home_layer_sizes.append(per_home_layers_base * layer_size_bytes)
        usual_home_size = per_home_layer_sizes[i]


print(f"Per Home Num Blocks: {per_home_num_blocks}")
print(f"Per Home Layer Sizes: {per_home_layer_sizes}")

total_activation_cnt_per_device = [int(total_train_chunks * per_home_num_blocks[i]) for i in range(N)]

dev_activation_stack = activations_capacity
home_activation_stack = [total_activation_cnt_per_device[i] - activations_capacity for i in range(N)]

typical_home_activation_size = home_activation_stack[0] * activation_size_bytes
   
per_home_total_size = [home_activation_stack[i] * activation_size_bytes + home_activation_stack[0] + 4 * (per_home_layer_sizes[i]) for i in range(N)]



memory_legend_text = (
    f"Memory Breakdown:\n\n\n"
    f" ----- USER INPUTS ----- \n"
    f" - Model Info:\n"
    f"     - Total Blocks (non-head): {total_layers}\n"
    f"     - Bitwidth: {bitwidth}\n"
    f"     - Attention Algo: {attn_type}\n"
    f"     - Dims:\n"
    f"         - Model Dim: {model_dim}\n"
    f"         - KV Dim: {kv_dim}\n"
    f"         - Per-Expert Dim: {expert_dim}\n"
    f"         - Num Experts: {num_experts}\n"
    f"         - Active Experts: {active_experts}\n"
    f"         - Vocab Size: {vocab_size}\n\n"
    f" - Training Parameters:\n"
    f"     - Sequence Length: {seqlen}\n"
    f"     - Training Token Ratio: {train_token_ratio}\n"
    f"     - Min Chunk Size: {min_chunk_size}\n\n"
    f" - Num Devices: {N}\n\n"
    f" - Max Device Memory Bytes: {max_device_memory_bytes / (1 << 30):.2f} GB\n\n\n"  
    f" ----- Full Training Overview ----- \n"
    f" - Memory Requirements:\n"
    f"     - Full-Model Memory Requirements:\n"
    f"         - Model Size: {train_model_size / (1 << 30):.2f} GB\n"
    f"         - Gradient Size: {train_gradient_size / (1 << 30):.2f} GB\n"
    f"         - Optimizer State Size: {(2 * train_model_size) / (1 << 30):.2f} GB\n"
    f"         - Activation Size: {train_activation_size / (1 << 30):.2f} GB\n"
    f"         - Context (Non-Train) Size: {train_context_size / (1 << 30):.2f} GB\n"
    f"     - TOTAL: {aggregate_memory_size / (1 << 30):.2f} GB\n\n"
    f" ----- Derived Configuration ----- \n"
    f" - Chunk Type: {chunk_type}\n"
    f"     - Chunk Size: {chunk_size}\n"
    f"         - Total Chunks: {total_chunks}\n"
    f"         - Train Chunks: {total_train_chunks}\n"
    f" - Chunk Memory Info (Layer-Wise):\n"
    f"     - Activation Size: {(activation_size_bytes)/ 1e6:.2f} MB\n" 
    f"     - Context Size: {(chunk_context_size_bytes)/ 1e6:.2f} MB\n"
    f"     - Output Size: {(output_size_bytes)/ 1e6:.2f} MB\n"
    f"     - Workspace Size: {(chunk_workspace_size)/ 1e6:.2f} MB\n\n"
    f" - Device Memory Partitions (Typical):\n"
    f"     - Activation Capacity: {activations_capacity}\n"
    f"         - Allocation: {(activations_capacity * activation_size_bytes)/ (1 << 30):.3f} GB\n"
    f"     - Layer Capacity: {layer_capacity}\n"
    f"         - Allocation: {(layer_capacity * layer_size_bytes)/ (1 << 30):.3f} GB\n"
    f"     - Grad Layer Capacity: {grad_layer_capacity}\n"
    f"         - Allocation: {(grad_layer_capacity * layer_size_bytes)/ (1 << 30):.3f} GB\n"
    f"     - Context Buffer Capacity: {context_buffer_capacity}\n"
    f"         - Allocation: {(context_buffer_capacity * per_layer_full_context_size)/ (1 << 30):.3f} GB\n"
    f"     - Grad Context Buffer Capacity: {grad_context_buffer_capacity}\n"
    f"         - Allocation: {(grad_context_buffer_capacity * per_layer_full_context_size)/ (1 << 30):.3f} GB\n"
    f"     - Transition Capacity (Inp/Out): {transitions_capacity}\n"
    f"         - Allocation: {2 * transitions_capacity * output_size_bytes / (1 << 30):.3f} GB\n\n"     
    f" *** TOTAL PER-DEVICE MEMORY: {typical_device_memory_size / (1 << 30):.2f} GB ***\n\n\n"
    f" - Home Memory Partitions (Typical):\n"
    f"     - Activation Size: {typical_home_activation_size / (1 << 30):.2f} GB\n"
    f"         - Home Activations Saved: {home_activation_stack[0]}\n"
    f"     - Model Shard Size: {per_home_layer_sizes[0] / (1 << 30):.2f} GB\n"
    f"     - Gradient Shard Size: {per_home_layer_sizes[0] / (1 << 30):.2f} GB\n"
    f"     - Optimizer State Size: {2 * per_home_layer_sizes[0] / (1 << 30):.2f} GB\n\n"
    f" *** TOTAL PER-HOME MEMORY: {per_home_total_size[0] / (1 << 30):.2f} GB ***\n\n"
)

matmul_flop_ratio = total_matmul_flops / total_flops if total_flops > 0 else 0
attn_flop_ratio = total_attn_flops / total_flops if total_flops > 0 else 0
practical_efficiency = matmul_efficiency * matmul_flop_ratio + attn_efficiency * attn_flop_ratio

compute_legend_text = (
    f"Compute Breakdown:\n\n\n"
    f" ----- USER INPUTS ----- \n"
    f" - Compute Constants:\n"
    f"     - Hardware Theoretical MAX: {int(hardware_max_flops / 1e12)} TFLOPs\n"
    f"     - Hardware Memory BW: {int(hardware_mem_bw_bytes_sec / (1 << 30))} GB/s\n"
    f"     - Peak Matmul Efficiency: {matmul_efficiency}\n"
    f"     - Peak Attn Efficiency: {attn_efficiency}\n"
    f" - Communication Constants:\n"
    f"     - Device-to-Home BW (Gb/s): {home_bw_gbit_sec}\n"
    f"     - Peer-to-Peer BW (Gb/s): {peer_bw_gbit_sec}\n\n"
    f" ----- Full Training Overview ----- \n"
    f" - FLOP Breakdown\n"     
    f"     - Total TFLOPs: {total_flops:.3e}\n"
    f"         - FWD TFLOPs: {total_fwd_flops:.3e}\n"
    f"         - Head TFLOPs: {total_head_flops:.3e}\n"
    f"         - BWD TFLOPs: {total_bwd_flops:.3e}\n"
    f"         - Overall Matmul TFLOPs: {total_matmul_flops:.3e}\n"
    f"         - Overall Attn TFLOPs: {total_attn_flops:.3e}\n\n"
    f" ----- Derived Simulation Config ----- \n"
    f" - Simulation Speed: {cycles_per_second / 1000:.3f} K cycles per second\n"
    f"   - C0 Computation: {computationFrames} Cycles\n"
    f"   - C{total_chunks-1} Computation: {max_computationFrames} Cycles\n"
    f"   - Head Computation: {headFrames} Cycles\n"
    f"   - BwdW Computation: {bwdWFrames} Cycles\n"
    f"   - Layer Transfer: {layerTransferFrames} Cycles\n"
    f"   - Head Transfer: {headTransferFrames} Cycles\n"
    f"   - Activation Transfer: {savedActivationsFrames} Cycles\n"
    f"   - Per-Chunk Context Transfer: {contextTransferCycleText}\n"
    f"   - Block Transition: {activationTransitionFrames} Cycles\n\n"
    f" *** RUNTIME LOWER-BOUND: {int(((total_matmul_flops / N) / (hardware_max_flops * matmul_efficiency) + (total_attn_flops / N) / (hardware_max_flops * attn_efficiency)) * cycles_per_second)} Cycles ***\n\n"
    f" *** THROUGHPUT UPPER-BOUND: {int((hardware_max_flops * practical_efficiency) / 1e12)} TFLOPS ***\n\n"
)

# Define padding from figure edges (e.g., 2% padding)
left_pad = 0.02
right_pad = 0.98
top_pad = 0.98
bottom_pad = 0.02 # Not used here, but for reference

# Memory Legend (Top Left)
at_memory = AnchoredText(
    memory_legend_text,
    loc='upper left',                     # Anchor point is the upper left of the text box
    bbox_to_anchor=(left_pad, top_pad),   # Position: slightly inset from top-left figure corner
    prop=dict(size=6),                    # Slightly larger text? Adjust as needed.
    frameon=True,
    pad=0.4,
    borderpad=0.5,
    bbox_transform=fig.transFigure        # IMPORTANT: Use figure coordinates
)
at_memory.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at_memory.patch.set_facecolor((1, 1, 1, 0.85)) # Slightly less transparent maybe?
at_memory.patch.set_edgecolor('black')
ax.add_artist(at_memory) # Still add it to the axes artist list

# Compute Legend (Top Right)
at_compute = AnchoredText(
    compute_legend_text,
    loc='upper right',                    # Anchor point is the upper right of the text box
    bbox_to_anchor=(right_pad, top_pad),  # Position: slightly inset from top-right figure corner
    prop=dict(size=6),                    # Slightly larger text? Adjust as needed.
    frameon=True,
    pad=0.4,
    borderpad=0.5,
    bbox_transform=fig.transFigure        # IMPORTANT: Use figure coordinates
)
at_compute.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at_compute.patch.set_facecolor((1, 1, 1, 0.85)) # Slightly less transparent maybe?
at_compute.patch.set_edgecolor('black')
ax.add_artist(at_compute) # Still add it to the axes artist list




# --- Initial Artist Setup ---
outer_circle_positions = []
inner_node_centers = []
stall_node_positions = []
unit_directions = []
node_transfer_distances = []
cmap = plt.get_cmap('rainbow_r')
norm = mcolors.Normalize(vmin=0, vmax=N - 1)
device_artists = {}
edge_artists = {}
device_label_artists = {}
initial_cycle = 0
title_obj = ax.set_title(f"Cycle {initial_cycle}", fontsize=title_fontsize, fontweight='bold')

# --- Create Nodes and Edges (Code unchanged from previous correct version) ---
for i in range(N):
    angle = np.linspace(0, 2 * np.pi, N, endpoint=False)[i]
    unit_dir = np.array([np.cos(angle), np.sin(angle)])
    unit_directions.append(unit_dir)
    outer_pos = center_pos + total_distance * unit_dir
    outer_circle_positions.append(outer_pos)
    color = cmap(norm(i))
    outer_circle = patches.Circle(outer_pos, radius=outer_node_radius, fc=color, ec='black', alpha=device_opacity, zorder=2)
    ax.add_patch(outer_circle)
    device_artists[f'circle_{i}'] = outer_circle
    outer_label = ax.text(outer_pos[0], outer_pos[1], f'D{i}', ha='center', va='center', fontsize=7, zorder=3)
    device_label_artists[f'circle_{i}'] = outer_label
    inner_center = center_pos + inner_radius * unit_dir
    inner_node_centers.append(inner_center)
    inner_square_side = inner_node_radius * np.sqrt(2)
    inner_square_bottom_left = inner_center - np.array([inner_square_side / 2, inner_square_side / 2])
    inner_square = patches.Rectangle(inner_square_bottom_left, inner_square_side, inner_square_side, fc=color, ec='black', alpha=inner_node_opacity, zorder=2)
    ax.add_patch(inner_square)
    device_artists[f'inner_square_{i}'] = inner_square
    inner_label = ax.text(inner_center[0], inner_center[1], f'D{i} Home', ha='center', va='center', fontsize=6, zorder=3)
    device_label_artists[f'inner_label_{i}'] = inner_label
    stall_node_pos = outer_pos + unit_dir * stall_node_distance_offset
    stall_node_positions.append(stall_node_pos)
    stall_node = RegularPolygon(stall_node_pos, numVertices=8, radius=stall_node_radius, orientation=np.pi/8, fc=COLOR_STALL_NODE_FILL, ec=color, lw=stall_node_border_width, zorder=2, visible=False)
    ax.add_patch(stall_node)
    device_artists[f'stall_node_{i}'] = stall_node
    stall_label = ax.text(stall_node_pos[0], stall_node_pos[1], "", ha='center', va='center', fontsize=stall_node_fontsize, fontweight='semibold', color='white', zorder=3, visible=False)
    device_label_artists[f'stall_label_{i}'] = stall_label
    finish_indicator_color = (0.0, 0.8, 0.0, stall_node_opacity) # Green, semi-transparent
    finish_indicator = patches.Circle(stall_node_pos, radius=stall_node_radius, fc=finish_indicator_color, ec=color, lw=stall_node_border_width, zorder=2, visible=False)
    ax.add_patch(finish_indicator)
    device_artists[f'finish_indicator_{i}'] = finish_indicator
    inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
    outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
    dist = np.linalg.norm(outer_edge_conn_point - inner_edge_conn_point)
    node_transfer_distances.append(dist)
arrow_style_str = f'-|>,head_length={head_len},head_width={head_wid}'
for i in range(N):
    unit_dir = unit_directions[i]
    inner_center = inner_node_centers[i]
    outer_pos = outer_circle_positions[i]
    inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
    outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
    radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
    edge_offset = radial_perp_vector * arrow_offset_dist
    start_pos_arrow_out = inner_edge_conn_point + edge_offset
    start_pos_arrow_in = outer_edge_conn_point - edge_offset
    arrow_out = patches.FancyArrowPatch(posA=start_pos_arrow_out, posB=start_pos_arrow_out, arrowstyle=arrow_style_str, color=COLOR_INBOUND_DEFAULT, linestyle='dashed', mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_out)
    label_out = ax.text(start_pos_arrow_out[0], start_pos_arrow_out[1], "", color=COLOR_INBOUND_DEFAULT, fontsize=edge_label_fontsize, ha='center', va='bottom', zorder=4)
    edge_artists[f'in_{i}'] = (arrow_out, label_out)
    arrow_in = patches.FancyArrowPatch(posA=start_pos_arrow_in, posB=start_pos_arrow_in, arrowstyle=arrow_style_str, color=COLOR_OUTBOUND_FWD_ACTIVATION, linestyle='dashed', mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_in)
    label_in = ax.text(start_pos_arrow_in[0], start_pos_arrow_in[1], "", color=COLOR_OUTBOUND_FWD_ACTIVATION, fontsize=edge_label_fontsize, ha='center', va='top', zorder=4)
    edge_artists[f'out_{i}'] = (arrow_in, label_in)
    start_pos_ring = outer_pos
    arrow_ring = patches.FancyArrowPatch(posA=start_pos_ring, posB=start_pos_ring, arrowstyle=arrow_style_str, color=COLOR_RING_CCW, linestyle='solid', mutation_scale=mut_scale, lw=edge_linewidth, zorder=1, connectionstyle=f"arc3,rad=0.2")
    ax.add_patch(arrow_ring)
    label_ring = ax.text(start_pos_ring[0], start_pos_ring[1], "", color=COLOR_RING_CCW, fontsize=edge_label_fontsize, ha='center', va='center', zorder=4)
    edge_artists[f'ring_{i}'] = (arrow_ring, label_ring)
    arc_radius = outer_node_radius * compute_arc_radius_scale
    compute_arc = Arc(outer_pos, width=2*arc_radius, height=2*arc_radius, angle=0, theta1=0.0, theta2=0.0, color=COLOR_COMPUTE_DEFAULT, lw=compute_arc_linewidth, zorder=1, visible=False)
    ax.add_patch(compute_arc)
    edge_artists[f'compute_{i}'] = compute_arc
world_angle_to_prev = np.zeros(N)
world_angle_to_next = np.zeros(N)
for i in range(N):
    pos_i = outer_circle_positions[i]
    prev_idx = (i - 1 + N) % N
    pos_prev = outer_circle_positions[prev_idx]
    vec_i_to_prev = pos_prev - pos_i
    world_angle_to_prev[i] = np.degrees(np.arctan2(vec_i_to_prev[1], vec_i_to_prev[0]))
    next_idx = (i + 1) % N
    pos_next = outer_circle_positions[next_idx]
    vec_i_to_next = pos_next - pos_i
    world_angle_to_next[i] = np.degrees(np.arctan2(vec_i_to_next[1], vec_i_to_next[0]))

# --- Device Class ---
class Device:
    def __init__(self, device_id, layer_capacity, activations_capacity, transitions_capacity, head_transitions_capacity, total_devices, total_layers, total_chunks):
        self.device_id = device_id
        self.device_has_started = False
        self.device_start_time = 0
        self.device_has_finished = False
        self.device_finish_time = 0 # Will mark when last compute task finishes
        self.layer_capacity = layer_capacity
        self.activations_capacity = activations_capacity
        self.transitions_capacity = transitions_capacity
        self.head_transitions_capacity = head_transitions_capacity
        self.total_devices = total_devices
        self.total_layers = total_layers
        self.total_chunks = total_chunks
        self.cur_weights = [-1 for _ in range(self.layer_capacity)]
        self.cur_weight_write_ptr = 0
        self.activations_buffer = [-1 for _ in range(self.activations_capacity)]
        self.activations_write_ptr = 0
        self.activations_empty_slot_ind = 0
        ## represents the order in which actiavtions are processed
        self.activations_stack = []
        self.cur_saved_activations_num = 0

        ## FOR NOW IGNORE THE 'RESULT' MEMORY SPACE, AND ASSUME HEAD DEVICE/LAST LAYER BLOCK
        ## CAN HOLD ALL THE REQUIRED TRANSITIONS....
        if (self.device_id == 0) or(self.device_id == self.total_layers % self.total_devices) or (((self.total_layers - 1) % self.total_devices) == self.device_id):
            self.transitions_capacity = self.total_chunks


        self.transitions_inbound_buffer = [-1 for _ in range(self.transitions_capacity)]
        self.transitions_outbound_buffer = [-1 for _ in range(self.transitions_capacity)]
        self.transitions_inbound_empty_slot_ind = 0
        self.transitions_outbound_empty_slot_ind = 0

        ## this is for where the last block puts its output transitions
        ## exists on device id = (head_device_id
        self.head_input_transitions_buffer = [-1 for _ in range(self.head_transitions_capacity)]
        self.head_input_transitions_empty_slot_ind = 0
        
        ## this is for where the head block puts its gradient transitions
        ## exists on device id = (head_device_id - 1) % self.total_devices
        self.head_output_transitions_buffer = [-1 for _ in range(self.head_transitions_capacity)]
        self.head_output_transitions_empty_slot_ind = 0
        
        self.context_buffer = [-1 for _ in range(self.total_chunks)]
        self.home_storage = set()
        self.computation_queue = []
        self.outbound_queue = []
        self.inbound_queue = []
        self.peer_transfer_queue = []
        self.is_computing = False
        self.is_stalled = False
        self.stall_start_time = 0
        self.cur_computation_start_time = 0
        self.cur_computation_duration = 0
        self.current_computation_type = None
        self.current_computation_layer_id = -1
        self.is_outbound_transferring = False
        self.cur_outbound_start_time = 0
        self.cur_outbound_duration = 0
        self.cur_outbound_edge = ""
        self.is_inbound_transferring = False
        self.cur_inbound_start_time = 0
        self.cur_inbound_duration = 0
        self.cur_inbound_edge = ""
        self.is_peer_transferring = False
        self.cur_peer_transfer_start_time = 0
        self.cur_peer_transfer_duration = 0
        self.cur_ring_edge = ""
        self.cur_peer_transfer_details = None
        self.computing_status = "Idle"
        self.stall_reason = ""
        

        ## 
        self.cur_fwd_computation_num = 0

        ## these represent the first activation that will be not be sent back to home storage
        ## but rather kept within activations buffer
        self.activations_stack_cutoff_ind = -1
        
        ## tracking the next items to prefetch
        ## weight applies to both forward and backward passes
        self.next_weight_prefetch_layer_id = -1
        self.next_bwd_weight_prefetch_layer_id = -1
        ## this applies to bwd pass only
        ## it represents the next item to be prefetched
        ## (and should be initialized to stack_cutoff_ind - 1)
        ## it is decremented until it reaches -1, at which point
        self.activations_stack_next_ind = -1
     
        ## this applies only to device containing the head...
        ## it is the chunk id of the last chunk the head will process
        ## the head process chunks in sequential order until the final
        ## chunk has transitioning inbound, in which case it will switch
        ## to processing chunks in reverse order
        self.head_final_chunk_id = -1
        self._initialize_tasks_and_state()

    def _initialize_tasks_and_state(self): # Unchanged        
        self.cur_weight_write_ptr = 0
        cur_layer_id = self.device_id

        total_activations = 0

        ## add all of the layers that this device needs
        cur_layer_id = self.device_id
        all_layers = [cur_layer_id]
        i = 0
        while cur_layer_id <= self.total_layers:
            self.home_storage.add((-1, cur_layer_id, False))
            all_layers.append(cur_layer_id)
            if i < self.layer_capacity:
                self.cur_weights[i] = (0, cur_layer_id)
            else:
                if i == self.layer_capacity:
                    self.next_weight_prefetch_layer_id = cur_layer_id
            cur_layer_id += self.total_devices
            i += 1

        ## initialize the bwd weight prefetch layer id
        if len(all_layers) > self.layer_capacity:
            self.next_bwd_weight_prefetch_layer_id = all_layers[len(all_layers) - self.layer_capacity - 1]
        else:
            self.next_bwd_weight_prefetch_layer_id = -1

        cur_layer_id = self.device_id
        ## forward pass
        while cur_layer_id < self.total_layers:
            transfer_direction = 1
            for i in range(self.total_chunks):
                self.computation_queue.append((i, cur_layer_id, False, False, transfer_direction, computation_times_frames[i]))
                if (i % train_chunk_freq == 0) or (i == self.total_chunks - 1):
                    self.activations_stack.append((i, cur_layer_id))
            ## strictly less as we will process head differently
            if cur_layer_id + self.total_devices < self.total_layers:
                cur_layer_id += self.total_devices
            else:
                ## here we can initialize the context buffer which should contain all chunks' context 
                ## for the last block layer...
                for i in range(self.total_chunks):
                    self.context_buffer[i] = (0, i, cur_layer_id)
                break
        
        total_activations = len(self.activations_stack)
        activation_ind_cutoff = total_activations - self.activations_capacity

        self.activations_stack_cutoff_ind = activation_ind_cutoff
        self.activations_stack_next_ind = activation_ind_cutoff

        ## add potential head task
        if cur_layer_id + self.total_devices == self.total_layers:
            
            transfer_direction = -1
            ## determine head cutoff
            ## this is heuristic to try and be productive as possible before the
            ## the final chunk has transitioned into the head device in which case we can 
            ## flip the direction of chunk processing

            total_chunk_inbound_frames = sum([computation_times_frames[i] for i in range(0, self.total_chunks, train_chunk_freq)])
            cutoff_chunk_id = int(total_chunk_inbound_frames / 2 / headFrames)
            """
            head_diff = headFrames - computation_times_frames[0]
            if head_diff <= 0:
                cutoff_chunk_id = self.total_chunks // 2
            else:
                
                cutoff_chunk_id = math.ceil((total_chunk_frames / 2) / headFrames)
            """
            
            print(f"Cutoff Chunk ID: {cutoff_chunk_id}")

            for i in range(cutoff_chunk_id):
                if (i % train_chunk_freq == 0) or (i == self.total_chunks - 1):
                    self.computation_queue.append((i, self.total_layers, False, False, transfer_direction, headFrames))
            for i in range(self.total_chunks - 1, cutoff_chunk_id - 1, -1):
                if (i % train_chunk_freq == 0) or (i == self.total_chunks - 1):
                    self.computation_queue.append((i, self.total_layers, False, False, transfer_direction, headFrames))
                    ## ensure we actually compute the final head chunk...
                    self.head_final_chunk_id = i
        
        ## now start backward pass...
        while cur_layer_id >= 0:
            if cur_layer_id > 0:
                transfer_direction = -1
            else:
                transfer_direction = 0
            for i in range(self.total_chunks - 1, -1, -1):
                if (i % train_chunk_freq == 0) or (i == self.total_chunks - 1):
                    ## add the BwdX first
                    self.computation_queue.append((i, cur_layer_id, True, False, transfer_direction, computation_times_frames_bwd[i]))
                    ## add the BwdW second
                    ## bwd W doesn't have an outbound transfer
                    self.computation_queue.append((i, cur_layer_id, False, True, 0, bwdWFrames))
            
            cur_layer_id -= self.total_devices
         
       

    def handle_completed_transfers(self, T, all_devices):
        if self.is_inbound_transferring and self.inbound_queue and (self.cur_inbound_start_time + self.cur_inbound_duration <= T):
            inbound_item = self.inbound_queue.pop(0)
            chunk_id, layer_id, is_grad, is_context, target_idx, duration = inbound_item
            if (chunk_id == -1):
                self.cur_weights[target_idx] = (0, layer_id)
            else:
                if is_context:
                    self.context_buffer[target_idx] = (0, chunk_id, layer_id)
                else:
                    self.activations_buffer[target_idx] = (0, chunk_id, layer_id)
            self.is_inbound_transferring = False
            self.cur_inbound_edge = ""

        if self.is_outbound_transferring and self.outbound_queue and (self.cur_outbound_start_time + self.cur_outbound_duration <= T):
            outbound_item = self.outbound_queue.pop(0)
            chunk_id, layer_id, is_grad, is_only_context, duration = outbound_item
            
            if (chunk_id >= 0) and (not is_only_context):
                activations_ind = self.activations_buffer.index((-2, chunk_id, layer_id))
                self.activations_buffer[activations_ind] = -1
                ## wasteful, but keeping orderly with setting -1 to be the first free slot in linear order...
                ## we know there will be at least one free slot because we just set one...
                first_free_idx = self.activations_buffer.index(-1)
                self.activations_empty_slot_ind = first_free_idx
            
            self.home_storage.add((chunk_id, layer_id, is_grad))
            self.is_outbound_transferring = False
            self.cur_outbound_edge = ""

        if self.is_peer_transferring and self.peer_transfer_queue and (self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T):
            
            peer_item = self.peer_transfer_queue.pop(0)
            peer_id, cid, lid, is_grad, duration = peer_item
            
            peer_dev = all_devices[peer_id]
           
            ## indicate as freed up in self outbound buffer
            outbound_idx_to_free = self.transitions_outbound_buffer.index((0, cid, lid, is_grad))
            self.transitions_outbound_buffer[outbound_idx_to_free] = -1
            ## we know there will be at least one free slot because we just set one...
            first_free_idx = self.transitions_outbound_buffer.index(-1)
            self.transitions_outbound_empty_slot_ind = first_free_idx

            
            ## indicate as completed in peer inbound buffer

            ## meant that this was a gradient transition from output,
            ## so the receiving device has special buffer to put it in
            if lid == self.total_layers:
                inbound_idx_to_update = peer_dev.head_output_transitions_buffer.index((-2, cid, lid, is_grad))
                peer_dev.head_output_transitions_buffer[inbound_idx_to_update] = (0, cid, lid, is_grad)
            else:
                ## this means that this is a forward transition from last block to the head
                ## so the head has a special buffer to put it in
                if lid == self.total_layers - 1 and not is_grad:
                    inbound_idx_to_update = peer_dev.head_input_transitions_buffer.index((-2, cid, lid, is_grad))
                    peer_dev.head_input_transitions_buffer[inbound_idx_to_update] = (0, cid, lid, is_grad)
                else:
                    inbound_idx_to_update = peer_dev.transitions_inbound_buffer.index((-2, cid, lid, is_grad))
                    peer_dev.transitions_inbound_buffer[inbound_idx_to_update] = (0, cid, lid, is_grad)

            self.is_peer_transferring = False
            self.cur_ring_edge = ""
            self.cur_peer_transfer_details = None

    def handle_computation_depends(self, T): # Unchanged logic, refactored style
        if not self.computation_queue:
            return False
        
        next_task = self.computation_queue[0]
        cid, lid, bX, bW, tdir, dur = next_task
        has_deps = False
        is_fwd = (not bX) and (not bW)

        self.stall_reason = ""

        if is_fwd:
            if lid == self.total_layers:
                computation_type_str = "Head"
                weight_str = "Wgt: Head"
            else:
                computation_type_str = "Fwd"
                weight_str = f"Wgt: L{lid}"
            has_weight = ((0, lid) in self.cur_weights)

            ## set to true for layer 0
            has_input_transition = True
            input_transition_key = None
            if lid > 0:
                input_transition_key = (0, cid, lid - 1, False)
                if lid < self.total_layers:
                    has_input_transition = input_transition_key in self.transitions_inbound_buffer
                else:
                    has_input_transition = input_transition_key in self.head_input_transitions_buffer
            has_deps = has_weight and has_input_transition
            if not has_deps:
                if not has_weight and not has_input_transition:
                    self.stall_reason = f"Missing:\nWgt\nAct. Stream"
                elif not has_weight:
                    self.stall_reason = f"Missing:\nWgt"
                else:
                    self.stall_reason = f"Missing:\nAct. Stream"
        elif bX:
            computation_type_str = "Bwd X"
            has_weight = ((0, lid) in self.cur_weights) 
            upstream_grad_key = (0, cid, lid + 1, True)
            context_key = (0, 0, lid)
            if lid < self.total_layers - 1:
                has_upstream_grad = upstream_grad_key in self.transitions_inbound_buffer
            else:
                has_upstream_grad = upstream_grad_key in self.head_output_transitions_buffer
            fwd_act_key = (0, cid, lid)
            has_fwd_activation = fwd_act_key in self.activations_buffer
            has_context = context_key in self.context_buffer
            has_deps = has_weight and has_upstream_grad and has_context and has_fwd_activation
            if not has_deps:
                missing = []
                if not has_weight:
                    missing.append(f"Wgt")
                    print(f"Missing Wgt")
                    print(f"Cur Weights: {self.cur_weights}")
                    print(f"Cur Weight Write Ptr: {self.cur_weight_write_ptr}")
                    print(f"Cur Inbound Queue: {self.inbound_queue}")
                    print("\n\n\n\n\n")
                if not has_context:
                    missing.append(f"Fwd Context")
                if not has_upstream_grad:
                    missing.append(f"Grad Stream")
                if not has_fwd_activation:
                    missing.append(f"Fwd Act")
                self.stall_reason = "Missing:\n" + "\n".join(missing)
        elif bW:
            computation_type_str = "Bwd W"
            ## assuming gradient activation is always present, 
            ## because this is previously computied for bX and 
            ## we always scheduling bW immediately after bX
            ## so we reuse the same grad activation buffer
            has_fwd_activation = ((0, cid, lid) in self.activations_buffer)
            has_deps = has_fwd_activation
            if not has_deps:
                self.stall_reason = "Missing:\nFwd Act"
        
        

        ## ensure that there is available space in the activations buffer if during fwd pass...
        if is_fwd and self.activations_empty_slot_ind is None:
            self.stall_reason = "Congested:\nAct. Buffer Full"
            has_deps = False

        ## ensure that there is room in the transition outbound buffer
        has_outbound_transition = True
        if bW or (bX and lid == 0):
            has_outbound_transition = False
        if has_deps and has_outbound_transition and self.transitions_outbound_empty_slot_ind is None:
            self.stall_reason = "Congested:\nOutbound Transition Buffer Full"
            has_deps = False


        if has_deps:
            if not self.device_has_started:
                self.device_start_time = T
                self.device_has_started = True
            
            if self.is_stalled and TO_PRINT:
                print(f"T={T}, Dev {self.device_id}: UNSTALL -> Comp C{cid},L{lid},{computation_type_str}. Stalled for {T - self.stall_start_time} cycles.")

            self.cur_computation_start_time = T
            self.cur_computation_duration = dur
            self.is_computing = True
            self.is_stalled = False
            self.stall_reason = ""
            self.current_computation_type = computation_type_str
            self.current_computation_layer_id = lid
            self.computing_status = f"COMPUTING:\n{computation_type_str}\nC{cid},L{lid}"

            is_grad_out = (not is_fwd) or (lid == self.total_layers)
            
            if has_outbound_transition:
                self.transitions_outbound_buffer[self.transitions_outbound_empty_slot_ind] = (-2, cid, lid, is_grad_out)
                if -1 in self.transitions_outbound_buffer:
                    self.transitions_outbound_empty_slot_ind = self.transitions_outbound_buffer.index(-1)
                else:
                    self.transitions_outbound_empty_slot_ind = None
            
        else:
            if not self.is_stalled:
                self.is_stalled = True
                self.stall_start_time = T
                self.computing_status = f"STALL:\n{computation_type_str}\nC{cid},L{lid}"
            self.is_computing = False
            self.current_computation_type = None
            self.current_computation_layer_id = -1
            return False

    def handle_bwd_prefetch_weight(self):
        
        ## after last chunk is processed on head, replace head weight with prior weight
        #W assumes that the 
        if self.next_bwd_weight_prefetch_layer_id >= 0:
            self.inbound_queue.append((-1, self.next_bwd_weight_prefetch_layer_id, False, False, self.cur_weight_write_ptr, layerTransferFrames))
            self.cur_weights[self.cur_weight_write_ptr] = (-2, self.next_bwd_weight_prefetch_layer_id)
            self.next_bwd_weight_prefetch_layer_id -= self.total_devices
             ## work backwards when replacing weights....
            self.cur_weight_write_ptr = (self.cur_weight_write_ptr - 1) % self.layer_capacity
        return

    def handle_bwd_prefetch_context(self, chunk_id, cur_layer_id, next_layer_id):
        ## prefetch the context for the next layer
        if self.context_buffer[chunk_id] != (0, chunk_id, cur_layer_id):
            print("Trying to prefetch next context, but chunk id not on current layer...\n")
            return
        
        ## if activations buffer was deep enought to already contain context,
        ## then we don't need to prefetch it...
        if (0, chunk_id, next_layer_id) in self.activations_buffer:
            self.context_buffer[chunk_id] = (0, chunk_id, next_layer_id)
            return
        
        ## otherwise, we need to prefetch the context...
        self.context_buffer[chunk_id] = (-2, chunk_id, next_layer_id)
        self.inbound_queue.append((chunk_id, next_layer_id, False, True, chunk_id, contextTransferFrames))
        return

    def handle_bwd_prefetch_fwd_act(self):
        if self.activations_stack_next_ind >= 0:
            cid, lid = self.activations_stack[self.activations_stack_next_ind]
            next_act_idx = self.activations_empty_slot_ind
            self.inbound_queue.append((cid, lid, False, False, next_act_idx, savedActivationsFrames))
            self.activations_buffer[next_act_idx] = (-2, cid, lid)
            if -1 in self.activations_buffer:
                self.activations_empty_slot_ind = self.activations_buffer.index(-1)
            else:
                self.activations_empty_slot_ind = None
            self.activations_stack_next_ind -= 1
        return

    def handle_computation(self, T):
        completed_tasks = 0
        if not self.is_computing and not self.is_stalled and len(self.computation_queue) > 0:
            self.handle_computation_depends(T)

        ## finished computing!
        elif self.is_computing and (self.cur_computation_start_time + self.cur_computation_duration <= T):
            task = self.computation_queue.pop(0)
            completed_tasks += 1
            cid, lid, bX, bW, tdir, dur = task
            is_head = (lid == self.total_layers)
            is_fwd = (not bX) and (not bW) and (not is_head)
            
            task_type_str = self.current_computation_type


            if TO_PRINT:
                print(f"T={T}, Dev {self.device_id}: FINISHED Comp -> C{cid},L{lid},{task_type_str}")

            ## remove this from the transitoins inbound buffer now!
            
            peer_id = (self.device_id + tdir) % self.total_devices
            is_grad_in = bX or bW
            is_grad_out = bX or is_head
            
            depend_transition = lid + 1 if is_grad_in else lid - 1

            ## handle the transition buffers....
            
            ## makring input transition as free...

            ## remove the input transition from the inbound buffer

            ## no inp/out peer transitions for bW, computed immediately after bX...

            ## the input transition is used doring bW which follows bX,
            ## co can't clear it until bW is done...
            if (not bX) and (lid > 0 or is_grad_in):

                if (lid == self.total_layers):
                    inp_idx_to_free = self.head_input_transitions_buffer.index((0, cid, depend_transition, is_grad_in))
                    self.head_input_transitions_buffer[inp_idx_to_free] = -1
                    first_free_idx = self.head_input_transitions_buffer.index(-1)
                    self.head_input_transitions_empty_slot_ind = first_free_idx
                elif (lid == self.total_layers - 1) and is_grad_in:
                    inp_idx_to_free = self.head_output_transitions_buffer.index((0, cid, depend_transition, is_grad_in))
                    self.head_output_transitions_buffer[inp_idx_to_free] = -1
                    first_free_idx = self.head_output_transitions_buffer.index(-1)
                    self.head_output_transitions_empty_slot_ind = first_free_idx
                else:
                    inp_idx_to_free = self.transitions_inbound_buffer.index((0, cid, depend_transition, is_grad_in))
                    self.transitions_inbound_buffer[inp_idx_to_free] = -1
                    ## wasteful, but keeping orderly with setting -1 to be the first free slot in linear order...
                    ## we know there will be at least one free slot because we just set one...
                    first_free_idx = self.transitions_inbound_buffer.index(-1)
                    self.transitions_inbound_empty_slot_ind = first_free_idx
            
            ## we can clear the output transition as soon as fwd, bX, or head is done...
            ## no output transition for bW, computed immediately after bX...

            has_outbound_transition = True
            if bW or (bX and lid == 0):
                has_outbound_transition = False
            ## don't pass non training chunks to head...
            if is_fwd and (lid == self.total_layers - 1) and (cid % train_chunk_freq != 0) and (cid != self.total_chunks - 1):
                has_outbound_transition = False
            if has_outbound_transition:
                out_idx_to_update = self.transitions_outbound_buffer.index((-2, cid, lid, is_grad_out))
                self.transitions_outbound_buffer[out_idx_to_update] = (0, cid, lid, is_grad_out)
                ## append this to the peer transfer queue
                self.peer_transfer_queue.append((peer_id, cid, lid, is_grad_out, activationTransitionFrames))
            


            if is_fwd: # FWD Finished

                ## determine if we need to send back activations or not...
                if ((cid % train_chunk_freq == 0) or (cid == self.total_chunks - 1)):
                    ## now we need to save the activations in activation buffer
                    ## if at the tail end of forward computation, save the activations down!
                    if (self.cur_saved_activations_num > self.activations_stack_cutoff_ind):
                        self.activations_buffer[self.activations_empty_slot_ind] = (0, cid, lid)
                    else:
                        ## mark as moving into home storage
                        self.activations_buffer[self.activations_empty_slot_ind] = (-2, cid, lid)
                        self.outbound_queue.append((cid, lid, False, False, savedActivationsFrames))
                        self.cur_saved_activations_num += 1

                    if -1 in self.activations_buffer:
                        self.activations_empty_slot_ind = self.activations_buffer.index(-1)
                    else:
                        self.activations_empty_slot_ind = None
                else:
                    self.outbound_queue.append((cid, lid, False, True, contextTransferFrames))

                ## check to see if we finished a layer and should prefetch next weights...
                if (cid == self.total_chunks - 1) and self.next_weight_prefetch_layer_id <= self.total_layers:
                    layer_transfer_time = layerTransferFrames
                    if self.next_weight_prefetch_layer_id == self.total_layers:
                        layer_transfer_time = headTransferFrames
                    self.inbound_queue.append((-1, self.next_weight_prefetch_layer_id, False, False, self.cur_weight_write_ptr, layer_transfer_time))
                    self.cur_weights[self.cur_weight_write_ptr] = (-2, self.next_weight_prefetch_layer_id)
                    self.next_weight_prefetch_layer_id += self.total_devices
                    ## if we know we have a valid next weight prefetch layer id,
                    ## then we can increment the weight write pointer

                    ## otherwise this was our last weight, and we want to evict starting from the location we just
                    ## fetched in...
                    if (self.next_weight_prefetch_layer_id <= self.total_layers):
                        self.cur_weight_write_ptr = (self.cur_weight_write_ptr + 1) % self.layer_capacity

                self.cur_fwd_computation_num += 1
            elif is_head: # Head finished                
                if (cid == self.head_final_chunk_id):
                    self.handle_bwd_prefetch_weight()
            elif bX: # BwdX finished, can replace current context now

                if (lid - self.total_devices >= 0):
                    self.handle_bwd_prefetch_context(cid, lid, lid - self.total_devices)

                ## if this is the last chunk, now prefetch the next required weight
                ## which replaces the current weight
                if cid == 0:
                    self.handle_bwd_prefetch_weight()

            elif bW: # BwdW finished, can prefetch the next required activaton
                ## now prefetch the next required activation, which replaces the current activation

                ## set current activation we just used to be -1...
                act_idx = self.activations_buffer.index((0, cid, lid))
                self.activations_buffer[act_idx] = -1
                self.activations_empty_slot_ind = self.activations_buffer.index(-1)

                self.handle_bwd_prefetch_fwd_act()

                if (cid == 0):
                    self.outbound_queue.append((-1, lid, True, False, layerTransferFrames))

            # Reset compute state after any task finishes
            self.is_computing = False
            self.computing_status = "Idle"
            self.current_computation_type = None
            self.current_computation_layer_id = -1

            # Immediately try to start the next task if available
            if len(self.computation_queue) > 0:
                self.handle_computation_depends(T)
            else: # No more compute tasks for this device
                if not self.device_has_finished:
                    self.device_finish_time = T # Record time last compute task finished
                    self.device_has_finished = True
                    self.computing_status = "Finished Comp" # Indicate compute done, may still transfer
                    if TO_PRINT:
                        print(f"T={T}, Dev {self.device_id}: >>> FINISHED ALL COMPUTE TASKS <<<")

        elif self.is_stalled:
            # Re-check dependencies if stalled
            self.handle_computation_depends(T)

        return completed_tasks
            

    def handle_new_transfers(self, T, all_devices):
        if not self.is_inbound_transferring and self.inbound_queue:
            item = self.inbound_queue[0]
            chunk_id, layer_id, is_grad, is_context, target_idx, duration = item

            if chunk_id == -1: # Weight or Head state fetch
                 actual_storage_key = (-1, layer_id, is_grad)
            else: # Activation fetch
                 actual_storage_key = (chunk_id, layer_id, False)

            is_in_storage = actual_storage_key in self.home_storage
            if is_in_storage:
                self.is_inbound_transferring = True
                self.cur_inbound_start_time = T
                self.cur_inbound_duration = duration

                edge_color = COLOR_INBOUND_DEFAULT
                if chunk_id == -1:
                     if layer_id == total_layers:
                         label_lid_str = f"Wgt:\nHead"
                     else:
                         label_lid_str = f"Wgt:\nL{layer_id}"
                     self.cur_inbound_edge = label_lid_str
                     edge_color = COLOR_INBOUND_WEIGHT
                else:
                    if is_context:
                        self.cur_inbound_edge = f"Ctx:\nC{chunk_id},L{layer_id}"
                        edge_color = COLOR_INBOUND_BWD_FETCHED_CTX
                    else:
                        self.cur_inbound_edge = f"Act:\nC{chunk_id},L{layer_id}"
                        edge_color = COLOR_INBOUND_BWD_FETCHED_ACTIVATION


                arrow_vis, _ = edge_artists[f'in_{self.device_id}']
                arrow_vis.set_color(edge_color)
            else:
                if TO_PRINT:
                    print(f"T={T}, Dev {self.device_id}: ERROR - Inbound request {item} with key {actual_storage_key} not found in storage. Removing from queue.")

        if not self.is_outbound_transferring and self.outbound_queue:
            item = self.outbound_queue[0]
            chunk_id, layer_id, is_grad, is_only_context, duration = item
            self.is_outbound_transferring = True
            self.cur_outbound_start_time = T
            self.cur_outbound_duration = duration

            edge_color = COLOR_OUTBOUND_DEFAULT
            if chunk_id >= 0: # Activation save
                if not is_only_context:
                    self.cur_outbound_edge = f"Act:\nC{chunk_id},L{layer_id}"
                    edge_color = COLOR_OUTBOUND_FWD_ACTIVATION
                else:
                    self.cur_outbound_edge = f"Ctx:\nC{chunk_id},L{layer_id}"
                    edge_color = COLOR_OUTBOUND_FWD_CTX
            elif is_grad: # Weight Gradient save
                if layer_id == total_layers:
                     label_lid_str = f"Grad:\nHead"
                else:
                     label_lid_str = f"Grad:\nL{layer_id}"
                self.cur_outbound_edge = label_lid_str
                edge_color = COLOR_OUTBOUND_WGT_GRAD
            else: # Weight save (shouldn't happen often unless evicting?)
                self.cur_outbound_edge = f"UNKNOWN?"
                edge_color = COLOR_OUTBOUND_DEFAULT # Or maybe a different color?

            arrow_vis, _ = edge_artists[f'out_{self.device_id}']
            arrow_vis.set_color(edge_color)

        if not self.is_peer_transferring and self.peer_transfer_queue:
            item = self.peer_transfer_queue[0]
            peer_id, chunk_id, layer_id, is_grad_trans, duration = item

            ## need to check if the peer device has room for the transfer
            peer_dev = all_devices[peer_id]
            
            ## this means a graident transition coming from the head going to prior last block
            if layer_id == self.total_layers:
                peer_empty_slot_ind = peer_dev.head_output_transitions_empty_slot_ind
                peer_trans_buffer = peer_dev.head_output_transitions_buffer
            else:
                ## this means a forward transition coming from last block to the head
                if layer_id == self.total_layers - 1 and not is_grad_trans:
                    peer_empty_slot_ind = peer_dev.head_input_transitions_empty_slot_ind
                    peer_trans_buffer = peer_dev.head_input_transitions_buffer
                else:
                    peer_empty_slot_ind = peer_dev.transitions_inbound_empty_slot_ind
                    peer_trans_buffer = peer_dev.transitions_inbound_buffer
            if peer_empty_slot_ind is None:
                if TO_PRINT:
                    print(f"T={T}, Dev {self.device_id}: ERROR - Peer device {peer_id} has no empty slots in inbound buffer. Skipping transfer this cycle.")
                return
            
            ## indicate that the peer device is receiving data
            peer_trans_buffer[peer_empty_slot_ind] = (-2, chunk_id, layer_id, is_grad_trans)

            ## get next empty slot
            if -1 in peer_trans_buffer:
                first_free_idx = peer_trans_buffer.index(-1)
                if layer_id == self.total_layers:
                    peer_dev.head_output_transitions_empty_slot_ind = first_free_idx
                elif layer_id == self.total_layers - 1 and not is_grad_trans:
                    peer_dev.head_input_transitions_empty_slot_ind = first_free_idx
                else:
                    peer_dev.transitions_inbound_empty_slot_ind = first_free_idx
            else:
                if layer_id == self.total_layers:
                    peer_dev.head_output_transitions_empty_slot_ind = None
                elif layer_id == self.total_layers - 1 and not is_grad_trans:
                    peer_dev.head_input_transitions_empty_slot_ind = None
                else:
                    peer_dev.transitions_inbound_empty_slot_ind = None
            

            self.is_peer_transferring = True
            self.cur_peer_transfer_start_time = T
            self.cur_peer_transfer_duration = duration
            self.cur_peer_transfer_details = (peer_id, chunk_id, layer_id, is_grad_trans)

            if layer_id == total_layers:
                 label_lid_str = "Head"
            else:
                 label_lid_str = str(layer_id)

            if is_grad_trans:
                edge_color = COLOR_RING_CW
                connection_style_ring = f"arc3,rad=-0.2" # Negative radius for CW arc
                if layer_id == total_layers:
                    self.cur_ring_edge = f"Grad:\nC{chunk_id},Head"
                else:
                    self.cur_ring_edge = f"Grad:\nC{chunk_id},L{layer_id}"
            else: # Forward activation/output transfer
                edge_color = COLOR_RING_CCW
                connection_style_ring = f"arc3,rad=0.2" # Positive radius for CCW arc
                self.cur_ring_edge = f"Out:\nC{chunk_id},L{layer_id}"

            ## don't draw self ring edge, makes it look weird
            ## only applies for N = 1...
            if peer_id == self.device_id:
                self.cur_ring_edge = ""

            arrow_vis, _ = edge_artists[f'ring_{self.device_id}']
            arrow_vis.set_color(edge_color)
            arrow_vis.set_connectionstyle(connection_style_ring)


# --- Global Simulation State ---
all_devices = {}
total_tasks = 0
total_completed_tasks = {}
total_computation_time = 0
current_frame_index = 0
animation_paused = False
completion_text_artist = None
target_cycle = None
# ***** FIX: Track exact computation completion time *****
simulation_computation_finish_time = None
# ******************************************************

# --- Simulation Reset Function ---
def reset_simulation():
    """ Resets the simulation state and visual elements. """
    global all_devices, total_tasks, total_completed_tasks, total_computation_time
    global current_frame_index, animation_paused, completion_text_artist, target_cycle
    global simulation_computation_finish_time
    global N

    if TO_PRINT:
        print("\n" + "="*20 + " Resetting Simulation " + "="*20)

    all_devices = {i: Device(i, layer_capacity, activations_capacity, transitions_capacity, head_transitions_capacity,
                             N, total_layers, total_chunks)
                   for i in range(N)}

    total_tasks = sum(len(d.computation_queue) for d in all_devices.values())
    total_computation_time = sum(task[-1] for i in range(N) for task in all_devices[i].computation_queue)

    total_completed_tasks = {-1: 0} # Initialize with base case for frame 0
    current_frame_index = 0
    animation_paused = False
    target_cycle = None
    simulation_computation_finish_time = None

    if completion_text_artist is not None:
        if completion_text_artist.axes is not None:
            completion_text_artist.remove()
        completion_text_artist = None

    # Reset visuals
    for i in range(N):
        unit_dir = unit_directions[i]
        inner_center = inner_node_centers[i]
        outer_pos = outer_circle_positions[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
        edge_offset = radial_perp_vector * arrow_offset_dist
        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
        start_pos_arrow_out = inner_edge_conn_point + edge_offset
        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
        start_pos_arrow_in = outer_edge_conn_point - edge_offset
        start_pos_ring = outer_pos

        arrow_in_vis, label_in_vis = edge_artists[f'in_{i}']
        label_in_vis.set_text("")
        arrow_in_vis.set_color(COLOR_INBOUND_DEFAULT)
        arrow_in_vis.set_positions(start_pos_arrow_out, start_pos_arrow_out)

        arrow_out_vis, label_out_vis = edge_artists[f'out_{i}']
        label_out_vis.set_text("")
        arrow_out_vis.set_color(COLOR_OUTBOUND_FWD_ACTIVATION)
        arrow_out_vis.set_positions(start_pos_arrow_in, start_pos_arrow_in)

        arrow_ring, label_ring = edge_artists[f'ring_{i}']
        label_ring.set_text("")
        arrow_ring.set_color(COLOR_RING_CCW)
        arrow_ring.set_positions(start_pos_ring, start_pos_ring)
        arrow_ring.set_connectionstyle(f"arc3,rad=0.2") # Default to CCW

        if f'stall_node_{i}' in device_artists:
            device_artists[f'stall_node_{i}'].set_visible(False)
        if f'stall_label_{i}' in device_label_artists:
            device_label_artists[f'stall_label_{i}'].set_text("")
            device_label_artists[f'stall_label_{i}'].set_visible(False)

        # --- ADDED: Reset Finish Indicator ---
        if f'finish_indicator_{i}' in device_artists:
             device_artists[f'finish_indicator_{i}'].set_visible(False)
        # --- END ADDED ---

        if f'circle_{i}' in device_label_artists:
            # Initial status should be Idle
            device_label_artists[f'circle_{i}'].set_text(f'D{i}\nIdle')
        if f'inner_label_{i}' in device_label_artists:
            device_label_artists[f'inner_label_{i}'].set_text(f'D{i}\nHome')
        if f'compute_{i}' in edge_artists:
            compute_arc = edge_artists[f'compute_{i}']
            compute_arc.set_visible(False)
            compute_arc.theta1 = 0.0
            compute_arc.theta2 = 0.0

    title_obj.set_text(f'Cycle {current_frame_index}')
    if TO_PRINT:
        print(f"Reset complete. Total tasks: {total_tasks}")
        print("="*60 + "\n")

# --- Update Function ---
def update(frame):
    """ Main animation loop function, called for each frame (cycle). """
    global all_devices, total_completed_tasks, current_frame_index, completion_text_artist
    global animation_paused, target_cycle
    # ***** FIX: Use new state variable *****
    global simulation_computation_finish_time
    # *************************************

    # --- Initial Checks & Artist Setup ---
    if animation_paused:
        # Collect all artists to ensure they are returned for blitting if needed
        all_artists = [title_obj]
        for i in range(N):
              all_artists.extend([
                  device_label_artists[f'circle_{i}'], device_label_artists[f'inner_label_{i}'],
                  device_artists[f'stall_node_{i}'], device_label_artists[f'stall_label_{i}'],
                   device_artists[f'finish_indicator_{i}'],
                  edge_artists[f'in_{i}'][0], edge_artists[f'in_{i}'][1],
                  edge_artists[f'out_{i}'][0], edge_artists[f'out_{i}'][1],
                  edge_artists[f'ring_{i}'][0], edge_artists[f'ring_{i}'][1],
                  edge_artists[f'compute_{i}']
              ])
        if completion_text_artist:
            all_artists.append(completion_text_artist)
        return all_artists # Return current state when paused

    T = current_frame_index

    # Check if target cycle reached
    if target_cycle is not None and T == target_cycle:
        print(f"Reached target cycle {T}, pausing.")
        if ani.event_source is not None:
            ani.event_source.stop()
        animation_paused = True
        target_cycle = None # Clear target once reached
        title_obj.set_text(f'Cycle {T} (Paused)')
        fig.canvas.draw_idle()
        # Return artists even when pausing here
        all_artists = [title_obj]
        for i in range(N):
             all_artists.extend([
                 device_label_artists[f'circle_{i}'], device_label_artists[f'inner_label_{i}'],
                 device_artists[f'stall_node_{i}'], device_label_artists[f'stall_label_{i}'],
                 device_artists[f'finish_indicator_{i}'],
                 edge_artists[f'in_{i}'][0], edge_artists[f'in_{i}'][1],
                 edge_artists[f'out_{i}'][0], edge_artists[f'out_{i}'][1],
                 edge_artists[f'ring_{i}'][0], edge_artists[f'ring_{i}'][1],
                 edge_artists[f'compute_{i}']
             ])
        if completion_text_artist:
            all_artists.append(completion_text_artist)
        return all_artists

    # --- Update Title and Task Count ---
    title_obj.set_text(f'Cycle {T}')
    artists_to_update = [title_obj]

    # Ensure task count exists for the current frame, inheriting from previous if needed
    if T not in total_completed_tasks:
        last_known_frame = T - 1
        if last_known_frame < 0:
             last_known_frame = -1 # Handle base case T=0

        total_completed_tasks[T] = total_completed_tasks.get(last_known_frame, 0)


    # --- Simulation Logic Order ---
    # 1. Complete Transfers: Check for finished transfers and update buffers.
    #    Peer completion directly writes to the receiver's buffer.
    for i in range(N):
        all_devices[i].handle_completed_transfers(T, all_devices)

    # 2. Handle Computation: Check for finished computations, process outputs,
    #    check dependencies for next task, handle stalls.
    newly_completed_this_cycle = 0
    for i in range(N):
        newly_completed_this_cycle += all_devices[i].handle_computation(T)
    # Update total completed tasks *after* all devices have potentially finished a task
    total_completed_tasks[T] = total_completed_tasks.get(T, 0) + newly_completed_this_cycle

    # 3. Start New Transfers: Initiate pending transfers if channels are free.
    for i in range(N):
        all_devices[i].handle_new_transfers(T, all_devices)

    # --- Update Visuals ---
    for i in range(N):
        device = all_devices[i]
        unit_dir = unit_directions[i]
        inner_center = inner_node_centers[i]
        outer_pos = outer_circle_positions[i]
        transfer_dist_i = node_transfer_distances[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
        edge_offset = radial_perp_vector * arrow_offset_dist

        # Update Device Labels and Stall Nodes
        outer_label_artist = device_label_artists[f'circle_{i}']
        outer_label_artist.set_text(f'D{i}\n{device.computing_status}')
        inner_label_artist = device_label_artists[f'inner_label_{i}']
        inner_label_artist.set_text(f"D{i}\nHome")
        stall_node_artist = device_artists[f'stall_node_{i}']
        stall_label_artist = device_label_artists[f'stall_label_{i}']
        finish_indicator_artist = device_artists[f'finish_indicator_{i}'] # Get the new artist

        if device.device_has_finished:
            # Show finish indicator and hide stall indicator
            finish_indicator_artist.set_visible(True)
            stall_node_artist.set_visible(False)
            stall_label_artist.set_visible(False)
        elif device.is_stalled and device.stall_reason:
            # Show stall indicator and hide finish indicator
            #wrapped_stall_reason = textwrap.fill(device.stall_reason.replace("\n", " "), width=15)
            stall_label_artist.set_text(device.stall_reason)
            stall_label_artist.set_visible(True)
            stall_node_artist.set_visible(True)
            finish_indicator_artist.set_visible(False)
        else:
            # Hide both stall and finish indicators
            stall_label_artist.set_text("")
            stall_label_artist.set_visible(False)
            stall_node_artist.set_visible(False)
            finish_indicator_artist.set_visible(False)

        # Update Arrows (Inbound, Outbound, Ring)
        arrow_vis_inbound, label_vis_inbound = edge_artists[f'in_{i}']
        arrow_vis_outbound, label_vis_outbound = edge_artists[f'out_{i}']
        arrow_ring, label_ring = edge_artists[f'ring_{i}']

        # Inbound Arrow Update
        len_in_prog = 0.0
        cur_inbound_edge_text = ""
        color_inbound = arrow_vis_inbound.get_edgecolor() # Get current color set by handle_new_transfers
        if device.is_inbound_transferring and device.cur_inbound_duration > 0:
            prog_frac = min(1.0, (T - device.cur_inbound_start_time) / device.cur_inbound_duration)
            len_in_prog = prog_frac * transfer_dist_i
            cur_inbound_edge_text = device.cur_inbound_edge
        label_vis_inbound.set_color(color_inbound)
        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
        start_vis_inbound = inner_edge_conn_point + edge_offset
        end_vis_inbound = start_vis_inbound + unit_dir * len_in_prog
        arrow_vis_inbound.set_positions(start_vis_inbound, end_vis_inbound)
        label_perp_offset_in = radial_perp_vector * label_offset_distance
        midpoint_vis_in = (start_vis_inbound + end_vis_inbound) / 2
        label_pos_vis_in = midpoint_vis_in + label_perp_offset_in * 2 # Offset label perpendicular to arrow
        label_vis_inbound.set_position(label_pos_vis_in)
        label_vis_inbound.set_text(cur_inbound_edge_text)

        # Outbound Arrow Update
        len_out_prog = 0.0
        cur_outbound_edge_text = ""
        color_outbound = arrow_vis_outbound.get_edgecolor() # Get current color
        if device.is_outbound_transferring and device.cur_outbound_duration > 0:
            prog_frac = min(1.0, (T - device.cur_outbound_start_time) / device.cur_outbound_duration)
            len_out_prog = prog_frac * transfer_dist_i
            cur_outbound_edge_text = device.cur_outbound_edge
        label_vis_outbound.set_color(color_outbound)
        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
        start_vis_outbound = outer_edge_conn_point - edge_offset
        end_vis_outbound = start_vis_outbound - unit_dir * len_out_prog
        arrow_vis_outbound.set_positions(start_vis_outbound, end_vis_outbound)
        label_perp_offset_out = radial_perp_vector * label_offset_distance
        midpoint_vis_out = (start_vis_outbound + end_vis_outbound) / 2
        label_pos_vis_out = midpoint_vis_out - label_perp_offset_out * 2 # Offset label perpendicular
        label_vis_outbound.set_position(label_pos_vis_out)
        label_vis_outbound.set_text(cur_outbound_edge_text)

        # Ring Arrow Update
        len_ring_prog = 0.0
        peer_device_id = -1
        cur_ring_edge_text = ""
        color_ring = arrow_ring.get_edgecolor() # Get current color
        if device.is_peer_transferring and device.cur_peer_transfer_duration > 0:
            if device.cur_peer_transfer_details:
                peer_device_id, cid_ring, lid_ring, isg_ring = device.cur_peer_transfer_details
                prog_frac = min(1.0, (T - device.cur_peer_transfer_start_time) / device.cur_peer_transfer_duration)
                len_ring_prog = prog_frac
                cur_ring_edge_text = device.cur_ring_edge
            else: # Should not happen if state is consistent
                if TO_PRINT:
                     print(f"T={T} Dev {i}: WARN - is_peer_transferring but cur_peer_transfer_details is None")
                peer_device_id = -1
                len_ring_prog = 0.0
                cur_ring_edge_text = ""
        label_ring.set_color(color_ring)
        start_pos_ring_geo = outer_pos
        current_end_point_ring = start_pos_ring_geo
        # Calculate start/end points on the circumferences for smooth connection
        if peer_device_id != -1:
            target_pos_ring_geo_center = outer_circle_positions[peer_device_id]
            vec_to_target = target_pos_ring_geo_center - start_pos_ring_geo
            dist_to_target = np.linalg.norm(vec_to_target)
            if dist_to_target > 1e-6:
                dir_to_target = vec_to_target / dist_to_target
                # Adjust start/end points to be on the circle edges
                start_pos_ring_geo = outer_pos + dir_to_target * outer_node_radius
                target_pos_ring_geo = target_pos_ring_geo_center - dir_to_target * outer_node_radius
                # Interpolate along the adjusted line segment
                current_end_point_ring = start_pos_ring_geo + (target_pos_ring_geo - start_pos_ring_geo) * len_ring_prog
            else: # Target is the same node? (Shouldn't happen for ring transfers)
                current_end_point_ring = start_pos_ring_geo # Keep endpoint at start

        arrow_ring.set_positions(start_pos_ring_geo, current_end_point_ring)
        # Position label near the middle of the visible arrow path
        label_pos_ring = (start_pos_ring_geo + current_end_point_ring) / 2
        if len_ring_prog > 1e-6: # Avoid division by zero if arrow hasn't started
            edge_vec = current_end_point_ring - start_pos_ring_geo
            norm = np.linalg.norm(edge_vec)
            if norm > 1e-6:
                perp_vec = np.array([-edge_vec[1], edge_vec[0]]) / norm # Perpendicular vector
                conn_style = arrow_ring.get_connectionstyle()
                offset_direction_multiplier = 1.0 # Default for CCW / outward arc
                # Check if arc is CW / inward (rad is negative)
                if isinstance(conn_style, str) and 'rad=-' in conn_style:
                    offset_direction_multiplier = -1.0
                # Offset the label away from the arc
                label_pos_ring = label_pos_ring + perp_vec * label_offset_distance * 3 * offset_direction_multiplier
        label_ring.set_position(label_pos_ring)
        label_ring.set_text(cur_ring_edge_text)

        # Computation Arc Update
        compute_arc = edge_artists[f'compute_{i}']
        progress_frac = 0.0
        compute_color = COLOR_COMPUTE_DEFAULT
        if device.is_computing and device.cur_computation_duration > 0:
            progress_frac = min(1.0, (T - device.cur_computation_start_time) / device.cur_computation_duration)
            comp_type = device.current_computation_type
            comp_lid = device.current_computation_layer_id

            # Set color based on computation type
            if comp_type == "Fwd":
                 compute_color = COLOR_COMPUTE_FWD
            elif comp_type == "Bwd X":
                 compute_color = COLOR_COMPUTE_BWD_X
            elif comp_type == "Bwd W":
                 compute_color = COLOR_COMPUTE_BWD_W
            elif comp_type == "Head":
                 compute_color = COLOR_COMPUTE_HEAD
            else:
                 compute_color = COLOR_COMPUTE_DEFAULT

            # Determine arc angles based on computation type and pre-calculated world angles
            theta1_for_arc, theta2_for_arc = 0.0, 0.0
            total_sweep_angle = 0.0

            if comp_type == "Fwd":
                # Arc sweeps from previous device direction towards next device direction (CCW)
                angle_start_abs = world_angle_to_prev[i]
                # Special case for Layer 0: Start opposite the next device
                if comp_lid == 0:
                    angle_start_abs = world_angle_to_next[i] - 180
                angle_end_target_abs = world_angle_to_next[i]
                total_sweep_angle = (angle_end_target_abs - angle_start_abs + 360) % 360
                if N==1:
                     total_sweep_angle=360 # Full circle if only one device
                theta1_for_arc = angle_start_abs
                theta2_for_arc = angle_start_abs + progress_frac * total_sweep_angle
            elif comp_type == "Head":
                # Head computation can be visualized as a full circle sweep
                angle_start_abs = world_angle_to_prev[i] # Arbitrary start, e.g., from prev
                total_sweep_angle = 360.0
                theta1_for_arc = angle_start_abs
                theta2_for_arc = angle_start_abs + progress_frac * total_sweep_angle
            elif comp_type == "Bwd X" or comp_type == "Bwd W":
                # Arc sweeps from next device direction towards previous device direction (CW)
                angle_start_vis = world_angle_to_next[i] # Start visually where FWD ended
                angle_end_target_vis = world_angle_to_prev[i] # Target where FWD started
                total_sweep_angle = (angle_start_vis - angle_end_target_vis + 360) % 360 # Positive CW sweep
                if N==1:
                     total_sweep_angle=360 # Full circle if only one device
                current_cw_sweep = progress_frac * total_sweep_angle
                angle_end_vis_current = angle_start_vis - current_cw_sweep # Current end point moving CW
                # Matplotlib Arc draws CCW, so theta1 < theta2. We define the swept part.
                theta1_for_arc = angle_end_vis_current
                theta2_for_arc = angle_start_vis

            # Update arc visibility and properties
            # Show arc only if there's progress and a non-zero sweep angle
            if progress_frac > 1e-6 and total_sweep_angle > 1e-6 :
                compute_arc.theta1 = theta1_for_arc
                compute_arc.theta2 = theta2_for_arc
                compute_arc.set_visible(True)
                compute_arc.set_edgecolor(compute_color)
            else:
                compute_arc.set_visible(False)
        else: # Not computing
            compute_arc.set_visible(False)

        # Add updated artists for this device
        artists_to_update.extend([ outer_label_artist, inner_label_artist, stall_node_artist, stall_label_artist,
                                  finish_indicator_artist, arrow_vis_inbound, label_vis_inbound, arrow_vis_outbound, label_vis_outbound,
                                  arrow_ring, label_ring, compute_arc ])

    # --- Check for Simulation Completion (Based on Computation Tasks) ---
    current_total_completed = total_completed_tasks.get(T, 0)
    # ***** FIX: Define completion based *only* on computation tasks *****
    is_computation_complete = (current_total_completed >= total_tasks)
    # Record the time when computation first completes
    if is_computation_complete and simulation_computation_finish_time is None:
        simulation_computation_finish_time = T
    # Check if computation was already complete last cycle
    was_computation_complete_last_cycle = False
    if T > 0:
        was_computation_complete_last_cycle = (total_completed_tasks.get(T - 1, 0) >= total_tasks)
    # Stop the animation the first time computation completes
    should_stop_animation = is_computation_complete and not was_computation_complete_last_cycle
    # ********************************************************************

    # Display completion text only once, when computation first finishes
    if current_total_completed >= total_tasks and completion_text_artist is None:
        if TO_PRINT:
             print(f"T={T}: Completed all {current_total_completed}/{total_tasks} tasks!")
        # Calculate stats
        start_bubble = sum(d.device_start_time for d in all_devices.values() if d.device_has_started)
        # Ensure finish time is used correctly, even if simulation stops *exactly* at T
        stop_bubble = sum(max(0, T - d.device_finish_time) for d in all_devices.values() if d.device_has_finished)

        total_dev_time = T * N if T > 0 else 0
        steady_time = total_dev_time - stop_bubble - start_bubble if total_dev_time > 0 else 0

        overall_eff = 0.0
        if total_dev_time > 0:
             overall_eff = (total_computation_time / total_dev_time * 100)

        steady_eff = 0.0
        if steady_time > 0:
             steady_eff = (total_computation_time / steady_time * 100)

        runtime_in_seconds = T / cycles_per_second

        completion_text = (
             f"Simulation Complete!\nFinal Cycle Count: {T}\nRuntime: {runtime_in_seconds:.3f} seconds\n\n"
             f"Problem:\nTotal Tasks: {total_tasks}\n"
             f"Total Task Computation Cycles: {total_computation_time}\n"
             f"Utilized {N} devices for aggregate {total_dev_time} cycles\n\n"
             f"Pipeline:\nFill Cycles: {start_bubble}\n"
             f"Flush Cycles: {stop_bubble}\n"
             f"Steady-State Cycles: {steady_time}\n\n"
             f"Pipeline Efficiency:\n% Active Overall: {overall_eff:.2f}%\n"
             f"% Active during Steady-State: {steady_eff:.2f}%\n\n"
             f"Compute Throughput:\n"
             f"Advertised Upper-Bound: {int(hardware_max_flops / 1e12)} TFLOPS\n"
             f"True Upper-Bound: {int((hardware_max_flops * practical_efficiency) / 1e12)} TFLOPS\n"
             f"Achieved Throughput: {int(((total_flops / N) / runtime_in_seconds) / 1e12) if N > 0 and runtime_in_seconds > 0 else 0} TFLOPS\n\n"
        )
        completion_text_artist = ax.text(0.5, 0.5, completion_text, transform=ax.transAxes,
                                         ha='center', va='center', fontsize=14, color='navy', fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.5', fc=(0.9, 0.9, 1, 0.9), ec='black'),
                                         zorder=10)
        if TO_PRINT:
            print(completion_text)
        artists_to_update.append(completion_text_artist)
        # ***** FIX: Update title one last time when showing completion text *****
        title_obj.set_text(f'Cycle {T} (Complete)')
        # ************************************************************************

    # Stop the animation timer if computation just completed
    if should_stop_animation and ani is not None and ani.event_source is not None:
        if not animation_paused: # Only stop if it was running
            ani.event_source.stop()
            # Use the actual computation finish time in the log message
            final_time = simulation_computation_finish_time if simulation_computation_finish_time is not None else T
            print(f"Animation Complete (Computation Finished) at Cycle {final_time} - Paused")
        animation_paused = True # Ensure state reflects paused
        target_cycle = None # Clear any run-to target
        # ***** FIX: Ensure final title reflects the recorded completion time *****
        final_time = simulation_computation_finish_time if simulation_computation_finish_time is not None else T
        title_obj.set_text(f'Cycle {final_time} (Complete)')
        # ***********************************************************************

    # --- Increment Frame Index & Handle Max Frames ---
    if not animation_paused:
        current_frame_index += 1
        # Max frames check remains the same
        if current_frame_index >= max_frames:
            print(f"Max frames ({max_frames}) reached, stopping animation.")
            if ani.event_source is not None:
                ani.event_source.stop()
            animation_paused = True
            target_cycle = None
            title_obj.set_text(f'Cycle {T} (Max Frames)')
            if completion_text_artist is None: # Show max frames message only if not already complete
                current_T_max = T # Use the last frame index before hitting max
                current_total_completed_max = total_completed_tasks.get(current_T_max, 0)
                completion_text = (f"Max Frames Reached!\nFinal Cycle: {current_T_max}\n\n"
                                   f"Tasks Completed: {current_total_completed_max} / {total_tasks}\n")
                completion_text_artist = ax.text(0.5, 0.5, completion_text, transform=ax.transAxes,
                                                 ha='center', va='center', fontsize=14, color='maroon', fontweight='bold',
                                                 bbox=dict(boxstyle='round,pad=0.5', fc=(1, 0.9, 0.9, 0.95), ec='black'), zorder=10)
                artists_to_update.append(completion_text_artist)

    return artists_to_update


# --- Create Animation ---
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=initial_frame_interval, blit=False, repeat=False, save_count=max_frames)

# --- Widgets --- (Setup unchanged)
fig.subplots_adjust(bottom=0.25, right=0.85)
ax_slider = fig.add_axes([0.15, 0.15, 0.7, 0.03])
ax_restart = fig.add_axes([0.15, 0.09, 0.2, 0.04])
ax_pause = fig.add_axes([0.40, 0.09, 0.2, 0.04])
ax_play = fig.add_axes([0.65, 0.09, 0.2, 0.04])
ax_textbox = fig.add_axes([0.15, 0.03, 0.5, 0.04])
ax_runto_btn = fig.add_axes([0.70, 0.03, 0.15, 0.04])
btn_restart = Button(ax_restart, 'Restart')
btn_pause = Button(ax_pause, 'Pause')
btn_play = Button(ax_play, 'Play')
slider_speed = Slider(ax=ax_slider, label='Speed Level', valmin=min_speed_level, valmax=max_speed_level, valinit=initial_speed_level, valstep=1)
# Calculate a slightly more robust initial guess
initial_run_to_cycle_guess = 0
if N > 0 and total_chunks > 0:
     # Estimate: fill time + last chunk time on last device
     fill_cycles = (N - 1) * (computationFrames + activationTransitionFrames) if N > 1 else 0
     last_chunk_cycles = max_computationFrames # Computation of last chunk
     initial_run_to_cycle_guess = fill_cycles + last_chunk_cycles
else:
     initial_run_to_cycle_guess = computationFrames * total_chunks # Fallback for simpler cases

textbox_runto = TextBox(ax_textbox, "Run to Cycle:", initial=str(initial_run_to_cycle_guess), textalignment="center")
btn_runto = Button(ax_runto_btn, 'Run')

# --- Define Widget Callback Functions --- (Logic adjusted slightly for new completion state, refactored style)
def pause_animation(event):
    global animation_paused, target_cycle
    if not animation_paused:
        if ani.event_source is not None:
            ani.event_source.stop()
        animation_paused = True
        target_cycle = None
        # ***** FIX: Use recorded time if complete *****
        final_time = simulation_computation_finish_time if simulation_computation_finish_time is not None else current_frame_index
        if simulation_computation_finish_time is not None:
            status_text = " (Complete)"
        else:
            status_text = " (Paused)"
        title_obj.set_text(f'Cycle {final_time}{status_text}')
        # *******************************************
        fig.canvas.draw_idle()
        print("Animation Paused")
    else:
        print("Animation already paused.")

def play_animation(event):
    global animation_paused, target_cycle
    target_cycle = None # Playing clears any run-to target
    # ***** FIX: Check completion based on computation time *****
    is_computation_finished = (simulation_computation_finish_time is not None)
    is_at_max_frames = (current_frame_index >= max_frames)
    # *********************************************************
    if animation_paused and not is_computation_finished and not is_at_max_frames:
        if ani.event_source is not None:
            title_obj.set_text(f'Cycle {current_frame_index}') # Update title before starting
            ani.event_source.start()
            animation_paused = False
            print("Animation Resumed")
        else:
            print("Error: Animation event source not found.")
    elif is_computation_finished:
        print("Animation already complete.")
    elif is_at_max_frames:
        print("Animation stopped at max frames.")
    elif not animation_paused:
        print("Animation already playing.")
    else:
        print("Cannot play animation (unknown reason).")

def update_speed(val):
    global animation_paused # Use our global state flag
    speed_level = slider_speed.val
    new_interval = calculate_interval(speed_level, min_speed_level, max_speed_level, min_interval, max_interval)

    # Use our own state flag to determine if the animation was running
    was_running = not animation_paused
    timer_stopped = False # Keep track if we successfully stopped the timer

    if ani.event_source is not None:
        # Always stop the timer before changing interval
        ani.event_source.stop()
        timer_stopped = True # Record that we stopped it
        # Set the interval on the timer object directly
        ani.event_source.interval = new_interval
        # Crucially, also update the interval stored in the Animation object itself
        ani._interval = new_interval
    else:
        print("Error: Could not access animation timer.")
        return # Cannot proceed without a timer

    # Determine if we should resume based on the original state and current conditions
    is_computation_finished = (simulation_computation_finish_time is not None)
    is_at_max_frames = (current_frame_index >= max_frames)
    # Should resume only if it was running, not finished, not at max frames, and not targeting a specific cycle
    should_resume = was_running and not is_computation_finished and not is_at_max_frames and target_cycle is None

    if should_resume:
        ani.event_source.start()
        animation_paused = False # Update our state flag
    else:
        # If we stopped the timer but shouldn't resume, make sure our state reflects paused
        animation_paused = True
        # Update the title appropriately if paused due to speed change
        if not is_computation_finished and not is_at_max_frames:
            # Check if paused because we *were* running towards a target cycle
            if target_cycle is not None and current_frame_index != target_cycle:
                 # Keep the "Paused" status in title if still aiming for a target
                 pass # Title likely already correct from run_to or initial pause
            else: # Paused normally
                 title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
                 fig.canvas.draw_idle()
        # If complete, ensure title reflects that
        elif is_computation_finished:
            title_obj.set_text(f'Cycle {simulation_computation_finish_time} (Complete)')
            fig.canvas.draw_idle()
        # If at max frames, ensure title reflects that
        elif is_at_max_frames:
             title_obj.set_text(f'Cycle {current_frame_index} (Max Frames)')
             fig.canvas.draw_idle()


    print(f"Speed Level: {int(round(speed_level))}, Interval set to: {new_interval} ms")


def restart_animation_callback(event): # Unchanged logic, reset handles new state var
    global animation_paused
    print("Restart button clicked.")
    if ani.event_source is not None:
        ani.event_source.stop() # Stop existing timer first

    reset_simulation() # Reset state and visuals
    fig.canvas.draw_idle() # Update display with reset state

    # Flush events to ensure the reset is rendered before starting again
    try:
        fig.canvas.flush_events()
    except AttributeError:
        # Some backends might not have flush_events
        pass

    # Start the new animation timer
    if ani.event_source is not None:
        ani.event_source.start()
        animation_paused = False # Update state
        print("Simulation reset and playing from Cycle 0.")
    else:
        print("Error: Cannot restart animation timer.")
        # Ensure state is paused if timer couldn't start
        animation_paused = True
        title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
        fig.canvas.draw_idle()

def run_to_cycle_callback(event): # Logic adjusted for new completion state, refactored style
    global target_cycle, animation_paused, current_frame_index
    input_text = textbox_runto.text
    try:
        requested_cycle = int(input_text)
    except ValueError:
        print(f"Invalid input: '{input_text}'. Please enter an integer.")
        textbox_runto.set_val(str(current_frame_index)) # Reset textbox
        return

    if requested_cycle < 0:
        print(f"Invalid input: {requested_cycle}. Cycle must be non-negative.")
        textbox_runto.set_val(str(current_frame_index)) # Reset textbox
        return

    if requested_cycle >= max_frames:
        print(f"Target {requested_cycle} >= max frames ({max_frames}). Clamping to {max_frames - 1}.")
        requested_cycle = max_frames - 1
        textbox_runto.set_val(str(requested_cycle)) # Update textbox with clamped value

    print(f"Attempting to run to cycle: {requested_cycle}")

    # Pause current animation if it's running
    if ani.event_source is not None and not animation_paused:
        print("Stopping current animation...")
        ani.event_source.stop()
        animation_paused = True

    # ***** FIX: Check completion based on computation time *****
    is_computation_finished = (simulation_computation_finish_time is not None)
    # *********************************************************

    needs_restart = False
    # Reset if target is in past OR if computation is already finished
    if requested_cycle <= current_frame_index or is_computation_finished:
        # Avoid restart if already paused at the exact target cycle
        if not (requested_cycle == current_frame_index and animation_paused and not is_computation_finished):
            print("Target in past or computation complete. Restarting simulation...")
            if ani.event_source is not None:
                ani.event_source.stop() # Ensure timer stopped before reset
            reset_simulation()
            needs_restart = True
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except AttributeError:
                pass

    target_cycle = requested_cycle
    print(f"Target set to cycle {target_cycle}.")

    # Re-check completion state after potential reset
    is_computation_finished_after_reset = (simulation_computation_finish_time is not None)

    # Determine if we should run the animation
    should_run = (current_frame_index < target_cycle) and not is_computation_finished_after_reset and (current_frame_index < max_frames)

    if should_run:
        if ani.event_source is not None:
            print("Starting animation to reach target.")
            title_obj.set_text(f'Cycle {current_frame_index}') # Set title before starting
            ani.event_source.start()
            animation_paused = False # Update state
        else:
            print("Error: Animation event source not found.")
            target_cycle = None # Clear target if cannot run
            animation_paused = True
            title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
            fig.canvas.draw_idle()
    else:
        # Cannot run or already at/past target
        print(f"Cannot run or already at/past target ({current_frame_index}). Ensuring paused.")
        if ani.event_source is not None and not animation_paused: # Ensure paused if needed
             ani.event_source.stop()
        animation_paused = True

        if current_frame_index == target_cycle:
            target_cycle = None # Clear target if reached

        # Update title based on the final state
        final_title = f'Cycle {current_frame_index}'
        if is_computation_finished_after_reset:
            final_title = f'Cycle {simulation_computation_finish_time} (Complete)'
        elif current_frame_index >= max_frames:
            final_title += " (Max Frames)"
        elif target_cycle is None and animation_paused: # Normal pause state
             final_title += " (Paused)"
        # Removed target_cycle check here as it's cleared if reached
        title_obj.set_text(final_title)
        fig.canvas.draw_idle()


# --- Connect Widgets & Cursor Logic (Unchanged logic, refactored style) ---
btn_restart.on_clicked(restart_animation_callback)
btn_pause.on_clicked(pause_animation)
btn_play.on_clicked(play_animation)
slider_speed.on_changed(update_speed)
btn_runto.on_clicked(run_to_cycle_callback)

widget_axes = [ax_restart, ax_pause, ax_play, ax_slider, ax_textbox, ax_runto_btn]
is_cursor_over_widget = False

def on_hover(event):
    global is_cursor_over_widget
    currently_over = False
    if event.inaxes in widget_axes:
        currently_over = True

    if currently_over != is_cursor_over_widget:
        if currently_over:
            new_cursor = Cursors.HAND
        else:
            new_cursor = Cursors.POINTER
        try:
            if fig.canvas:
                fig.canvas.set_cursor(new_cursor)
        except Exception as e:
            if TO_PRINT:
                print(f"Minor error setting cursor: {e}")
        is_cursor_over_widget = currently_over

fig.canvas.mpl_connect('motion_notify_event', on_hover)

print("Initializing display...")
reset_simulation() # Initial setup

"""
print(f"Starting animation saving process (up to {max_frames} frames)...")
print("This may take a significant amount of time depending on max_frames and DPI.")
try:
    # Reduce DPI significantly for faster saving and smaller file size
    ani.save('32k_100pct_32b_x8_80gb_fastpeer_ring.mp4', writer='ffmpeg', dpi=150, progress_callback=lambda i, n: print(f'Saving frame {i+1}/{n}', end='\r') if (i+1)%10==0 else None)
    print("\nAnimation saving finished successfully.")

except Exception as e:
    print("\n--- Error during animation saving ---")
    print(f"Error Type: {type(e)}")
    print(f"Error Representation: {repr(e)}")
    print(f"Error String: {str(e)}")
    # Add traceback for full context
    import traceback
    print("\n--- Traceback ---")
    traceback.print_exc()
    print("-----------------\n")


# Explicitly close the plot to free memory - Still good practice
plt.close(fig)
print("Figure closed.")
"""


# --- Display ---
print("Showing plot...")
plt.show()
print("Plot window closed.")