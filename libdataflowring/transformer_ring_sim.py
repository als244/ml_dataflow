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

# --- Simulation Parameters ---
## making 100 microseconds equivalent to 1 cycle
cycles_per_second = 1e4

# --- Speed/Interval Parameters ---
min_speed_level = 1
max_speed_level = 100
min_interval = 1
max_interval = 100
initial_speed_level = 50

## Smaller Model Parameters
N = 8 # Number of devices
total_layers = 64

do_backward = True
# True=Pipeline Parallel (BWD Reversed), False=Data Parallel(BWD Same Order)
are_chunks_same_seq = True

vocab_size = 151646
model_dim = 5120
kv_factor = .125
kv_dim = int(model_dim * kv_factor)
ffn_multiplier = 5.4
ffn_dim = int(model_dim * ffn_multiplier)
num_experts = 1
active_experts = 1
expert_dim = int(ffn_dim // num_experts)
chunk_size = 1536

## FP8
bitwidth = 16
dtype_bytes = bitwidth / 8

seqlen_thousands = 32
seqlen = (1 << 10) * seqlen_thousands


## hardware configs

## compute configs
## H100 TFLOPS

## FP16 = 989 TFLOPS
## FP8 = 989 * 2 TFLOPS
hardware_max_flops = int((989 * (2 / dtype_bytes)) * 1e12)
matmul_efficiency = 0.7
attn_efficiency = 0.6


## communication configs
home_bw_gb_sec = 400
peer_transfer_bw_gb_sec = 100

## THese config params not implemented yet...
attn_type = "Exact"

total_chunks = math.ceil(seqlen / chunk_size)
train_chunk_freq = 1

num_sequences = 1


matmul_attn_flops_per_layer =2 * (model_dim * model_dim) + (4 * model_dim * kv_dim * chunk_size)
matmul_ffn_flops_per_layer = 3 * (2 * model_dim * ffn_dim * chunk_size * active_experts)

base_flops_per_layer = matmul_attn_flops_per_layer + matmul_ffn_flops_per_layer

## this is multiplied by the chunk_size + cur seq len
flops_per_attn_chunk_mult = 2 * chunk_size * model_dim

## head does fwd + bwd
head_flops = 2 * (2 * vocab_size * model_dim * chunk_size)


total_matmul_flops = 0
total_attn_flops = 0

total_fwd_flops = 0
total_bwd_flops = 0



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
    computation_times_frames[i] = int(computation_times_sec[i] * cycles_per_second)

    per_layer_chunk_flops += layer_flops

    if do_backward:
        ## attention layer for bwd x has double the flops...
        bwd_x_flops = layer_flops + attn_flops
        computation_times_sec_bwd[i] = base_flops_per_layer / (hardware_max_flops * matmul_efficiency) + (2 * attn_flops) / (hardware_max_flops * attn_efficiency)
        computation_times_frames_bwd[i] = int(computation_times_sec_bwd[i] * cycles_per_second)

        per_layer_chunk_flops += bwd_x_flops

        total_matmul_flops += total_layers * base_flops_per_layer
        total_attn_flops += 2 * total_layers * attn_flops

        per_layer_chunk_flops += bwd_w_flops
        total_matmul_flops += total_layers * base_flops_per_layer

        total_bwd_flops += total_layers * bwd_x_flops + total_layers * bwd_w_flops

    total_chunk_flops += total_layers * per_layer_chunk_flops
    total_chunk_flops += head_flops

    total_fwd_flops += head_flops / 2
    total_bwd_flops += head_flops / 2

    total_matmul_flops += head_flops

    total_flops += total_chunk_flops

    prev_seq_len = cur_seq_len



attn_block_size_bytes = dtype_bytes * (2 * model_dim * model_dim + 4 * model_dim * kv_dim)
ffn_block_size_bytes = dtype_bytes * (3 * model_dim * expert_dim * num_experts)
layer_size_bytes = attn_block_size_bytes + ffn_block_size_bytes

layer_transfer_time_sec = (layer_size_bytes * 8) / (home_bw_gb_sec * 1e9)

## model input, query, key, value, attn output, attn ouut output
attn_activation_bytes = dtype_bytes * 4 * (model_dim * chunk_size) + 2 * (kv_dim * chunk_size)
ffn_activation_bytes = dtype_bytes * (expert_dim * chunk_size * active_experts)
activation_size_bytes = attn_activation_bytes + ffn_activation_bytes

activation_transfer_time_sec = (activation_size_bytes * 8) / (home_bw_gb_sec * 1e9)

output_size_bytes = dtype_bytes * (model_dim * chunk_size)

transition_transfer_time_sec = (output_size_bytes * 8) / (peer_transfer_bw_gb_sec * 1e9)



layerTransferFrames = int(layer_transfer_time_sec * cycles_per_second) # Cycles to transfer weights
headFrames = int(head_computation_times_sec * cycles_per_second)
computationFrames = computation_times_frames[0] # Cycles per compute task
max_computationFrames = computation_times_frames[total_chunks-1]
savedActivationsFrames = int(activation_transfer_time_sec * cycles_per_second) # Cycles to transfer activations (save/fetch)
activationTransitionFrames = int(transition_transfer_time_sec * cycles_per_second) # Cycles to transfer activations/grads between devices

layer_capacity = 2 # Max weights per device

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



## NOT USING THESE CORRECTLY FOR NOW....
activations_capacity = 4 # Max CHECKPOINTED activations kept resident from FWD per device
transitions_capacity = total_chunks # Buffer size for incoming FWD data from peers
grad_activations_capacity = total_chunks # Buffer size for local dL/dA results

max_frames = 100000 # Limit animation length for performance if needed


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
COLOR_OUTBOUND_DEFAULT = 'gray'
COLOR_OUTBOUND_FWD_ACTIVATION = 'magenta'
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
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
lim = total_distance + outer_node_radius + stall_node_distance_offset + stall_node_radius + 0.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axis('off')
center_pos = np.array([0, 0])

# --- Legend Text ---
wrap_width = 40

legend_text = (
    f"Simulated Configuration:\n\n"
    f"      - Num Compute Devices: {N}\n\n"
    f"      - Model Info:\n"
    f"         - Total Blocks (non-head): {total_layers}\n"
    f"         - Bitwidth: {bitwidth}\n"
    f"         - Attention Algo: {attn_type}\n"
    f"         - Model Dim: {model_dim}\n"
    f"         - KV Dim: {kv_dim}\n"
    f"         - Per-Expert Dim: {expert_dim}\n"
    f"         - Num Experts: {num_experts}\n"
    f"         - Active Experts: {active_experts}\n"
    f"         - Vocab Size: {vocab_size}\n\n"
    f"      - Compute Environment Constants:\n"
    f"         - Hardware Theoretical MAX TFLOPs: {int(hardware_max_flops / 1e12)} TFLOPs\n"
    f"         - Matmul Efficiency: {matmul_efficiency}\n"
    f"         - Attn Efficiency: {attn_efficiency}\n"
    f"      - Communication Environment Constants:\n"
    f"         - Device-to-Home BW (Gb/s): {home_bw_gb_sec}\n"
    f"         - Peer-to-Peer BW (Gb/s): {peer_transfer_bw_gb_sec}\n\n"
    f"      - Training Parameters:\n"
    f"         - Sequence Length: {seqlen}\n"
    f"         - Chunk Training Frequency: {train_chunk_freq}\n"
    f"         - Num Sequences: {num_sequences}\n\n"
    f"      - FLOP Breakdown\n"     
    f"          - Total TFLOPs: {int(total_flops / 1e12)}\n"
    f"              - FWD TFLOPs: {int(total_fwd_flops / 1e12)} TFLOPs\n"
    f"              - BWD TFLOPs: {int(total_bwd_flops / 1e12)} TFLOPs\n"
    f"              - Overall Matmul TFLOPs: {int(total_matmul_flops / 1e12)} TFLOPs\n"
    f"              - Overall Attn TFLOPs: {int(total_attn_flops / 1e12)} TFLOPs\n\n"
    f"      - Dataflow Parameters:\n"
    f"          - Per-Device Layer Capacity: {layer_capacity}\n"
    f"          - Chunk Size: {chunk_size}\n"
    f"             - Total Chunks: {total_chunks}\n\n"
    f"      - Derived Cycles ({int(cycles_per_second / 1000)}k cycles per second):\n"
    f"         - C0 Computation: {computationFrames} Cycles\n"
    f"         - C{total_chunks-1} Computation: {max_computationFrames} Cycles\n"
    f"         - Head Computation: {headFrames} Cycles\n"
    f"         - Layer Transfer: {layerTransferFrames} Cycles\n"
    f"         - Activation Transfer: {savedActivationsFrames} Cycles\n"
    f"         - Block Transition: {activationTransitionFrames} Cycles\n\n"
    f"       - Runtime Lower-Bound:\n"
    f"         - Based on Matmul/Attn Efficiency: {int(((total_matmul_flops / N) / (hardware_max_flops * matmul_efficiency) + (total_attn_flops / N) / (hardware_max_flops * attn_efficiency)) * cycles_per_second)} cycles\n"
)
wrapped_legend_text = legend_text
at = AnchoredText(wrapped_legend_text, loc='upper left', bbox_to_anchor=(1.01, 1.01),
                  prop=dict(size=8), frameon=True, pad=0.4, borderpad=0.5,
                  bbox_transform=ax.transAxes)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at.patch.set_facecolor((1, 1, 1, 0.8))
at.patch.set_edgecolor('black')
ax.add_artist(at)


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
    def __init__(self, device_id, layer_capacity, activations_capacity, transitions_capacity, total_devices, total_layers, total_chunks):
        self.device_id = device_id
        self.device_has_started = False
        self.device_start_time = 0
        self.device_has_finished = False
        self.device_finish_time = 0 # Will mark when last compute task finishes
        self.layer_capacity = layer_capacity
        self.activations_capacity = activations_capacity
        self.transitions_capacity = transitions_capacity
        self.total_devices = total_devices
        self.total_layers = total_layers
        self.total_chunks = total_chunks
        self.bwd_grad_transitions_capacity = total_chunks
        self.grad_activations_capacity = grad_activations_capacity
        self.bwd_fetched_activation_buffer_capacity = self.total_chunks * 2
        self.cur_ready_weights = [-1 for _ in range(layer_capacity)]
        self.cur_weight_replace_ind = 0
        self.fwd_transitions_buffer = [-1 for _ in range(self.transitions_capacity)]
        self.fwd_transitions_write_ptr = 0
        self.bwd_grad_transitions_buffer = [-1 for _ in range(self.bwd_grad_transitions_capacity)]
        self.bwd_grad_transitions_write_ptr = 0
        self.resident_checkpoint_activations = [-1 for _ in range(self.activations_capacity)]
        self.resident_checkpoint_write_ptr = 0
        self.bwd_fetched_activation_buffer = [-1 for _ in range(self.bwd_fetched_activation_buffer_capacity)]
        self.bwd_fetched_activation_write_ptr = 0
        self.cur_ready_grad_activations = [-1 for _ in range(self.grad_activations_capacity)]
        self.cur_grad_activations_write_ptr = 0
        self.cur_model_outputs = [-1 for _ in range(self.total_chunks)]
        self.cur_model_outputs_write_ptr = 0
        self.cur_head_outputs = [-1 for _ in range(self.total_chunks)]
        self.cur_head_outputs_write_ptr = 0
        self.local_last_fwd_activations = {}
        self.outbound_storage = set()
        self.computation_queue = []
        self.outbound_queue = deque()
        self.inbound_queue = deque()
        self.peer_transfer_queue = deque()
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
        self.has_reversed = False
        self.last_fwd_layer_on_device = -1
        self.bwd_prefetch_trigger_layer_id = -1
        self.next_weight_layer_id = -1
        self.head_final_chunk_id = -1
        self._initialize_tasks_and_state()

    def _initialize_tasks_and_state(self): # Unchanged
        initially_loaded_weights = []
        for i in range(self.layer_capacity):
            layer_id_to_add = self.device_id + i * self.total_devices
            if layer_id_to_add < self.total_layers:
                self.cur_ready_weights[i] = layer_id_to_add
                self.outbound_storage.add((-1, layer_id_to_add, False))
                initially_loaded_weights.append(layer_id_to_add)
        self.cur_weight_replace_ind = 0
        cur_layer_id = self.device_id
        while cur_layer_id < self.total_layers:
            self.last_fwd_layer_on_device = cur_layer_id
            if (-1, cur_layer_id, False) not in self.outbound_storage:
                self.outbound_storage.add((-1, cur_layer_id, False))
            for i in range(self.total_chunks):
                if cur_layer_id < self.total_layers - 1:
                    transfer_direction = 1
                else:
                    transfer_direction = 0
                # Overwrite condition
                if cur_layer_id == self.total_layers - 1:
                    transfer_direction = 1
                self.computation_queue.append((i, cur_layer_id, False, False, transfer_direction, computation_times_frames[i]))
            cur_layer_id += self.total_devices
        self.next_weight_layer_id = self.device_id + self.layer_capacity * self.total_devices
        if self.last_fwd_layer_on_device != -1 and self.next_weight_layer_id > self.last_fwd_layer_on_device:
            if self.last_fwd_layer_on_device + self.total_devices == self.total_layers:
                self.next_weight_layer_id = self.total_layers
            else:
                self.next_weight_layer_id = None
        elif self.last_fwd_layer_on_device == -1:
            self.next_weight_layer_id = None

        if do_backward:
            head_layer_conceptual_id = self.total_layers
            is_head_device = (self.last_fwd_layer_on_device != -1 and self.last_fwd_layer_on_device + self.total_devices == self.total_layers)
            if is_head_device:
                cur_layer_id_for_head_task = head_layer_conceptual_id
                self.outbound_storage.add((-1, cur_layer_id_for_head_task, False))
                if self.layer_capacity > len(initially_loaded_weights):
                    head_wgt_idx = len(initially_loaded_weights)
                    self.cur_ready_weights[head_wgt_idx] = cur_layer_id_for_head_task
                head_task_queue_order = []
                if are_chunks_same_seq:
                    ## half of the total chu
                    cutoff_chunk_id = self.total_chunks // 2
                    head_task_queue_order.extend(range(cutoff_chunk_id))
                    head_task_queue_order.extend(range(self.total_chunks - 1, cutoff_chunk_id - 1, -1))
                    self.head_final_chunk_id = cutoff_chunk_id
                else:
                    # Assuming cutoff_chunk_id needs definition outside the if for this branch
                    cutoff_chunk_id = self.total_chunks // 2 # Or some other default
                    head_task_queue_order.extend(range(cutoff_chunk_id, self.total_chunks))
                    self.head_final_chunk_id = self.total_chunks - 1
                for i in head_task_queue_order:
                    self.computation_queue.append((i, cur_layer_id_for_head_task, False, False, -1, headFrames))
                cur_layer_id_for_bwd_loop = self.last_fwd_layer_on_device
            else:
                cur_layer_id_for_bwd_loop = self.last_fwd_layer_on_device

            if cur_layer_id_for_bwd_loop >= 0:
                current_bwd_layer = cur_layer_id_for_bwd_loop
                while current_bwd_layer >= 0:
                    if are_chunks_same_seq:
                        chunk_order = range(self.total_chunks - 1, -1, -1)
                    else:
                        chunk_order = range(self.total_chunks)

                    for i in chunk_order:
                        if current_bwd_layer > 0:
                            transfer_direction = -1
                        else:
                            transfer_direction = 0
                        self.computation_queue.append((i, current_bwd_layer, True, False, transfer_direction, computation_times_frames_bwd[i]))
                    for i in chunk_order:
                        self.computation_queue.append((i, current_bwd_layer, False, True, 0, computation_times_frames[0]))
                    current_bwd_layer -= self.total_devices

    def handle_completed_transfers(self, T, all_devices): # Unchanged logic, split lines
        if self.is_inbound_transferring and self.inbound_queue and (self.cur_inbound_start_time + self.cur_inbound_duration <= T):
            inbound_item = self.inbound_queue.popleft()
            chunk_id, layer_id, is_grad, target_idx, duration, target_buffer_type = inbound_item
            item_type, status, buffer_name = 'Unknown', 'OK', "N/A"
            if target_buffer_type == 'weight':
                item_type, buffer_name = 'Wgt', "Ready Wgt"
                if 0 <= target_idx < self.layer_capacity:
                    if self.cur_ready_weights[target_idx] in [-1, -2] or self.cur_ready_weights[target_idx] == layer_id:
                        self.cur_ready_weights[target_idx] = layer_id
                    else:
                        status = f'STALE Wgt (Idx {target_idx} holds L{self.cur_ready_weights[target_idx]}, got L{layer_id})'
                else:
                    status = f'ERROR Wgt (Invalid Idx {target_idx})'
            elif target_buffer_type == 'bwd_fetched_act':
                item_type, buffer_name = 'Bwd Fetched Act', "Bwd Fetched Act Buf"
                act_tuple = (chunk_id, layer_id)
                effective_idx = target_idx % self.bwd_fetched_activation_buffer_capacity
                if 0 <= effective_idx < self.bwd_fetched_activation_buffer_capacity:
                    current_val = self.bwd_fetched_activation_buffer[effective_idx]
                    if current_val == -2 or current_val == act_tuple:
                        self.bwd_fetched_activation_buffer[effective_idx] = act_tuple
                    elif current_val != -1:
                        status = f'STALE Bwd Fetched Act (Idx {effective_idx} holds {current_val}, got {act_tuple})'
                    else:
                        self.bwd_fetched_activation_buffer[effective_idx] = act_tuple
                        status = f'WARN Bwd Fetched Act (Idx {effective_idx} was empty, placed {act_tuple})'
                else:
                    status = f'ERROR Bwd Fetched Act (Invalid Idx {effective_idx})'
            else:
                status = f'ERROR Unknown Buffer Type ({target_buffer_type}) for Inbound'

            if TO_PRINT or 'ERROR' in status or 'STALE' in status or 'WARN' in status:
                if chunk_id != -1:
                    log_msg = f"T={T}, Dev {self.device_id}: RX INBOUND {item_type} C{chunk_id},L{layer_id}"
                else:
                    log_msg = f"T={T}, Dev {self.device_id}: RX INBOUND {item_type} L{layer_id}"

                log_msg += f" -> Buf '{buffer_name}' Idx {target_idx} Complete."
                if status != 'OK':
                    log_msg = log_msg.replace("Complete.", status)
                print(log_msg)

            self.is_inbound_transferring = False
            self.cur_inbound_edge = ""

        if self.is_outbound_transferring and self.outbound_queue and (self.cur_outbound_start_time + self.cur_outbound_duration <= T):
            outbound_item = self.outbound_queue.popleft()
            chunk_id, layer_id, is_grad, duration = outbound_item
            storage_key = (chunk_id, layer_id, is_grad)
            if storage_key not in self.outbound_storage:
                self.outbound_storage.add(storage_key)
                if chunk_id >= 0:
                    item_type = 'Act'
                elif is_grad:
                    item_type = 'WgtGrad'
                else:
                    item_type = 'Wgt'

                if TO_PRINT:
                    print(f"T={T}, Dev {self.device_id}: TX OUTBOUND {item_type} C{chunk_id},L{layer_id} -> Storage COMPLETE. Key: {storage_key}")

            self.is_outbound_transferring = False
            self.cur_outbound_edge = ""

        if self.is_peer_transferring and self.peer_transfer_queue and (self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T):
            peer_item = self.peer_transfer_queue.popleft()
            peer_id, chunk_id, layer_id, is_grad, duration = peer_item
            peer_dev = all_devices[peer_id]
            item_key = (chunk_id, layer_id, is_grad)
            status, target_buffer_name, target_idx = "OK", "N/A", -1
            is_output_to_head_input = (layer_id == self.total_layers - 1) and (not is_grad)
            is_grad_from_head_output = (layer_id == self.total_layers) and is_grad
            target_buffer, write_ptr, buffer_capacity = None, -1, 0

            if is_output_to_head_input:
                target_buffer_name, target_buffer, write_ptr, buffer_capacity = "Model Outputs (Head In)", peer_dev.cur_model_outputs, peer_dev.cur_model_outputs_write_ptr, peer_dev.total_chunks
            elif is_grad_from_head_output:
                target_buffer_name, target_buffer, write_ptr, buffer_capacity = "Head Outputs (Bwd In)", peer_dev.cur_head_outputs, peer_dev.cur_head_outputs_write_ptr, peer_dev.total_chunks
            elif is_grad:
                target_buffer_name, target_buffer, write_ptr, buffer_capacity = "Bwd Grad Transitions", peer_dev.bwd_grad_transitions_buffer, peer_dev.bwd_grad_transitions_write_ptr, peer_dev.bwd_grad_transitions_capacity
            else:
                target_buffer_name, target_buffer, write_ptr, buffer_capacity = "Fwd Transitions", peer_dev.fwd_transitions_buffer, peer_dev.fwd_transitions_write_ptr, peer_dev.transitions_capacity

            if target_buffer is not None and 0 <= write_ptr < buffer_capacity:
                target_idx = write_ptr
                if target_buffer[target_idx] != -1:
                    status = f'WARN Peer Overwrite (Buf {target_buffer_name}, Idx {target_idx} held {target_buffer[target_idx]}, got {item_key})'
                target_buffer[target_idx] = item_key
                if is_output_to_head_input:
                    peer_dev.cur_model_outputs_write_ptr = (target_idx + 1) % buffer_capacity
                elif is_grad_from_head_output:
                    peer_dev.cur_head_outputs_write_ptr = (target_idx + 1) % buffer_capacity
                elif is_grad:
                    peer_dev.bwd_grad_transitions_write_ptr = (target_idx + 1) % buffer_capacity
                else:
                    peer_dev.fwd_transitions_write_ptr = (target_idx + 1) % buffer_capacity
            else:
                status = f'ERROR Peer TX (Invalid Idx {write_ptr} for Peer {peer_id} Buf {target_buffer_name} size {buffer_capacity})'

            item_type = 'Unknown'
            if is_grad:
                if layer_id < self.total_layers:
                    item_type = 'Bwd Grad'
                else:
                    item_type = 'Head Grad'
            else:
                item_type = 'Fwd Output'

            if TO_PRINT or 'ERROR' in status or 'WARN' in status:
                log_msg = f"T={T}, Dev {self.device_id}: TX PEER SEND {item_type} C{chunk_id},L{layer_id} -> Dev {peer_id} Complete. (Data {item_key} written to Dev {peer_id} Buf '{target_buffer_name}' Idx {target_idx}). Status: {status}"
                print(log_msg)

            self.is_peer_transferring = False
            self.cur_ring_edge = ""
            self.cur_peer_transfer_details = None

    def handle_computation_depends(self, T): # Unchanged logic, refactored style
        if not self.computation_queue:
            return False
        next_task = self.computation_queue[0]
        cid, lid, bX, bW, tdir, dur = next_task
        has_deps = False
        is_fwd = (not bX) and (not bW) and (lid < self.total_layers)
        is_head = (lid == self.total_layers)
        computation_type_str = "Unknown"
        self.stall_reason = ""

        if is_fwd:
            computation_type_str = "Fwd"
            has_weight = (lid in self.cur_ready_weights)
            has_input_transition = True
            input_transition_key = None
            if lid > 0:
                input_transition_key = (cid, lid - 1, False)
                has_input_transition = any(isinstance(item, tuple) and item == input_transition_key for item in self.fwd_transitions_buffer if item != -1)
            has_deps = has_weight and has_input_transition
            if not has_deps:
                missing = []
                if not has_weight:
                    is_pending = any(q[0]==-1 and q[1]==lid and q[5]=='weight' for q in self.inbound_queue)
                    if is_pending:
                        missing.append(f"Weight L{lid} (Pend)")
                    else:
                        missing.append(f"Weight L{lid}")
                if lid > 0 and not has_input_transition:
                    missing.append(f"Input Fwd Act C{cid},L{lid-1}")

                if missing:
                    self.stall_reason = "Missing:\n" + "\n".join(missing)
                else:
                    self.stall_reason = "Unknown FWD Dep"

        elif is_head:
            computation_type_str = "Head"
            head_weight_lid = self.total_layers
            has_weight = (head_weight_lid in self.cur_ready_weights)
            input_key = (cid, self.total_layers - 1, False)
            has_input_from_last_block = any(isinstance(item, tuple) and item == input_key for item in self.cur_model_outputs if item != -1)
            has_deps = has_weight and has_input_from_last_block
            if not has_deps:
                missing = []
                if not has_weight:
                    is_pending = any(q[0]==-1 and q[1]==head_weight_lid and q[5]=='weight' for q in self.inbound_queue)
                    if is_pending:
                        missing.append(f"Head State L{head_weight_lid} (Pend)")
                    else:
                        missing.append(f"Head State L{head_weight_lid}")
                if not has_input_from_last_block:
                    missing.append(f"Input Act C{cid},L{self.total_layers-1}")

                if missing:
                    self.stall_reason = "Missing:\n" + "\n".join(missing)
                else:
                    self.stall_reason = "Unknown HEAD Dep"

        elif bX:
            computation_type_str = "Bwd X"
            has_weight = (lid in self.cur_ready_weights)
            has_upstream_grad = False
            upstream_grad_key = (cid, lid + 1, True)
            upstream_buffer_name, source_buffer_check = "None", None
            if lid == self.total_layers - 1:
                upstream_buffer_name, source_buffer_check = "Head Outputs", self.cur_head_outputs
            elif lid < self.total_layers - 1:
                upstream_buffer_name, source_buffer_check = "Bwd Grad Transitions", self.bwd_grad_transitions_buffer
            if source_buffer_check is not None:
                has_upstream_grad = any(isinstance(item, tuple) and item == upstream_grad_key for item in source_buffer_check if item != -1)
            has_deps = has_weight and has_upstream_grad
            if not has_deps:
                missing = []
                if not has_weight:
                    is_pending = any(q[0]==-1 and q[1]==lid and q[5]=='weight' for q in self.inbound_queue)
                    if is_pending:
                        missing.append(f"Weight L{lid} (Pend)")
                    else:
                         missing.append(f"Weight L{lid}")
                if not has_upstream_grad:
                    missing.append(f"Upstream Grad C{cid},L{lid+1} (from {upstream_buffer_name})")

                if missing:
                    self.stall_reason = "Missing:\n" + "\n".join(missing)
                else:
                    self.stall_reason = "Unknown BwdX Dep"

        elif bW:
            computation_type_str = "Bwd W"
            fwd_act_key, act_grad_key = (cid, lid), (cid, lid)
            has_fwd_activation, fwd_act_source = False, "None"
            if fwd_act_key in self.local_last_fwd_activations:
                has_fwd_activation, fwd_act_source = True, "Local Dict"
            elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.resident_checkpoint_activations if item != -1):
                has_fwd_activation, fwd_act_source = True, "Resident Chkpt Buf"
            elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.bwd_fetched_activation_buffer if item != -1):
                has_fwd_activation, fwd_act_source = True, "Bwd Fetched Buf"

            is_fwd_act_pending_fetch, pending_slot = False, -1
            if not has_fwd_activation:
                for idx, item in enumerate(self.bwd_fetched_activation_buffer):
                    if item == -2:
                        # Check if this slot is currently being filled by an active inbound transfer
                        is_slot_active = False
                        if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                            active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                            if active_in_buf_type == 'bwd_fetched_act' and active_in_target_idx == idx and active_in_cid == cid and active_in_lid == lid:
                                if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                    is_slot_active = True

                        # Check if this slot is queued but not yet active
                        is_slot_queued = False
                        if not is_slot_active:
                           for q_idx, q in enumerate(self.inbound_queue):
                               if q_idx == 0 and self.is_inbound_transferring: # Already checked active transfer
                                   continue
                               q_cid, q_lid, _, q_target_idx, _, q_buf_type = q
                               if q_buf_type == 'bwd_fetched_act' and q_target_idx == idx and q_cid == cid and q_lid == lid:
                                   is_slot_queued = True
                                   break

                        if is_slot_active or is_slot_queued:
                            is_fwd_act_pending_fetch, pending_slot = True, idx
                            break # Found the pending slot

            has_activation_grad = any(isinstance(item, tuple) and item == act_grad_key for item in self.cur_ready_grad_activations if item != -1)
            has_deps = has_fwd_activation and has_activation_grad
            if not has_deps:
                missing = []
                if not has_fwd_activation:
                    if is_fwd_act_pending_fetch:
                        pend_msg = f" (Pend Slot {pending_slot})"
                    else:
                        pend_msg = " (Not Found/Queued?)"
                    missing.append(f"Fwd Act C{cid},L{lid}{pend_msg}")
                if not has_activation_grad:
                    missing.append(f"Act Grad C{cid},L{lid}")

                if missing:
                    self.stall_reason = "Missing:\n" + "\n".join(missing)
                else:
                    self.stall_reason = "Unknown BwdW Dep"

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

            consumed_items = [] # Consumption logic remains the same
            if is_fwd and lid > 0 and input_transition_key:
                for idx, item in enumerate(self.fwd_transitions_buffer):
                    if isinstance(item, tuple) and item == input_transition_key:
                        self.fwd_transitions_buffer[idx] = -1
                        consumed_items.append(f"FwdTrans[{idx}]:{input_transition_key}")
                        break
            elif is_head:
                input_key = (cid, self.total_layers - 1, False)
                for idx, item in enumerate(self.cur_model_outputs):
                    if isinstance(item, tuple) and item == input_key:
                        self.cur_model_outputs[idx] = -1
                        consumed_items.append(f"ModelOut[{idx}]:{input_key}")
                        break
            elif bX:
                upstream_grad_key = (cid, lid + 1, True)
                source_buffer, buf_name = None, "N/A"
                if lid == self.total_layers - 1:
                    source_buffer, buf_name = self.cur_head_outputs, "HeadOut"
                elif lid < self.total_layers - 1:
                    source_buffer, buf_name = self.bwd_grad_transitions_buffer, "BwdGradTrans"
                if source_buffer is not None:
                    for idx, item in enumerate(source_buffer):
                        if isinstance(item, tuple) and item == upstream_grad_key:
                            source_buffer[idx] = -1
                            consumed_items.append(f"{buf_name}[{idx}]:{upstream_grad_key}")
                            break
            elif bW:
                fwd_act_key = (cid, lid)
                consumed_fwd_act = False
                if fwd_act_key in self.local_last_fwd_activations:
                    del self.local_last_fwd_activations[fwd_act_key]
                    consumed_items.append(f"LocalActDict:{fwd_act_key}")
                    consumed_fwd_act = True
                elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.resident_checkpoint_activations if item != -1):
                    for idx, item in enumerate(self.resident_checkpoint_activations):
                        if isinstance(item, tuple) and item == fwd_act_key:
                            self.resident_checkpoint_activations[idx] = -1
                            consumed_items.append(f"ResidentAct[{idx}]:{fwd_act_key}")
                            consumed_fwd_act = True
                            break
                elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.bwd_fetched_activation_buffer if item != -1):
                    for idx, item in enumerate(self.bwd_fetched_activation_buffer):
                         if isinstance(item, tuple) and item == fwd_act_key:
                             self.bwd_fetched_activation_buffer[idx] = -1
                             consumed_items.append(f"BwdFetchedAct[{idx}]:{fwd_act_key}")
                             consumed_fwd_act = True
                             break
                act_grad_key = (cid, lid)
                for idx, item in enumerate(self.cur_ready_grad_activations):
                    if isinstance(item, tuple) and item == act_grad_key:
                        self.cur_ready_grad_activations[idx] = -1
                        consumed_items.append(f"GradAct[{idx}]:{act_grad_key}")
                        break

            if TO_PRINT and consumed_items:
                print(f"      T={T}, Dev {self.device_id}: Consumed -> {', '.join(consumed_items)}")
            return True
        else:
            if not self.is_stalled:
                self.is_stalled = True
                self.stall_start_time = T
                self.computing_status = f"STALL:\n{computation_type_str}\nC{cid},L{lid}"
            self.is_computing = False
            self.current_computation_type = None
            self.current_computation_layer_id = -1
            return False

    def queue_bulk_bwd_prefetches(self, T, is_head_trigger=False): # Refactored style
        if not self.has_reversed:
            if TO_PRINT:
                print(f"T={T}, Dev {self.device_id}: WARNING - queue_bulk_bwd_prefetches called before reversing.")
            return

        if TO_PRINT:
            if is_head_trigger:
                trigger_type = "HEAD"
            else:
                trigger_type = "BWD_X"
            print(f"T={T}, Dev {self.device_id}: === Bulk BWD Prefetch Triggered ({trigger_type}) ===")
            print(f"  State Before -> Trigger Layer ID (Completed): L{self.bwd_prefetch_trigger_layer_id}, Next Weight Target: L{self.next_weight_layer_id}")

        layer_just_finished_task = self.bwd_prefetch_trigger_layer_id

        if is_head_trigger:
            layer_just_finished_task = self.last_fwd_layer_on_device
            if layer_just_finished_task < 0:
                if TO_PRINT:
                    print(f"  HEAD Trigger SKIPPED - Device had no forward layers.")
                return
        elif layer_just_finished_task < 0 :
            if TO_PRINT:
                print(f"  BWD_X Trigger SKIPPED - Tracked layer ID ({layer_just_finished_task}) is invalid.")
            return

        next_weight_layer_to_fetch = layer_just_finished_task - self.total_devices
        next_activation_layer_to_fetch = next_weight_layer_to_fetch

        if TO_PRINT:
            if is_head_trigger:
                 print(f"  Trigger Origin Layer: LHead(->{self.last_fwd_layer_on_device})")
            else:
                 print(f"  Trigger Origin Layer: L{layer_just_finished_task}")
            print(f"  Next Weight Layer to Fetch: L{next_weight_layer_to_fetch}")
            print(f"  Next Activation Layer to Fetch: L{next_activation_layer_to_fetch}")

        # Phase 1: Fetch Critical Activation for the layer just completed
        layer_for_phase_1 = layer_just_finished_task
        if layer_for_phase_1 >= 0:
            if are_chunks_same_seq:
                critical_chunk_id = self.total_chunks - 1
            else:
                critical_chunk_id = 0

            act_tuple_p1 = (critical_chunk_id, layer_for_phase_1)
            storage_key_act_p1 = (critical_chunk_id, layer_for_phase_1, False)

            if TO_PRINT:
                print(f"  Phase 1: Checking Critical Activation for L{layer_for_phase_1} -> Chunk {critical_chunk_id}")

            is_in_local = act_tuple_p1 in self.local_last_fwd_activations
            is_in_resident = any(isinstance(item, tuple) and item == act_tuple_p1 for item in self.resident_checkpoint_activations if item != -1)
            is_in_fetched = any(isinstance(item, tuple) and item == act_tuple_p1 for item in self.bwd_fetched_activation_buffer if item != -1)

            # Check if queued in inbound_queue (but not active)
            is_pending_queue = False
            for q_idx, q in enumerate(self.inbound_queue):
                 if q[0] == critical_chunk_id and q[1] == layer_for_phase_1 and q[5] == 'bwd_fetched_act':
                     if not (self.is_inbound_transferring and q_idx == 0): # Exclude the active transfer
                         is_pending_queue = True
                         break

            # Check if marked as pending (-2) in buffer and associated with an *active* transfer
            is_pending_buffer = False
            if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                 active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                 if active_in_buf_type == 'bwd_fetched_act' and active_in_cid == critical_chunk_id and active_in_lid == layer_for_phase_1:
                      eff_idx = active_in_target_idx % self.bwd_fetched_activation_buffer_capacity
                      if 0 <= eff_idx < len(self.bwd_fetched_activation_buffer) and self.bwd_fetched_activation_buffer[eff_idx] == -2:
                           if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                is_pending_buffer = True

            needs_fetch_p1 = (storage_key_act_p1 in self.outbound_storage and
                              not is_in_local and not is_in_resident and not is_in_fetched and
                              not is_pending_queue and not is_pending_buffer)

            if needs_fetch_p1:
                target_idx_p1 = self.bwd_fetched_activation_write_ptr
                effective_target_idx_p1 = target_idx_p1 % self.bwd_fetched_activation_buffer_capacity
                slot_is_pending = False

                # Check if the *specific target slot* is pending an active transfer
                if 0 <= effective_target_idx_p1 < len(self.bwd_fetched_activation_buffer) and self.bwd_fetched_activation_buffer[effective_target_idx_p1] == -2:
                    if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                        active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                        if active_in_buf_type == 'bwd_fetched_act' and active_in_target_idx == target_idx_p1: # Use original target_idx here
                             if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                 slot_is_pending = True

                if slot_is_pending:
                    if TO_PRINT:
                         print(f"  Phase 1: WARNING - Target BWD fetch slot {target_idx_p1} (Effective: {effective_target_idx_p1}) is already pending an active transfer! Skipping.")
                else:
                    if TO_PRINT:
                        print(f"    Queueing Act C{critical_chunk_id},L{layer_for_phase_1} -> Bwd Fetched Idx {target_idx_p1} (Effective: {effective_target_idx_p1})")
                    self.inbound_queue.append((critical_chunk_id, layer_for_phase_1, False, target_idx_p1, savedActivationsFrames, 'bwd_fetched_act'))
                    if 0 <= effective_target_idx_p1 < len(self.bwd_fetched_activation_buffer):
                        self.bwd_fetched_activation_buffer[effective_target_idx_p1] = -2 # Mark as pending
                    self.bwd_fetched_activation_write_ptr = (target_idx_p1 + 1) % self.bwd_fetched_activation_buffer_capacity
            elif TO_PRINT:
                status_p1 = []
                if is_in_local: status_p1.append("Local")
                if is_in_resident: status_p1.append("Resident")
                if is_in_fetched: status_p1.append("Fetched")
                if is_pending_queue or is_pending_buffer: status_p1.append("Pending")
                status_str = ', '.join(status_p1) if status_p1 else 'Not In Storage?'
                print(f"    Activation C{critical_chunk_id},L{layer_for_phase_1} already available ({status_str}).")

        # Phase 2: Fetch Weight for the *next* BWD layer
        layer_for_phase_2 = next_weight_layer_to_fetch
        queued_weight_fetch = False
        target_idx_p2 = -1
        if layer_for_phase_2 >= 0:
            if TO_PRINT:
                print(f"  Phase 2: Checking Weight for NEXT BWD L{layer_for_phase_2}")
            storage_key_wgt = (-1, layer_for_phase_2, False)
            is_in_buffer = (layer_for_phase_2 in self.cur_ready_weights)

            # Check if queued (but not active)
            is_in_queue = False
            for q_idx, q in enumerate(self.inbound_queue):
                if q[0] == -1 and q[1] == layer_for_phase_2 and q[5] == 'weight':
                    if not (self.is_inbound_transferring and q_idx == 0):
                         is_in_queue = True
                         break

            # Check if marked as pending (-2) in buffer and associated with an *active* transfer
            is_pending_buffer = False
            if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                 active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                 if active_in_buf_type == 'weight' and active_in_lid == layer_for_phase_2:
                      if 0 <= active_in_target_idx < len(self.cur_ready_weights) and self.cur_ready_weights[active_in_target_idx] == -2:
                           if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                is_pending_buffer = True

            needs_fetch_p2 = (storage_key_wgt in self.outbound_storage and
                              not is_in_buffer and not is_in_queue and not is_pending_buffer)

            if needs_fetch_p2: # Weight Protection Logic starts
                current_layer_weight_idx = -1
                try:
                    current_layer_weight_idx = self.cur_ready_weights.index(layer_just_finished_task)
                except ValueError:
                    current_layer_weight_idx = -1

                candidate_idx = self.cur_weight_replace_ind
                weight_in_candidate_slot = self.cur_ready_weights[candidate_idx] if 0 <= candidate_idx < self.layer_capacity else -1
                target_idx_p2 = -1 # Reset target index

                if candidate_idx == current_layer_weight_idx and current_layer_weight_idx != -1:
                    # Candidate slot holds the weight needed for the *current* layer's BWD W pass. Protect it.
                    if self.layer_capacity > 1:
                        alternative_idx = (candidate_idx + 1) % self.layer_capacity
                        weight_in_alternative_slot = self.cur_ready_weights[alternative_idx]
                        slot_is_pending = False
                        # Check if alternative slot is pending an active transfer
                        if self.cur_ready_weights[alternative_idx] == -2:
                             if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                                  active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                                  if active_in_buf_type == 'weight' and active_in_target_idx == alternative_idx:
                                       if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                            slot_is_pending = True

                        if weight_in_alternative_slot != layer_just_finished_task and not slot_is_pending:
                             target_idx_p2 = alternative_idx # Use alternative slot
                             if TO_PRINT:
                                 print(f"    LRU slot {candidate_idx} holds protected L{layer_just_finished_task}. Using alternative slot {target_idx_p2}.")
                        else:
                            if TO_PRINT:
                                 print(f"    LRU slot {candidate_idx} holds protected L{layer_just_finished_task}. Alternative slot {alternative_idx} also holds it, is pending, or capacity is 1. Cannot prefetch L{layer_for_phase_2} now.")
                            target_idx_p2 = -1 # Cannot use alternative
                    else: # Layer capacity is 1, cannot protect
                        if TO_PRINT:
                            print(f"    LRU slot {candidate_idx} holds protected L{layer_just_finished_task}. Layer capacity is 1. Cannot prefetch L{layer_for_phase_2} now.")
                        target_idx_p2 = -1
                else:
                    # Candidate slot does not hold the protected weight, check if it's pending
                    slot_is_pending = False
                    if weight_in_candidate_slot == -2:
                        if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                             active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                             if active_in_buf_type == 'weight' and active_in_target_idx == candidate_idx:
                                  if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                       slot_is_pending = True

                    if not slot_is_pending:
                        target_idx_p2 = candidate_idx # Use the candidate slot
                        if TO_PRINT:
                             print(f"    Using LRU slot {target_idx_p2} for eviction.")
                    else:
                         if TO_PRINT:
                              print(f"    LRU slot {candidate_idx} is already pending an active transfer. Cannot prefetch L{layer_for_phase_2} now.")
                         target_idx_p2 = -1

                # Queue the fetch if a valid target index was found
                if target_idx_p2 != -1:
                    weight_to_evict = self.cur_ready_weights[target_idx_p2]
                    if weight_to_evict >= 0:
                        evict_msg = f"L{weight_to_evict}"
                    elif weight_to_evict == -2:
                        evict_msg = 'Pending'
                    else:
                        evict_msg = 'Empty'

                    if TO_PRINT:
                        print(f"    Queueing Wgt L{layer_for_phase_2} -> Target Idx {target_idx_p2} (Evicting {evict_msg})")
                    self.inbound_queue.append((-1, layer_for_phase_2, False, target_idx_p2, layerTransferFrames, 'weight'))
                    self.cur_ready_weights[target_idx_p2] = -2 # Mark target slot as pending
                    self.cur_weight_replace_ind = (target_idx_p2 + 1) % self.layer_capacity
                    queued_weight_fetch = True
            elif TO_PRINT:
                status_p2 = []
                if is_in_buffer: status_p2.append("InBuffer")
                if is_in_queue or is_pending_buffer: status_p2.append("Pending")
                status_str = ', '.join(status_p2) if status_p2 else 'Not In Storage?'
                print(f"    Weight L{layer_for_phase_2} already available ({status_str}).")

        # Phase 3: Fetch Bulk Activations for the *next* BWD layer
        layer_for_phase_3 = next_activation_layer_to_fetch
        if layer_for_phase_3 >= 0:
            if TO_PRINT:
                print(f"  Phase 3: Checking Bulk Activations for L{layer_for_phase_3} -> Bwd Fetched Buffer")

            # Determine the critical chunk ID fetched in Phase 1 for this layer (if applicable)
            critical_chunk_id_p1_for_p3_layer = -1
            if layer_for_phase_1 == layer_for_phase_3:
                if are_chunks_same_seq:
                    critical_chunk_id_p1_for_p3_layer = self.total_chunks - 1
                else:
                    critical_chunk_id_p1_for_p3_layer = 0

            bulk_chunk_order = []
            if are_chunks_same_seq:
                bulk_chunk_order = [c for c in range(self.total_chunks - 1, -1, -1) if c != critical_chunk_id_p1_for_p3_layer]
            else:
                bulk_chunk_order = [c for c in range(self.total_chunks) if c != critical_chunk_id_p1_for_p3_layer]

            num_queued_p3 = 0
            for chunk_id in bulk_chunk_order:
                act_tuple_p3 = (chunk_id, layer_for_phase_3)
                storage_key_act_p3 = (chunk_id, layer_for_phase_3, False)

                is_in_local_p3 = act_tuple_p3 in self.local_last_fwd_activations
                is_in_resident_p3 = any(isinstance(item, tuple) and item == act_tuple_p3 for item in self.resident_checkpoint_activations if item != -1)
                is_in_fetched_p3 = any(isinstance(item, tuple) and item == act_tuple_p3 for item in self.bwd_fetched_activation_buffer if item != -1)

                # Check if queued (but not active)
                is_pending_queue_p3 = False
                for q_idx, q in enumerate(self.inbound_queue):
                    if q[0] == chunk_id and q[1] == layer_for_phase_3 and q[5] == 'bwd_fetched_act':
                         if not (self.is_inbound_transferring and q_idx == 0):
                             is_pending_queue_p3 = True
                             break

                # Check if marked as pending (-2) in buffer and associated with an *active* transfer
                is_pending_buffer_p3 = False
                if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                     active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                     if active_in_buf_type == 'bwd_fetched_act' and active_in_cid == chunk_id and active_in_lid == layer_for_phase_3:
                          eff_idx = active_in_target_idx % self.bwd_fetched_activation_buffer_capacity
                          if 0 <= eff_idx < len(self.bwd_fetched_activation_buffer) and self.bwd_fetched_activation_buffer[eff_idx] == -2:
                               if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                    is_pending_buffer_p3 = True

                needs_fetch_p3 = (storage_key_act_p3 in self.outbound_storage and
                                  not is_in_local_p3 and not is_in_resident_p3 and not is_in_fetched_p3 and
                                  not is_pending_queue_p3 and not is_pending_buffer_p3)

                if needs_fetch_p3:
                    target_idx_p3 = self.bwd_fetched_activation_write_ptr
                    effective_target_idx_p3 = target_idx_p3 % self.bwd_fetched_activation_buffer_capacity
                    slot_is_pending = False

                    # Check if the *specific target slot* is pending an active transfer
                    if 0 <= effective_target_idx_p3 < len(self.bwd_fetched_activation_buffer) and self.bwd_fetched_activation_buffer[effective_target_idx_p3] == -2:
                        if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                             active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                             if active_in_buf_type == 'bwd_fetched_act' and active_in_target_idx == target_idx_p3: # Use original target_idx
                                  if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                      slot_is_pending = True

                    if slot_is_pending:
                        if TO_PRINT:
                             print(f"  Phase 3: WARNING - Target BWD fetch slot {target_idx_p3} (Effective: {effective_target_idx_p3}) is already pending an active transfer! Skipping C{chunk_id},L{layer_for_phase_3}.")
                    else:
                        self.inbound_queue.append((chunk_id, layer_for_phase_3, False, target_idx_p3, savedActivationsFrames, 'bwd_fetched_act'))
                        if 0 <= effective_target_idx_p3 < len(self.bwd_fetched_activation_buffer):
                            self.bwd_fetched_activation_buffer[effective_target_idx_p3] = -2 # Mark as pending
                        self.bwd_fetched_activation_write_ptr = (target_idx_p3 + 1) % self.bwd_fetched_activation_buffer_capacity
                        num_queued_p3 += 1
            if TO_PRINT:
                print(f"    Queued {num_queued_p3} bulk activations for L{layer_for_phase_3}.")
        elif TO_PRINT:
            print("  Phase 3: No valid layer to fetch bulk activations for.")

        # Update the next weight layer ID we'll need after the one we just (tried to) queue
        potential_next_weight_id = next_weight_layer_to_fetch - self.total_devices
        if potential_next_weight_id < 0:
            self.next_weight_layer_id = None
        else:
            self.next_weight_layer_id = potential_next_weight_id

        if TO_PRINT:
            next_wgt_str = f"L{self.next_weight_layer_id}" if self.next_weight_layer_id is not None else 'None'
            print(f"  Updated State -> Next Trigger Layer ID Target: L{layer_just_finished_task - self.total_devices}, Next Weight Target: {next_wgt_str}")
            print(f"T={T}, Dev {self.device_id}: === Bulk BWD Prefetch Queuing Complete ===")

    def handle_computation(self, T, all_devices): # Refactored style
        completed_tasks = 0
        if not self.is_computing and not self.is_stalled and len(self.computation_queue) > 0:
            self.handle_computation_depends(T)
        elif self.is_computing and (self.cur_computation_start_time + self.cur_computation_duration <= T):
            task = self.computation_queue.pop(0)
            completed_tasks += 1
            cid, lid, bX, bW, tdir, dur = task
            is_head = (lid == self.total_layers)
            is_fwd = (not bX) and (not bW) and (not is_head)
            task_type_str = self.current_computation_type

            if TO_PRINT:
                print(f"T={T}, Dev {self.device_id}: FINISHED Comp -> C{cid},L{lid},{task_type_str}")

            if is_fwd: # FWD Finished
                act_tuple = (cid, lid)
                save_locally_hot, save_resident = False, False
                if lid == self.last_fwd_layer_on_device:
                    save_locally_hot = True

                if do_backward and self.activations_capacity > 0:
                    if are_chunks_same_seq and cid >= self.total_chunks - self.activations_capacity:
                        save_resident = True
                    elif not are_chunks_same_seq and cid < self.activations_capacity:
                        save_resident = True

                if save_locally_hot:
                    self.local_last_fwd_activations[act_tuple] = True

                if save_resident:
                    res_idx = self.resident_checkpoint_write_ptr
                    if 0 <= res_idx < self.activations_capacity:
                        self.resident_checkpoint_activations[res_idx] = act_tuple
                        self.resident_checkpoint_write_ptr = (res_idx + 1) % self.activations_capacity

                if do_backward:
                    storage_key = (cid, lid, False)
                    is_in_queue = any(q[0]==cid and q[1]==lid and q[2]==False for q in self.outbound_queue)
                    if storage_key not in self.outbound_storage and not is_in_queue:
                        self.outbound_queue.append((cid, lid, False, savedActivationsFrames))

                if tdir != 0: # FWD Peer Transfer
                    target_peer_id = (self.device_id + tdir + self.total_devices) % self.total_devices
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, False, activationTransitionFrames))
                    if TO_PRINT:
                        dest_type = "Head Input" if lid == self.total_layers - 1 else "Fwd Transition"
                        print(f"      Queueing FWD Output C{cid},L{lid} -> Peer {target_peer_id} ({dest_type})")

                if cid == self.total_chunks - 1: # FWD Prefetching & Reversal Logic
                    layer_to_fetch = self.next_weight_layer_id
                    needs_fetch = False
                    if layer_to_fetch is not None and layer_to_fetch <= self.total_layers :
                        storage_key_wgt = (-1, layer_to_fetch, False)
                        in_storage = storage_key_wgt in self.outbound_storage
                        if in_storage:
                            is_in_buffer = (layer_to_fetch in self.cur_ready_weights)
                            # Check if queued (but not active)
                            is_in_queue = False
                            for q_idx, q in enumerate(self.inbound_queue):
                                if q[0]==-1 and q[1]==layer_to_fetch and q[5]=='weight':
                                    if not (self.is_inbound_transferring and q_idx==0):
                                        is_in_queue = True
                                        break
                            # Check if marked pending in buffer and associated with active transfer
                            is_pending_buffer = False
                            if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                                active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                                if active_in_buf_type == 'weight' and active_in_lid == layer_to_fetch:
                                    if 0 <= active_in_target_idx < len(self.cur_ready_weights) and self.cur_ready_weights[active_in_target_idx] == -2:
                                        if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                            is_pending_buffer = True

                            if not is_in_buffer and not is_in_queue and not is_pending_buffer:
                                needs_fetch = True

                    if needs_fetch:
                        target_idx_fwd = self.cur_weight_replace_ind
                        slot_is_pending = False
                        # Check if target slot is pending an active transfer
                        if self.cur_ready_weights[target_idx_fwd] == -2:
                            if self.is_inbound_transferring and len(self.inbound_queue) > 0:
                                active_in_cid, active_in_lid, _, active_in_target_idx, _, active_in_buf_type = self.inbound_queue[0]
                                if active_in_buf_type == 'weight' and active_in_target_idx == target_idx_fwd:
                                    if T < (self.cur_inbound_start_time + self.cur_inbound_duration):
                                        slot_is_pending = True
                        if slot_is_pending:
                            if TO_PRINT:
                                print(f"  FWD Prefetch: WARNING - Target weight slot {target_idx_fwd} is already pending an active transfer! Skipping L{layer_to_fetch}.")
                        else:
                            weight_to_evict = self.cur_ready_weights[target_idx_fwd]
                            if weight_to_evict >= 0:
                                evict_msg = f"L{weight_to_evict}"
                            elif weight_to_evict == -2:
                                evict_msg = 'Pending'
                            else:
                                evict_msg = 'Empty'

                            if TO_PRINT:
                                print(f"      Queueing FWD Wgt L{layer_to_fetch} -> Target Idx {target_idx_fwd} (Evicting {evict_msg})")
                            self.inbound_queue.append((-1, layer_to_fetch, False, target_idx_fwd, layerTransferFrames, 'weight'))
                            self.cur_ready_weights[target_idx_fwd] = -2 # Mark as pending
                            self.cur_weight_replace_ind = (target_idx_fwd + 1) % self.layer_capacity

                    next_layer_id_after_fetch = layer_to_fetch + self.total_devices if layer_to_fetch is not None else None
                    will_reverse = do_backward and not self.has_reversed and (layer_to_fetch is None or next_layer_id_after_fetch > self.total_layers or layer_to_fetch == self.total_layers)

                    if will_reverse:
                        self.has_reversed = True
                        self.bwd_fetched_activation_write_ptr = 0
                        self.cur_grad_activations_write_ptr = 0
                        self.bwd_grad_transitions_write_ptr = 0
                        if TO_PRINT:
                            print(f"T={T}, Dev {self.device_id}: >>> REVERSING state after FWD C{cid},L{lid}.")
                        if layer_to_fetch == self.total_layers:
                            self.bwd_prefetch_trigger_layer_id = self.total_layers # Trigger is the head layer itself
                        else:
                            self.bwd_prefetch_trigger_layer_id = self.last_fwd_layer_on_device # Trigger is the last layer computed FWD
                        self.next_weight_layer_id = None # No more FWD weights needed
                    elif not self.has_reversed:
                        # Update next weight ID for FWD prefetching
                        if next_layer_id_after_fetch is not None and next_layer_id_after_fetch <= self.total_layers:
                            self.next_weight_layer_id = next_layer_id_after_fetch
                        else:
                            self.next_weight_layer_id = None

            elif is_head: # Head finished
                if tdir != 0: # Head Peer Transfer (Send Grad to Prev Device)
                    target_peer_id = (self.device_id + tdir + self.total_devices) % self.total_devices
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, True, activationTransitionFrames))
                    if TO_PRINT:
                        print(f"      Queueing Head Grad C{cid},L{lid} -> Peer {target_peer_id} (Head Output)")

                if cid == self.head_final_chunk_id: # Head Post-computation & BWD Trigger
                    # Queue save of Head state/gradient
                    storage_key = (-1, lid, True) # True for gradient/state
                    is_in_queue = any(q[0]==-1 and q[1]==lid and q[2]==True for q in self.outbound_queue)
                    if storage_key not in self.outbound_storage and not is_in_queue:
                        self.outbound_queue.append((-1, lid, True, layerTransferFrames)) # Using layerTransferFrames as placeholder size

                    # Trigger BWD prefetching for the *first* BWD layer on this device
                    if do_backward and self.has_reversed:
                         self.bwd_prefetch_trigger_layer_id = lid # Set trigger ID to the Head layer ID
                         self.queue_bulk_bwd_prefetches(T, is_head_trigger=True)
                    elif TO_PRINT and not self.has_reversed:
                         print(f"T={T}, Dev {self.device_id}: WARNING - Head finished final chunk C{cid} but device has not reversed!")

            elif bX: # BwdX finished
                idx_grad = self.cur_grad_activations_write_ptr
                grad_act_tuple = (cid, lid)
                if 0 <= idx_grad < self.grad_activations_capacity:
                    self.cur_ready_grad_activations[idx_grad] = grad_act_tuple
                    self.cur_grad_activations_write_ptr = (idx_grad + 1) % self.grad_activations_capacity

                if tdir != 0: # BwdX Peer Transfer (Send dL/dX to Prev Device)
                    target_peer_id = (self.device_id + tdir + self.total_devices) % self.total_devices
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, True, activationTransitionFrames))
                    if TO_PRINT:
                        print(f"      Queueing Bwd Grad (dL/dX) C{cid},L{lid} -> Peer {target_peer_id} (Bwd Grad Trans)")

                # Trigger BWD prefetch for the *next* layer after processing the 'critical' chunk for BWD X
                if are_chunks_same_seq:
                    trigger_chunk_id = 0 # Trigger after processing the *last* chunk's grad input (C0 for reversed)
                else:
                    trigger_chunk_id = self.total_chunks - 1 # Trigger after processing the *last* chunk's grad input (C_last for fwd order)

                if cid == trigger_chunk_id and self.has_reversed: # Check if reversed state is active
                    if TO_PRINT:
                        print(f"T={T}, Dev {self.device_id}: BwdX C{cid},L{lid} finished critical chunk, triggering next prefetch.")
                    self.bwd_prefetch_trigger_layer_id = lid # Set trigger to the layer just finished BWD X
                    self.queue_bulk_bwd_prefetches(T, is_head_trigger=False)

            elif bW: # BwdW finished
                # After the BWD W pass for the *last* chunk completes, queue the weight grad save
                if are_chunks_same_seq:
                     last_chunk_id_for_bwdW = 0 # Last chunk processed in BWD W is C0
                else:
                     last_chunk_id_for_bwdW = self.total_chunks - 1 # Last chunk is C_last

                if cid == last_chunk_id_for_bwdW:
                    storage_key = (-1, lid, True) # True indicates gradient
                    is_in_queue = any(q[0]==-1 and q[1]==lid and q[2]==True for q in self.outbound_queue)
                    if storage_key not in self.outbound_storage and not is_in_queue:
                        self.outbound_queue.append((-1, lid, True, layerTransferFrames))

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

    def handle_new_transfers(self, T): # Refactored style
        if not self.is_inbound_transferring and self.inbound_queue:
            item = self.inbound_queue[0]
            cid, lid, isg, target_idx, duration, target_buffer_type = item

            if cid == -1: # Weight or Head state fetch
                 actual_storage_key = (-1, lid, isg)
            else: # Activation fetch
                 actual_storage_key = (cid, lid, False)

            is_in_storage = actual_storage_key in self.outbound_storage
            if is_in_storage:
                self.is_inbound_transferring = True
                self.cur_inbound_start_time = T
                self.cur_inbound_duration = duration

                edge_color = COLOR_INBOUND_DEFAULT
                if target_buffer_type == 'weight':
                     if lid == total_layers:
                         label_lid_str = f"Wgt:\nHead"
                     else:
                         label_lid_str = f"Wgt:\nL{lid}"
                     self.cur_inbound_edge = label_lid_str
                     edge_color = COLOR_INBOUND_WEIGHT
                elif target_buffer_type == 'bwd_fetched_act':
                     self.cur_inbound_edge = f"Act:\nC{cid},L{lid}"
                     edge_color = COLOR_INBOUND_BWD_FETCHED_ACTIVATION
                else:
                     self.cur_inbound_edge = f"UNKNOWN\nC{cid},L{lid}"

                arrow_vis, _ = edge_artists[f'in_{self.device_id}']
                arrow_vis.set_color(edge_color)
            else:
                if TO_PRINT:
                    print(f"T={T}, Dev {self.device_id}: ERROR - Inbound request {item} with key {actual_storage_key} not found in storage. Removing from queue.")
                self.inbound_queue.popleft() # Remove invalid request

        if not self.is_outbound_transferring and self.outbound_queue:
            item = self.outbound_queue[0]
            cid, lid, isg, duration = item
            self.is_outbound_transferring = True
            self.cur_outbound_start_time = T
            self.cur_outbound_duration = duration

            edge_color = COLOR_OUTBOUND_DEFAULT
            if cid >= 0: # Activation save
                self.cur_outbound_edge = f"Act:\nC{cid},L{lid}"
                edge_color = COLOR_OUTBOUND_FWD_ACTIVATION
            elif isg: # Weight Gradient save
                if lid == total_layers:
                     label_lid_str = f"Grad:\nHead"
                else:
                     label_lid_str = f"Grad:\nL{lid}"
                self.cur_outbound_edge = label_lid_str
                edge_color = COLOR_OUTBOUND_WGT_GRAD
            else: # Weight save (shouldn't happen often unless evicting?)
                self.cur_outbound_edge = f"Wgt:\nL{lid}"
                edge_color = COLOR_OUTBOUND_DEFAULT # Or maybe a different color?

            arrow_vis, _ = edge_artists[f'out_{self.device_id}']
            arrow_vis.set_color(edge_color)

        if not self.is_peer_transferring and self.peer_transfer_queue:
            item = self.peer_transfer_queue[0]
            pid, cid, lid, isg, duration = item
            self.is_peer_transferring = True
            self.cur_peer_transfer_start_time = T
            self.cur_peer_transfer_duration = duration
            self.cur_peer_transfer_details = (pid, cid, lid, isg)

            if lid == total_layers:
                 label_lid_str = "Head"
            else:
                 label_lid_str = str(lid)

            is_gradient_data = isg
            if is_gradient_data:
                edge_color = COLOR_RING_CW
                connection_style_ring = f"arc3,rad=-0.2" # Negative radius for CW arc
                if lid == total_layers:
                    self.cur_ring_edge = f"Grad:\nC{cid},Head"
                else:
                    self.cur_ring_edge = f"Grad:\nC{cid},L{lid}"
            else: # Forward activation/output transfer
                edge_color = COLOR_RING_CCW
                connection_style_ring = f"arc3,rad=0.2" # Positive radius for CCW arc
                self.cur_ring_edge = f"Out:\nC{cid},L{lid}"

            arrow_vis, _ = edge_artists[f'ring_{self.device_id}']
            arrow_vis.set_color(edge_color)
            arrow_vis.set_connectionstyle(connection_style_ring)

    def print_buffer_status(self): # Refactored style
        if 'current_frame_index' in globals():
             current_T = current_frame_index
        else:
             current_T = 'N/A'
        print(f"--- Dev {self.device_id} Buffer Status (T={current_T}) ---")
        def format_item(item):
            if isinstance(item, tuple):
                if len(item) == 3:
                    id1, id2, flag = item
                    flag_str = 'G' if flag else 'A'
                    return f"({id1},{id2},{flag_str})"
                elif len(item) == 2:
                    return f"({item[0]},{item[1]})"
                else:
                    return str(item) # Should not happen based on usage
            elif item == -2:
                return 'Pend' # Represents a slot pending data arrival
            elif item >= 0:
                return f"L{item}" # Represents a loaded layer weight/state
            else: # item == -1
                return '_' # Represents an empty slot

        wgt_buf_str = ', '.join([format_item(w) for w in self.cur_ready_weights])
        print(f"  Ready Wgt:                  [{wgt_buf_str}] (Next Evict Idx: {self.cur_weight_replace_ind})")
        fwd_trans_buf_str = ', '.join([format_item(t) for t in self.fwd_transitions_buffer])
        print(f"  Fwd Transitions Buf:        [{fwd_trans_buf_str}] (Cap: {self.transitions_capacity}, Next Write Idx: {self.fwd_transitions_write_ptr})")
        bwd_grad_trans_buf_str = ', '.join([format_item(t) for t in self.bwd_grad_transitions_buffer])
        print(f"  Bwd Grad Transitions Buf:   [{bwd_grad_trans_buf_str}] (Cap: {self.bwd_grad_transitions_capacity}, Next Write Idx: {self.bwd_grad_transitions_write_ptr})")
        print(f"  Local Last Fwd Act (Hot):   {sorted(list(self.local_last_fwd_activations.keys()))}")
        res_act_buf_str = ', '.join([format_item(a) for a in self.resident_checkpoint_activations])
        print(f"  Resident Chkpt Act Buf:     [{res_act_buf_str}] (Cap: {self.activations_capacity}, Next Write Idx: {self.resident_checkpoint_write_ptr})")
        bwd_fetch_buf_str = ', '.join([format_item(a) for a in self.bwd_fetched_activation_buffer])
        print(f"  BWD Fetched Act Buf:        [{bwd_fetch_buf_str}] (Cap: {self.bwd_fetched_activation_buffer_capacity}, Next Write Idx: {self.bwd_fetched_activation_write_ptr})")
        grad_act_buf_str = ', '.join([format_item(g) for g in self.cur_ready_grad_activations])
        print(f"  Ready GradAct (dL/dA):      [{grad_act_buf_str}] (Cap: {self.grad_activations_capacity}, Next Write Idx: {self.cur_grad_activations_write_ptr})")
        model_out_buf_str = ', '.join([format_item(m) for m in self.cur_model_outputs])
        print(f"  Model Outputs (Head In):    [{model_out_buf_str}] (Next Write Idx: {self.cur_model_outputs_write_ptr})")
        head_out_buf_str = ', '.join([format_item(h) for h in self.cur_head_outputs])
        print(f"  Head Outputs (Bwd In):      [{head_out_buf_str}] (Next Write Idx: {self.cur_head_outputs_write_ptr})")
        print(f"  Queue Lengths: Comp={len(self.computation_queue)}, In={len(self.inbound_queue)}, Out={len(self.outbound_queue)}, PeerSend={len(self.peer_transfer_queue)}")
        next_wgt_str = f"L{self.next_weight_layer_id}" if self.next_weight_layer_id is not None else 'None'
        print(f"  State: Reversed={self.has_reversed}, Last BWD Trigger Layer=L{self.bwd_prefetch_trigger_layer_id}, Next Wgt Target={next_wgt_str}")
        print(f"-------------------------------------")


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
    # ***** FIX: Reset new state variable *****
    global simulation_computation_finish_time
    # *****************************************
    global N

    if TO_PRINT:
        print("\n" + "="*20 + " Resetting Simulation " + "="*20)

    all_devices = {i: Device(i, layer_capacity, activations_capacity, transitions_capacity,
                              N, total_layers, total_chunks)
                   for i in range(N)}

    total_tasks = sum(len(d.computation_queue) for d in all_devices.values())
    total_computation_time = sum(task[-1] for i in range(N) for task in all_devices[i].computation_queue)

    total_completed_tasks = {-1: 0} # Initialize with base case for frame 0
    current_frame_index = 0
    animation_paused = False
    target_cycle = None
    # ***** FIX: Reset new state variable *****
    simulation_computation_finish_time = None
    # *****************************************

    if completion_text_artist is not None:
        if completion_text_artist.axes is not None:
            completion_text_artist.remove()
        completion_text_artist = None

    # Reset visuals (Unchanged logic, just layout)
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
        if f'circle_{i}' in device_label_artists:
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
        newly_completed_this_cycle += all_devices[i].handle_computation(T, all_devices)
    # Update total completed tasks *after* all devices have potentially finished a task
    total_completed_tasks[T] = total_completed_tasks.get(T, 0) + newly_completed_this_cycle

    # 3. Start New Transfers: Initiate pending transfers if channels are free.
    for i in range(N):
        all_devices[i].handle_new_transfers(T)

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
        if device.is_stalled and device.stall_reason:
            wrapped_stall_reason = textwrap.fill(device.stall_reason.replace("\n", " "), width=15)
            stall_label_artist.set_text(wrapped_stall_reason)
            stall_label_artist.set_visible(True)
            stall_node_artist.set_visible(True)
        else:
            stall_label_artist.set_text("")
            stall_label_artist.set_visible(False)
            stall_node_artist.set_visible(False)

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
                                  arrow_vis_inbound, label_vis_inbound, arrow_vis_outbound, label_vis_outbound,
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

        matmul_flop_ratio = total_matmul_flops / total_flops if total_flops > 0 else 0
        attn_flop_ratio = total_attn_flops / total_flops if total_flops > 0 else 0

        practical_efficiency = matmul_efficiency * matmul_flop_ratio + attn_efficiency * attn_flop_ratio

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
             f"Upper-Bound based on Matmul/Attn Efficiency: {int((hardware_max_flops * practical_efficiency) / 1e12)} TFLOPS\n"
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
                if TO_PRINT:
                    print("\n" + "="*20 + " Max Frames Reached " + "="*20)
                    print(completion_text)
                    for i in range(N):
                        all_devices[i].print_buffer_status()
                    print("="*60 + "\n")

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

# --- Display ---
print("Initializing display...")
reset_simulation() # Initial setup
print("Showing plot...")
plt.show()
print("Plot window closed.")