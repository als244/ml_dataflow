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
#import pdb; pdb.set_trace()


# --- Global Control Flags ---
TO_PRINT = True # ENABLE DEBUG PRINTING

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

# --- Simulation Parameters ---
N = 8 # Number of devices
computationFrames = 50 # Cycles per compute task
layerTransferFrames = N * computationFrames # Cycles to transfer weights
savedActivationsFrames = computationFrames # Cycles to transfer activations (save/fetch)
activationTransitionFrames = 5 # Cycles to transfer activations/grads between devices

total_layers = 32
total_chunks = 16
layer_capacity = 2 # Max weights per device
activations_capacity = 4 # Max CHECKPOINTED activations kept resident from FWD per device
transitions_capacity = N # Buffer size for incoming data from peers
grad_activations_capacity = total_chunks # Buffer size for local dL/dA results

do_backward = True
are_chunks_same_seq = True # True=Pipeline Parallel (BWD Reversed), False=Data Parallel(BWD Same Order)

max_frames = 100000 # Limit animation length for performance if needed

# --- Speed/Interval Parameters ---
min_speed_level = 1
max_speed_level = 100
min_interval = 1
max_interval = 100
initial_speed_level = 50

def calculate_interval(speed_level, s_min, s_max, i_min, i_max):
    """Linearly maps speed level (s_min to s_max) to interval (i_max to i_min)."""
    if s_max == s_min:
        interval_range = 0 # Avoid division by zero if s_max == s_min
    else:
        interval_range = i_min - i_max

    if s_max == s_min: # Handle edge case where speed range is zero
        return i_min
    else:
        speed_range = s_max - s_min
        if speed_range == 0:
            return i_min # Should not happen if s_max != s_min, but safety check
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
# Network -> Device (Visually Outward from Center)
COLOR_INBOUND_DEFAULT = 'gray' # Default color if type unknown
COLOR_INBOUND_WEIGHT = 'olive'
COLOR_INBOUND_BWD_FETCHED_ACTIVATION = 'deepskyblue' # Activations fetched during BWD pass

# Device -> Network (Visually Inward toward Center)
COLOR_OUTBOUND_DEFAULT = 'gray' # Default color if type unknown
COLOR_OUTBOUND_FWD_ACTIVATION = 'magenta' # Forward pass activation saving to storage
COLOR_OUTBOUND_WGT_GRAD = 'turquoise' # Weight gradient saving to storage

# Ring Transfers (Device -> Device)
COLOR_RING_CCW = 'indigo' # FWD Activations / Head Inputs
COLOR_RING_CW = 'maroon'  # BWD Gradients

# --- Computation Arc Colors ---
COLOR_COMPUTE_DEFAULT = 'gray'
COLOR_COMPUTE_FWD = 'darkgreen'
COLOR_COMPUTE_BWD_X = 'orangered'
COLOR_COMPUTE_BWD_W = 'teal'
COLOR_COMPUTE_HEAD = 'lawngreen'


# Use the stall_node_opacity defined earlier directly in the RGBA tuple
COLOR_STALL_NODE_FILL = (1.0, 0.0, 0.0, stall_node_opacity) # Red with alpha


# --- Setup ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
lim = total_distance + outer_node_radius + stall_node_distance_offset + stall_node_radius + 0.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axis('off')
center_pos = np.array([0, 0])

# --- Legend Text ---
wrap_width = 40 # Adjust as needed for legend box width
legend_text = (
    f"Simulated Configuration:\n\n"
    f"      - Num Layers: {total_layers}\n"
    f"      - Num Devices: {N}\n"
    f"      - Num Chunks: {total_chunks}\n"
)

if are_chunks_same_seq:
    legend_text += f"      - Chunk Order: Pipeline Parallel (BWD Reversed)\n"
else:
    legend_text += f"      - Chunk Order: Data Parallel (BWD Same Order)\n"

if do_backward:
    legend_text += f"      - Do Backward: Yes\n"
else:
    legend_text += f"      - Do Backward: No\n"

legend_text += (
    f"      - Per-Device Layer (Weight) Capacity: {layer_capacity}\n"
    f"      - Per-Device Resident Activation Capacity: {activations_capacity}\n"
    f"      - Per-Device Transition Capacity: {transitions_capacity}\n"
    f"      - Per-Device Grad Activation Capacity: {grad_activations_capacity}\n"
    f"      - Constants:\n"
    f"            - Layer Computation: {computationFrames} Cycles\n"
    f"            - Layer Transfer: {layerTransferFrames} Cycles\n"
    f"            - Activation Transfer: {savedActivationsFrames} Cycles\n"
    f"            - Block Transition: {activationTransitionFrames} Cycles\n"
)


# Wrap legend text
wrapped_legend_text = legend_text

at = AnchoredText(wrapped_legend_text, loc='upper left', bbox_to_anchor=(1.01, 1.01),
                  prop=dict(size=10), frameon=True, pad=0.4, borderpad=0.5,
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
edge_artists = {} # Arrows AND compute arcs
device_label_artists = {}

initial_cycle = 0
title_obj = ax.set_title(f"Cycle {initial_cycle}", fontsize=title_fontsize, fontweight='bold')

# --- Create Outer, Inner, and Stall Nodes ---
for i in range(N):
    angle = np.linspace(0, 2 * np.pi, N, endpoint=False)[i]
    unit_dir = np.array([np.cos(angle), np.sin(angle)])
    unit_directions.append(unit_dir)
    outer_pos = center_pos + total_distance * unit_dir
    outer_circle_positions.append(outer_pos)
    color = cmap(norm(i))

    # Outer Circle (Device)
    outer_circle = patches.Circle(outer_pos, radius=outer_node_radius, fc=color, ec='black', alpha=device_opacity, zorder=2)
    ax.add_patch(outer_circle)
    device_artists[f'circle_{i}'] = outer_circle
    outer_label = ax.text(outer_pos[0], outer_pos[1], f'D{i}', ha='center', va='center', fontsize=7, zorder=3)
    device_label_artists[f'circle_{i}'] = outer_label

    # Inner Square (Home/Storage)
    inner_center = center_pos + inner_radius * unit_dir
    inner_node_centers.append(inner_center)
    inner_square_side = inner_node_radius * np.sqrt(2)
    inner_square_bottom_left = inner_center - np.array([inner_square_side / 2, inner_square_side / 2])
    inner_square = patches.Rectangle(inner_square_bottom_left, inner_square_side, inner_square_side,
                                     fc=color, ec='black', alpha=inner_node_opacity, zorder=2)
    ax.add_patch(inner_square)
    device_artists[f'inner_square_{i}'] = inner_square
    inner_label = ax.text(inner_center[0], inner_center[1], f'D{i} Home', ha='center', va='center', fontsize=6, zorder=3)
    device_label_artists[f'inner_label_{i}'] = inner_label


    stall_node_pos = outer_pos + unit_dir * stall_node_distance_offset
    stall_node_positions.append(stall_node_pos)
    stall_node = RegularPolygon(stall_node_pos, numVertices=8, radius=stall_node_radius,
                                 orientation=np.pi/8,
                                 fc=COLOR_STALL_NODE_FILL, ec=color, lw=stall_node_border_width,
                                 zorder=2, visible=False)
    ax.add_patch(stall_node)
    device_artists[f'stall_node_{i}'] = stall_node
    stall_label = ax.text(stall_node_pos[0], stall_node_pos[1], "",
                          ha='center', va='center', fontsize=stall_node_fontsize, fontweight='semibold',
                          color='white', zorder=3, visible=False)
    device_label_artists[f'stall_label_{i}'] = stall_label

    # Connection points and distance calculation
    inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
    outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
    dist = np.linalg.norm(outer_edge_conn_point - inner_edge_conn_point)
    node_transfer_distances.append(dist)

# --- Create Edges ---
arrow_style_str = f'-|>,head_length={head_len},head_width={head_wid}'
for i in range(N):
    unit_dir = unit_directions[i]
    inner_center = inner_node_centers[i]
    outer_pos = outer_circle_positions[i]
    inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
    outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
    radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
    edge_offset = radial_perp_vector * arrow_offset_dist

    # Start positions WITH offset
    start_pos_arrow_out = inner_edge_conn_point + edge_offset # Start for arrow pointing OUTWARD (INBOUND data)
    start_pos_arrow_in = outer_edge_conn_point - edge_offset # Start for arrow pointing INWARD (OUTBOUND data)

    # Create INBOUND arrow object (Visually Inner -> Outer)
    arrow_out = patches.FancyArrowPatch(posA=start_pos_arrow_out, posB=start_pos_arrow_out,
                                         arrowstyle=arrow_style_str, color=COLOR_INBOUND_DEFAULT, linestyle='dashed',
                                         mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_out)
    label_out = ax.text(start_pos_arrow_out[0], start_pos_arrow_out[1], "", color=COLOR_INBOUND_DEFAULT, fontsize=edge_label_fontsize, ha='center', va='bottom', zorder=4)
    edge_artists[f'in_{i}'] = (arrow_out, label_out)

    # Create OUTBOUND arrow object (Visually Outer -> Inner)
    arrow_in = patches.FancyArrowPatch(posA=start_pos_arrow_in, posB=start_pos_arrow_in,
                                         arrowstyle=arrow_style_str, color=COLOR_OUTBOUND_FWD_ACTIVATION, linestyle='dashed',
                                         mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_in)
    label_in = ax.text(start_pos_arrow_in[0], start_pos_arrow_in[1], "", color=COLOR_OUTBOUND_FWD_ACTIVATION, fontsize=edge_label_fontsize, ha='center', va='top', zorder=4)
    edge_artists[f'out_{i}'] = (arrow_in, label_in)

    # Create RING arrow object (Device -> Device)
    start_pos_ring = outer_pos
    arrow_ring = patches.FancyArrowPatch(posA=start_pos_ring, posB=start_pos_ring,
                                          arrowstyle=arrow_style_str, color=COLOR_RING_CCW, linestyle='solid',
                                          mutation_scale=mut_scale, lw=edge_linewidth, zorder=1,
                                          connectionstyle=f"arc3,rad=0.2")
    ax.add_patch(arrow_ring)
    label_ring = ax.text(start_pos_ring[0], start_pos_ring[1], "", color=COLOR_RING_CCW, fontsize=edge_label_fontsize, ha='center', va='center', zorder=4)
    edge_artists[f'ring_{i}'] = (arrow_ring, label_ring)

    # --- Create Computation Progress Arc ---
    arc_radius = outer_node_radius * compute_arc_radius_scale
    compute_arc = Arc(outer_pos, width=2*arc_radius, height=2*arc_radius,
                      angle=0, theta1=0.0, theta2=0.0,
                      color=COLOR_COMPUTE_DEFAULT, lw=compute_arc_linewidth, zorder=1,
                      visible=False)
    ax.add_patch(compute_arc)
    edge_artists[f'compute_{i}'] = compute_arc

# --- Pre-calculate Relative Angles for Arcs ---
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


# --- Device Class (Corrected Version) ---
# --- Device Class (MODIFIED parts only) ---
class Device:
    def __init__(self, device_id, layer_capacity, activations_capacity, transitions_capacity, total_devices, total_layers, total_chunks):
        # === Basic Properties ===
        # ... (no changes) ...
        self.device_id = device_id
        self.device_has_started = False; self.device_start_time = 0
        self.device_has_finished = False; self.device_finish_time = 0

        # === Capacities and Constants ===
        # ... (no changes) ...
        self.layer_capacity = layer_capacity
        self.activations_capacity = activations_capacity
        self.transitions_capacity = transitions_capacity # Capacity for FWD transitions buffer
        self.grad_activations_capacity = grad_activations_capacity
        self.total_devices = total_devices
        self.total_layers = total_layers
        self.total_chunks = total_chunks

        # === Local Buffers ===
        self.cur_ready_weights = [-1 for _ in range(layer_capacity)]
        self.cur_weight_replace_ind = 0

        # --- RENAMED/NEW Transition Buffers ---
        # Buffer for incoming FORWARD activations from peer (Y_l-1)
        self.fwd_transitions_buffer = [-1 for _ in range(self.transitions_capacity)]
        self.fwd_transitions_write_ptr = 0

        # Buffer for incoming BACKWARD gradients from peer (dL/dX_l+1)
        self.bwd_grad_transitions_buffer = [-1 for _ in range(self.total_chunks)] # Capacity total_chunks
        self.bwd_grad_transitions_write_ptr = 0
        # --- END RENAMED/NEW ---

        # Buffer for activations kept resident from FWD pass
        self.resident_checkpoint_activations = [-1 for _ in range(self.activations_capacity)]
        self.resident_checkpoint_write_ptr = 0

        # Buffer for activations fetched during BWD pass
        self.bwd_fetched_activation_buffer = [-1 for _ in range(self.total_chunks)]
        self.bwd_fetched_activation_write_ptr = 0

        # Buffer for locally computed Activation Gradients (dL/dA)
        self.cur_ready_grad_activations = [-1 for _ in range(self.grad_activations_capacity)]
        self.cur_grad_activations_write_ptr = 0

        # Buffers for cross-device/block transfers
        self.cur_model_outputs = [-1 for _ in range(self.total_chunks)] # Input to Head
        self.cur_model_outputs_write_ptr = 0
        self.cur_head_outputs = [-1 for _ in range(self.total_chunks)] # Input to last block BwdX
        self.cur_head_outputs_write_ptr = 0

        # Local cache for activations of the very last FWD layer processed
        self.local_last_fwd_activations = {}

        # === Queues & State ===
        # Inbound Queue Item: (chunk_id, layer_id, is_grad, target_idx, duration, target_buffer_type)
        # target_buffer_type: 'weight', 'bwd_fetched_act'
        # Peer Transfer Queue Item: (peer_id, chunk_id, layer_id, is_grad, target_idx, duration)
        # Note: target_idx for peer transfers now refers implicitly to the correct buffer based on is_grad
        self.computation_queue = []; self.outbound_queue = []; self.inbound_queue = []; self.peer_transfer_queue = []
        # ... (rest of state variables unchanged) ...
        self.is_computing = False; self.is_stalled = False; self.stall_start_time = 0
        self.cur_computation_start_time = 0; self.cur_computation_duration = 0
        self.current_computation_type = None; self.current_computation_layer_id = -1
        self.is_outbound_transferring = False; self.cur_outbound_start_time = 0; self.cur_outbound_duration = 0
        self.is_inbound_transferring = False; self.cur_inbound_start_time = 0; self.cur_inbound_duration = 0
        self.is_peer_transferring = False; self.cur_peer_transfer_start_time = 0; self.cur_peer_transfer_duration = 0
        self.cur_outbound_edge = ""; self.cur_inbound_edge = ""; self.cur_ring_edge = ""
        self.computing_status = "Idle"; self.stall_reason = ""
        self.has_reversed = False
        self.last_fwd_layer_on_device = -1
        self.bwd_prefetch_trigger_layer_id = -1
        self.next_weight_layer_id = -1
        self.outbound_storage = set()

        # === Task Queue Setup ===
        # ... (Task queue setup logic remains the same, just ensure Device args match if called externally) ...
        initially_loaded_weights = []
        for i in range (layer_capacity):
            layer_id_to_add = device_id + i * total_devices
            if layer_id_to_add < total_layers:
                self.cur_ready_weights[i] = layer_id_to_add
                self.outbound_storage.add((-1, layer_id_to_add, False))
                initially_loaded_weights.append(layer_id_to_add)
        self.cur_weight_replace_ind = 0

        cur_layer_id = device_id
        while cur_layer_id < total_layers:
            self.last_fwd_layer_on_device = cur_layer_id
            if (-1, cur_layer_id, False) not in self.outbound_storage:
                self.outbound_storage.add((-1, cur_layer_id, False))
            for i in range (total_chunks):
                self.computation_queue.append((i, cur_layer_id, False, False, 1, computationFrames))
            cur_layer_id += self.total_devices

        self.next_weight_layer_id = self.device_id + layer_capacity * self.total_devices
        if self.last_fwd_layer_on_device != -1 and self.next_weight_layer_id > self.last_fwd_layer_on_device:
             self.next_weight_layer_id = total_layers
        elif self.last_fwd_layer_on_device == -1:
             self.next_weight_layer_id = None
        self.bwd_prefetch_trigger_layer_id = self.last_fwd_layer_on_device

        self.head_final_chunk_id = -1
        if do_backward:
            head_layer_conceptual_id = total_layers
            is_head_device = (self.last_fwd_layer_on_device != -1 and
                              self.last_fwd_layer_on_device + self.total_devices == total_layers)
            if is_head_device:
                cur_layer_id_for_bwd_loop = head_layer_conceptual_id
                self.outbound_storage.add((-1, cur_layer_id_for_bwd_loop, False))
                if self.layer_capacity > len(initially_loaded_weights):
                     head_wgt_idx = len(initially_loaded_weights)
                     self.cur_ready_weights[head_wgt_idx] = cur_layer_id_for_bwd_loop
                cutoff_chunk_id = self.total_chunks // 2
                head_task_queue_order = []
                if are_chunks_same_seq:
                     head_task_queue_order.extend(range(cutoff_chunk_id))
                     head_task_queue_order.extend(range(self.total_chunks - 1, cutoff_chunk_id - 1, -1))
                     self.head_final_chunk_id = cutoff_chunk_id
                else:
                     head_task_queue_order.extend(range(self.total_chunks))
                     self.head_final_chunk_id = self.total_chunks - 1
                for i in head_task_queue_order: self.computation_queue.append((i, cur_layer_id_for_bwd_loop, False, False, -1, computationFrames))
                cur_layer_id_for_bwd_loop = self.last_fwd_layer_on_device
            else: cur_layer_id_for_bwd_loop = self.last_fwd_layer_on_device

            if cur_layer_id_for_bwd_loop >= 0:
                current_bwd_layer = cur_layer_id_for_bwd_loop
                while current_bwd_layer >= 0:
                    if are_chunks_same_seq: chunk_order = range(total_chunks - 1, -1, -1)
                    else: chunk_order = range(total_chunks)
                    for i in chunk_order: # BwdX
                        transfer_direction = -1 if current_bwd_layer > 0 else 0
                        self.computation_queue.append((i, current_bwd_layer, True, False, transfer_direction, computationFrames))
                    for i in chunk_order: # BwdW
                        self.computation_queue.append((i, current_bwd_layer, False, True, 0, computationFrames))
                    current_bwd_layer -= self.total_devices


    # --- handle_completed_transfers (MODIFIED for Peer Transfers) ---
    def handle_completed_transfers(self, T, all_devices):
        # Inbound Transfer Completion (from Storage) - No change needed here
        if self.is_inbound_transferring and (self.cur_inbound_start_time + self.cur_inbound_duration <= T):
            inbound_item = self.inbound_queue.pop(0)
            chunk_id, layer_id, is_grad, target_idx, duration, target_buffer_type = inbound_item
            item_type = 'Unknown'; status = 'OK'; buffer_name = "N/A"
            if target_buffer_type == 'weight':
                item_type = 'Wgt'; buffer_name = "Ready Wgt"
                if 0 <= target_idx < self.layer_capacity:
                    if self.cur_ready_weights[target_idx] in [-1, -2] or self.cur_ready_weights[target_idx] == layer_id: self.cur_ready_weights[target_idx] = layer_id
                    else: status = f'STALE Wgt (Idx {target_idx} holds L{self.cur_ready_weights[target_idx]}, got L{layer_id})'
                else: status = f'ERROR Wgt (Invalid Idx {target_idx})'
            elif target_buffer_type == 'bwd_fetched_act':
                item_type = 'Bwd Fetched Act'; buffer_name = "Bwd Fetched Act Buf"
                act_tuple = (chunk_id, layer_id); effective_idx = target_idx % self.total_chunks
                if 0 <= effective_idx < self.total_chunks:
                     if self.bwd_fetched_activation_buffer[effective_idx] in [-1, -2] or self.bwd_fetched_activation_buffer[effective_idx] == act_tuple: self.bwd_fetched_activation_buffer[effective_idx] = act_tuple
                     else: status = f'STALE Bwd Fetched Act (Idx {effective_idx} holds {self.bwd_fetched_activation_buffer[effective_idx]}, got {act_tuple})'
                else: status = f'ERROR Bwd Fetched Act (Invalid Idx {effective_idx})'
            else: status = f'ERROR Unknown Buffer Type ({target_buffer_type}) for Inbound'
            if TO_PRINT or 'ERROR' in status or 'STALE' in status:
                 log_msg = f"T={T}, Dev {self.device_id}: RX INBOUND {item_type} "
                 log_msg += f"C{chunk_id},L{layer_id}" if chunk_id != -1 else f"L{layer_id}"
                 log_msg += f" -> Buf '{buffer_name}' Idx {target_idx} Complete."; log_msg = log_msg.replace("Complete.", status) if 'ERROR' in status or 'STALE' in status else log_msg
                 print(log_msg)
            self.is_inbound_transferring = False; self.cur_inbound_edge = ""

        # Outbound Transfer Completion (to Storage) - No change needed here
        if self.is_outbound_transferring and (self.cur_outbound_start_time + self.cur_outbound_duration <= T):
             outbound_item = self.outbound_queue.pop(0)
             chunk_id, layer_id, is_grad, duration = outbound_item; storage_key = (chunk_id, layer_id, is_grad)
             if storage_key not in self.outbound_storage:
                 self.outbound_storage.add(storage_key); item_type = 'Act' if chunk_id >= 0 else ('WgtGrad' if is_grad else 'Wgt')
                 if TO_PRINT: print(f"T={T}, Dev {self.device_id}: TX OUTBOUND {item_type} C{chunk_id},L{layer_id} -> Storage COMPLETE. Key: {storage_key}")
             self.is_outbound_transferring = False; self.cur_outbound_edge = ""

        # Peer Transfer Completion (Device -> Peer) - MODIFIED Buffer Handling
        if self.is_peer_transferring and (self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T):
             peer_item = self.peer_transfer_queue.pop(0)
             peer_id, chunk_id, layer_id, is_grad, target_idx, duration = peer_item # target_idx is the write pointer value *at the time of queuing*
             peer_dev = all_devices[peer_id]
             item_type = 'Unknown'; loc = "Unknown Buffer"; status = 'OK'
             is_output_to_head_input = (layer_id == self.total_layers - 1) and (not is_grad)
             is_grad_from_head_output = (layer_id == self.total_layers) and is_grad
             item_key = (chunk_id, layer_id, is_grad)

             # --- Determine target buffer based on is_grad and context ---
             target_buffer = None
             buffer_capacity = 0
             if is_output_to_head_input: # Fwd output to Head
                 loc = "Model Outputs (Head In)"; item_type = 'Fwd Output'
                 target_buffer = peer_dev.cur_model_outputs
                 buffer_capacity = peer_dev.total_chunks
             elif is_grad_from_head_output: # Grad output from Head
                 loc = "Head Outputs (Bwd In)"; item_type = 'Head Grad'
                 target_buffer = peer_dev.cur_head_outputs
                 buffer_capacity = peer_dev.total_chunks
             elif is_grad: # <<< Backward gradient (dL/dX) transfer between layers >>>
                 loc = "Bwd Grad Transitions"; item_type = 'Bwd Grad'
                 target_buffer = peer_dev.bwd_grad_transitions_buffer # Target NEW buffer
                 buffer_capacity = peer_dev.total_chunks # Use its capacity
             else: # <<< Forward activation (Y_l) transfer between layers >>>
                 loc = "Fwd Transitions"; item_type = 'Fwd Output'
                 target_buffer = peer_dev.fwd_transitions_buffer # Target original FWD buffer
                 buffer_capacity = peer_dev.transitions_capacity # Use its capacity
             # --- End target buffer determination ---

             # Validate index and write to buffer
             if target_buffer is not None and 0 <= target_idx < buffer_capacity:
                 if target_buffer[target_idx] in [-1, -2] or target_buffer[target_idx] == item_key:
                     target_buffer[target_idx] = item_key
                 else: status = f'STALE Peer Data (Buf {loc}, Idx {target_idx} holds {target_buffer[target_idx]}, got {item_key})'
             elif target_buffer is None: status = f'ERROR Peer Target Buffer Undefined'
             else: status = f'ERROR Peer Data (Invalid Idx {target_idx} for Buf {loc} size {buffer_capacity})'

             if TO_PRINT or 'ERROR' in status or 'STALE' in status:
                 log_msg = f"T={T}, Dev {self.device_id}: TX PEER {item_type} C{chunk_id},L{lid} -> Dev {peer_id} Buf '{loc}' Idx {target_idx} Complete."; log_msg = log_msg.replace("Complete.", status) if 'ERROR' in status or 'STALE' in status else log_msg
                 print(log_msg)
             self.is_peer_transferring = False; self.cur_ring_edge = ""


    # --- handle_computation_depends (MODIFIED dependency check/consumption) ---
    def handle_computation_depends(self, T):
        if not self.computation_queue: return False
        next_task = self.computation_queue[0]
        cid, lid, bX, bW, tdir, dur = next_task
        has_deps = False; is_fwd = (not bX) and (not bW) and (lid < self.total_layers); is_head = (lid == self.total_layers)
        computation_type_str = "Unknown"; self.stall_reason = ""

        # === Check Dependencies ===
        if is_fwd:
            computation_type_str = "Fwd"; has_weight = (lid in self.cur_ready_weights)
            has_input_transition = True; input_transition_key = None
            if lid > 0:
                input_transition_key = (cid, lid - 1, False)
                # Check FORWARD transition buffer
                has_input_transition = any(isinstance(item, tuple) and item == input_transition_key for item in self.fwd_transitions_buffer if item != -1)
            has_deps = has_weight and has_input_transition
            if not has_deps:
                missing = [];
                if not has_weight: missing.append(f"Weight L{lid}")
                if lid > 0 and not has_input_transition: missing.append(f"Input Fwd Act C{cid},L{lid-1}") # Clarify Fwd Act
                self.stall_reason = "Missing:\n" + "\n".join(missing) if missing else "Unknown FWD Dep"

        elif is_head:
            computation_type_str = "Head"; head_weight_lid = self.total_layers; has_weight = (head_weight_lid in self.cur_ready_weights)
            input_key = (cid, self.total_layers - 1, False)
            has_input_from_last_block = any(isinstance(item, tuple) and item == input_key for item in self.cur_model_outputs if item != -1)
            has_deps = has_weight and has_input_from_last_block
            if not has_deps:
                missing = [];
                if not has_weight: missing.append(f"Head State L{head_weight_lid}")
                if not has_input_from_last_block: missing.append(f"Input Act C{cid},L{self.total_layers-1}")
                self.stall_reason = "Missing:\n" + "\n".join(missing) if missing else "Unknown HEAD Dep"

        elif bX:
            computation_type_str = "Bwd X"; has_weight = (lid in self.cur_ready_weights)
            has_upstream_grad = False; upstream_grad_key = (cid, lid + 1, True); upstream_buffer_name = "None"
            source_buffer_check = None # Buffer to check for the gradient
            # Check appropriate buffer for upstream gradient
            if lid == self.total_layers - 1:
                 upstream_buffer_name = "Head Outputs"; source_buffer_check = self.cur_head_outputs
            elif lid < self.total_layers - 1:
                 upstream_buffer_name = "Bwd Grad Transitions"; source_buffer_check = self.bwd_grad_transitions_buffer # <<< Check NEW buffer >>>
            # else: source_buffer_check = None (lid < 0, invalid)

            if source_buffer_check is not None:
                has_upstream_grad = any(isinstance(item, tuple) and item == upstream_grad_key for item in source_buffer_check if item != -1)

            has_deps = has_weight and has_upstream_grad
            if not has_deps:
                 missing = [];
                 if not has_weight: missing.append(f"Weight L{lid}")
                 if not has_upstream_grad: missing.append(f"Upstream Grad C{cid},L{lid+1} (from {upstream_buffer_name})")
                 self.stall_reason = "Missing:\n" + "\n".join(missing) if missing else "Unknown BwdX Dep"

        elif bW:
            computation_type_str = "Bwd W"; fwd_act_key = (cid, lid); act_grad_key = (cid, lid)
            has_fwd_activation = False; fwd_act_source = "None"
            # Check order: Local Dict -> Resident Buf -> BWD Fetched Buf
            if fwd_act_key in self.local_last_fwd_activations: has_fwd_activation = True; fwd_act_source = "Local Dict"
            elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.resident_checkpoint_activations if item != -1): has_fwd_activation = True; fwd_act_source = "Resident Chkpt Buf"
            elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.bwd_fetched_activation_buffer if item != -1): has_fwd_activation = True; fwd_act_source = "Bwd Fetched Buf"
            has_activation_grad = any(isinstance(item, tuple) and item == act_grad_key for item in self.cur_ready_grad_activations if item != -1)
            has_deps = has_fwd_activation and has_activation_grad
            if not has_deps:
                 missing = [];
                 if not has_fwd_activation: missing.append(f"Fwd Act C{cid},L{lid} (Not found)")
                 if not has_activation_grad: missing.append(f"Act Grad C{cid},L{lid}")
                 self.stall_reason = "Missing:\n" + "\n".join(missing) if missing else "Unknown BwdW Dep"

        # === Update Device State ===
        if has_deps:
            # Start Computation
            if not self.device_has_started: self.device_start_time = T; self.device_has_started = True
            if self.is_stalled and TO_PRINT: print(f"T={T}, Dev {self.device_id}: UNSTALL -> Comp C{cid},L{lid},{computation_type_str}. Stalled for {T - self.stall_start_time} cycles.")
            self.cur_computation_start_time = T; self.cur_computation_duration = dur; self.is_computing = True; self.is_stalled = False; self.stall_reason = ""
            self.current_computation_type = computation_type_str; self.current_computation_layer_id = lid; self.computing_status = f"COMPUTING:\n{computation_type_str}\nC{cid},L{lid}"


            # Consume Dependencies - MODIFIED for BwdX
            consumed_items = []
            if is_fwd and lid > 0 and input_transition_key: # Consume from FWD buffer
                 for idx, item in enumerate(self.fwd_transitions_buffer): # <<< Check FWD buffer >>>
                     if isinstance(item, tuple) and item == input_transition_key: self.fwd_transitions_buffer[idx] = -1; consumed_items.append(f"FwdTrans[{idx}]:{input_transition_key}"); break
            elif is_head: # Consume from Model Output buffer
                 input_key = (cid, self.total_layers - 1, False)
                 for idx, item in enumerate(self.cur_model_outputs):
                     if isinstance(item, tuple) and item == input_key: self.cur_model_outputs[idx] = -1; consumed_items.append(f"ModelOut[{idx}]:{input_key}"); break
            elif bX: # Consume from Head Output or BWD Grad buffer
                 upstream_grad_key = (cid, lid + 1, True)
                 source_buffer = None; buf_name = "N/A"
                 if lid == self.total_layers - 1: source_buffer = self.cur_head_outputs; buf_name="HeadOut"
                 elif lid < self.total_layers - 1: source_buffer = self.bwd_grad_transitions_buffer; buf_name="BwdGradTrans" # <<< Consume from NEW buffer >>>
                 if source_buffer is not None:
                      for idx, item in enumerate(source_buffer):
                          if isinstance(item, tuple) and item == upstream_grad_key: source_buffer[idx] = -1; consumed_items.append(f"{buf_name}[{idx}]:{upstream_grad_key}"); break
            elif bW: # Consume Fwd Act and Act Grad
                 # ... (BwdW consumption logic unchanged) ...
                 fwd_act_key = (cid, lid); consumed_fwd_act = False
                 if fwd_act_key in self.local_last_fwd_activations: del self.local_last_fwd_activations[fwd_act_key]; consumed_items.append(f"LocalActDict:{fwd_act_key}"); consumed_fwd_act = True
                 elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.resident_checkpoint_activations if item != -1):
                     for idx, item in enumerate(self.resident_checkpoint_activations):
                         if isinstance(item, tuple) and item == fwd_act_key: self.resident_checkpoint_activations[idx] = -1; consumed_items.append(f"ResidentAct[{idx}]:{fwd_act_key}"); consumed_fwd_act = True; break
                 elif any(isinstance(item, tuple) and item == fwd_act_key for item in self.bwd_fetched_activation_buffer if item != -1):
                     for idx, item in enumerate(self.bwd_fetched_activation_buffer):
                          if isinstance(item, tuple) and item == fwd_act_key: self.bwd_fetched_activation_buffer[idx] = -1; consumed_items.append(f"BwdFetchedAct[{idx}]:{fwd_act_key}"); consumed_fwd_act = True; break
                 act_grad_key = (cid, lid)
                 for idx, item in enumerate(self.cur_ready_grad_activations):
                     if isinstance(item, tuple) and item == act_grad_key: self.cur_ready_grad_activations[idx] = -1; consumed_items.append(f"GradAct[{idx}]:{act_grad_key}"); break


            if TO_PRINT and consumed_items: print(f"  T={T}, Dev {self.device_id}: Consumed -> {', '.join(consumed_items)}")

        else: # Dependencies not met
            if not self.is_stalled:
                self.is_stalled = True; self.stall_start_time = T; self.computing_status = f"STALL:\n{computation_type_str}\nC{cid},L{lid}"
                if TO_PRINT: print(f"T={T}, Dev {self.device_id}: STALL on {computation_type_str} C{cid},L{lid}. Reason: {self.stall_reason.replace('Missing:', '').replace('Missing', '').replace('\\n', ', ')}")
            self.is_computing = False; self.current_computation_type = None; self.current_computation_layer_id = -1
        return has_deps


    # --- queue_bulk_bwd_prefetches ---
    # (No changes needed here, logic already targets bwd_fetched_activation_buffer correctly)
    def queue_bulk_bwd_prefetches(self, T, is_head_trigger=False):
        """ Queues backward prefetches for weights and activations needed soon. """
        # ... (Function content remains the same as previous corrected version) ...
        if not self.has_reversed:
            if TO_PRINT: print(f"T={T}, Dev {self.device_id}: WARNING - queue_bulk_bwd_prefetches called before reversing.")
            return
        if TO_PRINT:
            trigger_type = "HEAD" if is_head_trigger else "BWD_X"
            print(f"T={T}, Dev {self.device_id}: === Bulk BWD Prefetch Triggered ({trigger_type}) ===")
            print(f"  State Before -> Trigger Layer ID: L{self.bwd_prefetch_trigger_layer_id}, Next Weight Target: L{self.next_weight_layer_id}")
        layer_id_just_finished_task = self.bwd_prefetch_trigger_layer_id
        if is_head_trigger: layer_id_just_finished_task = self.total_layers
        if layer_id_just_finished_task < 0 and not is_head_trigger:
            if TO_PRINT: print(f"  BWD_X Trigger SKIPPED - Tracked layer ID ({layer_id_just_finished_task}) is invalid."); return
        if is_head_trigger:
            next_bwd_layer_on_device = self.last_fwd_layer_on_device
            if next_bwd_layer_on_device < 0:
                if TO_PRINT: print(f"  HEAD Trigger SKIPPED - Device had no forward layers."); return
        else: next_bwd_layer_on_device = layer_id_just_finished_task - self.total_devices
        target_prefetch_dep_layer = next_bwd_layer_on_device - self.total_devices
        if TO_PRINT:
            print(f"  Trigger Origin Layer: L{'Head' if is_head_trigger else layer_id_just_finished_task}")
            print(f"  Next BWD Layer (starts now): L{next_bwd_layer_on_device}")
            print(f"  Target Prefetch Dep Layer (for future): L{target_prefetch_dep_layer}")

        # Phase A: Queue Weights for next_bwd_layer_on_device
        layer_for_phase_A = next_bwd_layer_on_device
        if layer_for_phase_A >= 0:
            #...(Weight fetch logic unchanged)...
            if TO_PRINT: print(f"  Phase A: Checking Weight for NEXT BWD L{layer_for_phase_A}")
            storage_key_wgt = (-1, layer_for_phase_A, False); is_in_buffer = (layer_for_phase_A in self.cur_ready_weights)
            is_pending_in_buffer_slot = False; target_idx_A = self.cur_weight_replace_ind
            if self.cur_ready_weights[target_idx_A] == -2: is_pending_in_buffer_slot = any(q[0]==-1 and q[1]==layer_for_phase_A and q[3]==target_idx_A and q[5]=='weight' for q in self.inbound_queue)
            is_in_queue = any(q[0]==-1 and q[1]==layer_for_phase_A and q[5]=='weight' for q in self.inbound_queue)
            needs_fetch = (storage_key_wgt in self.outbound_storage and not is_in_buffer and not is_pending_in_buffer_slot and not is_in_queue)
            if needs_fetch:
                 weight_to_evict = self.cur_ready_weights[target_idx_A]
                 if TO_PRINT: print(f"    Queueing Wgt L{layer_for_phase_A} -> Target Idx {target_idx_A} (Evicting L{weight_to_evict if weight_to_evict != -1 else 'Empty'})")
                 self.inbound_queue.append((-1, layer_for_phase_A, False, target_idx_A, layerTransferFrames, 'weight'))
                 self.cur_ready_weights[target_idx_A] = -2; self.cur_weight_replace_ind = (target_idx_A + 1) % self.layer_capacity

        # Phase B: Queue Activations for BwdW of next_bwd_layer_on_device and target_prefetch_dep_layer
        layers_to_fetch_acts_for = [l for l in [next_bwd_layer_on_device, target_prefetch_dep_layer] if l is not None and l >= 0]
        if not layers_to_fetch_acts_for and TO_PRINT: print("  Phase B: No valid layers to fetch activations for.")
        for layer_id_act in layers_to_fetch_acts_for:
            if TO_PRINT: print(f"  Phase B: Checking Activations for L{layer_id_act} -> Bwd Fetched Buffer")
            if are_chunks_same_seq: chunk_order_b = range(total_chunks - 1, -1, -1)
            else: chunk_order_b = range(total_chunks)
            num_queued_B = 0
            for chunk_id in chunk_order_b:
                act_tuple = (chunk_id, layer_id_act); storage_key_act = (chunk_id, layer_id_act, False)
                is_available = False
                if act_tuple in self.local_last_fwd_activations: is_available = True # Hot cache
                elif any(isinstance(item, tuple) and item == act_tuple for item in self.resident_checkpoint_activations if item != -1): is_available = True # Resident
                elif any(isinstance(item, tuple) and item == act_tuple for item in self.bwd_fetched_activation_buffer if item != -1): is_available = True # Already fetched BWD
                else: # Check if pending/queued for BWD fetch buffer
                    is_pending_in_slot = False; target_idx_B = self.bwd_fetched_activation_write_ptr
                    if self.bwd_fetched_activation_buffer[target_idx_B] == -2: is_pending_in_slot = any(q[0]==chunk_id and q[1]==layer_id_act and q[3]==target_idx_B and q[5]=='bwd_fetched_act' for q in self.inbound_queue if q[0]!=-1)
                    if is_pending_in_slot: is_available = True
                    else: is_in_queue = any(q[0]==chunk_id and q[1]==layer_id_act and q[5]=='bwd_fetched_act' for q in self.inbound_queue if q[0]!=-1); is_available = is_in_queue

                needs_fetch = (storage_key_act in self.outbound_storage and not is_available)
                if needs_fetch:
                    target_idx_B = self.bwd_fetched_activation_write_ptr # Use correct pointer
                    if TO_PRINT: print(f"    Queueing Act C{chunk_id},L{layer_id_act} -> Bwd Fetched Idx {target_idx_B}")
                    self.inbound_queue.append((chunk_id, layer_id_act, False, target_idx_B, savedActivationsFrames, 'bwd_fetched_act'))
                    self.bwd_fetched_activation_buffer[target_idx_B] = -2
                    self.bwd_fetched_activation_write_ptr = (target_idx_B + 1) % self.total_chunks
                    num_queued_B += 1
            if TO_PRINT: print(f"    Queued {num_queued_B} activations for L{layer_id_act}.")

        self.bwd_prefetch_trigger_layer_id = next_bwd_layer_on_device
        self.next_weight_layer_id = target_prefetch_dep_layer
        if TO_PRINT:
             print(f"  Updated State -> Next Trigger Layer ID: L{self.bwd_prefetch_trigger_layer_id}, Next Weight Target: L{self.next_weight_layer_id}")
             print(f"T={T}, Dev {self.device_id}: === Bulk BWD Prefetch Queuing Complete ===")


    # --- handle_computation (MODIFIED Peer Transfer Target Ptrs) ---
    def handle_computation(self, T, all_devices):
        """ Handles starting, finishing, and managing consequences of computation tasks. """
        completed_tasks = 0
        if not self.is_computing and not self.is_stalled and len(self.computation_queue) > 0:
            self.handle_computation_depends(T)
        elif self.is_computing and (self.cur_computation_start_time + self.cur_computation_duration <= T):
            task = self.computation_queue.pop(0); completed_tasks += 1
            cid, lid, bX, bW, tdir, dur = task
            is_head = (lid == self.total_layers); is_fwd = (not bX) and (not bW) and (not is_head)
            task_type_str = self.current_computation_type
            if TO_PRINT: print(f"T={T}, Dev {self.device_id}: FINISHED Comp -> C{cid},L{lid},{task_type_str}")

            # Handle Outputs / Queue Transfers / Trigger Prefetches
            if is_fwd:
                act_tuple = (cid, lid); save_locally_hot = False; save_resident = False
                if lid == self.last_fwd_layer_on_device: save_locally_hot = True
                if do_backward and self.activations_capacity > 0:
                    if are_chunks_same_seq and cid >= self.total_chunks - self.activations_capacity: save_resident = True
                    elif not are_chunks_same_seq and cid < self.activations_capacity: save_resident = True
                if save_locally_hot: self.local_last_fwd_activations[act_tuple] = True; # if TO_PRINT: print(f"    Saving FWD Act C{cid},L{lid} -> Local Dict (Hot)")
                if save_resident:
                    res_idx = self.resident_checkpoint_write_ptr
                    if 0 <= res_idx < self.activations_capacity:
                        # if TO_PRINT: print(f"    Saving FWD Act C{cid},L{lid} -> Resident Chkpt Buf Idx {res_idx}")
                        self.resident_checkpoint_activations[res_idx] = act_tuple; self.resident_checkpoint_write_ptr = (res_idx + 1) % self.activations_capacity
                if do_backward:
                     storage_key = (cid, lid, False); is_in_queue = any(q[0]==cid and q[1]==lid and q[2]==False for q in self.outbound_queue)
                     if storage_key not in self.outbound_storage and not is_in_queue:
                         # if TO_PRINT: print(f"    Queueing FWD Act C{cid},L{lid} to Outbound Storage.")
                         self.outbound_queue.append((cid, lid, False, savedActivationsFrames))
                if tdir != 0: # Peer Transfer FWD Activation
                    target_peer_id = (self.device_id + tdir + self.total_devices) % self.total_devices; target_dev = all_devices[target_peer_id]
                    is_output_to_head_buffer = (lid == self.total_layers - 1)
                    if is_output_to_head_buffer:
                        # Target Head Input Buffer
                        idx = target_dev.cur_model_outputs_write_ptr # USE TARGET'S POINTER
                        self.peer_transfer_queue.append((target_peer_id, cid, lid, False, idx, activationTransitionFrames))
                        if TO_PRINT: print(f"    Queueing FWD Output C{cid},L{lid} -> Peer {target_peer_id} Head Input Idx {idx}")
                        target_dev.cur_model_outputs_write_ptr = (idx + 1) % target_dev.total_chunks # ADVANCE TARGET'S POINTER
                    else:
                        # Target FWD Transition Buffer
                        idx = target_dev.fwd_transitions_write_ptr # USE TARGET'S FWD POINTER
                        self.peer_transfer_queue.append((target_peer_id, cid, lid, False, idx, activationTransitionFrames))
                        if TO_PRINT: print(f"    Queueing FWD Output C{cid},L{lid} -> Peer {target_peer_id} Fwd Transition Idx {idx}")
                        target_dev.fwd_transitions_write_ptr = (idx + 1) % target_dev.transitions_capacity # ADVANCE TARGET'S FWD POINTER

            elif is_head: # Head computation done
                if tdir != 0: # Peer Transfer Head Gradient
                     target_peer_id = (self.device_id + tdir + self.total_devices) % self.total_devices; target_dev = all_devices[target_peer_id]
                     idx = target_dev.cur_head_outputs_write_ptr # USE TARGET'S POINTER
                     self.peer_transfer_queue.append((target_peer_id, cid, lid, True, idx, activationTransitionFrames))
                     if TO_PRINT: print(f"    Queueing Head Grad C{cid},L{lid} -> Peer {target_peer_id} Head Output Idx {idx}")
                     target_dev.cur_head_outputs_write_ptr = (idx + 1) % target_dev.total_chunks # ADVANCE TARGET'S POINTER
                if cid == self.head_final_chunk_id: # After last chunk
                     storage_key = (-1, lid, True); is_in_queue = any(q[0]==-1 and q[1]==lid and q[2]==True for q in self.outbound_queue)
                     if storage_key not in self.outbound_storage and not is_in_queue: self.outbound_queue.append((-1, lid, True, layerTransferFrames))
                     if do_backward and self.has_reversed:
                        self.bwd_prefetch_trigger_layer_id = lid; self.queue_bulk_bwd_prefetches(T, is_head_trigger=True)

            elif bX: # BwdX done
                idx_grad = self.cur_grad_activations_write_ptr
                if 0 <= idx_grad < self.grad_activations_capacity: self.cur_ready_grad_activations[idx_grad] = (cid, lid); self.cur_grad_activations_write_ptr = (idx_grad + 1) % self.grad_activations_capacity
                if tdir != 0:
                    target_peer_id = (self.device_id + tdir + self.total_devices) % self.total_devices; target_dev = all_devices[target_peer_id]
                    idx = target_dev.bwd_grad_transitions_write_ptr # <<< USE TARGET'S BWD GRAD POINTER >>>
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, True, idx, activationTransitionFrames)) # is_grad = True
                    if TO_PRINT: print(f"    Queueing Bwd Grad (dL/dX) C{cid},L{lid} -> Peer {target_peer_id} Bwd Grad Trans Idx {idx}")
                    target_dev.bwd_grad_transitions_write_ptr = (idx + 1) % target_dev.total_chunks # <<< ADVANCE TARGET'S BWD GRAD POINTER >>>

            elif bW: # BwdW done
                last_chunk_id_for_bwd = 0 if are_chunks_same_seq else self.total_chunks - 1
                if cid == last_chunk_id_for_bwd: # After last chunk, save final grad
                     storage_key = (-1, lid, True); is_in_queue = any(q[0]==-1 and q[1]==lid and q[2]==True for q in self.outbound_queue)
                     if storage_key not in self.outbound_storage and not is_in_queue: self.outbound_queue.append((-1, lid, True, layerTransferFrames))

            # Fwd Prefetching / Reversal Logic
            # ... (No changes needed in this section) ...
            if not self.has_reversed:
                is_last_fwd_chunk_for_layer = is_fwd and (cid == self.total_chunks - 1)
                if is_last_fwd_chunk_for_layer:
                     layer_to_fetch = self.next_weight_layer_id
                     if layer_to_fetch is not None and layer_to_fetch <= total_layers :
                          storage_key_wgt = (-1, layer_to_fetch, False); in_storage = storage_key_wgt in self.outbound_storage; needs_fetch = False
                          if in_storage:
                               is_in_buffer = (layer_to_fetch in self.cur_ready_weights); is_pending_in_buffer_slot = False; target_idx_fwd = self.cur_weight_replace_ind
                               if self.cur_ready_weights[target_idx_fwd] == -2: is_pending_in_buffer_slot = any(q[0]==-1 and q[1]==layer_to_fetch and q[3]==target_idx_fwd and q[5]=='weight' for q in self.inbound_queue)
                               is_in_queue = any(q[0]==-1 and q[1]==layer_to_fetch and q[5]=='weight' for q in self.inbound_queue)
                               if not is_in_buffer and not is_pending_in_buffer_slot and not is_in_queue: needs_fetch = True
                          if needs_fetch:
                               weight_to_evict = self.cur_ready_weights[target_idx_fwd]
                               # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: >>> QUEUING FWD Prefetch L{layer_to_fetch} -> Idx {target_idx_fwd} (Evicting L{weight_to_evict if weight_to_evict !=-1 else 'Empty'}).")
                               self.inbound_queue.append((-1, layer_to_fetch, False, target_idx_fwd, layerTransferFrames, 'weight'))
                               self.cur_ready_weights[target_idx_fwd] = -2; self.cur_weight_replace_ind = (target_idx_fwd + 1) % self.layer_capacity
                          next_layer_id_after_fetch = layer_to_fetch + self.total_devices; will_reverse = (do_backward and next_layer_id_after_fetch > total_layers)
                          if will_reverse:
                               self.has_reversed = True; self.bwd_fetched_activation_write_ptr = 0; self.cur_grad_activations_write_ptr = 0
                               if TO_PRINT: print(f"T={T}, Dev {self.device_id}: >>> REVERSING state after FWD C{cid},L{lid}.")
                               self.next_weight_layer_id = None
                          else:
                               self.next_weight_layer_id = next_layer_id_after_fetch
                               if not do_backward and self.next_weight_layer_id >= total_layers: self.next_weight_layer_id = None
                     else: # No valid next layer
                          if do_backward and not self.has_reversed and self.next_weight_layer_id is None:
                              self.has_reversed = True; self.bwd_fetched_activation_write_ptr = 0; self.cur_grad_activations_write_ptr = 0
                              if TO_PRINT: print(f"T={T}, Dev {self.device_id}: >>> REVERSING state after FWD C{cid},L{lid} (No further FWD layers).")

            # Reset computation state flags
            self.is_computing = False; self.computing_status = "Idle"; self.current_computation_type = None; self.current_computation_layer_id = -1
            # Try to start next task immediately
            if len(self.computation_queue) > 0: self.handle_computation_depends(T)
            else: # No more tasks
                 if not self.device_has_finished: self.device_finish_time = T; self.device_has_finished = True; self.computing_status = "Finished" #; if TO_PRINT: print(f"T={T}, Dev {self.device_id}: Finished all tasks.")
        elif self.is_stalled: self.handle_computation_depends(T) # Re-check dependencies
        return completed_tasks


    # --- handle_new_transfers ---
    # (No changes needed here, logic relies on flags set elsewhere)
    def handle_new_transfers(self, T):
        # ... (Function content remains the same as previous version) ...
        if not self.is_inbound_transferring and len(self.inbound_queue) > 0:
             item = self.inbound_queue[0]; cid, lid, isg, target_idx, duration, target_buffer_type = item
             actual_storage_key = (-1, lid, isg) if cid == -1 else (cid, lid, False); is_in_storage = actual_storage_key in self.outbound_storage
             if is_in_storage:
                 self.is_inbound_transferring = True; self.cur_inbound_start_time = T; self.cur_inbound_duration = duration
                 label_lid_str = "Head" if lid == total_layers else str(lid); edge_color = COLOR_INBOUND_DEFAULT
                 if target_buffer_type == 'weight': self.cur_inbound_edge = f"Wgt:L{label_lid_str}"; edge_color = COLOR_INBOUND_WEIGHT
                 elif target_buffer_type == 'bwd_fetched_act': self.cur_inbound_edge = f"Fetch:C{cid},L{lid}"; edge_color = COLOR_INBOUND_BWD_FETCHED_ACTIVATION
                 else: self.cur_inbound_edge = f"? C{cid},L{lid}"
                 arrow_vis, _ = edge_artists[f'in_{self.device_id}']; arrow_vis.set_color(edge_color)
                 # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: >>> STARTING Inbound Transfer {self.cur_inbound_edge} -> Buf '{target_buffer_type}' Idx {target_idx}.")
             else: self.inbound_queue.pop(0) # Remove invalid request
        if not self.is_outbound_transferring and len(self.outbound_queue) > 0:
             item = self.outbound_queue[0]; cid, lid, isg, duration = item
             self.is_outbound_transferring = True; self.cur_outbound_start_time = T; self.cur_outbound_duration = duration
             label_lid_str = "Head" if lid == total_layers else str(lid); edge_color = COLOR_OUTBOUND_FWD_ACTIVATION
             if cid >= 0: self.cur_outbound_edge = f"Act:C{cid},L{lid}"
             elif isg: self.cur_outbound_edge = f"Grad:L{label_lid_str}"; edge_color = COLOR_OUTBOUND_WGT_GRAD
             else: self.cur_outbound_edge = f"Wgt:L{lid}"; edge_color = COLOR_OUTBOUND_DEFAULT
             arrow_vis, _ = edge_artists[f'out_{self.device_id}']; arrow_vis.set_color(edge_color)
             # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: >>> STARTING Outbound Transfer {self.cur_outbound_edge} to Storage.")
        if not self.is_peer_transferring and len(self.peer_transfer_queue) > 0:
             item = self.peer_transfer_queue[0]; pid, cid, lid, isg, target_idx, duration = item
             self.is_peer_transferring = True; self.cur_peer_transfer_start_time = T; self.cur_peer_transfer_duration = duration
             label_lid_str = "Head" if lid == total_layers else str(lid)
             target_is_cw = (pid == (self.device_id - 1 + self.total_devices) % self.total_devices); edge_color = COLOR_RING_CW if target_is_cw else COLOR_RING_CCW
             connection_style_ring = f"arc3,rad={'-' if target_is_cw else ''}0.2"
             if isg: self.cur_ring_edge = f"Grad:C{cid},L{label_lid_str}"
             else: self.cur_ring_edge = f"Out:C{cid},L{lid}"
             arrow_vis, _ = edge_artists[f'ring_{self.device_id}']; arrow_vis.set_color(edge_color); arrow_vis.set_connectionstyle(connection_style_ring)
             # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: >>> STARTING Peer Transfer {self.cur_ring_edge} -> Peer {pid} Idx {target_idx}.")


    # --- print_buffer_status (MODIFIED with new buffer names) ---
    def print_buffer_status(self):
        current_T = current_frame_index if 'current_frame_index' in globals() else 'N/A'
        print(f"--- Dev {self.device_id} Buffer Status (T={current_T}) ---")
        wgt_buf_str = ', '.join([f"L{w}" if w >= 0 else ('Pend' if w == -2 else '_') for w in self.cur_ready_weights])
        print(f"  Ready Wgt: [{wgt_buf_str}] (Next Evict Idx: {self.cur_weight_replace_ind})")

        # Renamed Fwd Transitions Buffer
        fwd_trans_buf_str = ', '.join([f"({t[0]},{t[1]})" if isinstance(t, tuple) else ('Pend' if t == -2 else '_') for t in self.fwd_transitions_buffer])
        print(f"  Fwd Transitions Buf:    [{fwd_trans_buf_str}] (Cap: {self.transitions_capacity}, Next Write Idx: {self.fwd_transitions_write_ptr})")

        # New Bwd Grad Transitions Buffer
        bwd_grad_trans_buf_str = ', '.join([f"({t[0]},{t[1]})" if isinstance(t, tuple) else ('Pend' if t == -2 else '_') for t in self.bwd_grad_transitions_buffer])
        print(f"  Bwd Grad Transitions Buf: [{bwd_grad_trans_buf_str}] (Cap: {self.total_chunks}, Next Write Idx: {self.bwd_grad_transitions_write_ptr})")

        print(f"  Local Last Fwd Act (Hot): {sorted(list(self.local_last_fwd_activations.keys()))}")
        res_act_buf_str = ', '.join([f"({a[0]},{a[1]})" if isinstance(a, tuple) else ('Pend' if a == -2 else '_') for a in self.resident_checkpoint_activations])
        print(f"  Resident Chkpt Act Buf: [{res_act_buf_str}] (Cap: {self.activations_capacity}, Next Write Idx: {self.resident_checkpoint_write_ptr})")
        bwd_fetch_buf_str = ', '.join([f"({a[0]},{a[1]})" if isinstance(a, tuple) else ('Pend' if a == -2 else '_') for a in self.bwd_fetched_activation_buffer])
        print(f"  BWD Fetched Act Buf:    [{bwd_fetch_buf_str}] (Cap: {self.total_chunks}, Next Write Idx: {self.bwd_fetched_activation_write_ptr})")
        grad_act_buf_str = ', '.join([f"({g[0]},{g[1]})" if isinstance(g, tuple) else ('Pend' if g == -2 else '_') for g in self.cur_ready_grad_activations])
        print(f"  Ready GradAct (dL/dA):   [{grad_act_buf_str}] (Cap: {self.grad_activations_capacity}, Next Write Idx: {self.cur_grad_activations_write_ptr})")
        model_out_buf_str = ', '.join([f"({m[0]},{m[1]},{'G' if m[2] else 'A'})" if isinstance(m, tuple) else ('Pend' if m == -2 else '_') for m in self.cur_model_outputs])
        print(f"  Model Outputs (Head In): [{model_out_buf_str}] (Next Write Idx: {self.cur_model_outputs_write_ptr})")
        head_out_buf_str = ', '.join([f"({h[0]},{h[1]},{'G' if h[2] else 'A'})" if isinstance(h, tuple) else ('Pend' if h == -2 else '_') for h in self.cur_head_outputs])
        print(f"  Head Outputs (Bwd In):  [{head_out_buf_str}] (Next Write Idx: {self.cur_head_outputs_write_ptr})")
        print(f"  Queue Lengths: In={len(self.inbound_queue)}, Out={len(self.outbound_queue)}, Peer={len(self.peer_transfer_queue)}")
        print(f"  State: Reversed={self.has_reversed}, BWD Trigger Layer=L{self.bwd_prefetch_trigger_layer_id}, Next Wgt Target=L{self.next_weight_layer_id if self.next_weight_layer_id is not None else 'None'}")
        print(f"-------------------------------------")


# --- Global Simulation State ---
# ... (Remains the same) ...
all_devices = {i: Device(i, layer_capacity, activations_capacity, transitions_capacity,
                             N, total_layers, total_chunks) # Use N for total_devices argument
                   for i in range(N)}
total_tasks = 0
total_completed_tasks = {}
total_computation_time = 0
current_frame_index = 0
animation_paused = False
completion_text_artist = None
target_cycle = None

# --- Simulation Reset Function ---
# ... (Remains the same) ...
# --- Simulation Reset Function (Corrected) ---
def reset_simulation():
    """Resets the simulation state and visual elements."""
    global all_devices, total_tasks, total_completed_tasks, total_computation_time
    global current_frame_index, animation_paused, completion_text_artist, target_cycle
    global unit_directions, inner_node_centers, outer_circle_positions, inner_node_radius, outer_node_radius, arrow_offset_dist
    global N,total_chunks,total_layers

    if TO_PRINT:
        print("\n" + "="*20 + " Resetting Simulation " + "="*20)

    # --- CORRECTION HERE: Use N instead of total_devices ---
    all_devices = {i: Device(i, layer_capacity, activations_capacity, transitions_capacity,
                             N, total_layers, total_chunks) # Use N for total_devices argument
                   for i in range(N)}
    # --- END CORRECTION ---

    # Recalculate totals
    total_tasks = sum(len(d.computation_queue) for d in all_devices.values())
    total_computation_time = sum(task[-1] for i in range(N) for task in all_devices[i].computation_queue)

    # Reset state variables
    total_completed_tasks = {-1: 0}
    current_frame_index = 0
    animation_paused = False
    target_cycle = None

    # Remove completion text
    if completion_text_artist is not None:
        if completion_text_artist.axes is not None:
            completion_text_artist.remove()
        completion_text_artist = None

    # Reset visual elements
    for i in range(N):
        unit_dir = unit_directions[i]; inner_center = inner_node_centers[i]; outer_pos = outer_circle_positions[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]]); edge_offset = radial_perp_vector * arrow_offset_dist
        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius; start_pos_arrow_out = inner_edge_conn_point + edge_offset
        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius; start_pos_arrow_in = outer_edge_conn_point - edge_offset
        start_pos_ring = outer_pos

        # Reset Transfer Arrows
        arrow_in_vis, label_in_vis = edge_artists[f'in_{i}']
        label_in_vis.set_text("")
        arrow_in_vis.set_color(COLOR_INBOUND_DEFAULT) # Reset to default color
        arrow_in_vis.set_positions(start_pos_arrow_out, start_pos_arrow_out)

        arrow_out_vis, label_out_vis = edge_artists[f'out_{i}']
        label_out_vis.set_text("")
        arrow_out_vis.set_color(COLOR_OUTBOUND_FWD_ACTIVATION) # Reset to default outbound color
        arrow_out_vis.set_positions(start_pos_arrow_in, start_pos_arrow_in)

        arrow_ring, label_ring = edge_artists[f'ring_{i}']
        label_ring.set_text("")
        arrow_ring.set_color(COLOR_RING_CCW) # Reset to default CCW color
        arrow_ring.set_positions(start_pos_ring, start_pos_ring)
        arrow_ring.set_connectionstyle(f"arc3,rad=0.2") # Reset to default CCW style

        # Reset Stall Node
        if f'stall_node_{i}' in device_artists: device_artists[f'stall_node_{i}'].set_visible(False)
        if f'stall_label_{i}' in device_label_artists: device_label_artists[f'stall_label_{i}'].set_text(""); device_label_artists[f'stall_label_{i}'].set_visible(False)

        # Reset Device Labels
        if f'circle_{i}' in device_label_artists: device_label_artists[f'circle_{i}'].set_text(f'D{i}\nIdle')
        if f'inner_label_{i}' in device_label_artists: device_label_artists[f'inner_label_{i}'].set_text(f'D{i}\nHome')

        # Reset Computation Arc
        if f'compute_{i}' in edge_artists:
            compute_arc = edge_artists[f'compute_{i}']
            compute_arc.set_visible(False)
            compute_arc.theta1 = 0.0 # Reset angles
            compute_arc.theta2 = 0.0 # Reset angles

    title_obj.set_text(f'Cycle {current_frame_index}')

    if TO_PRINT:
        print(f"Reset complete. Total tasks: {total_tasks}")
        print("="*60 + "\n")

# --- Update Function ---
# (No changes needed in update function itself, color handling moved to handle_new_transfers)
def update(frame):
    global all_devices
    global total_completed_tasks, current_frame_index, completion_text_artist, animation_paused, target_cycle
    global world_angle_to_prev, world_angle_to_next

    if animation_paused:
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
        if completion_text_artist: all_artists.append(completion_text_artist)
        return all_artists

    T = current_frame_index

    if target_cycle is not None and T == target_cycle:
        print(f"Reached target cycle {T}, pausing.")
        if ani.event_source is not None: ani.event_source.stop()
        animation_paused = True; target_cycle = None
        title_obj.set_text(f'Cycle {T} (Paused)'); fig.canvas.draw_idle()
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
        if completion_text_artist: all_artists.append(completion_text_artist)
        return all_artists

    title_obj.set_text(f'Cycle {T}')
    artists_to_update = [title_obj]

    if T not in total_completed_tasks:
        last_known_frame = -1; valid_keys = [k for k in total_completed_tasks if k < T]
        if valid_keys: last_known_frame = max(valid_keys)
        total_completed_tasks[T] = total_completed_tasks.get(last_known_frame, 0)

    # Simulation Step Order: Complete -> Compute/Consume/Queue -> Start New Transfers
    print("\n\n\nALL DEVICES:",all_devices)
    for i in range(N): all_devices[i].handle_completed_transfers(T, all_devices)
    newly_completed = 0
    for i in range(N): newly_completed += all_devices[i].handle_computation(T, all_devices)
    total_completed_tasks[T] = total_completed_tasks.get(T, total_completed_tasks.get(T-1, 0)) + newly_completed
    for i in range(N): all_devices[i].handle_new_transfers(T) # This now sets arrow colors

    # Update Visuals
    for i in range(N):
        device = all_devices[i]; unit_dir = unit_directions[i]
        inner_center = inner_node_centers[i]; outer_pos = outer_circle_positions[i]
        transfer_dist_i = node_transfer_distances[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]]); edge_offset = radial_perp_vector * arrow_offset_dist

        # Labels & Stall Node
        outer_label_artist = device_label_artists[f'circle_{i}']
        outer_label_artist.set_text(f'D{i}\n{device.computing_status}')
        inner_label_artist = device_label_artists[f'inner_label_{i}']
        inner_label_artist.set_text(f"D{i}\nHome")
        stall_node_artist = device_artists[f'stall_node_{i}']
        stall_label_artist = device_label_artists[f'stall_label_{i}']
        if device.is_stalled and device.stall_reason:
             wrapped_stall_reason = textwrap.fill(device.stall_reason.replace("\n", " "), width=15)
             stall_label_artist.set_text(wrapped_stall_reason); stall_label_artist.set_visible(True); stall_node_artist.set_visible(True)
        else:
             stall_label_artist.set_text(""); stall_label_artist.set_visible(False); stall_node_artist.set_visible(False)

        # Transfer Arrows (Inbound, Outbound, Ring)
        arrow_vis_inbound, label_vis_inbound = edge_artists[f'in_{i}']
        arrow_vis_outbound, label_vis_outbound = edge_artists[f'out_{i}']
        arrow_ring, label_ring = edge_artists[f'ring_{i}']

        # Inbound Arrow Update
        len_in_prog = 0.0; cur_inbound_edge_text = ""
        color_inbound = arrow_vis_inbound.get_edgecolor() # Read color set by handle_new_transfers
        if device.is_inbound_transferring and device.cur_inbound_duration > 0:
             prog_frac = min(1.0, (T - device.cur_inbound_start_time) / device.cur_inbound_duration)
             len_in_prog = prog_frac * transfer_dist_i; cur_inbound_edge_text = device.cur_inbound_edge
        label_vis_inbound.set_color(color_inbound)
        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius; start_vis_inbound = inner_edge_conn_point + edge_offset
        end_vis_inbound = start_vis_inbound + unit_dir * len_in_prog
        arrow_vis_inbound.set_positions(start_vis_inbound, end_vis_inbound)
        label_perp_offset_in = radial_perp_vector * label_offset_distance; midpoint_vis_in = (start_vis_inbound + end_vis_inbound) / 2
        label_pos_vis_in = midpoint_vis_in + label_perp_offset_in * 2
        label_vis_inbound.set_position(label_pos_vis_in); label_vis_inbound.set_text(cur_inbound_edge_text)

        # Outbound Arrow Update
        len_out_prog = 0.0; cur_outbound_edge_text = ""
        color_outbound = arrow_vis_outbound.get_edgecolor() # Read color set by handle_new_transfers
        if device.is_outbound_transferring and device.cur_outbound_duration > 0:
             prog_frac = min(1.0, (T - device.cur_outbound_start_time) / device.cur_outbound_duration)
             len_out_prog = prog_frac * transfer_dist_i; cur_outbound_edge_text = device.cur_outbound_edge
        label_vis_outbound.set_color(color_outbound)
        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius; start_vis_outbound = outer_edge_conn_point - edge_offset
        end_vis_outbound = start_vis_outbound - unit_dir * len_out_prog
        arrow_vis_outbound.set_positions(start_vis_outbound, end_vis_outbound)
        label_perp_offset_out = radial_perp_vector * label_offset_distance; midpoint_vis_out = (start_vis_outbound + end_vis_outbound) / 2
        label_pos_vis_out = midpoint_vis_out - label_perp_offset_out * 2
        label_vis_outbound.set_position(label_pos_vis_out); label_vis_outbound.set_text(cur_outbound_edge_text)

        # Ring Arrow Update
        len_ring_prog = 0.0; peer_device_id = -1; cur_ring_edge_text = ""
        color_ring = arrow_ring.get_edgecolor() # Read color set by handle_new_transfers
        if device.is_peer_transferring and device.cur_peer_transfer_duration > 0:
            peer_item = device.peer_transfer_queue[0]; peer_device_id = peer_item[0]
            prog_frac = min(1.0, (T - device.cur_peer_transfer_start_time) / device.cur_peer_transfer_duration)
            len_ring_prog = prog_frac; cur_ring_edge_text = device.cur_ring_edge
        label_ring.set_color(color_ring)
        start_pos_ring_geo = outer_pos; current_end_point_ring = start_pos_ring_geo
        if peer_device_id != -1:
            target_pos_ring_geo_center = outer_circle_positions[peer_device_id]; vec = target_pos_ring_geo_center - start_pos_ring_geo
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                start_offset_dir = vec / norm
                start_pos_ring_geo = outer_pos + start_offset_dir * outer_node_radius
                target_pos_ring_geo = target_pos_ring_geo_center - start_offset_dir * outer_node_radius
                current_end_point_ring = start_pos_ring_geo + (target_pos_ring_geo - start_pos_ring_geo) * len_ring_prog
        arrow_ring.set_positions(start_pos_ring_geo, current_end_point_ring)
        label_pos_ring = (start_pos_ring_geo + current_end_point_ring) / 2
        if len_ring_prog > 1e-6:
            edge_vec = current_end_point_ring - start_pos_ring_geo; norm = np.linalg.norm(edge_vec)
            if norm > 1e-6:
                 perp_vec = np.array([-edge_vec[1], edge_vec[0]]) / norm
                 conn_style = arrow_ring.get_connectionstyle(); offset_direction_multiplier = 1.0
                 if isinstance(conn_style, str) and 'rad=-' in conn_style: offset_direction_multiplier = -1.0
                 label_pos_ring = label_pos_ring + perp_vec * label_offset_distance * 3 * offset_direction_multiplier
        label_ring.set_position(label_pos_ring); label_ring.set_text(cur_ring_edge_text)

        # Computation Arc
        compute_arc = edge_artists[f'compute_{i}']
        progress_frac = 0.0; compute_color = COLOR_COMPUTE_DEFAULT
        if device.is_computing and device.cur_computation_duration > 0:
            progress_frac = min(1.0, (T - device.cur_computation_start_time) / device.cur_computation_duration)
            comp_type = device.current_computation_type; comp_lid = device.current_computation_layer_id
            if comp_type == "Fwd":   compute_color = COLOR_COMPUTE_FWD
            elif comp_type == "Bwd X": compute_color = COLOR_COMPUTE_BWD_X
            elif comp_type == "Bwd W": compute_color = COLOR_COMPUTE_BWD_W
            elif comp_type == "Head":  compute_color = COLOR_COMPUTE_HEAD
            else: compute_color = COLOR_COMPUTE_DEFAULT
            theta1_for_arc = 0.0; theta2_for_arc = 0.0
            angle_start_abs = 0.0; angle_end_target_abs = 0.0; total_sweep_angle = 0.0
            if comp_type == "Fwd":
                 angle_start_abs = world_angle_to_prev[i]
                 if comp_lid == 0: angle_start_abs = world_angle_to_next[i] - 180
                 angle_end_target_abs = world_angle_to_next[i]; total_sweep_angle = (angle_end_target_abs - angle_start_abs + 360) % 360
                 if N==1: total_sweep_angle=360
                 theta1_for_arc = angle_start_abs; theta2_for_arc = angle_start_abs + progress_frac * total_sweep_angle
            elif comp_type == "Head":
                 angle_start_abs = world_angle_to_prev[i]; total_sweep_angle = 360.0
                 theta1_for_arc = angle_start_abs; theta2_for_arc = angle_start_abs + progress_frac * total_sweep_angle
            elif comp_type == "Bwd X" or comp_type == "Bwd W":
                 angle_start_vis = world_angle_to_next[i]; angle_end_target_vis = world_angle_to_prev[i]
                 total_sweep_angle = (angle_start_vis - angle_end_target_vis + 360) % 360
                 if N==1: total_sweep_angle=360
                 current_cw_sweep = progress_frac * total_sweep_angle; angle_end_vis_current = angle_start_vis - current_cw_sweep
                 theta1_for_arc = angle_end_vis_current; theta2_for_arc = angle_start_vis
            if abs(theta2_for_arc - theta1_for_arc) > 1e-6 or (progress_frac > 0 and total_sweep_angle > 1e-6) :
                 compute_arc.theta1 = theta1_for_arc; compute_arc.theta2 = theta2_for_arc
                 compute_arc.set_visible(True); compute_arc.set_edgecolor(compute_color)
            else: compute_arc.set_visible(False)
        else: compute_arc.set_visible(False)

        artists_to_update.extend([ outer_label_artist, inner_label_artist, stall_node_artist, stall_label_artist,
            arrow_vis_inbound, label_vis_inbound, arrow_vis_outbound, label_vis_outbound, arrow_ring, label_ring, compute_arc ])

    # Check for Completion & Display Text & Stop Animation
    current_total_completed = total_completed_tasks.get(T, 0)
    all_devices_finished = all(d.device_has_finished for d in all_devices.values())
    is_fully_complete = (current_total_completed >= total_tasks) and all_devices_finished
    was_complete_last_cycle = False
    if T > 0 :
        last_completed = total_completed_tasks.get(T-1, 0)
        was_complete_last_cycle = (last_completed >= total_tasks) and all(d.device_has_finished and d.device_finish_time < T for d in all_devices.values())
    should_stop_animation = is_fully_complete and not was_complete_last_cycle

    if is_fully_complete and completion_text_artist is None:
        start_times = [d.device_start_time for d in all_devices.values() if d.device_has_started]; finish_times = [d.device_finish_time for d in all_devices.values() if d.device_has_finished]
        first_start_time = min(start_times) if start_times else 0; latest_finish_time = max(finish_times) if finish_times else T
        total_dev_time = (latest_finish_time - first_start_time) * N if latest_finish_time > first_start_time else 0
        overall_eff = (total_computation_time / total_dev_time * 100) if total_dev_time > 0 else 0.0
        completion_text = ( f"Simulation Complete!\nFinal Cycle Count: {latest_finish_time}\n\n"
             f"Problem:\nTotal Tasks: {total_tasks}\nTotal Task Comp Time: {total_computation_time}\n"
             f"Utilized {N} devices for aggregate {total_dev_time:.0f} cycles\n\nEFFICIENCY:\nOverall: {overall_eff:.2f}%" )
        completion_text_artist = ax.text(0.5, 0.5, completion_text, transform=ax.transAxes, ha='center', va='center', fontsize=14, color='navy', fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.5', fc=(0.9, 0.9, 1, 0.95), ec='black'), zorder=10)
        if TO_PRINT: print("\n" + "="*20 + " Simulation Complete " + "="*20); print(completion_text); print("="*60 + "\n")
        artists_to_update.append(completion_text_artist)

    if should_stop_animation and ani is not None and ani.event_source is not None:
        if not animation_paused: ani.event_source.stop(); print(f"Animation Complete at Cycle {T} - Paused")
        animation_paused = True; target_cycle = None; title_obj.set_text(f'Cycle {T} (Complete)')

    # Increment Global Frame Index
    if not animation_paused:
        current_frame_index += 1
        if current_frame_index >= max_frames:
            print(f"Max frames ({max_frames}) reached, stopping animation.")
            if ani.event_source is not None: ani.event_source.stop()
            animation_paused = True; target_cycle = None; title_obj.set_text(f'Cycle {T} (Max Frames)')
            if completion_text_artist is None:
                 current_T = T; current_total_completed = total_completed_tasks.get(current_T, 0)
                 completion_text = ( f"Max Frames Reached!\nFinal Cycle: {current_T}\n\n" f"Tasks Completed: {current_total_completed} / {total_tasks}\n" )
                 completion_text_artist = ax.text(0.5, 0.5, completion_text, transform=ax.transAxes, ha='center', va='center', fontsize=14, color='maroon', fontweight='bold',
                                                 bbox=dict(boxstyle='round,pad=0.5', fc=(1, 0.9, 0.9, 0.95), ec='black'), zorder=10)
                 artists_to_update.append(completion_text_artist)

    return artists_to_update


# --- Create Animation ---
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=initial_frame_interval,
                              blit=False, repeat=False, save_count=max_frames)


# --- Widgets ---
# ... (Setup remains the same) ...
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
initial_run_to_cycle_guess = (N + total_chunks - 1) * computationFrames + activationTransitionFrames * N if N > 0 else computationFrames * total_chunks
textbox_runto = TextBox(ax_textbox, "Run to Cycle:", initial=str(initial_run_to_cycle_guess), textalignment="center")
btn_runto = Button(ax_runto_btn, 'Run')


# --- Define Widget Callback Functions ---
# ... (Callbacks remain the same) ...
def pause_animation(event):
    global animation_paused, target_cycle
    if not animation_paused:
        if ani.event_source is not None: ani.event_source.stop()
        animation_paused = True; target_cycle = None
        title_obj.set_text(f'Cycle {current_frame_index} (Paused)'); fig.canvas.draw_idle(); print("Animation Paused")
    else: print("Animation already paused.")

def play_animation(event):
    global animation_paused, target_cycle
    target_cycle = None
    current_completed = total_completed_tasks.get(current_frame_index, 0); all_devices_finished = all(d.device_has_finished for d in all_devices.values())
    can_play = animation_paused and not (current_completed >= total_tasks and all_devices_finished) and current_frame_index < max_frames
    if can_play:
        if ani.event_source is not None:
            title_obj.set_text(f'Cycle {current_frame_index}'); ani.event_source.start(); animation_paused = False; print("Animation Resumed")
        else: print("Error: Animation event source not found.")
    elif current_completed >= total_tasks and all_devices_finished: print("Animation already complete.")
    elif current_frame_index >= max_frames: print("Animation stopped at max frames.")
    elif not animation_paused: print("Animation already playing.")
    else: print("Cannot play animation (unknown reason).")

def update_speed(val):
    global animation_paused
    speed_level = slider_speed.val; new_interval = calculate_interval(speed_level, min_speed_level, max_speed_level, min_interval, max_interval); was_playing = not animation_paused
    if ani.event_source is not None:
        ani.event_source.stop(); ani.event_source.interval = new_interval; ani._interval = new_interval
    else: print("Error: Could not access animation timer."); return
    current_completed = total_completed_tasks.get(current_frame_index, 0); all_devices_finished = all(d.device_has_finished for d in all_devices.values())
    should_resume = was_playing and not (current_completed >= total_tasks and all_devices_finished) and current_frame_index < max_frames and target_cycle is None
    if should_resume: ani.event_source.start(); animation_paused = False
    else:
        animation_paused = True
        if not (current_completed >= total_tasks and all_devices_finished) and current_frame_index < max_frames:
             if target_cycle is None or current_frame_index != target_cycle:
                 is_complete = current_completed >= total_tasks and all_devices_finished; is_max_frames = current_frame_index >= max_frames
                 if not is_complete and not is_max_frames: title_obj.set_text(f'Cycle {current_frame_index} (Paused)'); fig.canvas.draw_idle()
    print(f"Speed Level: {int(round(speed_level))}, Interval set to: {new_interval} ms")

def restart_animation_callback(event):
    global animation_paused
    print("Restart button clicked.");
    if ani.event_source is not None: ani.event_source.stop()
    reset_simulation(); fig.canvas.draw_idle()
    try: fig.canvas.flush_events()
    except AttributeError: pass
    if not animation_paused:
        if ani.event_source is not None: ani.event_source.start(); print("Simulation reset and playing from Cycle 0.")
        else: print("Error: Cannot restart animation timer."); animation_paused = True; title_obj.set_text(f'Cycle {current_frame_index} (Paused)'); fig.canvas.draw_idle()
    else: print("Simulation reset and paused at Cycle 0."); title_obj.set_text(f'Cycle {current_frame_index} (Paused)'); fig.canvas.draw_idle()

def run_to_cycle_callback(event):
    global target_cycle, animation_paused, current_frame_index
    input_text = textbox_runto.text
    try: requested_cycle = int(input_text)
    except ValueError: print(f"Invalid input: '{input_text}'."); textbox_runto.set_val(str(current_frame_index)); return
    if requested_cycle < 0: print(f"Invalid input: {requested_cycle}. Must be non-negative."); textbox_runto.set_val(str(current_frame_index)); return
    if requested_cycle >= max_frames: print(f"Target {requested_cycle} >= max frames ({max_frames}). Clamping."); requested_cycle = max_frames - 1; textbox_runto.set_val(str(requested_cycle))
    print(f"Attempting to run to cycle: {requested_cycle}");
    if ani.event_source is not None and not animation_paused: print("Stopping current animation..."); ani.event_source.stop()
    needs_restart = False; current_completed = total_completed_tasks.get(current_frame_index, 0); all_devices_finished = all(d.device_has_finished for d in all_devices.values())
    if requested_cycle <= current_frame_index or (current_completed >= total_tasks and all_devices_finished):
        if not (requested_cycle == current_frame_index and animation_paused):
            print("Target in past or sim complete. Restarting...");
            if ani.event_source is not None: ani.event_source.stop()
            reset_simulation(); needs_restart = True; fig.canvas.draw_idle();
            try: fig.canvas.flush_events()
            except AttributeError: pass
    target_cycle = requested_cycle; print(f"Target set to cycle {target_cycle}.")
    current_completed_after_reset = total_completed_tasks.get(current_frame_index, 0); all_devices_finished_after_reset = all(d.device_has_finished for d in all_devices.values())
    should_run = (current_frame_index < target_cycle) and not (current_completed_after_reset >= total_tasks and all_devices_finished_after_reset) and (current_frame_index < max_frames)
    if should_run:
        if ani.event_source is not None: print("Starting animation to reach target."); title_obj.set_text(f'Cycle {current_frame_index}'); ani.event_source.start(); animation_paused = False
        else: print("Error: Animation event source not found."); target_cycle = None; animation_paused = True; title_obj.set_text(f'Cycle {current_frame_index} (Paused)'); fig.canvas.draw_idle()
    else:
        print(f"Cannot run or already at/past target ({current_frame_index}). Ensuring paused.");
        if ani.event_source is not None: ani.event_source.stop()
        animation_paused = True
        if current_frame_index == target_cycle: target_cycle = None # Clear target if met
        final_title = f'Cycle {current_frame_index}'
        if current_completed_after_reset >= total_tasks and all_devices_finished_after_reset: final_title += " (Complete)"
        elif current_frame_index >= max_frames: final_title += " (Max Frames)"
        elif target_cycle is None and animation_paused: final_title += " (Paused)"
        title_obj.set_text(final_title); fig.canvas.draw_idle()


# --- Connect Widgets to Callbacks ---
btn_restart.on_clicked(restart_animation_callback)
btn_pause.on_clicked(pause_animation)
btn_play.on_clicked(play_animation)
slider_speed.on_changed(update_speed)
btn_runto.on_clicked(run_to_cycle_callback)
# textbox_runto.on_submit(run_to_cycle_callback) # Optional


# --- Cursor Hover Logic ---
# ... (Remains the same) ...
widget_axes = [ax_restart, ax_pause, ax_play, ax_slider, ax_textbox, ax_runto_btn]
is_cursor_over_widget = False
def on_hover(event):
    global is_cursor_over_widget; currently_over = False
    if event.inaxes in widget_axes: currently_over = True
    if currently_over != is_cursor_over_widget:
        new_cursor = Cursors.HAND if currently_over else Cursors.POINTER
        try:
            if fig.canvas: fig.canvas.set_cursor(new_cursor)
        except Exception as e:
             if TO_PRINT: print(f"Minor error setting cursor: {e}")
        is_cursor_over_widget = currently_over
fig.canvas.mpl_connect('motion_notify_event', on_hover)


# --- Display ---
print("Initializing display...")
reset_simulation() # Reset before showing

print("Showing plot...")
plt.show()
print("Plot window closed.")