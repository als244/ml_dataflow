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

# --- Global Control Flags ---
TO_PRINT = False

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
N = 8
computationFrames = 50
layerTransferFrames = N * computationFrames
savedActivationsFrames = computationFrames
activationTransitionFrames = 5

total_devices = N
total_layers = 32
total_chunks = 16
layer_capacity = 2
activations_capacity = total_chunks - 1
transitions_capacity = N
grad_activations_capacity = total_chunks

do_backward = True
are_chunks_same_seq = True
save_grad_activation_in_layer_buffer = True

max_frames = 4800 # Limit animation length for performance if needed

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
COLOR_INBOUND_DEFAULT = 'olive'
COLOR_INBOUND_WEIGHT = 'olive'
COLOR_INBOUND_ACTIVATION = 'magenta' # For backward pass activations fetch

# Device -> Network (Visually Inward toward Center)
COLOR_OUTBOUND_DEFAULT = 'magenta' # For forward pass activation storage
COLOR_OUTBOUND_WGT_GRAD = 'turquoise' # For weight gradient storage

# Ring Transfers (Device -> Device)
# CCW = Counter-Clockwise (i -> i+1), typically forward activations/head inputs
# CW = Clockwise (i -> i-1), typically backward gradients
COLOR_RING_CCW = 'indigo' # Was FWD
COLOR_RING_CW = 'orange'  # Was BWD

# --- Computation Arc Colors ---
COLOR_COMPUTE_DEFAULT = 'gray'
COLOR_COMPUTE_FWD = 'darkgreen'
COLOR_COMPUTE_BWD_X = 'teal'
COLOR_COMPUTE_BWD_W = 'mediumturquoise'
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
    f"     - Num Layers: {total_layers}\n"
    f"     - Num Devices: {N}\n"
    f"     - Num Chunks: {total_chunks}\n"
)

if are_chunks_same_seq:
    legend_text += f"     - Chunks from Same Sequence: Yes\n"
else:
    legend_text += f"     - Chunks from Same Sequence: No\n"

if do_backward:
    legend_text += f"     - Do Backward: Yes\n"
    if save_grad_activation_in_layer_buffer:
        legend_text += f"         - Store Act Grads in Layer Buf: Yes\n"
    else:
        legend_text += f"         - Store Act Grads in Layer Buf: No\n"
else:
    legend_text += f"     - Do Backward: No\n"

legend_text += (
    f"     - Per-Device Layer Capacity: {layer_capacity}\n"
    f"     - Per-Device Layer Activations Capacity: {activations_capacity}\n"
    f"     - Per-Device Layer Transitions Capacity: {transitions_capacity}\n"
    f"     - Constants:\n"
    f"         - Layer Computation: {computationFrames} Cycles\n"
    f"         - Layer Transfer: {layerTransferFrames} Cycles\n"
    f"         - Activation Transfer: {savedActivationsFrames} Cycles\n"
    f"         - Block Transition: {activationTransitionFrames} Cycles\n"
)


# Wrap legend text - less critical than code ternaries but good practice
# wrapped_legend_text = "\n".join(textwrap.fill(line, wrap_width) for line in legend_text.splitlines())
wrapped_legend_text = legend_text # Keep original formatting if wrapping looks bad

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
stall_node_positions = [] # Store stall node positions
unit_directions = []
node_transfer_distances = []

cmap = plt.get_cmap('rainbow_r')
norm = mcolors.Normalize(vmin=0, vmax=N - 1)
device_artists = {}
edge_artists = {} # Will store arrows AND compute arcs
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
    outer_label = ax.text(outer_pos[0], outer_pos[1], f'D{i}', ha='center', va='center', fontsize=7, zorder=3, bbox=dict(boxstyle='round,pad=0.1', fc=(1,1,1,0.5), ec='none'))
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
                                 orientation=np.pi/8, # Rotate slightly so flat edge is top/bottom
                                 fc=COLOR_STALL_NODE_FILL,
                                 ec=color, # Border color matches device
                                 lw=stall_node_border_width,
                                 zorder=2,
                                 visible=False) # Initially hidden
    ax.add_patch(stall_node)
    device_artists[f'stall_node_{i}'] = stall_node
    stall_label = ax.text(stall_node_pos[0], stall_node_pos[1], "",
                             ha='center', va='center', fontsize=stall_node_fontsize, fontweight='semibold',
                             color='white', zorder=3, visible=False) # Initially hidden
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
    start_pos_arrow_out = inner_edge_conn_point + edge_offset # Base start for arrow pointing OUTWARD (Visually INBOUND)
    start_pos_arrow_in = outer_edge_conn_point - edge_offset # Base start for arrow pointing INWARD (Visually OUTBOUND)

    # Create arrow object that VISUALLY points OUTWARD (Inner -> Outer) - Represents INBOUND data state
    arrow_out = patches.FancyArrowPatch(posA=start_pos_arrow_out, posB=start_pos_arrow_out,
                                        arrowstyle=arrow_style_str, color=COLOR_INBOUND_DEFAULT, linestyle='dashed',
                                        mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_out)
    label_out = ax.text(start_pos_arrow_out[0], start_pos_arrow_out[1], "", color=COLOR_INBOUND_DEFAULT, fontsize=edge_label_fontsize, ha='center', va='bottom', zorder=4)
    edge_artists[f'in_{i}'] = (arrow_out, label_out) # NOTE: SWAPPED KEY - this handles INBOUND state

    # Create arrow object that VISUALLY points INWARD (Outer -> Inner) - Represents OUTBOUND data state
    arrow_in = patches.FancyArrowPatch(posA=start_pos_arrow_in, posB=start_pos_arrow_in,
                                       arrowstyle=arrow_style_str, color=COLOR_OUTBOUND_DEFAULT, linestyle='dashed',
                                       mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_in)
    label_in = ax.text(start_pos_arrow_in[0], start_pos_arrow_in[1], "", color=COLOR_OUTBOUND_DEFAULT, fontsize=edge_label_fontsize, ha='center', va='top', zorder=4)
    edge_artists[f'out_{i}'] = (arrow_in, label_in) # NOTE: SWAPPED KEY - this handles OUTBOUND state

    # Keep ring arrows (Device -> Device)
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
    angle_degrees = np.degrees(np.arctan2(unit_dir[1], unit_dir[0])) # Angle of the device in degrees
    compute_arc = Arc(outer_pos, width=2*arc_radius, height=2*arc_radius,
                      angle=angle_degrees, # Rotate the whole arc shape
                      theta1=0.0, theta2=0.0, # Start as zero-length arc, update will set real values
                      color=COLOR_COMPUTE_DEFAULT, lw=compute_arc_linewidth, zorder=1,
                      visible=False) # Initially hidden
    ax.add_patch(compute_arc)
    edge_artists[f'compute_{i}'] = compute_arc # Store the arc artist

# --- Pre-calculate Relative Angles for Arcs ---
def get_relative_angle_deg(center_pos, target_pos, base_angle_deg):
    """Calculates angle of target_pos relative to center_pos, then makes it relative to base_angle_deg."""
    if np.allclose(center_pos, target_pos):
        return 0 # Avoid atan2 error if points are identical
    world_angle_rad = np.arctan2(target_pos[1] - center_pos[1], target_pos[0] - center_pos[0])
    world_angle_deg = np.degrees(world_angle_rad)
    relative_angle_deg = (world_angle_deg - base_angle_deg + 180) % 360 - 180
    return relative_angle_deg

device_angles_deg = np.degrees(np.arctan2([ud[1] for ud in unit_directions], [ud[0] for ud in unit_directions]))
theta_rel_to_prev = np.zeros(N)
theta_rel_to_next = np.zeros(N)

for i in range(N):
    pos_i = outer_circle_positions[i]
    base_angle_i = device_angles_deg[i]

    # --- Angle towards PREVIOUS neighbor (where FWD/HEAD data comes FROM, where BWD data goes TO) ---
    prev_idx = (i - 1 + N) % N
    pos_prev = outer_circle_positions[prev_idx]
    # Calculate the point on device i's circumference that faces device i-1
    vec_i_to_prev = pos_prev - pos_i
    norm_i_to_prev = np.linalg.norm(vec_i_to_prev)
    if norm_i_to_prev > 1e-6:
        dir_i_to_prev = vec_i_to_prev / norm_i_to_prev
        # This is the point on i's circle where an arrow *from* prev would ideally land
        target_point_from_prev = pos_i + dir_i_to_prev * outer_node_radius
        theta_rel_to_prev[i] = get_relative_angle_deg(pos_i, target_point_from_prev, base_angle_i)
    else:
        theta_rel_to_prev[i] = 0 # Should not happen for N>1

    # --- Angle towards NEXT neighbor (where BWD data comes FROM, where FWD data goes TO) ---
    next_idx = (i + 1) % N
    pos_next = outer_circle_positions[next_idx]
    # Calculate the point on device i's circumference that faces device i+1
    vec_i_to_next = pos_next - pos_i
    norm_i_to_next = np.linalg.norm(vec_i_to_next)
    if norm_i_to_next > 1e-6:
        dir_i_to_next = vec_i_to_next / norm_i_to_next
        # This is the point on i's circle where an arrow *to* next would ideally start
        target_point_to_next = pos_i + dir_i_to_next * outer_node_radius
        theta_rel_to_next[i] = get_relative_angle_deg(pos_i, target_point_to_next, base_angle_i)
    else:
        theta_rel_to_next[i] = 0

# --- Device Class  ---
class Device:
    def __init__(self, device_id, layer_capacity, activations_capacity, transitions_capacity, total_devices, total_layers, total_chunks):
        self.device_id = device_id
        self.device_has_started = False
        self.device_start_time = 0
        self.device_has_finished = False
        self.device_finish_time = 0

        self.layer_capacity = layer_capacity
        self.activations_capacity = activations_capacity
        self.transitions_capacity = transitions_capacity

        self.total_devices = total_devices
        self.total_layers = total_layers
        self.total_chunks = total_chunks

        self.cur_ready_weights = [-1 for _ in range(layer_capacity)]
        self.cur_weight_replace_ind = 0

        self.cur_ready_transitions = [-1 for _ in range(transitions_capacity)]
        self.cur_transitions_replace_ind = 0

        self.cur_fetched_activations = [-1 for _ in range(self.total_chunks)]
        self.cur_fetched_activations_replace_ind = self.total_chunks - 1

        self.cur_ready_grad_activations = [-1 for _ in range(grad_activations_capacity)]
        self.cur_grad_activations_replace_ind = grad_activations_capacity - 1

        self.cur_model_outputs = [-1 for _ in range(self.total_chunks)]
        self.cur_model_outputs_replace_ind = 0

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

        self.is_inbound_transferring = False
        self.cur_inbound_start_time = 0
        self.cur_inbound_duration = 0

        self.is_peer_transferring = False
        self.cur_peer_transfer_start_time = 0
        self.cur_peer_transfer_duration = 0

        self.cur_outbound_edge = ""
        self.cur_inbound_edge = ""
        self.cur_ring_edge = ""

        self.computing_status = "Idle"
        self.stall_reason = ""

        self.next_weight_layer_id = device_id + layer_capacity * total_devices
        self.to_fetch_activations_next = True
        self.has_reversed = False

        self.outbound_storage = set()

        # --- Fwd/Bwd Setup Logic ---
        for i in range (layer_capacity):
            layer_id_to_add = device_id + i * total_devices
            if layer_id_to_add < total_layers:
                self.cur_ready_weights[i] = layer_id_to_add
                self.outbound_storage.add((-1, layer_id_to_add, False))

        cur_layer_id = device_id
        last_fwd_layer_on_device = -1
        while cur_layer_id < total_layers:
            last_fwd_layer_on_device = cur_layer_id
            if (-1, cur_layer_id, False) not in self.outbound_storage:
                self.outbound_storage.add((-1, cur_layer_id, False))
            for i in range (total_chunks):
                self.computation_queue.append((i, cur_layer_id, False, False, 1, computationFrames))
            cur_layer_id += self.total_devices

        self.next_saved_activations_layer_id = last_fwd_layer_on_device
        if are_chunks_same_seq:
            self.next_saved_activations_chunk_id = self.total_chunks - 1
        else:
            self.next_saved_activations_chunk_id = 0

        if do_backward:
            head_layer_conceptual_id = total_layers
            if last_fwd_layer_on_device != -1 and last_fwd_layer_on_device + self.total_devices == total_layers:
                cur_layer_id = head_layer_conceptual_id
                self.outbound_storage.add((-1, cur_layer_id, False))
                if are_chunks_same_seq:
                    chunk_order = range(total_chunks - 1, -1, -1)
                else:
                    chunk_order = range(total_chunks)
                for i in chunk_order:
                    self.computation_queue.append((i, cur_layer_id, False, False, -1, computationFrames))
                cur_layer_id = last_fwd_layer_on_device
            else:
                cur_layer_id = last_fwd_layer_on_device

            while cur_layer_id >= 0:
                if are_chunks_same_seq:
                    chunk_order = range(total_chunks - 1, -1, -1)
                else:
                    chunk_order = range(total_chunks)
                for i in chunk_order:
                    self.computation_queue.append((i, cur_layer_id, True, False, -1, computationFrames))
                for i in chunk_order:
                    self.computation_queue.append((i, cur_layer_id, False, True, 0, computationFrames))
                cur_layer_id -= self.total_devices

    def handle_completed_transfers(self, T, all_devices):
        # Check and handle inbound completion
        if self.is_inbound_transferring and (self.cur_inbound_start_time + self.cur_inbound_duration <= T):
            inbound_item = self.inbound_queue.pop(0)
            chunk_id, layer_id, is_grad, replace_ind, duration = inbound_item
            if chunk_id == -1:
                self.cur_ready_weights[replace_ind] = layer_id
                item_type = 'Wgt'
            else:
                self.cur_fetched_activations[replace_ind] = (chunk_id, layer_id)
                item_type = 'Act'
            if TO_PRINT: print(f"T={T}, Dev {self.device_id}: RX INBOUND {item_type} C{chunk_id},L{layer_id} -> Idx {replace_ind}.")
            self.is_inbound_transferring = False
            self.cur_inbound_edge = ""

        # Check and handle outbound completion
        if self.is_outbound_transferring and (self.cur_outbound_start_time + self.cur_outbound_duration <= T):
            outbound_item = self.outbound_queue.pop(0)
            chunk_id, layer_id, is_grad, duration = outbound_item
            self.outbound_storage.add((chunk_id, layer_id, is_grad))
            if chunk_id >= 0:
                item_type = 'Act'
            elif is_grad:
                item_type = 'Grad'
            else:
                item_type = 'Wgt'
            if TO_PRINT: print(f"T={T}, Dev {self.device_id}: TX OUTBOUND {item_type} C{chunk_id},L{layer_id} COMPLETE.")
            self.is_outbound_transferring = False
            self.cur_outbound_edge = ""

        # Check and handle peer transfer completion
        if self.is_peer_transferring and (self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T):
            peer_item = self.peer_transfer_queue.pop(0)
            peer_id, chunk_id, layer_id, is_grad, rep_ind, duration = peer_item
            peer_dev = all_devices[peer_id]
            is_to_head_input_buffer = (layer_id == self.total_layers - 1) and (not is_grad)
            if is_to_head_input_buffer:
                peer_dev.cur_model_outputs[rep_ind] = (chunk_id, layer_id, is_grad)
                loc = "Head Input"
                item_type = 'Out' # Output activations going to next block's input
            else:
                peer_dev.cur_ready_transitions[rep_ind] = (chunk_id, layer_id, is_grad)
                loc = "Trans"
                if is_grad:
                    item_type = 'Grad'
                else:
                    item_type = 'Out'

            if TO_PRINT: print(f"T={T}, Dev {self.device_id}: TX PEER {item_type} C{chunk_id},L{layer_id} -> Dev {peer_id} {loc} Idx {rep_ind}.")
            self.is_peer_transferring = False
            self.cur_ring_edge = ""

    def handle_computation_depends(self, T):
        if not self.computation_queue:
            return False # Nothing to compute

        cid, lid, bX, bW, tdir, dur = self.computation_queue[0]
        has_deps = False
        is_fwd = (not bX) and (not bW) and (lid < self.total_layers)
        is_head = (lid == self.total_layers)
        self.stall_reason = ""
        computation_type_str = "" # For setting self.current_computation_type

        if is_fwd:
            computation_type_str = "Fwd"
            has_weight = (lid in self.cur_ready_weights)
            if lid == 0:
                has_deps = has_weight
                if not has_deps:
                    self.stall_reason = f"Missing:\nWeights for L:{lid}"
                else:
                    self.stall_reason = "" # Clear reason if deps met
            else:
                has_input_transition = ((cid, lid - 1, False) in self.cur_ready_transitions)
                has_deps = has_weight and has_input_transition
                if not has_deps:
                    missing_items = []
                    if not has_weight: missing_items.append("Weights")
                    if not has_input_transition: missing_items.append("Act. Stream")
                    self.stall_reason = f"Missing:\n{'\n'.join(missing_items)}"
        elif is_head:
            computation_type_str = "Head"
            has_weight = (lid in self.cur_ready_weights)
            has_input_from_last_block = ((cid, lid - 1, False) in self.cur_model_outputs)
            has_deps = has_weight and has_input_from_last_block
            if not has_deps:
                missing_items = []
                if not has_weight: missing_items.append("Head Weights")
                if not has_input_from_last_block: missing_items.append("Act. Stream")
                self.stall_reason = f"Missing:\n{'\n'.join(missing_items)}"
                if not has_weight and has_input_from_last_block and TO_PRINT: print(f"HEAD WEIGHTS STALL! CURRENT READY WEIGHTS: {self.cur_ready_weights}")
        else: # Backward
            if bX:
                computation_type_str = "Bwd X"
                has_weight = (lid in self.cur_ready_weights)
                has_upstream_grad = ((cid, lid + 1, True) in self.cur_ready_transitions)
                has_deps = has_weight and has_upstream_grad
                if not has_deps:
                    missing_items = []
                    if not has_weight: missing_items.append("Weights")
                    if not has_upstream_grad: missing_items.append("Upstream Grad")
                    self.stall_reason = f"Missing:\n{'\n'.join(missing_items)}"
            elif bW:
                computation_type_str = "Bwd W"
                has_fwd_activation = ((cid, lid) in self.cur_fetched_activations)
                has_activation_grad = ((cid, lid) in self.cur_ready_grad_activations)
                has_deps = has_fwd_activation and has_activation_grad
                if not has_deps:
                    missing_items = []
                    if not has_fwd_activation: missing_items.append("Fwd Activations")
                    if not has_activation_grad: missing_items.append("Grad Activations")
                    self.stall_reason = f"Missing:\n{'\n'.join(missing_items)}"

        # Update state based on dependency check
        if has_deps:
            if not self.device_has_started:
                self.device_start_time = T
                self.device_has_started = True
            if self.is_stalled and TO_PRINT:
                print(f"T={T}, Dev {self.device_id}: UNSTALL -> Comp C{cid},L{lid},{computation_type_str}. Stalled for {T - self.stall_start_time}")

            self.cur_computation_start_time = T
            self.cur_computation_duration = dur
            self.is_computing = True
            self.is_stalled = False
            self.current_computation_type = computation_type_str
            self.current_computation_layer_id = lid
            self.computing_status = f"COMPUTING:\n{computation_type_str}\nC{cid},L{lid}"
            self.stall_reason = "" # Clear stall reason when starting computation
        else:
            if not self.is_stalled:
                self.is_stalled = True
                self.stall_start_time = T
                self.computing_status = f"STALL:\n{computation_type_str}\nC{cid},L{lid}"
                if TO_PRINT: print(f"T={T}, Dev {self.device_id}: STALL on Task {(cid, lid, bX, bW, tdir, dur)}") # Simplified print
            # Ensure computation state is false if stalled
            self.is_computing = False
            self.current_computation_type = None
            self.current_computation_layer_id = -1


        return has_deps

    def handle_bwd_fetch(self, T):
         if self.to_fetch_activations_next:
             act_needed = (self.next_saved_activations_chunk_id, self.next_saved_activations_layer_id)
             if act_needed[1] >= 0: # Check if layer ID is valid
                 idx_to_replace = self.cur_fetched_activations_replace_ind
                 current_val_at_idx = self.cur_fetched_activations[idx_to_replace]
                 # Need to fetch if index has old data AND the needed data is actually stored remotely
                 if current_val_at_idx != act_needed and current_val_at_idx != -1 and (act_needed[0], act_needed[1], False) in self.outbound_storage:
                     self.inbound_queue.append((act_needed[0], act_needed[1], False, idx_to_replace, savedActivationsFrames))
                     self.cur_fetched_activations[idx_to_replace] = -1 # Mark slot as fetching
                     self.cur_fetched_activations_replace_ind = (idx_to_replace - 1 + self.total_chunks) % self.total_chunks
                     if TO_PRINT: print(f"T={T}, Dev {self.device_id}: QUEUE BWD Fetch Act C{act_needed[0]},L{act_needed[1]} -> Idx {idx_to_replace}")

                     # Determine next activation to fetch
                     if are_chunks_same_seq:
                         next_cid = act_needed[0] - 1
                         if next_cid < 0:
                             self.next_saved_activations_layer_id -= self.total_devices
                             self.next_saved_activations_chunk_id = self.total_chunks - 1
                             self.to_fetch_activations_next = False # Switch to fetching weights next
                         else:
                             self.next_saved_activations_chunk_id = next_cid
                     else: # Not same sequence
                         next_cid = (act_needed[0] + 1) % self.total_chunks
                         if next_cid == 0:
                             self.next_saved_activations_layer_id -= self.total_devices
                             self.next_saved_activations_chunk_id = 0
                             self.to_fetch_activations_next = False # Switch to fetching weights next
                         else:
                             self.next_saved_activations_chunk_id = next_cid
                 # Else: Data already present, or not stored remotely (shouldn't happen in correct logic), or slot is free (-1) - advance logic anyway if slot is free? No, wait until fetch needed.
                 # If we didn't queue a fetch, we might need to advance the pointer if the data was already correct or slot free.
                 # Let's only advance the pointers *after* queuing the fetch. If no fetch is queued, we try again next cycle.

             else: # No more valid activation layers to fetch
                 self.to_fetch_activations_next = False # Switch to fetching weights next time

         else: # Fetch Weight
             wgt_needed = self.next_weight_layer_id
             if wgt_needed >= 0: # Check if layer ID is valid
                 idx_to_replace = self.cur_weight_replace_ind
                 current_val_at_idx = self.cur_ready_weights[idx_to_replace]
                 # Need to fetch if index has old data AND the needed data is actually stored remotely
                 if current_val_at_idx != wgt_needed and current_val_at_idx != -1 and (-1, wgt_needed, False) in self.outbound_storage:
                     self.inbound_queue.append((-1, wgt_needed, False, idx_to_replace, layerTransferFrames))
                     self.cur_ready_weights[idx_to_replace] = -1 # Mark slot as fetching
                     self.cur_weight_replace_ind = (idx_to_replace - 1 + self.layer_capacity) % self.layer_capacity
                     if TO_PRINT: print(f"T={T}, Dev {self.device_id}: QUEUE BWD Fetch Wgt L{wgt_needed} -> Idx {idx_to_replace}")
                     self.next_weight_layer_id -= self.total_devices
                     self.to_fetch_activations_next = True # Switch back to fetching activations
                 # Else: Weight already present or slot free, advance logic anyway? No, wait until fetch needed.

             else: # No more valid weight layers to fetch
                 self.to_fetch_activations_next = True # Switch back to activations (though likely done)


    def handle_computation(self, T, all_devices):
        completed_tasks = 0
        # Try to start computing if idle and queue has tasks
        if not self.is_computing and not self.is_stalled and len(self.computation_queue) > 0:
            # This call might set is_computing=True or is_stalled=True
            self.handle_computation_depends(T)

        # Check if ongoing computation is finished
        elif self.is_computing and (self.cur_computation_start_time + self.cur_computation_duration <= T):
            task = self.computation_queue.pop(0)
            completed_tasks += 1
            cid, lid, bX, bW, tdir, dur = task
            is_head = (lid == self.total_layers)
            is_fwd = (not bX) and (not bW) and (not is_head)

            # Determine task type string for printing
            task_type_str = ""
            if bX: task_type_str = 'X'
            elif bW: task_type_str = 'W'
            elif is_head: task_type_str = 'H'
            else: task_type_str = 'F' # is_fwd must be true

            if TO_PRINT: print(f"T={T}, Dev {self.device_id}: FINISHED Comp -> C{cid},L{lid},{task_type_str}")

            # Handle outputs/transfers based on task type
            if is_fwd:
                is_last_fwd_for_bwd_save = (lid == self.next_saved_activations_layer_id)
                should_outbound_save = True
                if is_last_fwd_for_bwd_save and save_grad_activation_in_layer_buffer:
                    should_outbound_save = False # Saved locally instead
                    idx = self.cur_fetched_activations_replace_ind
                    self.cur_fetched_activations[idx] = (cid, lid)
                    self.cur_fetched_activations_replace_ind = (idx - 1 + self.total_chunks) % self.total_chunks

                if should_outbound_save:
                    self.outbound_queue.append((cid, lid, False, savedActivationsFrames))

                target_peer_id = (self.device_id + tdir) % self.total_devices
                is_output_to_head_buffer = (lid == self.total_layers - 1)
                if is_output_to_head_buffer:
                    target_dev = all_devices[target_peer_id]
                    idx = target_dev.cur_model_outputs_replace_ind
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, False, idx, activationTransitionFrames))
                    target_dev.cur_model_outputs_replace_ind = (idx + 1) % target_dev.total_chunks
                else:
                    target_dev = all_devices[target_peer_id]
                    idx = target_dev.cur_transitions_replace_ind
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, False, idx, activationTransitionFrames))
                    target_dev.cur_transitions_replace_ind = (idx + 1) % target_dev.transitions_capacity

            elif is_head:
                target_peer_id = (self.device_id + tdir) % self.total_devices
                target_dev = all_devices[target_peer_id]
                idx = target_dev.cur_transitions_replace_ind
                self.peer_transfer_queue.append((target_peer_id, cid, lid, True, idx, activationTransitionFrames)) # Send grad (True)
                target_dev.cur_transitions_replace_ind = (idx + 1) % target_dev.transitions_capacity

            else: # Backward (bX or bW)
                if bX:
                    # Store activation gradient locally
                    idx = self.cur_grad_activations_replace_ind
                    self.cur_ready_grad_activations[idx] = (cid, lid)
                    self.cur_grad_activations_replace_ind = (idx - 1 + self.grad_activations_capacity) % self.grad_activations_capacity

                    # Send activation gradient to peer
                    target_peer_id = (self.device_id + tdir) % self.total_devices
                    target_dev = all_devices[target_peer_id]
                    idx = target_dev.cur_transitions_replace_ind
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, True, idx, activationTransitionFrames)) # Send grad (True)
                    target_dev.cur_transitions_replace_ind = (idx + 1) % target_dev.transitions_capacity
                elif bW:
                    # Check if this is the last chunk for this layer's weight gradient
                    is_last_chunk_for_wgrad = False
                    if are_chunks_same_seq:
                        if cid == 0: is_last_chunk_for_wgrad = True
                    else: # not same sequence
                        if cid == self.total_chunks - 1: is_last_chunk_for_wgrad = True

                    if is_last_chunk_for_wgrad:
                        self.outbound_queue.append((-1, lid, True, layerTransferFrames)) # Save weight gradient

            # Prefetching Logic
            if not self.has_reversed: # Still in forward or initial head phase
                is_last_fwd_chunk = (cid == self.total_chunks - 1)
                if is_fwd and is_last_fwd_chunk and self.next_weight_layer_id < total_layers: # Check layer validity
                    target_idx = self.cur_weight_replace_ind
                    needs_fetch = False
                    current_weight_at_idx = self.cur_ready_weights[target_idx]

                    # Determine if fetch is needed: slot has a different, valid weight AND needed weight exists in storage
                    if current_weight_at_idx != self.next_weight_layer_id and \
                       current_weight_at_idx != -1 and \
                       (-1, self.next_weight_layer_id, False) in self.outbound_storage:
                        needs_fetch = True

                    if needs_fetch:
                        self.inbound_queue.append((-1, self.next_weight_layer_id, False, target_idx, layerTransferFrames))
                        self.cur_ready_weights[target_idx] = -1 # Mark as fetching

                    # Always advance pointers after considering the *current* next_weight_layer_id
                    next_potential_layer_id = self.next_weight_layer_id + self.total_devices
                    self.cur_weight_replace_ind = (self.cur_weight_replace_ind + 1) % self.layer_capacity

                    # Check if we need to reverse for backward pass
                    if next_potential_layer_id >= total_layers and do_backward:
                        self.has_reversed = True
                        # Calculate first BWD weight layer needed based on last FWD save layer
                        first_bwd_w_layer = self.next_saved_activations_layer_id
                        # Calculate the layer ID for the slot that will be replaced *next* during BWD
                        self.next_weight_layer_id = first_bwd_w_layer - (self.layer_capacity - 1) * self.total_devices
                        # Adjust replace index for BWD (start replacing the oldest FWD weight)
                        # The index already advanced, so point it back one step relative to capacity
                        self.cur_weight_replace_ind = (self.cur_weight_replace_ind - 1 + self.layer_capacity) % self.layer_capacity # Corrected BWD start index
                    else:
                        self.next_weight_layer_id = next_potential_layer_id # Continue FWD prefetching

            elif do_backward: # Already reversed, handle backward prefetching
                 # Check if it's the last chunk computation for Head or Bwd W
                 is_last_chunk_head = False
                 if is_head:
                     if are_chunks_same_seq and cid == 0: is_last_chunk_head = True
                     elif not are_chunks_same_seq and cid == self.total_chunks - 1: is_last_chunk_head = True

                 is_last_chunk_bwd_w = False
                 if bW:
                     if are_chunks_same_seq and cid == 0: is_last_chunk_bwd_w = True
                     elif not are_chunks_same_seq and cid == self.total_chunks - 1: is_last_chunk_bwd_w = True

                 if is_last_chunk_head or is_last_chunk_bwd_w:
                     self.handle_bwd_fetch(T) # Trigger potential fetch (weight or activation)

            # Reset state after computation
            self.is_computing = False
            self.computing_status = "Idle"
            self.current_computation_type = None
            self.current_computation_layer_id = -1


            # Check if more tasks exist and try to start the next one
            if len(self.computation_queue) > 0:
                self.handle_computation_depends(T) # Try starting next task immediately
            else:
                # No more tasks, device is finished
                if not self.device_has_finished:
                    self.device_finish_time = T
                    self.device_has_finished = True
                    self.computing_status = "Finished"

        elif self.is_stalled:
            # If stalled, keep checking dependencies
            self.handle_computation_depends(T)

        return completed_tasks

    def handle_new_transfers(self, T):
        # Start inbound transfer if possible
        if not self.is_inbound_transferring and len(self.inbound_queue) > 0:
            item = self.inbound_queue[0]
            cid, lid, isg = item[0], item[1], item[2]
            duration = item[-1]
            # Check if the required data is actually available in central storage
            if (cid, lid, isg) in self.outbound_storage:
                self.is_inbound_transferring = True
                self.cur_inbound_start_time = T
                self.cur_inbound_duration = duration
                if cid == -1:
                    self.cur_inbound_edge = f"L:{lid}"
                else:
                    self.cur_inbound_edge = f"Act:C{cid},L{lid}"

        # Start outbound transfer if possible
        if not self.is_outbound_transferring and len(self.outbound_queue) > 0:
            item = self.outbound_queue[0]
            cid, lid, isg, duration = item
            self.is_outbound_transferring = True
            self.cur_outbound_start_time = T
            self.cur_outbound_duration = duration
            if cid >= 0: # Activation save
                self.cur_outbound_edge = f"Act:C{cid},L{lid}"
            else: # Weight or Gradient save
                if isg: # Gradient
                    self.cur_outbound_edge = f"Grad:L{lid}"
                else: # Weight
                    self.cur_outbound_edge = f"Wgt:L{lid}"

        # Start peer transfer if possible
        if not self.is_peer_transferring and len(self.peer_transfer_queue) > 0:
            item = self.peer_transfer_queue[0]
            pid, cid, lid, isg, ridx, duration = item
            self.is_peer_transferring = True
            self.cur_peer_transfer_start_time = T
            self.cur_peer_transfer_duration = duration
            if isg: # Gradient transfer
                self.cur_ring_edge = f"Grad:C{cid},L{lid}"
            else: # Output/Activation transfer
                self.cur_ring_edge = f"Out:C{cid},L{lid}"

# --- Global Simulation State ---
all_devices = {}
total_tasks = 0
total_completed_tasks = {} # Dict to store completed tasks per frame
total_computation_time = 0
current_frame_index = 0 # THIS IS THE MASTER TIME COUNTER
animation_paused = False  # Start playing automatically
completion_text_artist = None
target_cycle = None # Target cycle for "Run to Cycle"

# --- Simulation Reset Function (MODIFIED) ---
def reset_simulation():
    """Resets the simulation state and visual elements."""
    global all_devices, total_tasks, total_completed_tasks, total_computation_time
    global current_frame_index, animation_paused, completion_text_artist, target_cycle # Added target_cycle
    global unit_directions, inner_node_centers, outer_circle_positions, inner_node_radius, outer_node_radius, arrow_offset_dist

    if TO_PRINT:
        print("--- Resetting Simulation ---")

    # Recreate devices
    all_devices = {i: Device(i, layer_capacity, activations_capacity, transitions_capacity, total_devices, total_layers, total_chunks) for i in range(N)}

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
        arrow_in_vis.set_color(COLOR_INBOUND_DEFAULT)
        arrow_in_vis.set_positions(start_pos_arrow_out, start_pos_arrow_out)

        arrow_out_vis, label_out_vis = edge_artists[f'out_{i}']
        label_out_vis.set_text("")
        arrow_out_vis.set_color(COLOR_OUTBOUND_DEFAULT)
        arrow_out_vis.set_positions(start_pos_arrow_in, start_pos_arrow_in)

        arrow_ring, label_ring = edge_artists[f'ring_{i}']
        label_ring.set_text("")
        arrow_ring.set_color(COLOR_RING_CCW)
        arrow_ring.set_positions(start_pos_ring, start_pos_ring)
        arrow_ring.set_connectionstyle(f"arc3,rad=0.2")

        # Reset Stall Node
        if f'stall_node_{i}' in device_artists: device_artists[f'stall_node_{i}'].set_visible(False)
        if f'stall_label_{i}' in device_label_artists: device_label_artists[f'stall_label_{i}'].set_text(""); device_label_artists[f'stall_label_{i}'].set_visible(False)

        # Reset Device Labels
        if f'circle_{i}' in device_label_artists: device_label_artists[f'circle_{i}'].set_text(f'D{i}\nIdle')
        if f'inner_label_{i}' in device_label_artists: device_label_artists[f'inner_label_{i}'].set_text(f'D{i}\nHome')

        # --- Reset Computation Arc ---
        if f'compute_{i}' in edge_artists:
            compute_arc = edge_artists[f'compute_{i}']
            compute_arc.set_visible(False)
            compute_arc.theta1 = 0.0 # Reset angles
            compute_arc.theta2 = 0.0 # Reset angles


    title_obj.set_text(f'Cycle {current_frame_index}')

    if TO_PRINT:
        print(f"Reset complete. Total tasks: {total_tasks}")

# --- Update Function  ---
def update(frame): # Keep frame argument, but we'll ignore its value for T
    """Main animation update function."""
    global total_completed_tasks, current_frame_index, completion_text_artist, animation_paused, target_cycle # Added target_cycle
    global theta_rel_to_prev, theta_rel_to_next

    # --- Early Exit if Paused ---
    if animation_paused:
        # Collect existing artists without updating state
        all_artists = [title_obj]
        for i in range(N):
            all_artists.extend([
                device_label_artists[f'circle_{i}'], device_label_artists[f'inner_label_{i}'],
                device_artists[f'stall_node_{i}'], device_label_artists[f'stall_label_{i}'],
                edge_artists[f'in_{i}'][0], edge_artists[f'in_{i}'][1],
                edge_artists[f'out_{i}'][0], edge_artists[f'out_{i}'][1],
                edge_artists[f'ring_{i}'][0], edge_artists[f'ring_{i}'][1],
                edge_artists[f'compute_{i}'] # Add compute arc
            ])
        if completion_text_artist:
            all_artists.append(completion_text_artist)
        return all_artists

    # --- Use Global Frame Index ---
    T = current_frame_index

    # --- Check if Target Cycle Reached ---
    if target_cycle is not None and T == target_cycle:
        print(f"Reached target cycle {T}, pausing.")
        if ani.event_source is not None:
            ani.event_source.stop()
        animation_paused = True
        target_cycle = None # Clear the target
        title_obj.set_text(f'Cycle {T} (Paused)')
        fig.canvas.draw_idle()
        # Need to return artists for this paused frame
        all_artists = [title_obj]
        for i in range(N):
             all_artists.extend([
                 device_label_artists[f'circle_{i}'], device_label_artists[f'inner_label_{i}'],
                 device_artists[f'stall_node_{i}'], device_label_artists[f'stall_label_{i}'],
                 edge_artists[f'in_{i}'][0], edge_artists[f'in_{i}'][1],
                 edge_artists[f'out_{i}'][0], edge_artists[f'out_{i}'][1],
                 edge_artists[f'ring_{i}'][0], edge_artists[f'ring_{i}'][1],
                 edge_artists[f'compute_{i}'] # Add compute arc
             ])
        if completion_text_artist:
            all_artists.append(completion_text_artist)
        return all_artists # RETURN after pausing at target

    # --- Proceed with Normal Update ---
    title_obj.set_text(f'Cycle {T}')
    artists_to_update = [title_obj]

    # --- Update Task Completion Count ---
    # Ensure T exists, copying count from the last known frame if needed
    if T not in total_completed_tasks:
        last_known_frame = -1 # Default if T=0
        # Find the largest key smaller than T
        valid_keys = [k for k in total_completed_tasks if k < T]
        if valid_keys:
            last_known_frame = max(valid_keys)

        # Get count from last known frame, default to 0 if no previous frame exists
        total_completed_tasks[T] = total_completed_tasks.get(last_known_frame, 0)


    # Run Simulation Step
    # 1. Handle completions from previous cycle
    for i in range(N):
        all_devices[i].handle_completed_transfers(T, all_devices)
    # 2. Handle computations (potentially completing tasks and starting new ones)
    newly_completed = 0
    for i in range(N):
        newly_completed += all_devices[i].handle_computation(T, all_devices)
    total_completed_tasks[T] = total_completed_tasks.get(T-1, 0) + newly_completed # Update based on T-1 count
    # 3. Handle initiating new transfers based on state after computation
    for i in range(N):
        all_devices[i].handle_new_transfers(T)

    if T % 200 == 0 and T > 0: # Avoid printing at T=0 if not needed
        if TO_PRINT: print(f"Cycle {T} - Device 0 Weights: {all_devices[0].cur_ready_weights}")

    # --- Update Visuals ---
    for i in range(N):
        device = all_devices[i]
        unit_dir = unit_directions[i]
        inner_center = inner_node_centers[i]
        outer_pos = outer_circle_positions[i]
        transfer_dist_i = node_transfer_distances[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
        edge_offset = radial_perp_vector * arrow_offset_dist

        # Update Device Labels and Stall Node
        outer_label_artist = device_label_artists[f'circle_{i}']
        status_text = device.computing_status
        outer_label_artist.set_text(f'D{i}\n{status_text}')

        inner_label_artist = device_label_artists[f'inner_label_{i}']
        inner_label_artist.set_text(f"D{i}\nHome") # Usually static

        stall_node_artist = device_artists[f'stall_node_{i}']
        stall_label_artist = device_label_artists[f'stall_label_{i}']
        if device.is_stalled and device.stall_reason:
            stall_label_artist.set_text(device.stall_reason)
            stall_label_artist.set_visible(True)
            stall_node_artist.set_visible(True)
        else:
            stall_label_artist.set_text("")
            stall_label_artist.set_visible(False)
            stall_node_artist.set_visible(False)

        # Update Transfer Arrows (Inbound, Outbound, Ring)
        arrow_vis_inbound, label_vis_inbound = edge_artists[f'in_{i}']
        arrow_vis_outbound, label_vis_outbound = edge_artists[f'out_{i}']
        arrow_ring, label_ring = edge_artists[f'ring_{i}']

        # --- Inbound Arrow Update ---
        len_in_prog = 0.0
        cur_inbound_edge_text = ""
        color_inbound = COLOR_INBOUND_DEFAULT
        if device.is_inbound_transferring and device.cur_inbound_duration > 0:
            prog_frac = min(1.0, (T - device.cur_inbound_start_time) / device.cur_inbound_duration)
            len_in_prog = prog_frac * transfer_dist_i
            cur_inbound_edge_text = device.cur_inbound_edge
            is_act = "Act:" in cur_inbound_edge_text
            is_wgt = "L:" in cur_inbound_edge_text # Assuming 'L:' only used for weights inbound
            if is_act and device.has_reversed:
                color_inbound = COLOR_INBOUND_ACTIVATION
            elif is_wgt:
                color_inbound = COLOR_INBOUND_WEIGHT
            # else: color_inbound remains default

        arrow_vis_inbound.set_color(color_inbound)
        label_vis_inbound.set_color(color_inbound)
        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
        start_vis_inbound = inner_edge_conn_point + edge_offset
        end_vis_inbound = start_vis_inbound + unit_dir * len_in_prog
        arrow_vis_inbound.set_positions(start_vis_inbound, end_vis_inbound)
        label_perp_offset_in = radial_perp_vector * label_offset_distance
        midpoint_vis_in = (start_vis_inbound + end_vis_inbound) / 2
        label_pos_vis_in = midpoint_vis_in + label_perp_offset_in * 2 # Offset perpendicular outward
        label_vis_inbound.set_position(label_pos_vis_in)
        label_vis_inbound.set_text(cur_inbound_edge_text)


        # --- Outbound Arrow Update ---
        len_out_prog = 0.0
        cur_outbound_edge_text = ""
        color_outbound = COLOR_OUTBOUND_DEFAULT
        if device.is_outbound_transferring and device.cur_outbound_duration > 0:
            prog_frac = min(1.0, (T - device.cur_outbound_start_time) / device.cur_outbound_duration)
            len_out_prog = prog_frac * transfer_dist_i
            cur_outbound_edge_text = device.cur_outbound_edge
            if "Grad:L" in cur_outbound_edge_text:
                 color_outbound = COLOR_OUTBOUND_WGT_GRAD
            # else: color_outbound remains default

        arrow_vis_outbound.set_color(color_outbound)
        label_vis_outbound.set_color(color_outbound)
        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
        start_vis_outbound = outer_edge_conn_point - edge_offset
        end_vis_outbound = start_vis_outbound - unit_dir * len_out_prog # Move towards center
        arrow_vis_outbound.set_positions(start_vis_outbound, end_vis_outbound)
        label_perp_offset_out = radial_perp_vector * label_offset_distance
        midpoint_vis_out = (start_vis_outbound + end_vis_outbound) / 2
        label_pos_vis_out = midpoint_vis_out - label_perp_offset_out * 2 # Offset perpendicular inward
        label_vis_outbound.set_position(label_pos_vis_out)
        label_vis_outbound.set_text(cur_outbound_edge_text)


        # --- Ring Arrow Update ---
        len_ring_prog = 0.0
        peer_device_id = -1
        cur_ring_edge_text = ""
        color_ring = COLOR_RING_CCW # Default CCW
        connection_style_rad_sign = '' # Default positive for CCW
        connection_style_ring = f"arc3,rad=0.2" # Default CCW

        if device.is_peer_transferring and device.cur_peer_transfer_duration > 0:
            peer_item = device.peer_transfer_queue[0]
            peer_device_id = peer_item[0]
            prog_frac = min(1.0, (T - device.cur_peer_transfer_start_time) / device.cur_peer_transfer_duration)
            len_ring_prog = prog_frac
            cur_ring_edge_text = device.cur_ring_edge
            target_is_cw = (peer_device_id == (i - 1 + N) % N)
            if target_is_cw:
                color_ring = COLOR_RING_CW
                connection_style_rad_sign = '-'
            else:
                color_ring = COLOR_RING_CCW
                connection_style_rad_sign = '' # Explicitly positive/default
            connection_style_ring = f"arc3,rad={connection_style_rad_sign}0.2"

        arrow_ring.set_color(color_ring)
        label_ring.set_color(color_ring)
        arrow_ring.set_connectionstyle(connection_style_ring)

        start_pos_ring_geo = outer_pos
        current_end_point_ring = start_pos_ring_geo # Default to start if no transfer

        if peer_device_id != -1: # Calculate target point if transferring
            target_pos_ring_geo_center = outer_circle_positions[peer_device_id]
            vec = target_pos_ring_geo_center - start_pos_ring_geo
            norm = np.linalg.norm(vec)
            if norm > 1e-6: # Avoid division by zero
                # Calculate points on the circumference
                start_offset_dir = vec / norm
                start_pos_ring_geo = outer_pos + start_offset_dir * outer_node_radius
                target_pos_ring_geo = target_pos_ring_geo_center - start_offset_dir * outer_node_radius # Point on target circumference

                # Interpolate along the vector between circumference points
                current_end_point_ring = start_pos_ring_geo + (target_pos_ring_geo - start_pos_ring_geo) * len_ring_prog
            # else: remain at start_pos_ring_geo (shouldn't happen for distinct devices)

        arrow_ring.set_positions(start_pos_ring_geo, current_end_point_ring)

        # Position ring label
        label_pos_ring = (start_pos_ring_geo + current_end_point_ring) / 2 # Initial midpoint
        if len_ring_prog > 1e-6: # Offset label if arrow has length
             edge_vec = current_end_point_ring - start_pos_ring_geo
             norm = np.linalg.norm(edge_vec)
             if norm > 1e-6:
                 # Perpendicular vector: (-dy, dx)
                 perp_vec = np.array([-edge_vec[1], edge_vec[0]]) / norm
                 offset_direction_multiplier = 1.0
                 if connection_style_rad_sign == '-': # CW transfer
                     offset_direction_multiplier = -1.0
                 label_pos_ring = label_pos_ring + perp_vec * label_offset_distance * 3 * offset_direction_multiplier
             # else: Keep label at midpoint if vector is zero length

        label_ring.set_position(label_pos_ring)
        label_ring.set_text(cur_ring_edge_text)


        # --- Update Computation Arc ---
        compute_arc = edge_artists[f'compute_{i}']
        progress_frac = 0.0
        compute_color = COLOR_COMPUTE_DEFAULT

        if device.is_computing and device.cur_computation_duration > 0:
            progress_frac = min(1.0, (T - device.cur_computation_start_time) / device.cur_computation_duration)
            comp_type = device.current_computation_type
            comp_lid = device.current_computation_layer_id # Layer ID stored when computation started

            # --- Determine Arc Color ---
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

            # --- Determine Arc Angles (theta1, theta2) ---
            theta1_rel = 0.0
            theta2_target_rel = 0.0
            total_sweep = 0.0
            is_cw = False # Flag for clockwise sweep

            if comp_type == "Fwd":
                 if comp_lid == 0:
                     theta1_rel = -180.0
                 else:
                     theta1_rel = theta_rel_to_prev[i] # Start from direction of previous node
                 theta2_target_rel = theta_rel_to_next[i]
                 total_sweep = (theta2_target_rel - theta1_rel + 360) % 360
                 if total_sweep < 1e-6: total_sweep = 360.0 if N <= 2 else 0.0 # Avoid zero sweep? Maybe small value? Let's allow 0.
            elif comp_type == "Head":
                 theta1_rel = theta_rel_to_prev[i]
                 total_sweep = 360.0
                 theta2_target_rel = theta1_rel # Target angle isn't used directly for fixed sweep
            elif comp_type == "Bwd X" or comp_type == "Bwd W":
                 theta1_rel = theta_rel_to_next[i]
                 theta2_target_rel = theta_rel_to_prev[i]
                 total_sweep = (theta1_rel - theta2_target_rel + 360) % 360
                 is_cw = True
                 if total_sweep < 1e-6: total_sweep = 360.0 if N <= 2 else 0.0

            # --- Calculate Current End Angle ---
            current_sweep = progress_frac * total_sweep
            if is_cw:
                theta2_rel = theta1_rel - current_sweep
            else:
                theta2_rel = theta1_rel + current_sweep

            # --- Update Arc ---
            compute_arc.theta1 = theta1_rel
            compute_arc.theta2 = theta2_rel
            compute_arc.set_visible(True) # Visible when computing
            compute_arc.set_edgecolor(compute_color)

        else:
            compute_arc.set_visible(False)


        # Add all updated artists for this device
        artists_to_update.extend([
            outer_label_artist, inner_label_artist, stall_node_artist, stall_label_artist,
            arrow_vis_inbound, label_vis_inbound,
            arrow_vis_outbound, label_vis_outbound,
            arrow_ring, label_ring,
            compute_arc
        ])


    # Check for Completion
    current_total_completed = total_completed_tasks.get(T, 0)
    is_newly_complete = (current_total_completed >= total_tasks) and (total_completed_tasks.get(T-1, 0) < total_tasks)
    should_stop_animation = is_newly_complete

    # Display Completion Text
    if current_total_completed >= total_tasks and completion_text_artist is None:
        # Calculate final statistics
        start_bubble = sum(d.device_start_time for d in all_devices.values() if d.device_has_started)
        stop_bubble = sum(max(0, T - d.device_finish_time) for d in all_devices.values() if d.device_has_finished)

        if T > 0:
            total_dev_time = T * N
            steady_time = total_dev_time - stop_bubble - start_bubble
            if total_dev_time > 0:
                overall_eff = (total_computation_time / total_dev_time * 100)
            else:
                overall_eff = 0.0

            if steady_time > 0:
                steady_eff = (total_computation_time / steady_time * 100)
            else:
                steady_eff = 0.0
        else: # Avoid division by zero if T=0
            total_dev_time = 0
            steady_time = 0
            overall_eff = 0.0
            steady_eff = 0.0


        completion_text = (
            f"Simulation Complete!\nFinal Cycle Count: {T}\n\n"
            f"Problem:\nTotal Tasks: {total_tasks}\n"
            f"Total Task Computation Time: {total_computation_time}\n"
            f"Utilized {N} devices for aggregate {total_dev_time} cycles\n\n"
            f"Pipeline:\nFill Time: {start_bubble}\n"
            f"Flush Time: {stop_bubble}\n"
            f"Steady-State Time: {steady_time}\n\n"
            f"EFFICIENCY:\nOverall: {overall_eff:.2f}%\n"
            f"Steady-State: {steady_eff:.2f}%"
        )
        completion_text_artist = ax.text(0.5, 0.5, completion_text, transform=ax.transAxes,
                                         ha='center', va='center', fontsize=14, color='navy', fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.5', fc=(0.9, 0.9, 1, 0.95), ec='black'), zorder=10)
        if TO_PRINT: print(completion_text)
        artists_to_update.append(completion_text_artist)


    # Stop animation timer AFTER processing frame T if complete
    if should_stop_animation and ani is not None and ani.event_source is not None:
        if not animation_paused:
            ani.event_source.stop()
            print("Animation Complete - Paused")
        animation_paused = True
        target_cycle = None # Clear target on completion
        title_obj.set_text(f'Cycle {T}') # Ensure final cycle shown

    # --- Increment Global Frame Index for the *next* call ---
    if not animation_paused:
        current_frame_index += 1
        # Check max frames limit
        if current_frame_index >= max_frames:
            print(f"Max frames ({max_frames}) reached, stopping animation.")
            if ani.event_source is not None:
                ani.event_source.stop()
            animation_paused = True
            target_cycle = None # Clear target
            title_obj.set_text(f'Cycle {T} (Max Frames)')

    return artists_to_update

# --- Create Animation ---
# blit=False is generally safer when adding/removing complex artists or changing visibility frequently.
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=initial_frame_interval,
                              blit=False, repeat=False, save_count=max_frames)


# --- Widgets ---
fig.subplots_adjust(bottom=0.25, right=0.85)

# Original Widgets
ax_slider = fig.add_axes([0.15, 0.15, 0.7, 0.03]) # Moved slider up slightly
ax_restart = fig.add_axes([0.15, 0.09, 0.2, 0.04])
ax_pause = fig.add_axes([0.40, 0.09, 0.2, 0.04])
ax_play = fig.add_axes([0.65, 0.09, 0.2, 0.04])
ax_textbox = fig.add_axes([0.15, 0.03, 0.5, 0.04]) # Left, Bottom, Width, Height
ax_runto_btn = fig.add_axes([0.70, 0.03, 0.15, 0.04]) # Next to textbox

# Instantiate Widgets
btn_restart = Button(ax_restart, 'Restart')
btn_pause = Button(ax_pause, 'Pause')
btn_play = Button(ax_play, 'Play')
slider_speed = Slider(ax=ax_slider, label='Speed Level', valmin=min_speed_level, valmax=max_speed_level, valinit=initial_speed_level, valstep=1)

# Displays the first 'steady-state' cycle
initial_run_to_cycle_guess = (N + total_chunks - 1) * computationFrames + activationTransitionFrames * N
textbox_runto = TextBox(ax_textbox, "Run to Cycle:", initial=str(initial_run_to_cycle_guess), textalignment="center")
btn_runto = Button(ax_runto_btn, 'Run')


# --- Define Widget Callback Functions ---

def pause_animation(event):
    """Callback function for the Pause button."""
    global animation_paused, target_cycle
    if not animation_paused:
        if ani.event_source is not None:
            ani.event_source.stop()
        animation_paused = True
        target_cycle = None
        title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
        fig.canvas.draw_idle()
        print("Animation Paused")
    else:
        print("Animation already paused.")

def play_animation(event):
    """Callback function for the Play button."""
    global animation_paused, target_cycle
    target_cycle = None
    current_completed = total_completed_tasks.get(current_frame_index, 0)

    # Check if animation should play
    can_play = False
    if animation_paused and current_completed < total_tasks and current_frame_index < max_frames:
        can_play = True

    if can_play:
        if ani.event_source is not None:
            title_obj.set_text(f'Cycle {current_frame_index}') # Clear (Paused)
            ani.event_source.start()
            animation_paused = False
            print("Animation Resumed")
        else:
             print("Error: Animation event source not found.")
    # Provide feedback if play doesn't start
    elif current_completed >= total_tasks:
        print("Animation already complete.")
    elif current_frame_index >= max_frames:
        print("Animation stopped at max frames.")
    elif not animation_paused: # Already playing
        print("Animation already playing.")
    else: # Should not happen based on logic above, but catch-all
        print("Cannot play animation (unknown reason).")


def update_speed(val):
    """Callback function for the Speed slider."""
    global animation_paused
    speed_level = slider_speed.val
    new_interval = calculate_interval(speed_level, min_speed_level, max_speed_level, min_interval, max_interval)
    was_playing = not animation_paused # Check state BEFORE stopping timer

    # Stop the timer to change interval
    if ani.event_source is not None:
        ani.event_source.stop()
        ani.event_source.interval = new_interval
        ani._interval = new_interval # Internal interval needs update too
    else:
        print("Error: Could not access animation timer to update interval."); return

    current_completed = total_completed_tasks.get(current_frame_index, 0)
    # Determine if the animation should resume playing *after* speed change
    should_resume = False
    if was_playing and current_completed < total_tasks and current_frame_index < max_frames and target_cycle is None:
        # Resume only if it WAS playing, not finished, not maxed out, AND not currently running to a target cycle
        should_resume = True

    if should_resume:
        ani.event_source.start()
        animation_paused = False
        # Title is updated by the regular update function
    else:
        # Ensure state is paused if not resuming
        animation_paused = True
        # Update title to show paused state, unless already complete or maxed
        if current_completed < total_tasks and current_frame_index < max_frames:
            # Show (Paused) unless we are paused because we hit the target cycle
            # (The target cycle check handles its own title update)
            if target_cycle is None or current_frame_index != target_cycle:
                 title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
                 fig.canvas.draw_idle() # Redraw to show updated title


    print(f"Speed Level: {int(round(speed_level))}, Interval set to: {new_interval} ms")

def restart_animation_callback(event):
    """Callback function for the Restart button."""
    global animation_paused
    print("Restart button clicked.")
    if ani.event_source is not None:
        ani.event_source.stop() # Stop any current animation

    # IMPORTANT: Call reset simulation *before* trying to draw or restart timer
    reset_simulation() # Resets state including target_cycle, current_frame_index, and sets animation_paused = False initially

    # Redraw the figure in its reset state
    fig.canvas.draw_idle()
    try:
        # Flush events to ensure the draw happens before potentially restarting timer quickly
        fig.canvas.flush_events()
    except AttributeError:
        pass # Some backends might not have flush_events

    # Decide whether to start playing based on the initial state (which reset_simulation sets)
    if not animation_paused: # Should be true after reset
        if ani.event_source is not None:
            # Start the animation timer from the beginning
            ani.event_source.start()
            print("Simulation reset and playing from Cycle 0.")
        else:
            # If timer can't start, force paused state and update title
            print("Error: Cannot restart animation timer after reset.")
            animation_paused = True # Correct the state
            title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
            fig.canvas.draw_idle()
    else:
        # This case should ideally not happen if reset_simulation works correctly,
        # but handle it defensively.
        print("Simulation reset and paused at Cycle 0.")
        # Ensure title reflects the paused state
        title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
        fig.canvas.draw_idle()


def run_to_cycle_callback(event):
    """Callback for the 'Run' button next to the TextBox."""
    global target_cycle, animation_paused, current_frame_index

    # --- Input Parsing and Validation ---
    input_text = textbox_runto.text
    try:
        requested_cycle = int(input_text)
    except ValueError:
        print(f"Invalid input: '{input_text}'. Please enter an integer cycle number.")
        # Reset textbox to current frame if invalid input
        textbox_runto.set_val(str(current_frame_index))
        return

    if requested_cycle < 0:
        print(f"Invalid input: {requested_cycle}. Cycle number must be non-negative.")
        textbox_runto.set_val(str(current_frame_index))
        return

    # Optional: Check against max_frames
    if requested_cycle >= max_frames:
         print(f"Target cycle {requested_cycle} exceeds max frames ({max_frames}). Clamping to {max_frames-1}.")
         requested_cycle = max_frames - 1 # Clamp to last valid frame
         textbox_runto.set_val(str(requested_cycle)) # Update textbox with clamped value


    print(f"Attempting to run to cycle: {requested_cycle}")

    # --- Stop current animation regardless (if running) ---
    if ani.event_source is not None and not animation_paused:
        print("Stopping current animation to process Run To Cycle.")
        ani.event_source.stop()
        # We will set animation_paused later based on whether we restart or run

    # --- Restart Logic ---
    needs_restart = False
    current_completed = total_completed_tasks.get(current_frame_index, 0)
    # Restart if target is in the past OR simulation is already done
    if requested_cycle <= current_frame_index or current_completed >= total_tasks:
        print("Target cycle is in the past or simulation complete. Restarting...")
        # Stop timer explicitly before reset (belt-and-suspenders)
        if ani.event_source is not None: ani.event_source.stop()
        reset_simulation() # Resets index to 0, visuals, sets animation_paused=False, target=None
        needs_restart = True # Flag that we performed a reset
        # Draw the reset state immediately
        fig.canvas.draw_idle()
        try: fig.canvas.flush_events()
        except AttributeError: pass

    # --- Set Target ---
    target_cycle = requested_cycle
    print(f"Target set to cycle {target_cycle}.")

    # --- Start/Resume Animation if needed ---
    # We need to run if the target is ahead of the current frame index AND simulation not complete/maxed
    current_completed_after_reset = total_completed_tasks.get(current_frame_index, 0) # Re-check after potential reset
    should_run = (current_frame_index < target_cycle) and \
                 (current_completed_after_reset < total_tasks) and \
                 (current_frame_index < max_frames)

    if should_run:
        if ani.event_source is not None:
            print("Starting animation to reach target.")
            title_obj.set_text(f'Cycle {current_frame_index}') # Update title before starting
            ani.event_source.start()
            animation_paused = False # Correct the state variable
        else:
            print("Error: Animation event source not found. Cannot run to cycle.")
            target_cycle = None # Clear target if we can't start
            animation_paused = True # Stay paused if we can't run
            title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
            fig.canvas.draw_idle()
    else:
        # We are already at or past the target, or simulation finished/maxed before starting
        # This handles the case where reset happens and target is 0, or target > current but sim is done.
        print(f"Cannot run or already at/past target cycle {current_frame_index}. Ensuring paused state.")
        # Make sure animation is stopped
        if ani.event_source is not None:
             ani.event_source.stop() # Safe to call even if already stopped
        animation_paused = True # Ensure state is paused

        # If we reached the target *immediately* (e.g., target 0 after reset), clear target
        if current_frame_index == target_cycle:
            target_cycle = None

        # Update title to reflect paused state (or final state if complete/maxed)
        final_title = f'Cycle {current_frame_index}'
        if current_completed_after_reset >= total_tasks: final_title += " (Complete)"
        elif current_frame_index >= max_frames: final_title += " (Max Frames)"
        elif target_cycle is None : final_title += " (Paused)" # Paused because target was met or invalid

        title_obj.set_text(final_title)
        fig.canvas.draw_idle() # Redraw to show the correct state


# --- Connect Widgets to Callbacks ---
btn_restart.on_clicked(restart_animation_callback)
btn_pause.on_clicked(pause_animation)
btn_play.on_clicked(play_animation)
slider_speed.on_changed(update_speed)
btn_runto.on_clicked(run_to_cycle_callback)
# Optional: Connect TextBox submit (Enter key) to the same callback
# textbox_runto.on_submit(run_to_cycle_callback) # Uncomment if desired


# --- Cursor Hover Logic for Widgets ---
widget_axes = [ax_restart, ax_pause, ax_play, ax_slider, ax_textbox, ax_runto_btn]
is_cursor_over_widget = False

def on_hover(event):
    """Changes cursor when hovering over button or slider axes."""
    global is_cursor_over_widget
    currently_over = False
    if event.inaxes in widget_axes:
        currently_over = True

    if currently_over != is_cursor_over_widget: # Only change if state changes
        if currently_over:
            new_cursor = Cursors.HAND
        else:
            new_cursor = Cursors.POINTER
        try:
            # Check if canvas exists and is valid
            if fig.canvas and fig.canvas.manager and fig.canvas.manager.window:
                 fig.canvas.set_cursor(new_cursor)
        except Exception as e:
             # pass # Silently ignore cursor setting errors (can happen during close)
             if TO_PRINT: print(f"Minor error setting cursor: {e}")
        is_cursor_over_widget = currently_over

fig.canvas.mpl_connect('motion_notify_event', on_hover)

# --- Display ---
print("Initializing display...")
# Reset simulation *before* showing the plot to ensure initial state is correct
reset_simulation()

print("Showing plot...")
plt.show()
print("Plot window closed.")