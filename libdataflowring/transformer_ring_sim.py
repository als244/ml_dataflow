import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import RegularPolygon
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import Button, Slider
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
stall_node_opacity = 0.4
stall_node_border_width = 2.0
stall_node_fontsize = 5.5

# --- Transfer Parameters ---
edge_linewidth = 1.5
edge_label_fontsize = 7
label_offset_distance = 0.5


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
        return i_min
    speed_range = s_max - s_min
    interval_range = i_min - i_max
    if speed_range == 0:
        return i_min
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
COLOR_INBOUND_DEFAULT = 'darkgreen'
COLOR_INBOUND_WEIGHT = 'darkgreen'
COLOR_INBOUND_ACTIVATION = 'magenta' # For backward pass activations fetch

# Device -> Network (Visually Inward toward Center)
COLOR_OUTBOUND_DEFAULT = 'magenta' # For forward pass activation storage
COLOR_OUTBOUND_WGT_GRAD = 'lightgreen' # For weight gradient storage

# Ring Transfers (Device -> Device)
# CCW = Counter-Clockwise (i -> i+1), typically forward activations/head inputs
# CW = Clockwise (i -> i-1), typically backward gradients
COLOR_RING_CCW = 'indigo' # Was FWD
COLOR_RING_CW = 'orange'  # Was BWD

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
legend_text = (
    f"Simulated Configuration:\n\n"
    f"     - Num Layers: {total_layers}\n"
    f"     - Num Devices: {N}\n"
    f"     - Num Chunks: {total_chunks}\n"
    f"     - Chunks from Same Sequence: {are_chunks_same_seq}\n"
    f"     - Do Backward: {do_backward}\n"
    f"         - Store Activation Gradients in Layer Buffer: {save_grad_activation_in_layer_buffer}\n"
    f"     - Per-Device Layer Capacity: {layer_capacity}\n"
    f"     - Per-Device Layer Activations Capacity: {activations_capacity}\n"
    f"     - Per-Device Layer Transitions Capacity: {transitions_capacity}\n"
    f"     - Constants:\n"
    f"         - Layer Computation: {computationFrames} Cycles\n"
    f"         - Layer Transfer: {layerTransferFrames} Cycles\n"
    f"         - Activation Transfer: {savedActivationsFrames} Cycles\n"
    f"         - Block Transition: {activationTransitionFrames} Cycles\n"
)

at = AnchoredText(legend_text, loc='upper left', bbox_to_anchor=(1.01, 1.01),
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
edge_artists = {}
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
                          ha='center', va='center', fontsize=stall_node_fontsize,
                          color='black', zorder=3, visible=False) # Initially hidden
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
    # Note: We will swap the visual meaning later. arrow_out visualizes INBOUND, arrow_in visualizes OUTBOUND
    start_pos_arrow_out = inner_edge_conn_point + edge_offset # Base start for arrow pointing OUTWARD (Visually INBOUND)
    start_pos_arrow_in = outer_edge_conn_point - edge_offset # Base start for arrow pointing INWARD (Visually OUTBOUND)

    # Create arrow object that VISUALLY points OUTWARD (Inner -> Outer) - Represents INBOUND data state
    arrow_out = patches.FancyArrowPatch(posA=start_pos_arrow_out, posB=start_pos_arrow_out,
                                        arrowstyle=arrow_style_str, color=COLOR_INBOUND_DEFAULT, linestyle='dashed',
                                        mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_out)
    label_out = ax.text(start_pos_arrow_out[0], start_pos_arrow_out[1], "", color=COLOR_INBOUND_DEFAULT, fontsize=edge_label_fontsize, ha='center', va='bottom', zorder=4) # zorder=4 to be above nodes, va=bottom
    edge_artists[f'in_{i}'] = (arrow_out, label_out) # NOTE: SWAPPED KEY - this handles INBOUND state

    # Create arrow object that VISUALLY points INWARD (Outer -> Inner) - Represents OUTBOUND data state
    arrow_in = patches.FancyArrowPatch(posA=start_pos_arrow_in, posB=start_pos_arrow_in,
                                       arrowstyle=arrow_style_str, color=COLOR_OUTBOUND_DEFAULT, linestyle='dashed',
                                       mutation_scale=mut_scale, lw=edge_linewidth, zorder=1)
    ax.add_patch(arrow_in)
    label_in = ax.text(start_pos_arrow_in[0], start_pos_arrow_in[1], "", color=COLOR_OUTBOUND_DEFAULT, fontsize=edge_label_fontsize, ha='center', va='top', zorder=4) # zorder=4, va=top
    edge_artists[f'out_{i}'] = (arrow_in, label_in) # NOTE: SWAPPED KEY - this handles OUTBOUND state

    # Keep ring arrows (Device -> Device)
    start_pos_ring = outer_pos
    arrow_ring = patches.FancyArrowPatch(posA=start_pos_ring, posB=start_pos_ring,
                                         arrowstyle=arrow_style_str, color=COLOR_RING_CCW, linestyle='solid', # Default CCW color
                                         mutation_scale=mut_scale, lw=edge_linewidth, zorder=1,
                                         connectionstyle=f"arc3,rad=0.2") # Add curve
    ax.add_patch(arrow_ring)
    label_ring = ax.text(start_pos_ring[0], start_pos_ring[1], "", color=COLOR_RING_CCW, fontsize=edge_label_fontsize, ha='center', va='center', zorder=4) # zorder=4
    edge_artists[f'ring_{i}'] = (arrow_ring, label_ring)


# --- Device Class ---
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
        self.cur_model_outputs_replace_ind = 0 # Typo fixed previously

        self.computation_queue = []
        self.outbound_queue = []
        self.inbound_queue = []
        self.peer_transfer_queue = []

        self.is_computing = False
        self.is_stalled = False
        self.stall_start_time = 0
        self.cur_computation_start_time = 0
        self.cur_computation_duration = 0

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
        self.has_reversed = False # Tracks if pass direction has reversed, to know what to prefetch from home

        self.outbound_storage = set() # Stores (chunk_id, layer_id, is_grad) available for fetch

        # --- Fwd/Bwd Setup Logic ---
        # Initial weights
        for i in range (layer_capacity):
            layer_id_to_add = device_id + i * total_devices
            if layer_id_to_add < total_layers:
                self.cur_ready_weights[i] = layer_id_to_add
                self.outbound_storage.add((-1, layer_id_to_add, False)) # -1 chunk_id for weights

        # Forward pass setup
        cur_layer_id = device_id
        last_fwd_layer_on_device = -1
        while cur_layer_id < total_layers:
            last_fwd_layer_on_device = cur_layer_id
            # Ensure initial weights are in outbound storage if not already added
            if (-1, cur_layer_id, False) not in self.outbound_storage:
                self.outbound_storage.add((-1, cur_layer_id, False))
            # Add computation tasks for this layer
            for i in range (total_chunks):
                # Task: (chunk_id, layer_id, is_bwd_grad, is_bwd_wgt, peer_transfer_dir, duration)
                # peer_transfer_dir: +1 for CCW, -1 for CW, 0 for no ring transfer (W grad)
                self.computation_queue.append((i, cur_layer_id, False, False, 1, computationFrames))
            cur_layer_id += self.total_devices

        self.next_saved_activations_layer_id = last_fwd_layer_on_device # Last layer whose acts might be saved
        if are_chunks_same_seq:
              self.next_saved_activations_chunk_id = self.total_chunks - 1
        else:
              self.next_saved_activations_chunk_id = 0

        # Backward pass setup
        if do_backward:
            head_layer_conceptual_id = total_layers # For gradient input after loss
            # Check if this device handles the layer feeding into the conceptual head layer
            # This device is responsible for the "head gradient" computation if it holds the last layer
            if last_fwd_layer_on_device != -1 and last_fwd_layer_on_device + self.total_devices == total_layers:
                cur_layer_id = head_layer_conceptual_id
                # Add conceptual "head weight" to storage if needed (might represent loss state)
                self.outbound_storage.add((-1, cur_layer_id, False))
                if are_chunks_same_seq:
                    chunk_order = range(total_chunks - 1, -1, -1)
                else:
                      chunk_order = range(total_chunks)
                # Add "head computation" tasks (generates initial gradient)
                for i in chunk_order:
                    self.computation_queue.append((i, cur_layer_id, False, False, -1, computationFrames)) # Feeds grad CW
                # Start backward pass from the actual last layer on this device
                cur_layer_id = last_fwd_layer_on_device
            else:
                # If not handling the last layer, start BWD from the last fwd layer it did handle
                cur_layer_id = last_fwd_layer_on_device

            # Add actual backward pass computation tasks
            while cur_layer_id >= 0:
                if are_chunks_same_seq:
                    chunk_order = range(total_chunks - 1, -1, -1)
                else:
                      chunk_order = range(total_chunks)
                # Bwd Activation Gradient (X) computation tasks
                for i in chunk_order:
                    self.computation_queue.append((i, cur_layer_id, True, False, -1, computationFrames)) # is_bwd_grad=True, feeds grad CW
                # Bwd Weight Gradient (W) computation tasks
                for i in chunk_order:
                    self.computation_queue.append((i, cur_layer_id, False, True, 0, computationFrames)) # is_bwd_wgt=True, no ring transfer
                cur_layer_id -= self.total_devices
        # --- End Fwd/Bwd Setup ---

    def handle_completed_transfers(self, T, all_devices):
        """Processes completed transfers (Inbound, Outbound, Peer)"""
        # Check Inbound (Network -> Device)
        if self.is_inbound_transferring and (self.cur_inbound_start_time + self.cur_inbound_duration <= T):
            inbound_item = self.inbound_queue.pop(0)
            chunk_id, layer_id, is_grad, replace_ind, duration = inbound_item
            if chunk_id == -1: # It's a weight
                self.cur_ready_weights[replace_ind] = layer_id
                msg = f"Wgt L{layer_id}"
            else: # It's an activation
                self.cur_fetched_activations[replace_ind] = (chunk_id, layer_id)
                msg = f"Act C{chunk_id},L{layer_id}"

            if TO_PRINT:
                print(f"T={T}, Dev {self.device_id}: RX INBOUND {msg} -> Idx {replace_ind}.")

            self.is_inbound_transferring = False
            self.cur_inbound_edge = ""

        # Check Outbound (Device -> Network)
        if self.is_outbound_transferring and (self.cur_outbound_start_time + self.cur_outbound_duration <= T):
            outbound_item = self.outbound_queue.pop(0)
            chunk_id, layer_id, is_grad, duration = outbound_item
            # Add the completed item to persistent storage
            self.outbound_storage.add((chunk_id, layer_id, is_grad))

            if TO_PRINT:
                item_type = ""
                if chunk_id >= 0:
                    item_type = "Act"
                elif is_grad:
                    item_type = "Grad"
                else:
                    item_type = "Wgt"
                print(f"T={T}, Dev {self.device_id}: TX OUTBOUND {item_type} C{chunk_id},L{layer_id} COMPLETE.")

            self.is_outbound_transferring = False
            self.cur_outbound_edge = ""

        # Check Peer (Device -> Device)
        if self.is_peer_transferring and (self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T):
            peer_item = self.peer_transfer_queue.pop(0)
            peer_id, chunk_id, layer_id, is_grad, rep_ind, duration = peer_item
            peer_dev = all_devices[peer_id]

            # Determine if the transfer is going to the conceptual head input buffer
            is_to_head_input_buffer = (layer_id == self.total_layers - 1) and (not is_grad)

            if is_to_head_input_buffer:
                # Store in peer's model output buffer (acts as input to conceptual head)
                peer_dev.cur_model_outputs[rep_ind] = (chunk_id, layer_id, is_grad)
                loc = "Head Input"
            else:
                # Store in peer's transition buffer
                peer_dev.cur_ready_transitions[rep_ind] = (chunk_id, layer_id, is_grad)
                loc = "Trans"

            if TO_PRINT:
                  item_type = "Grad" if is_grad else "Out"
                  print(f"T={T}, Dev {self.device_id}: TX PEER {item_type} C{chunk_id},L{layer_id} -> Dev {peer_id} {loc} Idx {rep_ind}.")

            self.is_peer_transferring = False
            self.cur_ring_edge = ""

    def handle_computation_depends(self, T):
        # Checks if the next task in the queue has its dependencies met
        if not self.computation_queue: return False
        next_task = self.computation_queue[0]
        # Unpack task details for clarity
        cid, lid, bX, bW, tdir, dur = next_task
        has_deps = False
        is_fwd = (not bX) and (not bW) and (lid < self.total_layers)
        is_head = (lid == self.total_layers)

        self.stall_reason = ""
        # Check dependencies based on task type
        if is_fwd:
            has_weight = (lid in self.cur_ready_weights)
            if lid == 0: # First layer only needs weights
                if has_weight:
                    has_deps = True
                else:
                    self.stall_reason = f"Missing:\nWeights for L:{lid}"
            else: # Subsequent layers need weights and previous output
                has_input_transition = ((cid, lid - 1, False) in self.cur_ready_transitions)
                if has_weight and has_input_transition:
                    has_deps = True
                else:
                    if not has_weight and not has_input_transition:
                        self.stall_reason = f"Missing:\nWeights\nAct. Stream"
                    elif not has_weight:
                        self.stall_reason = f"Missing:\nWeights"
                    else:
                        self.stall_reason = f"Missing:\nAct. Stream"
        elif is_head:
            has_weight = (lid in self.cur_ready_weights) # Assumes head 'weights' use conceptual ID
            has_input_from_last_block = ((cid, lid - 1, False) in self.cur_model_outputs)
            if has_weight and has_input_from_last_block:
                has_deps = True
            else:
                if not has_weight and not has_input_from_last_block:
                    self.stall_reason = f"Missing:\nHead Weights\nAct. Stream"
                elif not has_weight:
                    self.stall_reason = f"Missing:\nHead Weights"
                    #print(f"HEAD WEIGHTS STALL! CURRENT READY WEIGHTS: {self.cur_ready_weights}")
                else:
                    self.stall_reason = f"Missing:\nAct. Stream"
        else: # Backward pass dependency checks
            if bX:
                has_weight = (lid in self.cur_ready_weights)
                has_upstream_grad = ((cid, lid + 1, True) in self.cur_ready_transitions)
                if has_weight and has_upstream_grad:
                    has_deps=True
                else:
                    if not has_weight and not has_upstream_grad:
                        self.stall_reason = f"Missing:\nWeights\nUpstream Grad"
                    elif not has_weight:
                        self.stall_reason = f"Missing:\nWeights"
                    else:
                        self.stall_reason = f"Missing:\nUpstream Grad"
            elif bW: # Needs activation grad + fwd activation
                has_fwd_activation = ((cid, lid) in self.cur_fetched_activations)
                has_activation_grad = ((cid, lid) in self.cur_ready_grad_activations)
                if has_fwd_activation and has_activation_grad:
                    has_deps=True
                else:
                    if not has_fwd_activation and not has_activation_grad:
                        self.stall_reason = f"Missing:\nFwd Activations\nGrad Activations"
                    elif not has_fwd_activation:
                        self.stall_reason = f"Missing:\nFwd Activations"
                    else:
                        self.stall_reason = f"Missing:\nGrad Activations"

        # Update device state based on dependency check
        if has_deps:
            # Dependencies met, schedule/start computation
            if not self.device_has_started:
                self.device_start_time = T
                self.device_has_started = True
            if self.is_stalled:
                if TO_PRINT:
                    stalled_duration = T - self.stall_start_time
                    task_type_str = 'X' if bX else ('W' if bW else ('H' if is_head else 'F'))
                    print(f"T={T}, Dev {self.device_id}: UNSTALL -> Comp C{cid},L{lid},{task_type_str}. Stalled for {stalled_duration}")
            # Update state to computing
            self.cur_computation_start_time = T
            self.cur_computation_duration = dur
            self.is_computing = True
            self.is_stalled = False
            task_type_str = 'Bwd X' if bX else ('Bwd W' if bW else ('Head' if is_head else 'Fwd'))
            self.computing_status =f"COMPUTING:\n{task_type_str}\nC{cid},L{lid}"
            self.stall_reason = ""
        else:
            # Dependencies not met, stall if not already stalled
            if not self.is_stalled:
                self.is_stalled = True
                self.stall_start_time = T
                task_type_str = 'Bwd X' if bX else ('Bwd W' if bW else ('Head' if is_head else 'Fwd'))
                self.computing_status = f"STALL:\n{task_type_str}\nC{cid},L{lid}"
                # Print stall message only once when stall begins (if enabled)
                if TO_PRINT:
                    print(f"T={T}, Dev {self.device_id}: STALL on Task {next_task}")
                    # Optional: More detailed print of missing dependencies
                    if is_fwd: print(f"     Need Wgt L{lid} (Have: {lid in self.cur_ready_weights}), Need Trans C{cid},L{lid-1} (Have: {(cid, lid - 1, False) in self.cur_ready_transitions})")
                    elif is_head: print(f"     Need Head Wgt L{lid} (Have: {lid in self.cur_ready_weights}), Need ModelOut C{cid},L{lid-1} (Have: {(cid, lid - 1, False) in self.cur_model_outputs})")
                    elif bX: print(f"     Need Wgt L{lid} (Have: {lid in self.cur_ready_weights}), Need UpGrad C{cid},L{lid+1} (Have: {(cid, lid + 1, True) in self.cur_ready_transitions})") # Corrected print
                    elif bW: print(f"     Need ActGrad C{cid},L{lid} (Have: {(cid, lid) in self.cur_ready_grad_activations}), Need FwdAct C{cid},L{lid} (Have: {(cid, lid) in self.cur_fetched_activations})")

        return has_deps


    def handle_bwd_fetch(self, T):
        """Handles prefetching for backward pass (activations and weights)."""
        # Decide whether to fetch activation or weight based on flag
        if self.to_fetch_activations_next:
            # --- Fetch Activation ---
            act_needed = (self.next_saved_activations_chunk_id, self.next_saved_activations_layer_id)

            # Check if there are more activations to fetch for this layer
            if act_needed[1] >= 0: # Layer ID is valid
                idx_to_replace = self.cur_fetched_activations_replace_ind
                current_val_at_idx = self.cur_fetched_activations[idx_to_replace]

                # Check if the needed activation is already present or being fetched
                if current_val_at_idx != act_needed and current_val_at_idx != -2: # -2 indicates fetch in progress
                    # Check if the needed activation exists in persistent storage
                    if (act_needed[0], act_needed[1], False) in self.outbound_storage:
                        # Queue the inbound fetch request
                        self.inbound_queue.append((act_needed[0], act_needed[1], False, idx_to_replace, savedActivationsFrames))
                        self.cur_fetched_activations[idx_to_replace] = -2 # Mark as fetch in progress

                        # Decrement replacement index (circular)
                        self.cur_fetched_activations_replace_ind = (idx_to_replace - 1 + self.total_chunks) % self.total_chunks

                        if TO_PRINT:
                            print(f"T={T}, Dev {self.device_id}: QUEUE BWD Fetch Act C{act_needed[0]},L{act_needed[1]} -> Idx {idx_to_replace}")

                        # Update next activation needed
                        if are_chunks_same_seq:
                            next_cid = act_needed[0] - 1
                            if next_cid < 0: # Move to previous layer
                                self.next_saved_activations_layer_id -= self.total_devices
                                self.next_saved_activations_chunk_id = self.total_chunks - 1 # Reset chunk ID
                                self.to_fetch_activations_next = False # Switch to fetching weight next
                            else: # Move to next chunk in sequence
                                self.next_saved_activations_chunk_id = next_cid
                        else: # Not same sequence (e.g., round-robin chunks)
                            next_cid = (act_needed[0] + 1) % self.total_chunks
                            if next_cid == 0: # Completed all chunks for this layer
                                self.next_saved_activations_layer_id -= self.total_devices
                                self.next_saved_activations_chunk_id = 0 # Reset chunk ID
                                self.to_fetch_activations_next = False # Switch to fetching weight next
                            else: # Move to next chunk
                                self.next_saved_activations_chunk_id = next_cid
                    # else: Activation not in storage, cannot fetch (will stall later)
                # else: Activation already present or being fetched, do nothing now
            else: # No more valid layers to fetch activations for
                self.to_fetch_activations_next = False # Should probably switch to weights or stop

        else: # --- Fetch Weight ---
            wgt_needed = self.next_weight_layer_id

            # Check if there are more weights to fetch
            if wgt_needed >= 0: # Layer ID is valid
                idx_to_replace = self.cur_weight_replace_ind
                current_val_at_idx = self.cur_ready_weights[idx_to_replace]

                # Check if the needed weight is already present or being fetched
                if current_val_at_idx != wgt_needed and current_val_at_idx != -2:
                     # Check if the needed weight exists in persistent storage
                    if (-1, wgt_needed, False) in self.outbound_storage:
                        # Queue the inbound fetch request
                        self.inbound_queue.append((-1, wgt_needed, False, idx_to_replace, layerTransferFrames))
                        self.cur_ready_weights[idx_to_replace] = -2 # Mark as fetch in progress

                        # Decrement replacement index (circular) - Assuming weights are fetched in reverse layer order for BWD
                        self.cur_weight_replace_ind = (idx_to_replace - 1 + self.layer_capacity) % self.layer_capacity

                        if TO_PRINT:
                            print(f"T={T}, Dev {self.device_id}: QUEUE BWD Fetch Wgt L{wgt_needed} -> Idx {idx_to_replace}")

                        # Update next weight needed (move to previous layer in sequence)
                        self.next_weight_layer_id -= self.total_devices
                        self.to_fetch_activations_next = True # Switch back to fetching activations next
                    # else: Weight not in storage, cannot fetch
                # else: Weight already present or being fetched
            else: # No more valid layers to fetch weights for
                self.to_fetch_activations_next = True # No more weights, maybe try activations again or stop? Logic might need refinement here.

    def handle_computation(self, T, all_devices):
        """Handles starting, checking completion, and processing output of computation tasks."""
        completed_tasks = 0

        # --- Try to start computation if idle and not stalled ---
        if not self.is_computing and not self.is_stalled and len(self.computation_queue) > 0:
            self.handle_computation_depends(T) # Check dependencies and potentially start

        # --- Check if ongoing computation is finished ---
        elif self.is_computing and (self.cur_computation_start_time + self.cur_computation_duration <= T):
            # Task finished, pop it from the queue
            task = self.computation_queue.pop(0)
            cid, lid, bX, bW, tdir, dur = task
            is_head = (lid == self.total_layers)
            is_fwd = (not bX) and (not bW) and (not is_head)

            if TO_PRINT:
                task_type_str = 'X' if bX else ('W' if bW else ('H' if is_head else 'F'))
                print(f"T={T}, Dev {self.device_id}: FINISHED Comp -> C{cid},L{lid},{task_type_str}")

            completed_tasks += 1

            # --- Process completed task output and queue transfers/prefetch ---
            if is_fwd: # --- Forward Pass Task Completion ---
                # Determine if this layer's activation should be saved locally or sent outbound
                is_last_fwd_for_bwd_save = (lid == self.next_saved_activations_layer_id)

                # Send activation outbound unless it's the last one AND we save locally
                if not is_last_fwd_for_bwd_save or not save_grad_activation_in_layer_buffer:
                    self.outbound_queue.append((cid, lid, False, savedActivationsFrames))
                    # if TO_PRINT: print(f"      -> Q Out Act {cid},{lid}")

                # Save activation locally if it's the last one AND configured to do so
                if is_last_fwd_for_bwd_save and save_grad_activation_in_layer_buffer:
                    idx = self.cur_fetched_activations_replace_ind
                    self.cur_fetched_activations[idx] = (cid, lid)
                    self.cur_fetched_activations_replace_ind = (idx - 1 + self.total_chunks) % self.total_chunks
                    # if TO_PRINT: print(f"      -> Store local Act {cid},{lid} -> Idx {idx}")

                # Queue peer transfer (CCW)
                target_peer_id = (self.device_id + tdir) % self.total_devices # tdir is +1 for Fwd
                is_output_to_head_buffer = (lid == self.total_layers - 1)

                if is_output_to_head_buffer: # Send to peer's head input buffer
                    idx = all_devices[target_peer_id].cur_model_outputs_replace_ind
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, False, idx, activationTransitionFrames))
                    all_devices[target_peer_id].cur_model_outputs_replace_ind = (idx + 1) % all_devices[target_peer_id].total_chunks # Advance peer's index
                    # if TO_PRINT: print(f"      -> Q Peer Out {cid},{lid} -> Dev{target_peer_id} HeadIn Idx {idx}")
                else: # Send to peer's transition buffer
                    idx = all_devices[target_peer_id].cur_transitions_replace_ind
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, False, idx, activationTransitionFrames))
                    all_devices[target_peer_id].cur_transitions_replace_ind = (idx + 1) % all_devices[target_peer_id].transitions_capacity # Advance peer's index
                    # if TO_PRINT: print(f"      -> Q Peer Out {cid},{lid} -> Dev{target_peer_id} Trans Idx {idx}")

            elif is_head: # --- Conceptual Head Task Completion ---
                # Generates initial gradient, send it via peer transfer (CW)
                target_peer_id = (self.device_id + tdir) % self.total_devices # tdir is -1 for Head
                idx = all_devices[target_peer_id].cur_transitions_replace_ind
                # Send gradient (is_grad=True) to peer's transition buffer
                self.peer_transfer_queue.append((target_peer_id, cid, lid, True, idx, activationTransitionFrames))
                all_devices[target_peer_id].cur_transitions_replace_ind = (idx + 1) % all_devices[target_peer_id].transitions_capacity # Advance peer's index
                # if TO_PRINT: print(f"      -> Q Peer Grad(H) {cid},{lid} -> Dev{target_peer_id} Trans Idx {idx}")

            else: # --- Backward Pass Task Completion ---
                if bX: # --- Bwd Activation Gradient Task ---
                    # Store the computed activation gradient locally
                    idx = self.cur_grad_activations_replace_ind
                    self.cur_ready_grad_activations[idx] = (cid, lid)
                    self.cur_grad_activations_replace_ind = (idx - 1 + self.grad_activations_capacity) % self.grad_activations_capacity
                    # if TO_PRINT: print(f"      -> Store ActGrad {cid},{lid} -> Idx {idx}")

                    # Send gradient via peer transfer (CW) to the next device in the sequence
                    target_peer_id = (self.device_id + tdir) % self.total_devices # tdir is -1 for bX
                    idx = all_devices[target_peer_id].cur_transitions_replace_ind
                    self.peer_transfer_queue.append((target_peer_id, cid, lid, True, idx, activationTransitionFrames))
                    all_devices[target_peer_id].cur_transitions_replace_ind = (idx + 1) % all_devices[target_peer_id].transitions_capacity # Advance peer's index
                    # if TO_PRINT: print(f"      -> Q Peer Grad(X) {cid},{lid} -> Dev{target_peer_id} Trans Idx {idx}")

                elif bW: # --- Bwd Weight Gradient Task ---
                    # Check if this is the last chunk for this weight gradient accumulation
                    is_last_chunk_for_wgrad = False
                    if are_chunks_same_seq and cid == 0:
                        is_last_chunk_for_wgrad = True
                    elif not are_chunks_same_seq and cid == self.total_chunks - 1:
                          is_last_chunk_for_wgrad = True

                    # If last chunk, queue outbound transfer for the accumulated weight gradient
                    if is_last_chunk_for_wgrad:
                        self.outbound_queue.append((-1, lid, True, layerTransferFrames)) # chunk_id=-1, is_grad=True
                        # if TO_PRINT: print(f"      -> Q Out WgtGrad L{lid}")
                    # else: Accumulation continues, do nothing specific here

            # --- Prefetching Logic ---
            if not self.has_reversed: # --- Forward Prefetching ---
                is_last_fwd_chunk = (cid == self.total_chunks - 1)
                # Trigger prefetch after completing the last chunk of a forward layer
                if is_fwd and is_last_fwd_chunk and self.next_weight_layer_id <= total_layers:
                    target_idx = self.cur_weight_replace_ind
                    needs_fetch = True
                    # Check if weight already present or fetch in progress
                    if self.cur_ready_weights[target_idx] == self.next_weight_layer_id or self.cur_ready_weights[target_idx] == -2:
                        needs_fetch = False

                    # Check if weight exists in storage
                    if needs_fetch and (-1, self.next_weight_layer_id, False) in self.outbound_storage:
                        self.inbound_queue.append((-1, self.next_weight_layer_id, False, target_idx, layerTransferFrames))
                        self.cur_ready_weights[target_idx] = -2 # Mark as fetching
                        # if TO_PRINT: print(f"      -> Q FWD Fetch Wgt L{self.next_weight_layer_id} -> Idx {target_idx}")

                    # Advance to next weight layer ID and replacement index
                    next_potential_layer_id = self.next_weight_layer_id + self.total_devices
                    self.cur_weight_replace_ind = (self.cur_weight_replace_ind + 1) % self.layer_capacity # Simple advance for FWD

                    # Check if time to reverse direction (after fetching last FWD weight)
                    if next_potential_layer_id >= total_layers and do_backward:
                        self.has_reversed = True
                        # Set up for first BWD weight fetch
                        first_bwd_w_layer = self.next_saved_activations_layer_id # Start BWD from layer where acts were saved
                        # Calculate the layer ID for the weight needed *before* the first BWD computation
                        # This depends on capacity and how far back we need to prefetch
                        self.next_weight_layer_id = first_bwd_w_layer - (self.layer_capacity - 1) * self.total_devices
                        # Reset BWD weight replacement index (start replacing from index before current target_idx)
                        self.cur_weight_replace_ind = (target_idx -1 + self.layer_capacity) % self.layer_capacity # Adjust for circular buffer logic in BWD fetch

                        # if TO_PRINT: print(f"      -> REVERSED direction. Next BWD Wgt Target: L{self.next_weight_layer_id}, Start Replace Idx: {self.cur_weight_replace_ind}")
                    else:
                          self.next_weight_layer_id = next_potential_layer_id # Continue FWD fetch sequence

            elif do_backward: # --- Backward Prefetching Trigger ---
                 # Trigger BWD prefetch after completing the last chunk of a Head task OR a BWD Weight task
                 is_last_chunk_head = is_head and ( (are_chunks_same_seq and cid == 0) or (not are_chunks_same_seq and cid == self.total_chunks - 1) )
                 is_last_chunk_bwd_w = bW and ( (are_chunks_same_seq and cid == 0) or (not are_chunks_same_seq and cid == self.total_chunks - 1) )

                 if is_last_chunk_head or is_last_chunk_bwd_w:
                       self.handle_bwd_fetch(T) # Call dedicated BWD fetch handler

            # --- Reset compute state and check next task ---
            self.is_computing = False
            self.computing_status = "Idle"
            self.stall_reason = "" # Clear stall reason on compute completion

            if len(self.computation_queue) > 0:
                  self.handle_computation_depends(T) # Check dependencies for the next task
            else: # No tasks left
                if not self.device_has_finished:
                    self.device_finish_time = T
                    self.device_has_finished = True
                    self.computing_status = "Finished"
                    # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: Device FINISHED ALL TASKS.")

        # --- Check dependencies again if stalled ---
        # (In case dependencies were met by transfers completed this cycle)
        elif self.is_stalled:
            self.handle_computation_depends(T)

        return completed_tasks

    def handle_new_transfers(self, T):
        """Initiates new transfers if channels are free and queues are not empty."""
        # --- Start Inbound Transfer (Network -> Device) ---
        if not self.is_inbound_transferring and len(self.inbound_queue) > 0:
            # Peek at the first item
            item = self.inbound_queue[0]
            cid, lid, isg = item[0], item[1], item[2]
            duration = item[-1] # Duration is last element

            # Check if the requested item is actually available in storage
            # (It should be, as fetch was queued based on this, but double-check)
            if (cid, lid, isg) in self.outbound_storage:
                self.is_inbound_transferring = True
                self.cur_inbound_start_time = T
                self.cur_inbound_duration = duration
                # Set edge label text
                if cid == -1: self.cur_inbound_edge = f"L:{lid}" # Weight
                else: self.cur_inbound_edge = f"Act:C{cid},L{lid}" # Activation

                # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: START INBOUND {'Wgt' if cid==-1 else 'Act'} C{cid},L{lid}")
            # else: Item disappeared from storage? Log error or handle? For now, transfer won't start.

        # --- Start Outbound Transfer (Device -> Network) ---
        if not self.is_outbound_transferring and len(self.outbound_queue) > 0:
            item = self.outbound_queue[0]
            cid, lid, isg, duration = item

            self.is_outbound_transferring = True
            self.cur_outbound_start_time = T
            self.cur_outbound_duration = duration

            # Set edge label text based on content
            if cid >= 0: # Activation
                 self.cur_outbound_edge = f"Act:C{cid},L{lid}"
            else: # Weight or Weight Gradient
                if isg: self.cur_outbound_edge = f"Grad:L{lid}" # Weight Gradient
                else: self.cur_outbound_edge = f"Wgt:L{lid}" # Weight (unlikely outbound?)

            # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: START OUTBOUND {self.cur_outbound_edge}")

        # --- Start Peer Transfer (Device -> Device) ---
        if not self.is_peer_transferring and len(self.peer_transfer_queue) > 0:
            item = self.peer_transfer_queue[0]
            pid, cid, lid, isg, ridx, duration = item

            self.is_peer_transferring = True
            self.cur_peer_transfer_start_time = T
            self.cur_peer_transfer_duration = duration

            # Set edge label text
            if isg: self.cur_ring_edge = f"Grad:C{cid},L{lid}" # Gradient
            else: self.cur_ring_edge = f"Out:C{cid},L{lid}" # Activation/Output

            # if TO_PRINT: print(f"T={T}, Dev {self.device_id}: START PEER {self.cur_ring_edge} -> Dev {pid}")


# --- Global Simulation State ---
all_devices = {}
total_tasks = 0
total_completed_tasks = {} # Dict to store completed tasks per frame
total_computation_time = 0
current_frame_index = 0 # THIS IS THE MASTER TIME COUNTER
animation_paused = False  # Start playing automatically
completion_text_artist = None

# --- Simulation Reset Function ---
def reset_simulation():
    """Resets the simulation state and visual elements."""
    global all_devices, total_tasks, total_completed_tasks, total_computation_time
    global current_frame_index, animation_paused, completion_text_artist
    # Access necessary global geometry info calculated during initial setup
    global unit_directions, inner_node_centers, outer_circle_positions, inner_node_radius, outer_node_radius, arrow_offset_dist

    if TO_PRINT:
        print("--- Resetting Simulation ---")

    # Recreate devices
    all_devices = {i: Device(i, layer_capacity, activations_capacity, transitions_capacity, total_devices, total_layers, total_chunks) for i in range(N)}

    # Recalculate totals
    total_tasks = sum(len(d.computation_queue) for d in all_devices.values())
    total_computation_time = sum(task[-1] for i in range(N) for task in all_devices[i].computation_queue)

    # Reset state variables
    total_completed_tasks = {-1: 0} # Initialize with base case for frame -1
    current_frame_index = 0 # RESET THE MASTER TIME COUNTER
    animation_paused = False # Reset to initial desired state (False = play)

    # Remove completion text if it exists
    if completion_text_artist is not None:
        if completion_text_artist.axes is not None:
            completion_text_artist.remove()
        completion_text_artist = None

    # Reset visual elements for each device/edge
    for i in range(N):
        # --- Recalculate static start positions for arrows ---
        unit_dir = unit_directions[i]
        inner_center = inner_node_centers[i]
        outer_pos = outer_circle_positions[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
        edge_offset = radial_perp_vector * arrow_offset_dist

        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
        start_pos_arrow_out = inner_edge_conn_point + edge_offset # Static start for vis-outward (Inbound)

        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
        start_pos_arrow_in = outer_edge_conn_point - edge_offset # Static start for vis-inward (Outbound)

        start_pos_ring = outer_pos # Static start for ring

        # --- Reset edge artists ---
        # Inbound Edge (Visually Outward from center)
        arrow_in_vis, label_in_vis = edge_artists[f'in_{i}']
        label_in_vis.set_text("")
        arrow_in_vis.set_color(COLOR_INBOUND_DEFAULT)
        arrow_in_vis.set_positions(start_pos_arrow_out, start_pos_arrow_out) # Zero length at static start

        # Outbound Edge (Visually Inward toward center)
        arrow_out_vis, label_out_vis = edge_artists[f'out_{i}']
        label_out_vis.set_text("")
        arrow_out_vis.set_color(COLOR_OUTBOUND_DEFAULT)
        arrow_out_vis.set_positions(start_pos_arrow_in, start_pos_arrow_in) # Zero length at static start

        # Ring Edge
        arrow_ring, label_ring = edge_artists[f'ring_{i}']
        label_ring.set_text("")
        arrow_ring.set_color(COLOR_RING_CCW) # Default to CCW color
        arrow_ring.set_positions(start_pos_ring, start_pos_ring) # Zero length at static start
        arrow_ring.set_connectionstyle(f"arc3,rad=0.2") # Reset curve

        # Reset stall node visuals
        if f'stall_node_{i}' in device_artists:
            device_artists[f'stall_node_{i}'].set_visible(False)
        if f'stall_label_{i}' in device_label_artists:
            device_label_artists[f'stall_label_{i}'].set_text("")
            device_label_artists[f'stall_label_{i}'].set_visible(False)

        # Reset main node labels
        if f'circle_{i}' in device_label_artists:
            device_label_artists[f'circle_{i}'].set_text(f'D{i}\nIdle')
        if f'inner_label_{i}' in device_label_artists:
            device_label_artists[f'inner_label_{i}'].set_text(f'D{i}\nHome') # Simplified label

    # Reset title using the now zero current_frame_index
    title_obj.set_text(f'Cycle {current_frame_index}')

    if TO_PRINT:
        print(f"Reset complete. Total tasks: {total_tasks}")

# --- Update Function (MODIFIED) ---
def update(frame): # Keep frame argument, but we'll ignore its value for T
    """Main animation update function."""
    global total_completed_tasks, current_frame_index, completion_text_artist, animation_paused

    # If paused, don't update simulation state, just return existing artists
    # This prevents state changes while visually paused.
    # Important: Do NOT increment current_frame_index if paused.
    if animation_paused:
        all_artists = [title_obj]
        for i in range(N):
            all_artists.extend([
                device_label_artists[f'circle_{i}'],
                device_label_artists[f'inner_label_{i}'],
                device_artists[f'stall_node_{i}'],
                device_label_artists[f'stall_label_{i}'],
                edge_artists[f'in_{i}'][0], edge_artists[f'in_{i}'][1],
                edge_artists[f'out_{i}'][0], edge_artists[f'out_{i}'][1],
                edge_artists[f'ring_{i}'][0], edge_artists[f'ring_{i}'][1],
            ])
        if completion_text_artist:
            all_artists.append(completion_text_artist)
        # Let pause_animation callback handle adding "(Paused)" text
        return all_artists # RETURN EARLY

    # --- Use Global Frame Index ---
    # Use the global state variable as the current simulation time T
    T = current_frame_index

    # Update title FIRST using the correct T for this cycle
    title_obj.set_text(f'Cycle {T}')
    artists_to_update = [title_obj] # Initialize the list of artists to return

    # Ensure completion count exists for this frame T
    if T not in total_completed_tasks:
        last_known_frame = max(k for k in total_completed_tasks if k < T) if any(k < T for k in total_completed_tasks) else -1
        total_completed_tasks[T] = total_completed_tasks.get(last_known_frame, 0)

    # --- Run Simulation Step (using T) ---
    for i in range(N):
        all_devices[i].handle_completed_transfers(T, all_devices)

    newly_completed = sum(all_devices[i].handle_computation(T, all_devices) for i in range(N))
    # Update total completed based on previous frame's count + newly completed THIS frame
    total_completed_tasks[T] = total_completed_tasks.get(T-1, 0) + newly_completed

    for i in range(N):
            all_devices[i].handle_new_transfers(T)

    # --- Update Visuals (using T and device states) ---
    for i in range(N):
        unit_dir = unit_directions[i]
        inner_center = inner_node_centers[i]
        outer_pos = outer_circle_positions[i]
        transfer_dist_i = node_transfer_distances[i]
        radial_perp_vector = np.array([-unit_dir[1], unit_dir[0]])
        edge_offset = radial_perp_vector * arrow_offset_dist

        device = all_devices[i]

        # --- Update Labels and Stall Node ---
        outer_label_artist = device_label_artists[f'circle_{i}']
        status_text = device.computing_status
        outer_label_artist.set_text(f'D{i}\n{status_text}')

        inner_label_artist = device_label_artists[f'inner_label_{i}']
        inner_label_artist.set_text(f"D{i}\nHome") # Simplified

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

        # --- Update Edges ---
        arrow_vis_inbound, label_vis_inbound = edge_artists[f'in_{i}']
        arrow_vis_outbound, label_vis_outbound = edge_artists[f'out_{i}']
        arrow_ring, label_ring = edge_artists[f'ring_{i}']

        # Inbound state
        len_in_prog = 0.0; cur_inbound_edge_text = ""; color_inbound = COLOR_INBOUND_DEFAULT
        if device.is_inbound_transferring and device.cur_inbound_duration > 0:
            prog_frac = min(1.0, (T - device.cur_inbound_start_time) / device.cur_inbound_duration)
            len_in_prog = prog_frac * transfer_dist_i
            cur_inbound_edge_text = device.cur_inbound_edge
            is_act = "Act:" in cur_inbound_edge_text; is_wgt = "L:" in cur_inbound_edge_text
            if is_act and device.has_reversed: color_inbound = COLOR_INBOUND_ACTIVATION
            elif is_wgt: color_inbound = COLOR_INBOUND_WEIGHT
        arrow_vis_inbound.set_color(color_inbound)
        label_vis_inbound.set_color(color_inbound)

        # Outbound state
        len_out_prog = 0.0; cur_outbound_edge_text = ""; color_outbound = COLOR_OUTBOUND_DEFAULT
        if device.is_outbound_transferring and device.cur_outbound_duration > 0:
            prog_frac = min(1.0, (T - device.cur_outbound_start_time) / device.cur_outbound_duration)
            len_out_prog = prog_frac * transfer_dist_i
            cur_outbound_edge_text = device.cur_outbound_edge
            is_wgt_grad = "Grad:L" in cur_outbound_edge_text
            if is_wgt_grad: color_outbound = COLOR_OUTBOUND_WGT_GRAD
        arrow_vis_outbound.set_color(color_outbound)
        label_vis_outbound.set_color(color_outbound)

        # Peer transfer state
        len_ring_prog = 0.0; peer_device_id = -1; cur_ring_edge_text = ""; color_ring = COLOR_RING_CCW; connection_style_ring = f"arc3,rad=0.2"
        if device.is_peer_transferring and device.cur_peer_transfer_duration > 0:
            peer_item = device.peer_transfer_queue[0]; peer_device_id = peer_item[0]
            prog_frac = min(1.0, (T - device.cur_peer_transfer_start_time) / device.cur_peer_transfer_duration)
            len_ring_prog = prog_frac
            cur_ring_edge_text = device.cur_ring_edge
            target_is_cw = (peer_device_id == (i - 1 + N) % N); target_is_ccw = (peer_device_id == (i + 1) % N)
            if target_is_cw: color_ring = COLOR_RING_CW; connection_style_ring = f"arc3,rad=-0.2"
            else: color_ring = COLOR_RING_CCW; connection_style_ring = f"arc3,rad=0.2"
        arrow_ring.set_color(color_ring)
        label_ring.set_color(color_ring)
        arrow_ring.set_connectionstyle(connection_style_ring)

        # Calculate Edge Positions
        inner_edge_conn_point = inner_center + unit_dir * inner_node_radius
        start_vis_inbound = inner_edge_conn_point + edge_offset
        end_vis_inbound = start_vis_inbound + unit_dir * len_in_prog

        outer_edge_conn_point = outer_pos - unit_dir * outer_node_radius
        start_vis_outbound = outer_edge_conn_point - edge_offset
        end_vis_outbound = start_vis_outbound - unit_dir * len_out_prog

        start_pos_ring_geo = outer_pos
        if peer_device_id != -1:
            target_pos_ring_geo = outer_circle_positions[peer_device_id]
            vec = target_pos_ring_geo - start_pos_ring_geo; norm = np.linalg.norm(vec)
            if norm > 1e-6:
                target_pos_ring_geo = start_pos_ring_geo + vec * (1 - outer_node_radius / norm)
                start_offset_dir = vec / norm
                start_pos_ring_geo = outer_pos + start_offset_dir * outer_node_radius # Offset start point
            current_end_point_ring = start_pos_ring_geo + (target_pos_ring_geo - start_pos_ring_geo) * len_ring_prog
        else:
            start_pos_ring_geo = outer_pos + unit_dir * outer_node_radius # Point outwards if no target
            current_end_point_ring = start_pos_ring_geo # Zero length

        # Calculate Label Positions
        label_perp_offset = radial_perp_vector * label_offset_distance
        midpoint_vis_in = (start_vis_inbound + end_vis_inbound) / 2; label_pos_vis_in = midpoint_vis_in + label_perp_offset * 2
        midpoint_vis_out = (start_vis_outbound + end_vis_outbound) / 2; label_pos_vis_out = midpoint_vis_out - label_perp_offset * 2
        midpoint_ring = (start_pos_ring_geo + current_end_point_ring) / 2; label_pos_ring = midpoint_ring
        if len_ring_prog > 1e-6:
            edge_vec = current_end_point_ring - start_pos_ring_geo; norm = np.linalg.norm(edge_vec)
            if norm > 1e-6:
                perp_vec = np.array([-edge_vec[1], edge_vec[0]]) / norm
                offset_factor = 1.0 if connection_style_ring.endswith("0.2") else -1.0
                label_pos_ring = midpoint_ring + perp_vec * label_offset_distance * 3 * offset_factor

        # Update Artists
        arrow_vis_inbound.set_positions(start_vis_inbound, end_vis_inbound)
        label_vis_inbound.set_position(label_pos_vis_in)
        label_vis_inbound.set_text(cur_inbound_edge_text)

        arrow_vis_outbound.set_positions(start_vis_outbound, end_vis_outbound)
        label_vis_outbound.set_position(label_pos_vis_out)
        label_vis_outbound.set_text(cur_outbound_edge_text)

        arrow_ring.set_positions(start_pos_ring_geo, current_end_point_ring)
        label_ring.set_position(label_pos_ring)
        label_ring.set_text(cur_ring_edge_text)

        # Add updated artists to the list
        artists_to_update.extend([outer_label_artist, inner_label_artist, stall_node_artist, stall_label_artist,
                                  arrow_vis_inbound, label_vis_inbound, arrow_vis_outbound, label_vis_outbound,
                                  arrow_ring, label_ring])

    # --- Check for Completion (using T) ---
    current_total_completed = total_completed_tasks.get(T, 0)
    is_newly_complete = (current_total_completed >= total_tasks) and (total_completed_tasks.get(T-1, 0) < total_tasks)
    should_stop_animation = is_newly_complete

    if current_total_completed >= total_tasks and completion_text_artist is None:
        if TO_PRINT:
            print(f"T={T}: Completed all {current_total_completed}/{total_tasks} tasks!")
        # Calculate stats
        start_bubble = sum(d.device_start_time for d in all_devices.values() if d.device_has_started)
        stop_bubble = sum(max(0, T - d.device_finish_time) for d in all_devices.values() if d.device_has_finished)
        total_dev_time = T * N if T > 0 else 0
        steady_time = total_dev_time - stop_bubble - start_bubble if total_dev_time > 0 else 0
        overall_eff = (total_computation_time / total_dev_time * 100) if total_dev_time > 0 else 0.0
        steady_eff = (total_computation_time / steady_time * 100) if steady_time > 0 else 0.0

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
                                         bbox=dict(boxstyle='round,pad=0.5', fc=(0.9, 0.9, 1, 0.95), ec='black'),
                                         zorder=10)
        if TO_PRINT:
            print(completion_text)
        artists_to_update.append(completion_text_artist)


    # Stop the animation timer AFTER processing frame T, BEFORE incrementing index
    if should_stop_animation and ani is not None and ani.event_source is not None:
        if not animation_paused: # Only stop if it was actually running
            ani.event_source.stop()
            print("Animation Complete - Paused")
        animation_paused = True # Ensure state is paused
        # Update title one last time for the final completed frame T
        title_obj.set_text(f'Cycle {T}')
        # No need to append title_obj again, it was done at the start


    # --- Increment Global Frame Index for the *next* call ---
    # Only increment if the animation isn't paused (it might have just become paused above)
    if not animation_paused:
        current_frame_index += 1
        # Optional: Stop if max_frames reached to prevent infinite run if completion condition fails
        if current_frame_index >= max_frames:
            print(f"Max frames ({max_frames}) reached, stopping animation.")
            ani.event_source.stop()
            animation_paused = True
            title_obj.set_text(f'Cycle {T} (Max Frames)')


    return artists_to_update


# --- Create Animation ---
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=initial_frame_interval,
                              blit=False, # Keep False for compatibility
                              repeat=False, save_count=max_frames)


# --- Widgets ---
fig.subplots_adjust(bottom=0.2, right=0.85)
ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
ax_restart = fig.add_axes([0.15, 0.05, 0.2, 0.04])
ax_pause = fig.add_axes([0.40, 0.05, 0.2, 0.04])
ax_play = fig.add_axes([0.65, 0.05, 0.2, 0.04])
btn_restart = Button(ax_restart, 'Restart')
btn_pause = Button(ax_pause, 'Pause')
btn_play = Button(ax_play, 'Play')
slider_speed = Slider(ax=ax_slider, label='Speed Level', valmin=min_speed_level, valmax=max_speed_level, valinit=initial_speed_level, valstep=1)

# --- Define Widget Callback Functions ---
def pause_animation(event):
    """Callback function for the Pause button."""
    global animation_paused
    if not animation_paused: # Only act if playing
        if ani.event_source is not None:
            ani.event_source.stop()
        animation_paused = True
        title_obj.set_text(f'Cycle {current_frame_index} (Paused)') # Use master index
        fig.canvas.draw_idle()
        print("Animation Paused")
    else:
        print("Animation already paused.")

def play_animation(event):
    """Callback function for the Play button."""
    global animation_paused
    # Check completion using the MASTER frame index
    current_completed = total_completed_tasks.get(current_frame_index, 0)
    if animation_paused and current_completed < total_tasks and current_frame_index < max_frames: # Check completion and max frames
        if ani.event_source is not None:
            title_obj.set_text(f'Cycle {current_frame_index}') # Update title before resuming
            ani.event_source.start()
            animation_paused = False
            print("Animation Resumed")
        else:
             print("Error: Animation event source not found.")
    elif current_completed >= total_tasks:
        print("Animation already complete.")
    elif current_frame_index >= max_frames:
        print("Animation stopped at max frames.")
    elif not animation_paused:
        print("Animation already playing.")

def update_speed(val):
    """Callback function for the Speed slider."""
    global animation_paused
    speed_level = slider_speed.val
    new_interval = calculate_interval(speed_level, min_speed_level, max_speed_level, min_interval, max_interval)
    was_playing = not animation_paused

    if ani.event_source is not None:
        ani.event_source.stop()
        ani.event_source.interval = new_interval
        ani._interval = new_interval
    else:
        print("Error: Could not access animation timer to update interval.")
        return # Exit if timer cannot be accessed

    # Check completion using the MASTER frame index
    current_completed = total_completed_tasks.get(current_frame_index, 0)

    # Only restart if it was playing before and not complete/maxed
    if was_playing and current_completed < total_tasks and current_frame_index < max_frames:
        ani.event_source.start()
        animation_paused = False # Ensure state is correct
    else:
        animation_paused = True # Stay paused if it wasn't playing or is complete/maxed
        # Update title to show paused if necessary and not complete/maxed
        if current_completed < total_tasks and current_frame_index < max_frames:
            title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
            fig.canvas.draw_idle()

    print(f"Speed Level: {int(round(speed_level))}, Interval set to: {new_interval} ms")

# --- Restart Callback (MODIFIED) ---
def restart_animation_callback(event):
    """Callback function for the Restart button."""
    global animation_paused
    print("Restart button clicked.")
    if ani.event_source is not None:
        ani.event_source.stop() # Stop current animation timer

    reset_simulation()        # Reset simulation variables and artists (this now sets animation_paused based on initial setting)

    # --- Draw the reset state immediately ---
    # This ensures "Cycle 0" is shown even if the animation starts paused.
    fig.canvas.draw_idle()
    try: # Flush events if possible to make redraw more immediate
        fig.canvas.flush_events()
        # print("Restart CB: Flushed events") # Optional debug
    except AttributeError:
        pass # Ignore if flush_events not available

    # If the desired initial state is *playing*, start the timer after reset
    if not animation_paused:
        if ani.event_source is not None:
            # Title is already set by reset_simulation and will be updated correctly
            # by the first call to update() which will now use T=0
            ani.event_source.start()
            print("Simulation reset and playing from Cycle 0.")
        else:
            print("Error: Cannot restart animation timer after reset.")
            animation_paused = True # Fallback to paused state
            # Title reflects the reset state (Cycle 0), add (Paused)
            title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
            fig.canvas.draw_idle() # Redraw with paused state
    else:
        # If the desired initial state is *paused*, we already drew frame 0 via draw_idle()
        # Title is already set correctly by reset_simulation
        print("Simulation reset and paused at Cycle 0.")
        # Optionally ensure "(Paused)" is added if needed, though pause_animation handles this better
        # title_obj.set_text(f'Cycle {current_frame_index} (Paused)')
        # fig.canvas.draw_idle()


# --- Connect Widgets to Callbacks ---
btn_restart.on_clicked(restart_animation_callback)
btn_pause.on_clicked(pause_animation)
btn_play.on_clicked(play_animation)
slider_speed.on_changed(update_speed)

# --- Cursor Hover Logic for Widgets ---
widget_axes = [ax_restart, ax_pause, ax_play, ax_slider] # Include slider axis
is_cursor_over_widget = False

def on_hover(event):
    """Changes cursor when hovering over button or slider axes."""
    global is_cursor_over_widget

    # Check if the event happened over one of the widget axes
    currently_over = event.inaxes in widget_axes

    if currently_over and not is_cursor_over_widget:
        # Mouse entered a widget axis: Change to hand cursor
        try: # Use try-except as set_cursor might fail in some backend/state combos
            fig.canvas.set_cursor(Cursors.HAND)
        except Exception as e:
            # print(f"Debug: Could not set cursor to HAND - {e}") # Optional debug
            pass # Ignore if setting cursor fails
        is_cursor_over_widget = True
    elif not currently_over and is_cursor_over_widget:
        # Mouse left a widget axis: Change back to default pointer
        try:
            fig.canvas.set_cursor(Cursors.POINTER)
        except Exception as e:
            # print(f"Debug: Could not set cursor to POINTER - {e}") # Optional debug
            pass
        is_cursor_over_widget = False

# Connect the motion_notify_event to the on_hover function
fig.canvas.mpl_connect('motion_notify_event', on_hover)

# --- Display ---
# Call reset_simulation first to set the initial state correctly
# This will set current_frame_index=0 and update the title initially.
# It also sets animation_paused=False by default.
print("Initializing display...")
reset_simulation()

# FuncAnimation starts the timer automatically if repeat=False (unless interval=None)
# If animation_paused was True after reset, the first update call would hit the early return.
# If animation_paused was False (default), it starts running and the first call to update will use T=0.

print("Showing plot...")
plt.show()
print("Plot window closed.")