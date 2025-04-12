# Code implementing the new outward edge logic

import numpy as np # Needed for ceil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from matplotlib.offsetbox import AnchoredText

# --- Parameters ---
N = 8
computationFrames = 36
layerTransferFrames = N * computationFrames
savedActivationsFrames = computationFrames
activationTransitionFrames = 2

layer_capacity = 2
total_devices = N
total_layers = 32
total_chunks = 16



max_frames = 4800 # Increased duration

# Arrow offset distance, head size, line width, fonts, label offset (same as before)

centralRadius = 0.2
transferDistance = 0.8
device_opacity = 0.8
total_distance = centralRadius + transferDistance
arrow_offset_dist = centralRadius * 0.15
head_len = 1
head_wid = 0.5
mut_scale = 6
edge_linewidth = 2.0
title_fontsize = 24
edge_label_fontsize = 9
label_offset_distance = 0.1
frame_interval = 80

# --- Setup ---
# ... (Identical Setup: fig, ax, geometry, title, devices, initial edge artists) ...
fig, ax = plt.subplots(figsize=(8, 8)); ax.set_aspect('equal')
lim = total_distance + 0.2; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axis('off')
center_pos = np.array([0, 0])


legend_text = (
    f"Simulated Configuration:\n\n"
    f"    - Num Layers: {total_layers}\n"
    f"    - Num Devices: {N}\n"
    f"    - Num Chunks: {total_chunks}\n"
    f"    - Per-Device Layer Capacity: {layer_capacity}\n\n"
    f"    Constants:\n"
    f"        - Layer Computation: {computationFrames} Cycles\n"
    f"        - Layer Transfer: {layerTransferFrames} Cycles\n"
    f"        - Activation Transfer: {savedActivationsFrames} Cycles\n"
    f"        - Block Transition: {activationTransitionFrames} Cycles\n"
)

# 2. Create the AnchoredText object
# loc='upper right' places it in the top-right corner
# pad=0.4, borderpad=0.5 control padding around text and box edge
# frameon=True draws a box around it
# prop sets font properties (optional)
at = AnchoredText(legend_text,
                  # loc='upper right', # Keep this or maybe change to 'upper left'
                  loc='upper left', # Anchor the box's top-left corner...
                  bbox_to_anchor=(1.01, 1.01), # ...slightly outside the axes top-right (x=1.01, y=1.01)
                  prop=dict(size=11), frameon=True,
                  pad=0.4, borderpad=0.5,
                  bbox_transform=ax.transAxes # Ensure coords are relative to axes
                  )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2") # Optional rounded box
at.patch.set_facecolor((1, 1, 1, 0.8)) # Semi-transparent white background
at.patch.set_edgecolor('black')

# 3. Add the artist to the axes
ax.add_artist(at)


circle_positions = np.array([ center_pos + total_distance * np.array([np.cos(angle), np.sin(angle)]) for angle in np.linspace(0, 2 * np.pi, N, endpoint=False)])
square_side = centralRadius * np.sqrt(2); square_bottom_left = center_pos - np.array([square_side / 2, square_side / 2])
device_artists = {}; edge_artists = {}; device_label_artists = {}
square = patches.Rectangle(square_bottom_left, square_side, square_side, fc='lightblue', ec='black', zorder=2); ax.add_patch(square)
device_artists['center'] = square; device_label_artists['center'] = ax.text(center_pos[0], center_pos[1], 'Center', ha='center', va='center', fontsize=8, zorder=3)
initial_cycle = 0; title_obj = ax.set_title(f"Cycle {initial_cycle}", fontsize=title_fontsize, fontweight='bold')
cmap = plt.get_cmap('rainbow_r'); norm = mcolors.Normalize(vmin=0, vmax=N - 1)
for i, pos in enumerate(circle_positions): # devices
    color = cmap(norm(i)); circle = patches.Circle(pos, radius=0.1, fc=color, ec='black', alpha=device_opacity, zorder=2); ax.add_patch(circle)
    device_artists[f'circle_{i}'] = circle; device_label_artists[f'circle_{i}'] = ax.text(pos[0], pos[1], f'C{i}', ha='center', va='center', fontsize=7, zorder=3, bbox=dict(boxstyle='round,pad=0.1', fc=(1,1,1,0.5), ec='none'))
arrow_style_str = f'-|>,head_length={head_len},head_width={head_wid}'
for i in range(N): # Create Edges
    pos = circle_positions[i]; next_pos = circle_positions[(i + 1) % N]
    direction_vector = pos - center_pos; norm_dir = np.linalg.norm(direction_vector)
    if norm_dir < 1e-9: norm_dir = 1
    unit_direction = direction_vector / norm_dir; edge_offset = np.array([-unit_direction[1], unit_direction[0]]) * arrow_offset_dist
    start_o_i = center_pos + unit_direction * centralRadius + edge_offset; start_i_o = pos - edge_offset; start_pos_ring = pos
    arrow_out = patches.FancyArrowPatch(posA=start_o_i, posB=start_o_i, arrowstyle=arrow_style_str, color='darkgreen', linestyle='dashed', mutation_scale=mut_scale, lw=edge_linewidth, zorder=1); ax.add_patch(arrow_out)
    label_out = ax.text(start_o_i[0], start_o_i[1], "", color='darkgreen', fontsize=edge_label_fontsize, ha='center', va='center'); edge_artists[f'out_{i}'] = (arrow_out, label_out)
    arrow_in = patches.FancyArrowPatch(posA=start_i_o, posB=start_i_o, arrowstyle=arrow_style_str, color='magenta', linestyle='dashed', mutation_scale=mut_scale, lw=edge_linewidth, zorder=1); ax.add_patch(arrow_in)
    label_in = ax.text(start_i_o[0], start_i_o[1], "", color='magenta', fontsize=edge_label_fontsize, ha='center', va='center'); edge_artists[f'in_{i}'] = (arrow_in, label_in)
    arrow_ring = patches.FancyArrowPatch(posA=start_pos_ring, posB=start_pos_ring, # Start invisible
                                       arrowstyle=arrow_style_str, color='blue', linestyle='solid',
                                       mutation_scale=mut_scale, lw=edge_linewidth, zorder=1) # Use zorder=1 like others
    ax.add_patch(arrow_ring)
    label_ring = ax.text(start_pos_ring[0], start_pos_ring[1], "", color='blue', fontsize=edge_label_fontsize, ha='center', va='center'); 
    edge_artists[f'ring_{i}'] = (arrow_ring, label_ring)




class Device:
    def __init__(self, device_id, layer_capacity, total_devices, total_layers, total_chunks):
        self.device_id = device_id
        self.device_has_started = False
        self.device_start_time = 0
        self.device_has_finished = False
        self.device_finish_time = 0
        self.layer_capacity = layer_capacity
        self.total_devices = total_devices
        self.total_layers = total_layers
        self.total_chunks = total_chunks
        self.cur_ready_weights = [-1 for _ in range(layer_capacity)]
        self.cur_weight_replace_ind = 0
        self.next_layer_id = device_id + layer_capacity * total_devices
        for i in range (layer_capacity):
            self.cur_ready_weights[i] = (device_id + i * total_devices)

        #
        self.cur_ready_activations = set()
        if device_id == 0:
            for i in range (total_chunks):
                self.cur_ready_activations.add((i, -1))
        self.computation_queue = []
        ## can schedule all tasks up front
        device_total_layers = total_layers // total_devices
        if device_id < total_layers % total_devices:
            device_total_layers += 1

        cur_layer_id = device_id
        for k in range (device_total_layers):
            for i in range (total_chunks):
                self.computation_queue.append((i, cur_layer_id, computationFrames))
            cur_layer_id += total_devices

        print(f"device Id: {self.device_id}\n\tComputation Queue: {self.computation_queue}\n\n")
        
        self.is_computing = False
        self.is_stalled = False
        self.stall_start_time = 0
        self.cur_computation_start_time = 0
        self.cur_computation_duration = 0
        self.is_outbound_transferring = False
        self.outbound_queue = []
        self.cur_outbound_start_time = 0
        self.cur_outbound_duration = 0
        self.is_inbound_transferring = False
        self.inbound_queue = []
        self.cur_inbound_start_time = 0
        self.cur_inbound_duration = 0
        self.is_peer_transferring = False
        self.peer_transfer_queue = []
        self.cur_peer_transfer_start_time = 0
        self.cur_peer_transfer_duration = 0
        

       

        self.cur_outbound_edge = ""
        self.cur_inbound_edge = ""
        self.cur_ring_edge = ""
        self.cur_computing_edge = ""

        

    ## ensure we call this before udpate_device, so that dependencies are updated
    ## in synchonrous manner, before any computations start at same time
    def handle_completed_transfers(self, T, all_devices):

        ## check inbound queue to see if layer transfer arrived
        if self.is_inbound_transferring and self.cur_inbound_start_time + self.cur_inbound_duration <= T:
            inbound_item = self.inbound_queue.pop(0)
            layer_id = inbound_item[0]
            replace_ind = inbound_item[1]
            self.cur_ready_weights[replace_ind] = layer_id
            
            self.is_inbound_transferring = False
            self.cur_inbound_edge = ""

        ## check outbound queue to see if we finished sending any outbound transfers to now start again this cycle
        if self.is_outbound_transferring and self.cur_outbound_start_time + self.cur_outbound_duration <= T:
            ## not doing anything with this yet...
            ## would be used to during bwd pass to refertch and track ssy memory...
            self.outbound_queue.pop(0)
            self.is_outbound_transferring = False
            self.cur_outbound_edge = ""


        ## check if peer transfer has completed and then mark that device as ready
        if self.is_peer_transferring and self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T:
            ## mark that device as ready in peer activation queue
            peer_transfer_item = self.peer_transfer_queue.pop(0)
            peer_device_id, chunk_id, layer_id, transer_duration = peer_transfer_item
            
            peer_device = all_devices[peer_device_id]
            peer_device.cur_ready_activations.add((chunk_id, layer_id))

            self.is_peer_transferring = False
            self.cur_ring_edge = ""
            

    def handle_computation(self, T):

        completed_tasks = 0

        if not self.is_computing and len(self.computation_queue) > 0:
            next_task = self.computation_queue[0]
            chunk_id, layer_id, comp_dur = next_task
            if layer_id in self.cur_ready_weights and (chunk_id, layer_id - 1) in self.cur_ready_activations:
                ## can schedule this task!
                if not self.device_has_started:
                    self.device_start_time = T
                    self.device_has_started = True
                #print(f"Scheduling Computing!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tChunk ID: {chunk_id}, Layer ID: {layer_id}\n\n")
                
                if self.is_stalled:
                    print(f"Unstalling!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tWas Stalled for: {T - self.stall_start_time} frames\n\n")
                
                self.cur_computation_start_time = T
                self.cur_computation_duration = comp_dur
                self.is_computing = True
                self.is_stalled = False
                self.computing_edge = f"Computing: Chunk ID: {chunk_id}, Layer ID: {layer_id}"
            else:
                if not self.is_stalled:
                    self.is_stalled = True
                    self.stall_start_time = T
                    self.stall_edge = f"Stalled: Chunk ID: {chunk_id}, Layer ID: {layer_id}"
                    print(f"Stalling!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tCannot Compute: Chunk ID: {chunk_id}, Layer ID: {layer_id}\n\n")

        ## or maybe we are already computing and need to check if we finished this task
        elif self.is_computing and self.cur_computation_start_time + self.cur_computation_duration <= T:
            ## finsihed computting this task
            ## pop from cur_computation_queue
            ## add to cur_ready_activations
            complted_task = self.computation_queue.pop(0)
            chunk_id, layer_id, comp_dur = complted_task

            #print(f"Finished Computing!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tChunk ID: {chunk_id}, Layer ID: {layer_id}\n\n")

            completed_tasks += 1

            self.outbound_queue.append((chunk_id, layer_id, savedActivationsFrames))

            if layer_id < self.total_layers - 1:
                ## will schedule this transfer later on
                self.peer_transfer_queue.append(((self.device_id + 1) % self.total_devices, chunk_id, layer_id, activationTransitionFrames))


            ## if this is the last chunk, add next layer to prefetch to inbound queue
            if self.next_layer_id < self.total_layers and (chunk_id == self.total_chunks - 1):
                self.inbound_queue.append((self.next_layer_id, self.cur_weight_replace_ind, layerTransferFrames))
                self.next_layer_id += self.total_devices
                self.cur_weight_replace_ind = ((self.cur_weight_replace_ind + 1) % self.layer_capacity)

            self.is_computing = False
            self.computing_edge = ""

            if len(self.computation_queue) > 0:
                next_task = self.computation_queue[0]
                chunk_id, layer_id, comp_dur = next_task
                if layer_id in self.cur_ready_weights and (chunk_id, layer_id - 1) in self.cur_ready_activations:
                    ## can schedule this task!
                    #print(f"Scheduling Computing!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tChunk ID: {chunk_id}, Layer ID: {layer_id}\n\n")
                    
                    ## can schedule this task!
                    if not self.device_has_started:
                        self.device_start_time = T
                        self.device_has_started = True
                    
                    if self.is_stalled:
                        print(f"Unstalling!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tWas Stalled for: {T - self.stall_start_time} frames\n\n")
                    
                    self.is_computing = True
                    self.is_stalled = False
                    self.cur_computation_start_time = T
                    self.cur_computation_duration = comp_dur
                    self.computing_edge = f"Computing: Chunk ID: {chunk_id}, Layer ID: {layer_id}"
                else:
                    if not self.is_stalled:
                        self.is_stalled = True
                        self.stall_start_time = T
                        self.stall_edge = f"Stalled: Chunk ID: {chunk_id}, Layer ID: {layer_id}"
                        print(f"Stalling!: Time: {T}\n\tdevice ID: {self.device_id}\n\t\tCannot Compute: Chunk ID: {chunk_id}, Layer ID: {layer_id}\n\n")
            else:
                if not self.device_has_finished:
                    self.device_finish_time = T
                    self.device_has_finished = True
        
        return completed_tasks

    def handle_new_transfers(self, T):

        ## schedule next transfers if any
        if not self.is_inbound_transferring and len(self.inbound_queue) > 0:
            self.is_inbound_transferring = True
            self.cur_inbound_start_time = T
            self.cur_inbound_duration = self.inbound_queue[0][-1]
            self.cur_inbound_edge = f"Layer: {self.inbound_queue[0][0]}"

        if not self.is_outbound_transferring and len(self.outbound_queue) > 0:
            self.is_outbound_transferring = True
            self.cur_outbound_start_time = T
            self.cur_outbound_duration = self.outbound_queue[0][-1]
            self.cur_outbound_edge = f"Activations:\nChunk: {self.outbound_queue[0][0]},Layer: {self.outbound_queue[0][1]}"


        if not self.is_peer_transferring and len(self.peer_transfer_queue) > 0:
            self.is_peer_transferring = True
            self.cur_peer_transfer_start_time = T
            self.cur_peer_transfer_duration = self.peer_transfer_queue[0][-1]
            self.cur_ring_edge = f"Chunk: {self.peer_transfer_queue[0][0]}\nLayer: {self.peer_transfer_queue[0][1]}"
        




all_devices = {i: Device(i, layer_capacity, total_devices, total_layers, total_chunks) for i in range(N)}

total_tasks = sum([len(all_devices[i].computation_queue) for i in range(N)])

total_completed_tasks = 0

total_computation_time = 0
for i in range(N):
    for task in all_devices[i].computation_queue:
        total_computation_time += task[-1]

# --- Define Update Function (NEW Outward Logic) ---
def update(frame):
    T = frame
    artists_to_update = []

    global total_completed_tasks

    # Update Title / device Labels (same as before)
    title_cycle_num = T; title_obj.set_text(f'Cycle {title_cycle_num}'); artists_to_update.append(title_obj)
    device_label_artists['center'].set_text(f'Center\nT={T}'); artists_to_update.append(device_label_artists['center'])
    for i in range(N): device_label_artists[f'circle_{i}'].set_text(f'C{i}\nT={T}'); artists_to_update.append(device_label_artists[f'circle_{i}'])

    ## first handle completed transfers
    for i in range(N):
        all_devices[i].handle_completed_transfers(T, all_devices)

    ## then handle computations
    for i in range(N):
        total_completed_tasks += all_devices[i].handle_computation(T)

    ## then handle new transfers
    for i in range(N):
        all_devices[i].handle_new_transfers(T)


    # --- Loop through each device i ---
    for i in range(N):
        # Geometry setup
        pos = circle_positions[i]; next_pos = circle_positions[(i + 1) % N]
        direction_vector = pos - center_pos; norm_dir = np.linalg.norm(direction_vector)
        if norm_dir < 1e-9: norm_dir = 1
        unit_direction = direction_vector / norm_dir
        radial_perp_vector = np.array([-unit_direction[1], unit_direction[0]])
        edge_offset = radial_perp_vector * arrow_offset_dist

        # Artists
        arrow_out, label_out = edge_artists[f'out_{i}']
        arrow_in, label_in = edge_artists[f'in_{i}']
        arrow_ring, label_ring = edge_artists[f'ring_{i}']

        ## get current device
        device = all_devices[i]

        ## get current device's edges
        cur_outbound_edge = device.cur_outbound_edge
        cur_inbound_edge = device.cur_inbound_edge
        cur_ring_edge = device.cur_ring_edge
        cur_computing_edge = device.cur_computing_edge


        ## get current fracs
        len_o_i = 0.0
        len_i_o = 0.0
        length_factor_ring = 0.0

        if device.is_outbound_transferring:
            
            len_i_o = ((T - device.cur_outbound_start_time) / device.cur_outbound_duration) * transferDistance
            #print(f"Time: {T}\n\tdevice ID: {device.device_id}\n\tCurrently Outbound Transferring: {cur_outbound_edge}\n\nStarted at: {device.cur_outbound_start_time}\n\tDuration: {device.cur_outbound_duration}\n\n Length: {len_i_o}\n\n")

        if device.is_inbound_transferring:
            len_o_i = ((T - device.cur_inbound_start_time) / device.cur_inbound_duration) * transferDistance
            #print(f"Time: {T}\n\tdevice ID: {device.device_id}\n\tCurrently Inbound Transferring: {cur_inbound_edge}\n\nStarted at: {device.cur_inbound_start_time}\n\tDuration: {device.cur_inbound_duration}\n\n Length: {len_o_i}\n\n")

        if device.is_peer_transferring:
            length_factor_ring = ((T - device.cur_peer_transfer_start_time) / device.cur_peer_transfer_duration)
            #print(f"Time: {T}\n\tdevice ID: {device.device_id}\n\tCurrently Ring Transferring: {cur_ring_edge}\n\nStarted at: {device.cur_peer_transfer_start_time}\n\tDuration: {device.cur_peer_transfer_duration}\n\n Length: {length_factor_ring}\n\n")


        # --- Calculate Positions ---
        start_o_i_pos = center_pos + unit_direction * centralRadius + edge_offset
        end_o_i_pos = center_pos + unit_direction * (centralRadius + len_o_i) + edge_offset
        start_i_o_pos = pos - edge_offset
        end_i_o_pos = center_pos + unit_direction * (total_distance - len_i_o) - edge_offset
        start_pos_ring_geo = pos # Renamed to avoid conflict
        current_end_point_ring = start_pos_ring_geo + (next_pos - start_pos_ring_geo) * length_factor_ring

        # --- Calculate Label Positions with Offset ---
        label_perp_offset = radial_perp_vector * label_offset_distance
        midpoint_out = (start_o_i_pos + end_o_i_pos) / 2; label_pos_out = midpoint_out + label_perp_offset
        midpoint_in = (start_i_o_pos + end_i_o_pos) / 2; label_pos_in = midpoint_in - label_perp_offset
        midpoint_ring = (start_pos_ring_geo + current_end_point_ring) / 2
        label_pos_ring = midpoint_ring
        if length_factor_ring > 1e-6 :
             edge_vec_ring = current_end_point_ring - start_pos_ring_geo; norm_edge_vec_ring = np.linalg.norm(edge_vec_ring)
             if norm_edge_vec_ring > 1e-6:
                unit_edge_vec_ring = edge_vec_ring / norm_edge_vec_ring
                perp_ring_vec = np.array([-unit_edge_vec_ring[1], unit_edge_vec_ring[0]])
                label_pos_ring = midpoint_ring + perp_ring_vec * label_offset_distance

        # --- Update Artists for device i ---
        arrow_out.set_positions(start_o_i_pos, end_o_i_pos); label_out.set_position(label_pos_out); label_out.set_text(cur_inbound_edge)
        arrow_in.set_positions(start_i_o_pos, end_i_o_pos); label_in.set_position(label_pos_in); label_in.set_text(cur_outbound_edge)

        arrow_ring.set_positions(start_pos_ring_geo, current_end_point_ring)
        label_ring.set_position(label_pos_ring); label_ring.set_text(cur_ring_edge)

        artists_to_update.extend([arrow_out, label_out, arrow_in, label_in, arrow_ring, label_ring])


    ## now update at this timestep...
    #print(f"Completed Tasks: {total_completed_tasks}\n\tTotal Tasks: {total_tasks}\n\tPercentage: {total_completed_tasks / total_tasks}\n\n")

    if (total_completed_tasks >= total_tasks):
        print(f"Completed all tasks at time: {T}!!\n\n")

        start_bubble_agg_time = 0
        for i in range(len(all_devices)):
            start_bubble_agg_time += all_devices[i].device_start_time
        
        stop_bubble_agg_time = 0
        for i in range(len(all_devices)):
            stop_bubble_agg_time += T - all_devices[i].device_finish_time

        total_dev_time = T * len(all_devices)

        steady_pipeline_time = total_dev_time - stop_bubble_agg_time - start_bubble_agg_time

       

        completion_text = f"Simulation Complete!\nFinal Cycle Count: {T}\n\n"
        completion_text += f"Problem:\nTotal Tasks: {total_tasks}\nTotal Task Computation Time (Raw Problem Time): {total_computation_time}\n\nUtilized {len(all_devices)} devices for an aggregate of {total_dev_time} computation cycles\n\n\n\n"
        completion_text += f"Pipeline:\nTotal Pipeline Fill Time: {start_bubble_agg_time}\nTotal Pipeline Flush Time: {stop_bubble_agg_time}\nStready-State Pipeline Time: {steady_pipeline_time}\n\n\n\n"
        completion_text += f"EFFICIENCY:\nOverall:{round(total_computation_time / total_dev_time, 5) * 100}%\nSteady-State:{round(total_computation_time / steady_pipeline_time, 5) * 100}%\n\n"

        completion_text_artist = ax.text(0.5, 0.5, completion_text, # Position (center)
                                         transform=ax.transAxes, # Use axes coordinates
                                         ha='center', va='center', # Center alignment
                                         fontsize=20, color='blue', fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.5', fc=(1,1,1,0.95), ec='black'), # White box
                                         zorder=10)
        
        print(completion_text)

        # Add the new text artist to the list to be returned FOR THIS FRAME
        artists_to_update.append(completion_text_artist)

        # Stop the animation events
        if ani is not None: # Check if ani object exists
             ani.event_source.stop()
    
    return artists_to_update

# --- Create Animation ---
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=frame_interval, blit=False, repeat=False)

# --- Display or Save ---
plt.show()