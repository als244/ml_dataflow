# simulation.py
import numpy as np
import traceback
import sys
import time
import textwrap
from collections import deque
import math


# --- Constants ---
# Keep calculation constants, remove drawing constants
# COLOR_* constants remain as they identify *types* of operations/transfers
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

diamond_color = '#B9F2FF'
COLOR_COMPUTE_BWD_W = 'yellow'
COLOR_COMPUTE_HEAD = 'lawngreen'
COLOR_STALL_NODE_FILL_HEX = '#FF0000' # Example Hex for frontend
COLOR_FINISH_NODE_FILL_HEX = '#00CC00' # Example Hex for frontend

# Remove Matplotlib-specific geometry constants if frontend handles layout
# inner_radius, inner_node_radius, etc. can be removed or kept if
# backend calculates positions (less ideal for frontend rendering)

# --- Helper Functions (Keep as is) ---
def round_up_to_multiple(n, m):
    return int(((n + m - 1) // m) * m)

def determine_chunk_size(seqlen, min_chunk_size):
    if seqlen > min_chunk_size:
        # Case 1: Find smallest DIVISOR of seqlen that is >= min_chunk_size.
        # Ensure the start_check_value is at least 1, as divisors are positive.
        start_check_value = max(1, min_chunk_size)
        for chunk_candidate in range(start_check_value, seqlen + 1):
            if seqlen % chunk_candidate == 0:
                return chunk_candidate
        return seqlen # Defensive return, but loop should always find a value.
    else: # Case 2: seqlen <= min_chunk_size
        # Find smallest MULTIPLE of seqlen that is >= min_chunk_size.
        if min_chunk_size % seqlen == 0:
            # If min_chunk_size is already a multiple of seqlen,
            # and we need chunk_size >= min_chunk_size, then min_chunk_size is the smallest.
            return min_chunk_size
        else:
            # min_chunk_size is not a multiple. Find the next multiple of seqlen
            # that is strictly greater than min_chunk_size. This will also be the
            # smallest multiple of seqlen that is >= min_chunk_size.
            num_complete_blocks = min_chunk_size // seqlen
            return (num_complete_blocks + 1) * seqlen

# --- Device Class (Keep logic, remove plotting references) ---
class Device:
    # --- __init__ ---
    # Keep __init__ mostly the same, ensure no plotting refs
    def __init__(self, device_id, layer_capacity, activations_capacity, transitions_capacity, head_transitions_capacity, total_devices, total_layers, num_seqs, seqlen, chunks_per_seq, seqs_per_chunk, total_chunks, final_chunk_in_seq_data_frac, computation_times_frames, computation_times_frames_last_block, computation_times_frames_bwd, headFrames, bwdWFrames, layerTransferFrames, gradLayerTransferFrames, headTransferFrames, savedActivationsFrames, activationTransitionFrames, contextTransferFrames):
        self.device_id = device_id
        self.device_has_started = False
        self.device_start_time = 0
        self.device_has_finished = False
        self.device_finish_time = 0
        self.layer_capacity = layer_capacity
        self.activations_capacity = activations_capacity
        self.transitions_capacity = transitions_capacity
        self.head_transitions_capacity = head_transitions_capacity
        self.total_devices = total_devices
        self.num_seqs = num_seqs
        self.seqlen = seqlen
        self.chunks_per_seq = chunks_per_seq
        self.seqs_per_chunk = seqs_per_chunk
        self.total_chunks = total_chunks
        self.final_chunk_in_seq_data_frac = final_chunk_in_seq_data_frac
        self.computation_times_frames = computation_times_frames
        self.computation_times_frames_bwd = computation_times_frames_bwd
        self.computation_times_frames_last_block = computation_times_frames_last_block
        self.headFrames = headFrames
        self.bwdWFrames = bwdWFrames
        self.layerTransferFrames = layerTransferFrames
        self.gradLayerTransferFrames = gradLayerTransferFrames
        self.headTransferFrames = headTransferFrames
        self.savedActivationsFrames = savedActivationsFrames
        self.activationTransitionFrames = activationTransitionFrames
        self.contextTransferFrames = contextTransferFrames
        self.total_layers = total_layers # Store this separately if needed (e.g., for head layer ID)

        # --- State Variables (Keep as is) ---
        self.cur_weights = [-1 for _ in range(self.layer_capacity)]
        self.cur_weight_write_ptr = 0
        self.activations_buffer = [-1 for _ in range(self.activations_capacity)]
        self.activations_write_ptr = 0
        self.activations_empty_slot_ind = 0
        self.activations_stack = []
        self.final_dev_activation_stack = set()


        # Adjust transitions capacity for special devices (Keep logic)
        is_head_device = (self.device_id == (self.total_layers % self.total_devices if self.total_devices > 0 else 0))
        is_last_block_device = (self.device_id == ((self.total_layers - 1) % self.total_devices if self.total_devices > 0 else 0))
        inp_transition_capacity = self.transitions_capacity
        out_transition_capacity = self.transitions_capacity
        current_head_input_transitions_capacity = 0
        current_head_output_transitions_capacity = 0
        if is_head_device:
             current_head_input_transitions_capacity = self.head_transitions_capacity
             inp_transition_capacity = current_head_input_transitions_capacity
        if is_last_block_device:
             current_head_output_transitions_capacity = self.head_transitions_capacity
             inp_transition_capacity = current_head_output_transitions_capacity

        self.transitions_inbound_buffer = [-1 for _ in range(inp_transition_capacity)]
        self.transitions_outbound_buffer = [-1 for _ in range(out_transition_capacity)]
        self.transitions_inbound_empty_slot_ind = 0 if inp_transition_capacity > 0 else None
        self.transitions_outbound_empty_slot_ind = 0 if out_transition_capacity > 0 else None
        self.head_input_transitions_buffer = [-1 for _ in range(current_head_input_transitions_capacity)] if current_head_input_transitions_capacity > 0 else None
        self.head_input_transitions_empty_slot_ind = 0 if current_head_input_transitions_capacity > 0 else None
        self.head_output_transitions_buffer = [-1 for _ in range(current_head_output_transitions_capacity)] if current_head_output_transitions_capacity > 0 else None
        self.head_output_transitions_empty_slot_ind = 0 if current_head_output_transitions_capacity > 0 else None

        self.context_buffer = [-1 for _ in range(self.total_chunks)]
        self.home_storage = set()
        self.fwd_computation_queue = deque()
        self.bwd_computation_queue = deque()
        self.current_task = None
        self.current_task_dir = 0
        self.outbound_queue = deque()
        self.inbound_queue = deque()
        self.peer_transfer_queue = deque()

        # --- Activity State (Keep as is) ---
        self.is_computing = False
        self.is_stalled = False
        self.stall_start_time = 0
        self.cur_computation_start_time = 0
        self.cur_computation_duration = 0
        self.current_computation_type = None # "Fwd", "Bwd X", "Bwd W", "Head"
        self.current_computation_layer_id = -1
        self.current_computation_chunk_id = -1

        self.is_outbound_transferring = False
        self.cur_outbound_start_time = 0
        self.cur_outbound_duration = 0
        self.cur_outbound_details = None # (chunk_id, layer_id, is_grad, is_context)

        self.is_inbound_transferring = False
        self.cur_inbound_start_time = 0
        self.cur_inbound_duration = 0
        self.cur_inbound_details = None # (chunk_id, layer_id, is_grad, is_context)

        self.is_peer_transferring = False
        self.cur_peer_transfer_start_time = 0
        self.cur_peer_transfer_duration = 0
        self.cur_peer_transfer_details = None # (peer_id, chunk_id, layer_id, is_grad)

        self.computing_status_text = "Idle" # Renamed for clarity vs is_computing flag
        self.stall_reason = ""
        self.cur_fwd_computation_num = 0
        self.activations_stack_next_ind = 0
        self.next_weight_prefetch_layer_id = -1
        self.next_bwd_weight_prefetch_layer_id = -1
        self.head_final_chunk_id = -1

        self._initialize_tasks_and_state()

    # --- Methods (_initialize_tasks_and_state, handle_*, etc.) ---
    # Keep the *logic* of these methods largely the same.
    # Make sure state updates modify self.* attributes correctly.
    # Remove print statements or replace with logging if desired.

    def _initialize_tasks_and_state(self):
        # Keep this method's logic largely the same for resetting state
        # and initializing queues/buffers/prefetch pointers.
        # No plotting refs should be here anyway.
        self.cur_weights = [-1 for _ in range(self.layer_capacity)]
        self.cur_weight_write_ptr = 0
        self.activations_buffer = [-1 for _ in range(self.activations_capacity)]
        self.activations_write_ptr = 0
        self.activations_empty_slot_ind = 0
        self.activations_stack = []
        self.final_dev_activation_stack = set()

        is_head_device = (self.device_id == (self.total_layers % self.total_devices if self.total_devices > 0 else 0))
        is_last_block_device = (self.device_id == ((self.total_layers - 1) % self.total_devices if self.total_devices > 0 else 0))
        inp_transition_capacity = self.transitions_capacity
        out_transition_capacity = self.transitions_capacity
        current_head_input_transitions_capacity = 0
        current_head_output_transitions_capacity = 0
        if is_head_device:
             current_head_input_transitions_capacity = self.head_transitions_capacity
             inp_transition_capacity = current_head_input_transitions_capacity
             out_transition_capacity = current_head_input_transitions_capacity
        if is_last_block_device:
             current_head_output_transitions_capacity = self.head_transitions_capacity
             inp_transition_capacity = current_head_output_transitions_capacity
             out_transition_capacity = current_head_output_transitions_capacity

        self.transitions_inbound_buffer = [-1 for _ in range(inp_transition_capacity)]
        self.transitions_outbound_buffer = [-1 for _ in range(out_transition_capacity)]
        self.transitions_inbound_empty_slot_ind = 0 if inp_transition_capacity > 0 else None
        self.transitions_outbound_empty_slot_ind = 0 if out_transition_capacity > 0 else None
        self.head_input_transitions_buffer = [-1 for _ in range(current_head_input_transitions_capacity)] if current_head_input_transitions_capacity > 0 else None
        self.head_input_transitions_empty_slot_ind = 0 if current_head_input_transitions_capacity > 0 else None
        self.head_output_transitions_buffer = [-1 for _ in range(current_head_output_transitions_capacity)] if current_head_output_transitions_capacity > 0 else None
        self.head_output_transitions_empty_slot_ind = 0 if current_head_output_transitions_capacity > 0 else None


        self.context_buffer = [-1 for _ in range(self.total_chunks)]
        self.home_storage = set()
        self.computation_queue = deque()
        self.outbound_queue = deque()
        self.inbound_queue = deque()
        self.peer_transfer_queue = deque()
        self.is_computing = False
        self.is_stalled = False
        self.device_has_started = False
        self.device_start_time = 0
        self.device_has_finished = False
        self.device_finish_time = 0
        self.computing_status_text = "Idle"
        self.stall_reason = ""
        self.current_computation_type = None
        self.current_computation_layer_id = -1
        self.current_computation_chunk_id = -1
        self.final_train_chunk = self.total_chunks - self.chunks_per_seq


        # --- Populate Home Storage and Initial Weights ---
        if self.total_devices == 0: return

        all_my_layers = []
        cur_layer_id = self.device_id
        layer_idx_in_home = 0
        while cur_layer_id < self.total_layers: # Only non-head layers
            self.home_storage.add((-1, cur_layer_id, False)) # Add layer weight to home storage
            all_my_layers.append(cur_layer_id)
            if layer_idx_in_home < self.layer_capacity:
                self.cur_weights[layer_idx_in_home] = (0, cur_layer_id) # Mark as loaded initially
            layer_idx_in_home += 1
            cur_layer_id += self.total_devices

        # Add head layer if this device hosts it
        head_device_id = self.total_layers % self.total_devices
        if self.device_id == head_device_id:
            self.home_storage.add((-1, self.total_layers, False)) # Add head 'layer' weight
            all_my_layers.append(self.total_layers) # Include head in the list
            if layer_idx_in_home < self.layer_capacity:
                 self.cur_weights[layer_idx_in_home] = (0, self.total_layers)

        # --- Initialize Prefetch Pointers ---
        if len(all_my_layers) > self.layer_capacity:
            self.next_weight_prefetch_layer_id = all_my_layers[self.layer_capacity]
            self.next_bwd_weight_prefetch_layer_id = all_my_layers[len(all_my_layers) - self.layer_capacity - 1] if len(all_my_layers) > self.layer_capacity else -1
        else:
            self.next_weight_prefetch_layer_id = -1
            self.next_bwd_weight_prefetch_layer_id = -1



        # --- Build Computation Queue ---
        # Forward Pass (Non-Head Layers)
        cur_layer_id = self.device_id
        while cur_layer_id < self.total_layers:
            transfer_direction = 1
            for i in range(self.total_chunks):
                 comp_time = self.computation_times_frames.get(i, 0)
                 
                 ## forward pass for all chunks on all layers except the last one passes output to the next device
                 self.fwd_computation_queue.append((i, cur_layer_id, False, False, transfer_direction, comp_time))
                 
            cur_layer_id += self.total_devices

        ## the activations are needed in reverse order of sequences
        ## and reverse order within each sequence...

        ## the nearest activation (final chunk to be appended) is the last chunk of the first sequence on the last layer...
        
        cur_layer_id = self.device_id

        while cur_layer_id < self.total_layers:
             ## for case of multiple seqs per chunk...
            cur_chunk_ind = self.total_chunks - 1 

            for seq_batch_id in range(self.num_seqs - 1, -1, -self.seqs_per_chunk):
                if (self.chunks_per_seq > 1):
                    seq_chunk_start = int(seq_batch_id * self.chunks_per_seq)
                    for chunk_ind in range(seq_chunk_start, seq_chunk_start + self.chunks_per_seq):
                        self.activations_stack.append((chunk_ind, cur_layer_id))
                else:
                    self.activations_stack.append((cur_chunk_ind, cur_layer_id))
                    cur_chunk_ind -= 1

            cur_layer_id += self.total_devices
        
       

        # Context Buffer Initialization
        last_layer_on_device = -1
        temp_layer = self.device_id
        while temp_layer < self.total_layers:
            last_layer_on_device = temp_layer
            temp_layer += self.total_devices

        self.last_block_layer_id = last_layer_on_device
        if last_layer_on_device != -1:
             for i in range(self.total_chunks):
                 # Mark context as 'available' ONLY for the last layer computed
                 # Assuming context is only needed for the *last* layer's output
                 # This might need refinement based on exact context dependencies
                 if (last_layer_on_device % self.total_devices) == self.device_id: # Ensure it's the last layer *on this device*
                    self.context_buffer[i] = (0, i, last_layer_on_device)

        # Activation Stack Cutoff
        total_activations = len(self.activations_stack)

        ## the index (within activations_stack) of the first activation that we will be saving (not sending back)
        ## in activations_buffer (preparing for turn-around)
        ## leaving 1 extra space for working activation...
        activation_ind_cutoff = max(0, (total_activations - self.activations_capacity))

        final_stack = self.activations_stack[activation_ind_cutoff + 1:]
        for act in final_stack:
            self.final_dev_activation_stack.add(act)

        ## the index (within activations_stack) of the next activation to be saved
        self.activations_stack_next_ind = activation_ind_cutoff

        # Head Task (if applicable)
        if self.device_id == head_device_id:

            transfer_direction = -1
            # Simple Head processing order: all training chunks sequentially
            # More complex logic (like reversal) could be added if needed
            for i in range(self.chunks_per_seq - 1, self.total_chunks, self.chunks_per_seq):
                for j in range(i, i - self.chunks_per_seq, -1):
                    self.fwd_computation_queue.append((j, self.total_layers, True, False, transfer_direction, self.headFrames))
                    self.head_final_chunk_id = j

        # Backward Pass
        cur_layer_id = last_layer_on_device # Start from last non-head layer
        # If head device, BWD starts from layer *before* head
        # Head layer's backward pass is implicitly handled by the 'Head' task itself generating the first gradient
        if self.device_id == head_device_id:
            head_layer_index = -1
            try:
                 # Find index of head layer if present
                 head_layer_index = all_my_layers.index(self.total_layers)
                 if head_layer_index > 0:
                     cur_layer_id = all_my_layers[head_layer_index - 1] # Start BWD from layer before head
                 else:
                      cur_layer_id = -1 # No non-head layers on head device
            except ValueError:
                 cur_layer_id = last_layer_on_device # Head not in list (shouldn't happen if head device)

        while cur_layer_id >= 0 :
            transfer_direction = -1
            if cur_layer_id == 0:
                transfer_direction = 0

            for i in range(self.chunks_per_seq - 1, self.total_chunks, self.chunks_per_seq):
                for j in range(i, i - self.chunks_per_seq, -1):
                    
                    bwdX_time = self.computation_times_frames_bwd.get(j, 0)
                    bwdW_time = self.bwdWFrames

                    current_transfer_dir_bX = transfer_direction if cur_layer_id > 0 else 0
                    self.bwd_computation_queue.append((j, cur_layer_id, True, False, current_transfer_dir_bX, bwdX_time)) # BwdX
                    self.bwd_computation_queue.append((j, cur_layer_id, False, True, 0, bwdW_time)) # BwdW

            cur_layer_id -= self.total_devices

    def _find_first_free_idx(self, buffer):
        try:
            return buffer.index(-1)
        except ValueError:
            return None

    # --- handle_completed_transfers ---
    # Keep logic essentially the same
    def handle_completed_transfers(self, T, all_devices):

        to_repeat = False
        # Inbound (Device <- Home)
        if self.is_inbound_transferring and self.inbound_queue and (self.cur_inbound_start_time + self.cur_inbound_duration <= T):
            item = self.inbound_queue.popleft()
            chunk_id, layer_id, is_grad, is_only_context, target_idx, duration = item
            # print(f"T={T}, Dev {self.device_id}: Completed INBOUND {item[:4]}") # Optional log

            if chunk_id == -1: # Weight or Head state
                if 0 <= target_idx < len(self.cur_weights):
                    self.cur_weights[target_idx] = (0, layer_id) # Mark as ready
                else: 
                    # print(f"T={T}, Dev {self.device_id}: ERROR - Invalid target index {target_idx} for weight inbound.")
                    # print(f"Cur weights: {self.cur_weights}")
                    pass
            elif is_only_context: # Context
                self.context_buffer[target_idx] = (0, chunk_id, layer_id)
            else: # Activation
                if 0 <= target_idx < len(self.activations_buffer):
                    self.activations_buffer[target_idx] = (0, chunk_id, layer_id)
                    if self.context_buffer[chunk_id] == (-2, chunk_id, layer_id):
                        self.context_buffer[chunk_id] = (0, chunk_id, layer_id)
                else: 
                    # print(f"T={T}, Dev {self.device_id}: ERROR - Invalid target index {target_idx} for activation inbound.")
                    # print(f"Activations buffer: {self.activations_buffer}")
                    pass

            self.is_inbound_transferring = False
            self.cur_inbound_details = None

            if duration == 0:
                to_repeat = True

        # Outbound (Device -> Home)
        if self.is_outbound_transferring and self.outbound_queue and (self.cur_outbound_start_time + self.cur_outbound_duration <= T):
            item = self.outbound_queue.popleft()
            chunk_id, layer_id, is_grad, is_only_context, duration = item
            # print(f"T={T}, Dev {self.device_id}: Completed OUTBOUND {item[:4]}") # Optional log

            storage_key = (chunk_id, layer_id, is_grad)
            
            self.home_storage.add(storage_key)

            if chunk_id >= 0 and not is_only_context: # Activation saved
                try:
                    idx_to_free = self.activations_buffer.index((-2, chunk_id, layer_id))
                    self.activations_buffer[idx_to_free] = -1
                    self.activations_empty_slot_ind = self._find_first_free_idx(self.activations_buffer)
                except ValueError: pass # print(f"T={T}, Dev {self.device_id}: WARN - Could not find activation ({chunk_id},{layer_id}) marked for outbound transfer.")

            self.is_outbound_transferring = False
            self.cur_outbound_details = None

            if duration == 0:
                to_repeat = True

        # Peer-to-Peer (Device -> Peer Device)
        if self.is_peer_transferring and self.peer_transfer_queue and (self.cur_peer_transfer_start_time + self.cur_peer_transfer_duration <= T):
            item = self.peer_transfer_queue.popleft()
            peer_id, cid, lid, is_grad, duration = item

            if 0 <= peer_id < len(all_devices):
                peer_dev = all_devices[peer_id]

                # Free self's outbound buffer slot
                try:
                    out_idx_to_free = self.transitions_outbound_buffer.index((0, cid, lid, is_grad))
                    self.transitions_outbound_buffer[out_idx_to_free] = -1
                    self.transitions_outbound_empty_slot_ind = self._find_first_free_idx(self.transitions_outbound_buffer)
                except ValueError: pass # print(f"T={T}, Dev {self.device_id}: WARN - Could not find outbound transition ({cid},{lid},{is_grad}) marked as ready.")

                # Update peer's inbound buffer slot
                buffer_to_check = None
                ## output of head is goes to the last block's special input buffer = head_input_transitions_buffer
                if lid == self.total_layers: buffer_to_check = peer_dev.head_output_transitions_buffer
                ## output of last block is goes to the head's special input buffer = head_output_transitions_buffer
                elif lid == self.total_layers - 1 and not is_grad: buffer_to_check = peer_dev.head_input_transitions_buffer
                else: buffer_to_check = peer_dev.transitions_inbound_buffer

                if buffer_to_check is not None:
                    try:
                        in_idx_to_update = buffer_to_check.index((-2, cid, lid, is_grad))
                        buffer_to_check[in_idx_to_update] = (0, cid, lid, is_grad)
                    except ValueError: pass # print(f"T={T}, Dev {self.device_id}: WARN - Peer {peer_id} could not find inbound transition ({cid},{lid},{is_grad}).")
                    except IndexError: pass # print(f"T={T}, Dev {self.device_id}: ERROR - Invalid index access for peer {peer_id} buffer.")
                # else: print(f"T={T}, Dev {self.device_id}: ERROR - Could not determine peer {peer_id}'s buffer.")
            # else: print(f"T={T}, Dev {self.device_id}: ERROR - Invalid peer_id {peer_id} for peer transfer.")

            self.is_peer_transferring = False
            self.cur_peer_transfer_details = None

            if duration == 0:
                to_repeat = True
        
        return to_repeat
            

    # --- handle_computation_depends ---
    # Keep logic the same, but update `computing_status_text`
    def handle_computation_depends(self, T):
        if not self.fwd_computation_queue and not self.bwd_computation_queue:
            self.computing_status_text = "Idle (No Tasks)" if not self.device_has_finished else "Finished Comp"
            self.is_stalled = False
            return False

        has_fwd_task = len(self.fwd_computation_queue) > 0
        has_bwd_task = len(self.bwd_computation_queue) > 0
       
        is_head_task = False
        if has_fwd_task:
            next_fwd_task = self.fwd_computation_queue[0]
            f_cid, f_lid, f_bX, f_bW, f_tdir, f_dur = next_fwd_task
            if f_lid == self.total_layers:
                is_head_task = True
       
        if has_bwd_task:
            next_bwd_task = self.bwd_computation_queue[0]
            b_cid, b_lid, b_bX, b_bW, b_tdir, b_dur = next_bwd_task


        computation_type_str = "???"
        self.stall_reason = ""

        ## first check if we have a bwd task that is ready...

        has_bwd_deps = False
        has_fwd_deps = False
        bwd_missing_items = []
        fwd_missing_items = []
        
        ## finishing head before doing bwd...
        #if not is_head_task and has_bwd_task and (b_bX or not has_fwd_task):
        if not is_head_task and has_bwd_task:  
            has_weight = (0, b_lid) in self.cur_weights
            has_input_transition = False
            input_transition_key = None
            input_buffer_to_check = None

            if b_bX:
                input_transition_key = (0, b_cid, b_lid + 1, True)
                if b_lid == self.total_layers - 1:
                    input_buffer_to_check = self.head_output_transitions_buffer
                else:
                    input_buffer_to_check = self.transitions_inbound_buffer
        
                # bW doesn't need a *new* input transition
                if input_transition_key and input_buffer_to_check:
                    has_input_transition = input_transition_key in input_buffer_to_check

            has_fwd_activation = True
            fwd_act_key = None

            if b_bW:
                fwd_act_key = (0, b_cid, b_lid)
                has_fwd_activation = fwd_act_key in self.activations_buffer

            
            has_context = True
            missing_ctx_chunk = -1

            cur_seq_start = b_cid - (b_cid % self.chunks_per_seq)
            if b_bX: # Simplified context check: assumes context for *this* layer `lid` is needed
                for i in range(b_cid, cur_seq_start - 1, -1):
                    if self.context_buffer[i] != (0, i, b_lid):
                        has_context = False
                        missing_ctx_chunk = i
                        break

            needs_outbound_trans_space = (b_tdir != 0)
       
            has_outbound_trans_space = True # Assume true if no transfer needed
            if needs_outbound_trans_space and (self.transitions_outbound_empty_slot_ind is None):
                has_outbound_trans_space = False

            if b_bX:
                computation_type_str = "Bwd X"
                has_bwd_deps = has_weight and has_input_transition and has_context and has_outbound_trans_space
                if not has_weight: bwd_missing_items.append("No Weight")
                if not has_input_transition: bwd_missing_items.append("No Grad. Stream")
                if not has_context: bwd_missing_items.append(f"No Ctx (Chunk: {missing_ctx_chunk})")
                if not has_outbound_trans_space: bwd_missing_items.append("Congested (Trans.)")
            elif b_bW:
                computation_type_str = "Bwd W"
                has_bwd_deps = has_fwd_activation
                if not has_fwd_activation: bwd_missing_items.append("No Fwd Act.")

            ## if we have a bwd task then schedule it!
            if has_bwd_deps:
                if not self.device_has_started:
                    self.device_has_started = True
                    self.device_start_time = T

                # if self.is_stalled: print(f"T={T}, Dev {self.device_id}: UNSTALL") # Optional log
                self.is_computing = True
                self.is_stalled = False
                self.stall_reason = ""
                self.cur_computation_start_time = T
                self.cur_computation_duration = b_dur
                self.current_computation_type = computation_type_str
                self.current_computation_layer_id = b_lid
                self.current_computation_chunk_id = b_cid
                # UPDATE STATUS TEXT FOR FRONTEND
                self.computing_status_text = f"COMP:\n{computation_type_str}\nC{b_cid},L{b_lid}"

                # Mark required input buffers as 'in use' / consumed (-2) (Keep logic)
                if input_transition_key and input_buffer_to_check and not b_bW:
                    try:
                        idx = input_buffer_to_check.index(input_transition_key)
                        input_buffer_to_check[idx] = (-2,) + input_transition_key[1:]
                        # Update empty slot index for that buffer
                        if input_buffer_to_check is self.transitions_inbound_buffer: self.transitions_inbound_empty_slot_ind = self._find_first_free_idx(self.transitions_inbound_buffer)
                        elif input_buffer_to_check is self.head_input_transitions_buffer: self.head_input_transitions_empty_slot_ind = self._find_first_free_idx(self.head_input_transitions_buffer)
                        elif input_buffer_to_check is self.head_output_transitions_buffer: self.head_output_transitions_empty_slot_ind = self._find_first_free_idx(self.head_output_transitions_buffer)
                    except ValueError: pass # print(f"T={T}, Dev {self.device_id}: ERROR - Could not find input transition {input_transition_key} to mark.")

                # Reserve space in outbound transition buffer is required
                # Already checked depedency that there is space in outbound buffer
                if needs_outbound_trans_space and has_outbound_trans_space:
                    is_grad_out = b_bX
                    self.transitions_outbound_buffer[self.transitions_outbound_empty_slot_ind] = (-2, b_cid, b_lid, is_grad_out)
                    self.transitions_outbound_empty_slot_ind = self._find_first_free_idx(self.transitions_outbound_buffer)

                return True, -1 # Computation successfully started
                
        
        ## if we didn't have a bwd task ready then check if we have a fwd task that is ready...
        has_fwd_deps = False
        if has_fwd_task:
            ## check if we have a fwd task that is ready...
            # --- Check Dependencies (Keep logic) ---
            has_weight = (0, f_lid) in self.cur_weights
            has_input_transition = False
            input_transition_key = None
            input_buffer_to_check = None
          
            input_transition_key = (0, f_cid, f_lid - 1, False)
            ## if head 
            if f_lid == self.total_layers:
                input_buffer_to_check = self.head_input_transitions_buffer
            else:
                input_buffer_to_check = self.transitions_inbound_buffer


            if f_lid == 0:
                input_buffer_to_check = None
                has_input_transition = True

            if input_transition_key and input_buffer_to_check:
                has_input_transition = input_transition_key in input_buffer_to_check


            needs_act_buffer_space = f_lid < self.total_layers
            has_act_buffer_space = True
            if needs_act_buffer_space and self.activations_empty_slot_ind is None:
                has_act_buffer_space = False

            needs_outbound_trans_space = (f_tdir != 0)
       
            has_outbound_trans_space = True # Assume true if no transfer needed
            if needs_outbound_trans_space and (self.transitions_outbound_empty_slot_ind is None):
                has_outbound_trans_space = False

            if f_lid == self.total_layers:
                computation_type_str = "Head"
                has_fwd_deps = has_weight and has_input_transition and has_outbound_trans_space
                if not has_weight: fwd_missing_items.append("No Weight")
                if not has_input_transition: fwd_missing_items.append("No Act. Stream")
                if not has_outbound_trans_space: fwd_missing_items.append("Congested (Trans.)")
            else:
                computation_type_str = "Fwd"
                has_fwd_deps = has_weight and has_input_transition and has_act_buffer_space and has_outbound_trans_space
                if not has_weight: fwd_missing_items.append("No Weight")
                if not has_input_transition: fwd_missing_items.append("No Act. Stream")
                if not has_act_buffer_space: fwd_missing_items.append("Congested (Act.)")
                if not has_outbound_trans_space: fwd_missing_items.append("Congested (Trans.)")

            if has_fwd_deps:
                if not self.device_has_started:
                    self.device_has_started = True
                    self.device_start_time = T

                # if self.is_stalled: print(f"T={T}, Dev {self.device_id}: UNSTALL") # Optional log
                self.is_computing = True
                self.is_stalled = False
                self.stall_reason = ""
                self.cur_computation_start_time = T
                self.cur_computation_duration = f_dur
               
                self.current_computation_type = computation_type_str
                self.current_computation_layer_id = f_lid
                self.current_computation_chunk_id = f_cid
                # UPDATE STATUS TEXT FOR FRONTEND
                self.computing_status_text = f"COMP:\n{computation_type_str}\nC{f_cid},L{f_lid}"

                # Mark required input buffers as 'in use' / consumed (-2) (Keep logic)
                if input_transition_key and input_buffer_to_check:
                    try:
                        idx = input_buffer_to_check.index(input_transition_key)
                        input_buffer_to_check[idx] = (-2,) + input_transition_key[1:]
                        # Update empty slot index for that buffer
                        if input_buffer_to_check is self.transitions_inbound_buffer: self.transitions_inbound_empty_slot_ind = self._find_first_free_idx(self.transitions_inbound_buffer)
                        elif input_buffer_to_check is self.head_input_transitions_buffer: self.head_input_transitions_empty_slot_ind = self._find_first_free_idx(self.head_input_transitions_buffer)
                    except ValueError: 
                        #print(f"T={T}, Dev {self.device_id}: ERROR - Could not find input transition {input_transition_key} to mark.")
                        pass

                # Reserve space in outbound transition buffer is required
                # Already checked depedency that there is space in outbound buffer
                if needs_outbound_trans_space and has_outbound_trans_space:
                    is_grad_out = f_lid == self.total_layers
                    self.transitions_outbound_buffer[self.transitions_outbound_empty_slot_ind] = (-2, f_cid, f_lid, is_grad_out)
                    self.transitions_outbound_empty_slot_ind = self._find_first_free_idx(self.transitions_outbound_buffer)

                return True, 1
        


        ## otherwise we are stalled
        if not self.is_stalled:
            self.is_stalled = True
            self.stall_start_time = T
            # print(f"T={T}, Dev {self.device_id}: STALL -> Waiting for C{cid},L{lid},{computation_type_str}. Missing: {missing_items}") # Optional Log

        self.is_computing = False
        if has_fwd_task:
            if is_head_task:
                self.stall_reason = "Head Blocked by:\n" + "\n".join(fwd_missing_items)
            else:
                self.stall_reason = "FWD Blocked by:\n" + "\n".join(fwd_missing_items)
            #if has_bwd_task:
            #    self.stall_reason += "\n\nBWD Blocked by:\n" + "\n".join(bwd_missing_items)
        else:
            self.stall_reason = "BWD Blocked by:\n" + "\n".join(bwd_missing_items)

        # UPDATE STATUS TEXT FOR FRONTEND
        if has_fwd_task:
            self.computing_status_text = f"STALL:\n{computation_type_str}\nC{f_cid},L{f_lid}"
        else:
            self.computing_status_text = f"STALL:\n{computation_type_str}\nC{b_cid},L{b_lid}"
            
        self.current_computation_type = None
        self.current_computation_layer_id = -1
        self.current_computation_chunk_id = -1
        return False, 0

    # --- Prefetching logic (handle_bwd_prefetch_*) ---
    # Keep logic the same.
    def handle_bwd_prefetch_weight(self):
        if self.next_bwd_weight_prefetch_layer_id >= 0:
            evict_layer_id = -1
            if 0 <= self.cur_weight_write_ptr < len(self.cur_weights):
                if isinstance(self.cur_weights[self.cur_weight_write_ptr], tuple) and len(self.cur_weights[self.cur_weight_write_ptr]) == 2:
                     evict_layer_id = self.cur_weights[self.cur_weight_write_ptr][1]
                else: pass
            else: return # Invalid pointer

            prefetch_key = (-1, self.next_bwd_weight_prefetch_layer_id, False)
            if prefetch_key not in self.home_storage:
                return

            transfer_time = self.headTransferFrames if self.next_bwd_weight_prefetch_layer_id == self.total_layers else self.layerTransferFrames
            weight_item = (-1, self.next_bwd_weight_prefetch_layer_id, False, False, self.cur_weight_write_ptr, transfer_time)

            insert_idx = 0
            if self.is_inbound_transferring: insert_idx = 1
            while insert_idx < len(self.inbound_queue):
                _, q_lid, _, _, _, _ = self.inbound_queue[insert_idx]
                if self.next_bwd_weight_prefetch_layer_id >= q_lid: 
                    break
                insert_idx += 1
            self.inbound_queue.insert(insert_idx, weight_item)

            if 0 <= self.cur_weight_write_ptr < len(self.cur_weights):
                 self.cur_weights[self.cur_weight_write_ptr] = (-2, self.next_bwd_weight_prefetch_layer_id)

            current_prefetch_layer = self.next_bwd_weight_prefetch_layer_id
            self.next_bwd_weight_prefetch_layer_id -= self.total_devices
            if self.next_bwd_weight_prefetch_layer_id < 0:
                self.next_bwd_weight_prefetch_layer_id = -1

            self.cur_weight_write_ptr = (self.cur_weight_write_ptr - 1 + self.layer_capacity) % self.layer_capacity

    def handle_bwd_prefetch_context(self, chunk_id, cur_layer_id, next_layer_id):
        if self.context_buffer[chunk_id] == (0, chunk_id, next_layer_id): 
            return # Already available

        if (0, chunk_id, next_layer_id) in self.activations_buffer:
            self.context_buffer[chunk_id] = (0, chunk_id, next_layer_id)
            return
        
        if self.context_buffer[chunk_id] == (-2, chunk_id, next_layer_id): 
            return # Already fetching
        
        if (-2, chunk_id, next_layer_id) in self.activations_buffer:
            self.context_buffer[chunk_id] = (-2, chunk_id, next_layer_id)
            return
        
        context_key = (chunk_id, next_layer_id, False)

        ctx_item = (chunk_id, next_layer_id, False, True, chunk_id, self.contextTransferFrames)

        insert_idx = 0
        if self.is_inbound_transferring: insert_idx = 1
        while insert_idx < len(self.inbound_queue):
            q_cid, q_lid, _, q_is_ctx, _, _ = self.inbound_queue[insert_idx]
            ## higher layers come first during bwds pass...
            if next_layer_id > q_lid: 
                break
            if q_lid == next_layer_id and q_cid != -1 and chunk_id > q_cid: 
                break
            if q_lid == next_layer_id and q_cid <= chunk_id and not q_is_ctx: 
                break
            insert_idx += 1

        self.inbound_queue.insert(insert_idx, ctx_item)

        self.context_buffer[chunk_id] = (-2, chunk_id, next_layer_id)

    def handle_bwd_prefetch_fwd_act(self):
        if self.activations_stack_next_ind >= 0:
            if self.activations_empty_slot_ind is None: return # Buffer full

            cid, lid = self.activations_stack[self.activations_stack_next_ind]
            next_act_idx = self.activations_empty_slot_ind

            act_key = (cid, lid, False)
            ## can't prefetch if it's not in home storage
            if act_key not in self.home_storage:
                 # Potentially try next one (careful with recursion)
                 # self.handle_bwd_prefetch_fwd_act()
                 return
            
            if (cid + 1) % self.chunks_per_seq == 0:
                act_item = (cid, lid, False, False, next_act_idx, round(self.final_chunk_in_seq_data_frac * self.savedActivationsFrames))
            else:
                act_item = (cid, lid, False, False, next_act_idx, self.savedActivationsFrames)

            insert_idx = 0
            if self.is_inbound_transferring: 
                insert_idx = 1
            while insert_idx < len(self.inbound_queue):
                 q_cid, q_lid, _, _, _, _ = self.inbound_queue[insert_idx]
                 if lid > q_lid:
                    break
                 insert_idx += 1
            self.inbound_queue.insert(insert_idx, act_item)

            self.activations_buffer[next_act_idx] = (-2, cid, lid)
            self.activations_empty_slot_ind = self._find_first_free_idx(self.activations_buffer)

            self.activations_stack_next_ind -= 1

    # --- handle_computation ---
    # Keep logic the same, update `computing_status_text`
    def handle_computation(self, T):
        completed_tasks = 0
        has_depends = False 
        comp_dir = 0
        if not self.is_computing and not self.is_stalled and (len(self.fwd_computation_queue) > 0 or len(self.bwd_computation_queue) > 0):
            has_depends, comp_dir = self.handle_computation_depends(T)
            if has_depends:
                if comp_dir == 1:
                    self.current_task = self.fwd_computation_queue[0]
                    self.current_task_dir = 1
                else:
                    self.current_task = self.bwd_computation_queue[0]
                    self.current_task_dir = -1

     
        if self.is_computing and (self.cur_computation_start_time + self.cur_computation_duration <= T):
            
            if self.current_task_dir == 1:
                self.fwd_computation_queue.popleft()
            else:
                self.bwd_computation_queue.popleft()

            completed_tasks += 1
            cid, lid, bX, bW, tdir, dur = self.current_task

            is_head = lid == self.total_layers
            is_fwd = not bX and not bW and not is_head
            task_type_str = self.current_computation_type # Get type set when started

            # print(f"T={T}, Dev {self.device_id}: FINISHED Comp -> C{cid},L{lid},{task_type_str}") # Optional Log

            # --- Post-Computation Actions & State Updates (Keep Logic) ---

            # 1. Free Consumed Input Buffers
            input_trans_consumed_key = None
            input_trans_buffer_to_free = None
            if is_fwd and lid > 0:
                 input_trans_consumed_key = (-2, cid, lid - 1, False)
                 input_trans_buffer_to_free = self.transitions_inbound_buffer
            elif is_head:
                 input_trans_consumed_key = (-2, cid, self.total_layers - 1, False)
                 input_trans_buffer_to_free = self.head_input_transitions_buffer
            elif bX:
                 input_trans_consumed_key = (-2, cid, lid + 1, True)
                 if lid == self.total_layers - 1: 
                     input_trans_buffer_to_free = self.head_output_transitions_buffer
                 else: 
                     input_trans_buffer_to_free = self.transitions_inbound_buffer

            if input_trans_consumed_key and input_trans_buffer_to_free:
                try:
                    idx = input_trans_buffer_to_free.index(input_trans_consumed_key)
                    input_trans_buffer_to_free[idx] = -1
                    if input_trans_buffer_to_free is self.transitions_inbound_buffer: self.transitions_inbound_empty_slot_ind = self._find_first_free_idx(self.transitions_inbound_buffer)
                    elif input_trans_buffer_to_free is self.head_input_transitions_buffer: self.head_input_transitions_empty_slot_ind = self._find_first_free_idx(self.head_input_transitions_buffer)
                    elif input_trans_buffer_to_free is self.head_output_transitions_buffer: self.head_output_transitions_empty_slot_ind = self._find_first_free_idx(self.head_output_transitions_buffer)
                except ValueError: 
                    #print(f"T={T} Dev {self.device_id} WARN: Couldn't find consumed input {input_trans_consumed_key} to free.")
                    pass

            # 2. Handle Output Activation/Context Saving (FWD)
            if is_fwd:
                should_keep_in_buffer = (cid, lid) in self.final_dev_activation_stack
                act_dest_idx = self.activations_empty_slot_ind
                
                if should_keep_in_buffer:
                    self.activations_buffer[act_dest_idx] = (0, cid, lid)
                else:
                    self.activations_buffer[act_dest_idx] = (-2, cid, lid)

                    if (cid + 1) % self.chunks_per_seq == 0:
                        self.outbound_queue.append((cid, lid, False, False, round(self.final_chunk_in_seq_data_frac * self.savedActivationsFrames)))
                    else:
                        self.outbound_queue.append((cid, lid, False, False, self.savedActivationsFrames))
                    
                self.activations_empty_slot_ind = self._find_first_free_idx(self.activations_buffer)

            # 3. Handle Output Peer Transition
            needs_outbound_trans = (tdir != 0)

            is_grad_out = bX or is_head
            if needs_outbound_trans:
                peer_id = (self.device_id + tdir) % self.total_devices
                ## all outbound transitions (in waiting pool until transfer), use the same buffer
                ## only the input buffers on head and last block use different input buffers....
                out_idx = self.transitions_outbound_buffer.index((-2, cid, lid, is_grad_out))
                self.transitions_outbound_buffer[out_idx] = (0, cid, lid, is_grad_out) # Mark ready
                # Use correct transition time (potentially different for head?) - assumed activationTransitionFrames for now
                trans_time = self.activationTransitionFrames # Simplification
                if (cid + 1) % self.chunks_per_seq == 0:
                    trans_time = round(self.final_chunk_in_seq_data_frac * trans_time)
                self.peer_transfer_queue.append((peer_id, cid, lid, is_grad_out, trans_time))


            # 4. Trigger Prefetches / Gradient Saves (Keep Logic)
            if is_fwd: # FWD Finished
                ## check to see if we finished a layer and should prefetch next weights...
                if (cid == self.total_chunks - 1) and self.next_weight_prefetch_layer_id >= 0 and self.next_weight_prefetch_layer_id <= self.total_layers:
                    layer_transfer_time = self.layerTransferFrames
                    if self.next_weight_prefetch_layer_id == self.total_layers:
                        layer_transfer_time = self.headTransferFrames
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
                    ## transfer head gradient...
                    self.outbound_queue.append((-1, lid, True, False, self.headTransferFrames))
                    self.handle_bwd_prefetch_weight()
            elif bX: # BwdX finished, can replace current context now

                if (lid - self.total_devices >= 0):
                    #print(f"Prefetching context for next layer of training chunk: {cid}, Cur Layer Id: {lid}, Next Layer Id: {lid - self.total_devices}")
                    self.handle_bwd_prefetch_context(cid, lid, lid - self.total_devices)

                ## could prefetch weight here but doing at end of bW for cleanliness

            elif bW: # BwdW finished, can prefetch the next required activaton
                ## now prefetch the next required activation, which replaces the current activation

                ## set current activation we just used to be -1...
                act_idx = self.activations_buffer.index((0, cid, lid))
                self.activations_buffer[act_idx] = -1
                self.activations_empty_slot_ind = self.activations_buffer.index(-1)

                self.handle_bwd_prefetch_fwd_act()

                if (cid == self.final_train_chunk):
                    self.outbound_queue.append((-1, lid, True, False, self.gradLayerTransferFrames))

                ## if this is the last chunk, now prefetch the next required weight
                ## which replaces the current weight
                if cid == self.final_train_chunk:
                    self.handle_bwd_prefetch_weight()

            # --- Reset Compute State & Check Next Task ---
            self.is_computing = False
            self.computing_status_text = "Idle" # Reset status text
            self.current_computation_type = None
            self.current_computation_layer_id = -1
            self.current_computation_chunk_id = -1
            self.current_task = None
            self.current_task_dir = 0

            if len(self.fwd_computation_queue) > 0 or len(self.bwd_computation_queue) > 0:
                has_depends, comp_dir = self.handle_computation_depends(T)
                if has_depends:
                    if comp_dir == 1:
                        self.current_task = self.fwd_computation_queue[0]
                        self.current_task_dir = 1
                    else:
                        self.current_task = self.bwd_computation_queue[0]
                        self.current_task_dir = -1
            elif not self.device_has_finished:
                self.device_finish_time = T
                self.device_has_finished = True
                self.computing_status_text = "Finished Comp"
                # print(f"T={T}, Dev {self.device_id}: >>> FINISHED ALL COMPUTE TASKS <<<") # Optional log

        elif self.is_stalled:
            has_depends, comp_dir = self.handle_computation_depends(T) # Re-check dependencies
            if has_depends:
                if comp_dir == 1:
                    self.current_task = self.fwd_computation_queue[0]
                    self.current_task_dir = 1
                else:
                    self.current_task = self.bwd_computation_queue[0]
                    self.current_task_dir = -1

        return completed_tasks

    # --- handle_new_transfers ---
    # Keep logic the same
    def handle_new_transfers(self, T, all_devices):
         # Start Inbound (Home -> Device) if possible
        if not self.is_inbound_transferring and self.inbound_queue:
            item = self.inbound_queue[0]
            chunk_id, layer_id, is_grad, is_context, target_idx, duration = item
            storage_key = (-1, layer_id, is_grad) if chunk_id == -1 else (chunk_id, layer_id, False)

            if storage_key in self.home_storage:
                self.is_inbound_transferring = True
                self.cur_inbound_start_time = T
                self.cur_inbound_duration = duration
                self.cur_inbound_details = item[:4]
            else:
                 # print(f"T={T}, Dev {self.device_id}: ERROR - Inbound request {item[:4]} key {storage_key} not in storage.")
                 self.inbound_queue.popleft() # Remove bad item

        # Start Outbound (Device -> Home) if possible
        if not self.is_outbound_transferring and self.outbound_queue:
            item = self.outbound_queue[0]
            chunk_id, layer_id, is_grad, is_only_context, duration = item
            self.is_outbound_transferring = True
            self.cur_outbound_start_time = T
            self.cur_outbound_duration = duration
            self.cur_outbound_details = item[:4]

        # Start Peer Transfer if possible
        if not self.is_peer_transferring and self.peer_transfer_queue:
            item = self.peer_transfer_queue[0]
            peer_id, cid, lid, is_grad_trans, duration = item

            peer_can_receive = False
            if 0 <= peer_id < len(all_devices):
                peer_dev = all_devices[peer_id]
                target_buffer = None
                target_empty_slot_attr = None
                # Determine TARGET buffer based on data type
                if lid == self.total_layers: # Grad FROM Head arriving
                    target_buffer = peer_dev.head_output_transitions_buffer # Peer receives in its head *output* buffer (confusing name maybe?)
                    target_empty_slot_attr = 'head_output_transitions_empty_slot_ind' # Check peer's corresponding index
                elif lid == self.total_layers - 1 and not is_grad_trans: # Fwd TO Head arriving
                    target_buffer = peer_dev.head_input_transitions_buffer # Peer receives in its head *input* buffer
                    target_empty_slot_attr = 'head_input_transitions_empty_slot_ind'
                else: # Regular transition arriving
                    target_buffer = peer_dev.transitions_inbound_buffer
                    target_empty_slot_attr = 'transitions_inbound_empty_slot_ind'

                if target_buffer is not None and hasattr(peer_dev, target_empty_slot_attr):
                    empty_slot_index = getattr(peer_dev, target_empty_slot_attr)
                    if empty_slot_index is not None and 0 <= empty_slot_index < len(target_buffer):
                        target_buffer[empty_slot_index] = (-2, cid, lid, is_grad_trans) # Mark peer slot receiving
                        setattr(peer_dev, target_empty_slot_attr, self._find_first_free_idx(target_buffer)) # Update peer's next slot
                        peer_can_receive = True

            if peer_can_receive:
                self.is_peer_transferring = True
                self.cur_peer_transfer_start_time = T
                self.cur_peer_transfer_duration = 0 if self.total_devices == 1 else duration
                self.cur_peer_transfer_details = (peer_id, cid, lid, is_grad_trans)


    def get_serializable_state(self):
        # Return a dictionary containing all internal state needed to resume
        # Convert deques to lists for JSON compatibility
        return {
            "device_id": self.device_id,
            "device_has_started": self.device_has_started,
            "device_start_time": self.device_start_time,
            "device_has_finished": self.device_has_finished,
            "device_finish_time": self.device_finish_time,
            # Buffers (save their content directly)
            "cur_weights": list(self.cur_weights), # Save list copy
            "activations_buffer": list(self.activations_buffer),
            "final_dev_activation_stack": list(self.final_dev_activation_stack),
            "transitions_inbound_buffer": list(self.transitions_inbound_buffer),
            "transitions_outbound_buffer": list(self.transitions_outbound_buffer),
            "head_input_transitions_buffer": list(self.head_input_transitions_buffer) if self.head_input_transitions_buffer is not None else None,
            "head_output_transitions_buffer": list(self.head_output_transitions_buffer) if self.head_output_transitions_buffer is not None else None,
            "context_buffer": list(self.context_buffer),
            # Pointers and indices
            "cur_weight_write_ptr": self.cur_weight_write_ptr,
            "activations_write_ptr": self.activations_write_ptr,
            "activations_empty_slot_ind": self.activations_empty_slot_ind,
            "transitions_inbound_empty_slot_ind": self.transitions_inbound_empty_slot_ind,
            "transitions_outbound_empty_slot_ind": self.transitions_outbound_empty_slot_ind,
            "head_input_transitions_empty_slot_ind": self.head_input_transitions_empty_slot_ind,
            "head_output_transitions_empty_slot_ind": self.head_output_transitions_empty_slot_ind,
            # Queues (convert deques to lists)
            "fwd_computation_queue": list(self.fwd_computation_queue),
            "bwd_computation_queue": list(self.bwd_computation_queue),
            "current_task": self.current_task,
            "current_task_dir": self.current_task_dir,
            "outbound_queue": list(self.outbound_queue),
            "inbound_queue": list(self.inbound_queue),
            "peer_transfer_queue": list(self.peer_transfer_queue),
            # Activity state
            "is_computing": self.is_computing,
            "is_stalled": self.is_stalled,
            "stall_start_time": self.stall_start_time,
            "cur_computation_start_time": self.cur_computation_start_time,
            "cur_computation_duration": self.cur_computation_duration,
            "current_computation_type": self.current_computation_type,
            "current_computation_layer_id": self.current_computation_layer_id,
            "current_computation_chunk_id": self.current_computation_chunk_id,
            "is_outbound_transferring": self.is_outbound_transferring,
            "cur_outbound_start_time": self.cur_outbound_start_time,
            "cur_outbound_duration" : self.cur_outbound_duration,
            "cur_outbound_details": self.cur_outbound_details,
            "is_inbound_transferring": self.is_inbound_transferring,
            "cur_inbound_start_time": self.cur_inbound_start_time,
            "cur_inbound_duration": self.cur_inbound_duration,
            "cur_inbound_details": self.cur_inbound_details,
            "is_peer_transferring": self.is_peer_transferring,
            "cur_peer_transfer_start_time": self.cur_peer_transfer_start_time,
            "cur_peer_transfer_duration": self.cur_peer_transfer_duration,
            "cur_peer_transfer_details": self.cur_peer_transfer_details,
            # Other state
            "computing_status_text": self.computing_status_text,
            "stall_reason": self.stall_reason,
            "cur_fwd_computation_num": self.cur_fwd_computation_num,
            "activations_stack": list(self.activations_stack), # Save stack
            "activations_stack_next_ind": self.activations_stack_next_ind,
            "next_weight_prefetch_layer_id": self.next_weight_prefetch_layer_id,
            "next_bwd_weight_prefetch_layer_id": self.next_bwd_weight_prefetch_layer_id,
            "last_block_layer_id": self.last_block_layer_id,
            "head_final_chunk_id": self.head_final_chunk_id,
            "home_storage": self.home_storage,
            "first_train_chunk": self.final_train_chunk
        }

    def load_from_serializable_state(self, dev_state):
        # Load state from dictionary, restoring internal attributes
        # Be careful with types and default values if keys are missing
        # This needs careful implementation matching get_serializable_state
        self.device_id = dev_state.get("device_id", self.device_id) # ID shouldn't change
        self.device_has_started = dev_state.get("device_has_started", False)
        self.device_start_time = dev_state.get("device_start_time", 0)
        self.device_has_finished = dev_state.get("device_has_finished", False)
        self.device_finish_time = dev_state.get("device_finish_time", 0)
        # Buffers (restore lists)
        self.cur_weights = list(dev_state.get("cur_weights", [-1] * self.layer_capacity))
        self.activations_buffer = list(dev_state.get("activations_buffer", [-1] * self.activations_capacity))
        self.final_dev_activation_stack = set(dev_state.get("final_dev_activation_stack", []))
        self.transitions_inbound_buffer = list(dev_state.get("transitions_inbound_buffer", [-1] * self.transitions_capacity))
        self.transitions_outbound_buffer = list(dev_state.get("transitions_outbound_buffer", [-1] * self.transitions_capacity))
        self.head_input_transitions_buffer = list(dev_state.get("head_input_transitions_buffer")) if dev_state.get("head_input_transitions_buffer") is not None else None
        self.head_output_transitions_buffer = list(dev_state.get("head_output_transitions_buffer")) if dev_state.get("head_output_transitions_buffer") is not None else None
        self.context_buffer = list(dev_state.get("context_buffer", [-1] * self.total_chunks))
        # Pointers and indices
        self.cur_weight_write_ptr = dev_state.get("cur_weight_write_ptr", 0)
        self.activations_write_ptr = dev_state.get("activations_write_ptr", 0)
        self.activations_empty_slot_ind = dev_state.get("activations_empty_slot_ind", 0 if self.activations_capacity > 0 else None)
        self.transitions_inbound_empty_slot_ind = dev_state.get("transitions_inbound_empty_slot_ind", 0 if self.transitions_capacity > 0 else None)
        self.transitions_outbound_empty_slot_ind = dev_state.get("transitions_outbound_empty_slot_ind", 0 if self.transitions_capacity > 0 else None)
        self.head_input_transitions_empty_slot_ind = dev_state.get("head_input_transitions_empty_slot_ind", 0 if self.head_input_transitions_buffer is not None else None)
        self.head_output_transitions_empty_slot_ind = dev_state.get("head_output_transitions_empty_slot_ind", 0 if self.head_output_transitions_buffer is not None else None)
        # Queues (restore deques from lists)
        self.fwd_computation_queue = deque(dev_state.get("fwd_computation_queue", []))
        self.bwd_computation_queue = deque(dev_state.get("bwd_computation_queue", []))
        self.current_task = dev_state.get("current_task", None)
        self.current_task_dir = dev_state.get("current_task_dir", 0)
        
        self.outbound_queue = deque(dev_state.get("outbound_queue", []))
        self.inbound_queue = deque(dev_state.get("inbound_queue", []))
        self.peer_transfer_queue = deque(dev_state.get("peer_transfer_queue", []))
        # Activity state
        self.is_computing = dev_state.get("is_computing", False)
        self.is_stalled = dev_state.get("is_stalled", False)
        self.stall_start_time = dev_state.get("stall_start_time", 0)
        self.cur_computation_start_time = dev_state.get("cur_computation_start_time", 0)
        self.cur_computation_duration = dev_state.get("cur_computation_duration", 0)
        self.current_computation_type = dev_state.get("current_computation_type", None)
        self.current_computation_layer_id = dev_state.get("current_computation_layer_id", -1)
        self.current_computation_chunk_id = dev_state.get("current_computation_chunk_id", -1)
        self.is_outbound_transferring = dev_state.get("is_outbound_transferring", False)
        self.cur_outbound_start_time = dev_state.get("cur_outbound_start_time", 0)
        self.cur_outbound_duration = dev_state.get("cur_outbound_duration", 0)
        self.cur_outbound_details = dev_state.get("cur_outbound_details", None)
        self.is_inbound_transferring = dev_state.get("is_inbound_transferring", False)
        self.cur_inbound_start_time = dev_state.get("cur_inbound_start_time", 0)
        self.cur_inbound_duration = dev_state.get("cur_inbound_duration", 0)
        self.cur_inbound_details = dev_state.get("cur_inbound_details", None)
        self.is_peer_transferring = dev_state.get("is_peer_transferring", False)
        self.cur_peer_transfer_start_time = dev_state.get("cur_peer_transfer_start_time", 0)
        self.cur_peer_transfer_duration = dev_state.get("cur_peer_transfer_duration", 0)
        self.cur_peer_transfer_details = dev_state.get("cur_peer_transfer_details", None)
        # Other state
        self.computing_status_text = dev_state.get("computing_status_text", "Idle")
        self.stall_reason = dev_state.get("stall_reason", "")
        self.cur_fwd_computation_num = dev_state.get("cur_fwd_computation_num", 0)
        self.activations_stack = list(dev_state.get("activations_stack", [])) # Restore list
        self.activations_stack_next_ind = dev_state.get("activations_stack_next_ind", 0)
        self.next_weight_prefetch_layer_id = dev_state.get("next_weight_prefetch_layer_id", -1)
        self.next_bwd_weight_prefetch_layer_id = dev_state.get("next_bwd_weight_prefetch_layer_id", -1)
        self.head_final_chunk_id = dev_state.get("head_final_chunk_id", -1)
        self.last_block_layer_id = dev_state.get("last_block_layer_id", -1)
        self.home_storage = set(dev_state.get("home_storage", []))
        self.final_train_chunk = dev_state.get("first_train_chunk", 0)
        # Ensure buffer empty slot indices are recalculated based on loaded buffers
        self._recalculate_empty_slot_indices()

    def _recalculate_empty_slot_indices(self):
        """ Helper to find first empty slot after loading buffers """
        self.activations_empty_slot_ind = self._find_first_free_idx(self.activations_buffer)
        self.transitions_inbound_empty_slot_ind = self._find_first_free_idx(self.transitions_inbound_buffer)
        self.transitions_outbound_empty_slot_ind = self._find_first_free_idx(self.transitions_outbound_buffer)
        self.head_input_transitions_empty_slot_ind = self._find_first_free_idx(self.head_input_transitions_buffer) if self.head_input_transitions_buffer is not None else None
        self.head_output_transitions_empty_slot_ind = self._find_first_free_idx(self.head_output_transitions_buffer) if self.head_output_transitions_buffer is not None else None

# --- Simulation Runner Class ---
class SimulationRunner:
    def __init__(self, params):
        self.params = params
        self.TO_PRINT = False # Hardcode for now

        # --- Extract and Calculate Parameters (Keep logic) ---
        self.cycle_rate_micros = params.get('cycle_rate_micros', 1000)
        self.N = params.get('N', 16)
        self.num_rings = params.get('num_rings', 8)
        self.num_seqs = params.get('num_seqs', 100)
        self.seqlen = params.get('seqlen', 4) * (1 << 10)
        self.max_attended_tokens = params.get('max_attended_tokens', 4) * (1 << 10)
        self.min_chunk_size = params.get('min_chunk_size', 16384)
        self.bitwidth = params.get('bitwidth', 8)
        self.attn_bitwidth = params.get('attn_bitwidth', 16)
        self.head_bitwidth = params.get('head_bitwidth', 8)
        self.grad_bitwidth = params.get('grad_bitwidth', 8)
        self.total_layers = params.get('total_layers', 59) # Store non-head count
        self.vocab_size = params.get('vocab_size', 128290)
        self.model_dim = params.get('model_dim', 7168)
        self.kv_dim = params.get('kv_dim', 512) # Recalculate kv_dim based on this
        self.num_experts = params.get('num_experts', 256)
        self.active_experts = params.get('active_experts', 9)
        self.expert_dim = params.get('expert_dim', 2048)
        self.attn_type = params.get('attn_type', "Exact")
        self.max_device_memory_bytes = params.get('max_device_memory_bytes', 8 * (1 << 30))
        self.hardware_max_flops = params.get('hardware_max_flops', 1989 * 1e12)
        self.hardware_mem_bw_bytes_sec = params.get('hardware_mem_bw_bytes_sec', 3350 * 1e9) # Typo fixed TB/s -> GB/s -> B/s
        self.matmul_efficiency = params.get('matmul_efficiency', 0.5)
        self.attn_fwd_efficiency = params.get('attn_fwd_efficiency', 0.3)
        self.attn_bwd_efficiency = params.get('attn_bwd_efficiency', 0.25)
        self.head_efficiency = params.get('head_efficiency', 0.8)
        self.home_bw_gbit_sec = params.get('home_bw_gbit_sec', 400)
        self.peer_bw_gbit_sec = params.get('peer_bw_gbit_sec', 100)
        self.chunk_type = params.get('chunk_type', "Equal Data")
        # --- Derived Parameters (Keep logic) ---
        self.dtype_bytes = self.bitwidth / 8.0 if self.bitwidth > 0 else 2.0

        self.hw_compute_bound = self.hardware_max_flops / self.hardware_mem_bw_bytes_sec if self.hardware_mem_bw_bytes_sec > 0 else float('inf')
        
        #print(f"Hardware Compute Bound: {self.hw_compute_bound}")
        
        ## determine chunk size based on arithmetic intensity of expert

        ## avg. expert has M value of: (active_experts * chunk_size) / (num_experts)
        ## and K value of: model_dim
        ## and N value of: expert_dim

        ## matmul arithmetic intensity is then given by:
        ## matmul_arith = (2 * M * K * N) / (((M * K) + (M * N) + (K * N)))

        ## we want matmul_arith >= hw_compute_bound
        ## solve for M:
        ## M >= (hw_compute_bound * K * N)) / (2 * K * N - hw_compute_bound * (K + N))

        ## Now let's solve for chunk_size:
        ## chunk_size >= (num_experts * hw_compute_bound * model_dim * expert_dim)) / (active_experts * (2 * model_dim * expert_dim - hw_compute_bound * (model_dim + expert_dim))

        ## Let's round up to the nearest multiple of 256
        comp_bound_min_chunk_size = math.ceil((self.num_experts * (self.hw_compute_bound * self.model_dim * self.expert_dim)) / (self.active_experts * (2 * self.model_dim * self.expert_dim - self.hw_compute_bound * (self.model_dim + self.expert_dim))))

        #print(f"Based on hw compute bound of {self.hw_compute_bound} FLOS/byte, along with FFN dims of {self.model_dim} x {self.expert_dim}, the base chunk size is {chunk_size_base}")
        if comp_bound_min_chunk_size < self.min_chunk_size:
            #print(f"Min chunk size is {self.min_chunk_size}, so using that instead...")
            comp_bound_min_chunk_size = self.min_chunk_size

        self.chunk_size = determine_chunk_size(self.seqlen, comp_bound_min_chunk_size)

        self.chunks_per_seq = math.ceil(self.seqlen / self.chunk_size) if self.chunk_size > 0 else 0
        self.seqs_per_chunk = max(1, int(self.chunk_size / self.seqlen))


        if (self.seqs_per_chunk == 1):
            self.total_chunks = int(self.chunks_per_seq * self.num_seqs)
        else:
            self.total_chunks = math.ceil(self.num_seqs / self.seqs_per_chunk)

            
        
                
        self.attn_block_size_bytes = self.dtype_bytes * (self.model_dim + (2 * self.model_dim * self.model_dim + 2 * self.model_dim * self.kv_dim))
        self.ffn_block_size_bytes = self.dtype_bytes * (self.model_dim + (3 * self.model_dim * self.expert_dim * self.num_experts))
        self.layer_size_bytes = self.attn_block_size_bytes + self.ffn_block_size_bytes

        self.grad_layer_dtype_bytes = self.grad_bitwidth / 8

        self.grad_layer_size_bytes = self.layer_size_bytes * (self.grad_layer_dtype_bytes / self.dtype_bytes)

        # 4 * for input/query/attn output/proj_output, 2 * for keys/values

        self.attention_dtype_bytes = self.attn_bitwidth / 8

        ## store Q, K, V, and attn_out in bf16/fp16
        self.attn_act_inps = self.attention_dtype_bytes * ((self.model_dim * self.chunk_size) + 2 * (self.kv_dim * self.chunk_size))

        self.attn_out = self.attention_dtype_bytes * ((self.model_dim * self.chunk_size))

        ## store result of matmul of attn out in set precision
        self.attn_out_proj = self.dtype_bytes * (self.model_dim * self.chunk_size)

        self.attn_activation_bytes = self.attn_act_inps + self.attn_out + self.attn_out_proj
        self.ffn_activation_bytes = self.dtype_bytes * (self.active_experts * (2 * self.chunk_size * self.expert_dim))


        self.activation_size_bytes = self.attn_activation_bytes + self.ffn_activation_bytes

        self.chunk_context_size_bytes = self.attention_dtype_bytes * 2 *(self.chunk_size * self.kv_dim)
        
        self.output_size_bytes = self.dtype_bytes * (self.model_dim * self.chunk_size)
        

        
        self.head_dtype_bytes = self.head_bitwidth / 8

        self.head_layer_size_bytes = self.head_dtype_bytes * (self.model_dim + (self.vocab_size * self.model_dim))

        self.per_layer_full_context_size = self.chunks_per_seq * self.chunk_context_size_bytes

           # Estimate transition/head buffer sizes based on chunks/N
        self.transitions_capacity = self.total_chunks
        # Head needs potentially more space for inputs/outputs across chunks
        self.head_transitions_capacity = max(self.transitions_capacity, self.total_chunks)

        transition_dev_mem = self.transitions_capacity * self.output_size_bytes # Use larger head capacity estimate

        ## head transitions capacity applies to device holding last layer and head
        special_transition_dev_mem = (self.head_transitions_capacity + self.transitions_capacity) * self.output_size_bytes

        # --- Determine Capacities (Keep simplified logic) ---
        self.layer_capacity = 2 if self.N <= self.total_layers else 1 # Base on non-head layers
        self.grad_layer_capacity = 2 if self.N <= self.total_layers else 1

        self.context_buffer_capacity = 2 # one for working forwards, one for recalling
        self.grad_context_buffer_capacity = 1
        self.per_layer_grad_context_size = self.chunks_per_seq * self.chunk_context_size_bytes

        base_dev_mem = self.layer_capacity * self.layer_size_bytes
        base_dev_mem += self.grad_layer_capacity * self.layer_size_bytes
        base_dev_mem += self.context_buffer_capacity * self.per_layer_full_context_size
        base_dev_mem += self.grad_context_buffer_capacity * self.per_layer_grad_context_size
        base_dev_mem += 2 * self.transitions_capacity * self.output_size_bytes
        base_dev_mem += 2 * self.activation_size_bytes

        
        model_dev_mem = self.layer_capacity * self.layer_size_bytes
        model_grad_dev_mem = self.grad_layer_capacity * self.layer_size_bytes
        context_dev_mem = (self.context_buffer_capacity) * self.per_layer_full_context_size
        grad_context_dev_mem = (self.grad_context_buffer_capacity) * self.per_layer_grad_context_size
        min_activations_dev_mem = 2 * self.activation_size_bytes

        error_string = f"Currently only supports:\n\nLayer (& Grad) Capacity = 2 ({(model_dev_mem + model_grad_dev_mem) / (1 << 30):.3f} GB)\n\nContext Buffer Capacity = 1 ({(context_dev_mem) / (1 << 30):.3f} GB)\nGrad Context Buffer Capacity = 1 ({(grad_context_dev_mem) / (1 << 30):.3f} GB)\n\nTransition Capacity = Num Devices ({(transition_dev_mem) / (1 << 30):.3f} GB)\n\nActivation Capacity >= 2 ({(min_activations_dev_mem) / (1 << 30):.3f} GB)\n\n\nThis requires at least {base_dev_mem / (1 << 30):.2f} GB of memory, but only {self.max_device_memory_bytes / (1 << 30):.2f} GB is available.\nMore generality coming soon...\n\nTry a Smaller Model or a Shorter Sequence (if Large Context Buffer).\n\n\n\nCannot run simulation with current configuration :("
        
        remain_dev_mem = self.max_device_memory_bytes
        remain_dev_mem -= model_dev_mem
        if remain_dev_mem < 0: 
            raise ValueError(f"\n\nError: 1st/6 Dev Mem Check.\nNot enough Dev Memory to hold Model Layers.\n\n\n{error_string}")


        remain_dev_mem -= model_grad_dev_mem
        if remain_dev_mem < 0: 
            raise ValueError(f"\n\nError: 2nd/6 Dev Mem Check.\nNot enough Dev Memory to hold Model Gradients.\n\n\n{error_string}")


        
        remain_dev_mem -= context_dev_mem
        if remain_dev_mem < 0: 
            raise ValueError(f"\n\nError: 3rd/6 Dev Mem Check.\nNot enough Dev Memory to hold Context Buffer.\n\n\n{error_string}")

        
        remain_dev_mem -= grad_context_dev_mem
        if remain_dev_mem < 0: 
            raise ValueError(f"\n\nError: 4th/6 Dev Mem Check.\nNot enough Dev Memory to hold Grad Context Buffer.\n\n\n{error_string}")

        remain_dev_mem -= transition_dev_mem
        if remain_dev_mem < 0: 
            raise ValueError(f"\n\nError: 5th/6 Dev Mem Check.\nNot enough Dev Memory to hold Transitions.\n\n\n{error_string}")


        base_activations_capacity = int(remain_dev_mem // self.activation_size_bytes) if self.activation_size_bytes > 0 else 0
        
        if base_activations_capacity < 2:
            raise ValueError(f"\n\nError: 6th/6 Dev Mem Check.\nNot enough Dev Memory to hold >= 2 Activations.\n\n\n{error_string}")

        ## max home layers is
        # 
        max_per_home_layers_base = math.ceil(self.total_layers / self.N)
        self.max_activations_capacity = (max_per_home_layers_base * self.total_chunks) + 1
        

        self.activations_capacity = int(min(base_activations_capacity, self.max_activations_capacity))
        
        remain_dev_mem -= self.activations_capacity * self.activation_size_bytes
        # --- Calculate Compute Times (Keep logic) ---
        
        ## smaller looks smoother and is more accurate, but takes longer to watch
        ## and becomes issue over network, working with 50-100 micros is good locally,
        ## (where it is smoother and can much faster without dealing with net latency)
        ## but using longer to look better for other clients
        self.cycles_per_second = 1e6 / self.cycle_rate_micros

        self.flops_per_attn_chunk_mult = 2 * self.chunk_size * self.model_dim
        self.base_flops_per_layer_matmul = 2 * self.chunk_size * (
             (2 * self.model_dim * self.model_dim) # Q, K, V, O projections
             + (2 * self.model_dim * self.kv_dim) # KV specific? Assume covered in QKVO for simplicity now
             + self.active_experts * (3 * self.model_dim * self.expert_dim) # FFN
         )

        self.computation_times_sec = {}
        self.computation_times_frames = {}
        self.computation_times_sec_bwd = {}
        self.computation_times_frames_bwd = {}
        self.computation_times_frames_last_block = {}
        self.total_fwd_attn_flops = 0
        self.total_fwd_matmul_flops = 0
        self.total_bwd_attn_flops = 0
        self.total_bwd_matmul_flops = 0
        self.total_fwd_flops = 0
        self.total_bwd_flops = 0
        self.total_head_flops = 0
        self.total_attn_flops = 0
        self.total_matmul_flops = 0
        self.total_compute_cycles = 0

        self.total_fwd_attn_pct = 0
        self.total_fwd_matmul_pct = 0
        self.total_bwd_attn_pct = 0
        self.total_bwd_matmul_pct = 0

        safe_flops = self.hardware_max_flops if self.hardware_max_flops > 0 else 1.0
        safe_matmul_eff = self.matmul_efficiency if self.matmul_efficiency > 0 else 1.0
        safe_attn_fwd_eff = self.attn_fwd_efficiency if self.attn_fwd_efficiency > 0 else 1.0
        safe_attn_bwd_eff = self.attn_bwd_efficiency if self.attn_bwd_efficiency > 0 else 1.0
        safe_head_eff = self.head_efficiency if self.head_efficiency > 0 else 1.0
          # Head FLOPS: fwd + bwd x + bwd w (assuming bwd similar to fwd)
        head_flops = 3 * (2 * self.chunk_size * self.model_dim * self.vocab_size)

        self.headTimeSec = head_flops / (safe_flops * safe_head_eff)
        
        self.headFrames = round(self.headTimeSec * self.cycles_per_second)

        # BwdW time based on matmul part only
        bwdW_flops = self.base_flops_per_layer_matmul
        bwdW_time = bwdW_flops / (safe_flops * safe_matmul_eff)
        self.bwdWTimeSec = bwdW_flops / (safe_flops * safe_matmul_eff)
        self.true_bwdWCycles = self.bwdWTimeSec * self.cycles_per_second
        self.bwdWFrames = max(1,round(self.bwdWTimeSec * self.cycles_per_second))

        self.fwd_attn_time = 0
        self.fwd_matmul_time = 0
        self.head_time = 0
        self.bwd_x_attn_time = 0
        self.bwd_x_matmul_time = 0
        self.bwd_w_time = 0

        prev_seq_len = 0

        is_train_chunk = True

     

        self.final_train_chunk = 0

        if self.chunks_per_seq > 1:
            self.final_chunk_in_seq_tokens = self.seqlen - (self.chunks_per_seq - 1) * self.chunk_size
            self.final_chunk_in_seq_data_frac = self.final_chunk_in_seq_tokens / self.chunk_size
        else:
            self.final_chunk_in_seq_tokens = self.chunk_size
            self.final_chunk_in_seq_data_frac = 1.0
    
        cur_chunk_id = 0

        for seq_batch_id in range(0, self.num_seqs, self.seqs_per_chunk):
            cur_seq_len = 0
            prev_seq_len = 0
            for chunk in range(self.chunks_per_seq):
                cur_seq_len = prev_seq_len + self.chunk_size
                if (cur_seq_len > self.seqlen):
                    cur_seq_len = self.seqlen
                if cur_seq_len > self.max_attended_tokens:
                    cur_seq_len = self.max_attended_tokens

                ## full chunk at attending the current seequence
                if self.chunks_per_seq > 1:
                    attn_flops = self.flops_per_attn_chunk_mult * cur_seq_len
                ## we have 1/more complete sequences each of cur_seq_len
                else:
                    attn_flops = self.seqs_per_chunk * 2 * self.model_dim * cur_seq_len * cur_seq_len

                self.total_attn_flops += self.total_layers * attn_flops
            

                layer_flops_fwd = self.base_flops_per_layer_matmul + attn_flops


                self.total_matmul_flops += self.total_layers * self.base_flops_per_layer_matmul
                self.total_fwd_flops += self.total_layers * layer_flops_fwd
                self.total_fwd_attn_flops += self.total_layers * attn_flops
                self.total_fwd_matmul_flops += self.total_layers * self.base_flops_per_layer_matmul

                matmul_time = self.base_flops_per_layer_matmul / (safe_flops * safe_matmul_eff)
                attn_time = attn_flops / (safe_flops * safe_attn_fwd_eff)
                self.fwd_attn_time += self.total_layers * attn_time
                self.fwd_matmul_time += self.total_layers * matmul_time

                self.computation_times_sec[cur_chunk_id] = matmul_time + attn_time
                if cur_chunk_id == 0:
                    self.true_first_chunk_cycles = self.computation_times_sec[cur_chunk_id] * self.cycles_per_second
                self.computation_times_frames[cur_chunk_id] = max(1,round(self.computation_times_sec[cur_chunk_id] * self.cycles_per_second))
                
            
                self.computation_times_frames_last_block[cur_chunk_id] = self.computation_times_frames[cur_chunk_id]
                self.total_compute_cycles += self.total_layers * self.computation_times_frames[cur_chunk_id]

                ## BWD ...

                # BwdX FLOPS = Fwd Matmul + 2 * Fwd Attn
                bwd_x_flops = self.base_flops_per_layer_matmul + 2.5 * attn_flops
                bwd_w_flops = self.base_flops_per_layer_matmul # Re-calc for clarity

                self.total_bwd_flops += self.total_layers * (bwd_x_flops + bwd_w_flops)

                bwd_attn_time = (2.5 * attn_flops) / (safe_flops * safe_attn_bwd_eff)

                bwd_x_time = matmul_time + bwd_attn_time # Bwd Attn is 2x Fwd Attn

                self.bwd_x_attn_time += self.total_layers * bwd_attn_time
                self.bwd_x_matmul_time += self.total_layers * matmul_time

                self.bwd_w_time += self.total_layers * self.bwdWTimeSec

                self.computation_times_sec_bwd[cur_chunk_id] = bwd_x_time
                self.computation_times_frames_bwd[cur_chunk_id] = max(1,round(bwd_x_time * self.cycles_per_second))
                self.total_compute_cycles += self.total_layers * self.computation_times_frames_bwd[cur_chunk_id]
                self.total_compute_cycles += self.total_layers * self.bwdWFrames # Add BwdW cycles

                # Accumulate FLOP types for BWD
                self.total_attn_flops += self.total_layers * 2.5 * attn_flops # For BwdX
                    
                self.total_bwd_attn_flops += self.total_layers * 2.5 * attn_flops # For BwdX


                self.total_matmul_flops += self.total_layers * (self.base_flops_per_layer_matmul) # For BwdX Matmul part
                self.total_matmul_flops += self.total_layers * bwd_w_flops # For BwdW

                self.total_bwd_matmul_flops += self.total_layers * (self.base_flops_per_layer_matmul) # For BwdX Matmul part
                self.total_bwd_matmul_flops += self.total_layers * bwd_w_flops # For BwdW

                # Add Head compute cycles only once per training chunk
                self.total_compute_cycles += self.headFrames
                self.total_matmul_flops += head_flops

                self.total_head_flops += head_flops

                self.head_time += self.headTimeSec

                prev_seq_len = cur_seq_len
                cur_chunk_id += 1

        # --- Total FLOPS (Final Calculation) ---
        self.total_flops = self.total_fwd_flops + self.total_bwd_flops + self.total_head_flops

        self.total_fwd_attn_pct = (self.total_fwd_attn_flops / self.total_flops) * 100
        self.total_fwd_matmul_pct = (self.total_fwd_matmul_flops / self.total_flops) * 100
        self.total_bwd_attn_pct = (self.total_bwd_attn_flops / self.total_flops) * 100
        self.total_bwd_matmul_pct = (self.total_bwd_matmul_flops / self.total_flops) * 100

        self.bwd_x_time = self.bwd_x_attn_time + self.bwd_x_matmul_time
        
        self.total_time = self.fwd_attn_time + self.fwd_matmul_time + self.head_time + self.bwd_x_time + self.bwd_w_time

        self.total_fwd_time = self.fwd_attn_time + self.fwd_matmul_time
        self.total_fwd_time_pct = self.total_fwd_time / self.total_time * 100

        self.total_bwd_x_time_pct = (self.bwd_x_time / self.total_time) * 100

        self.total_bwd_w_time_pct = (self.bwd_w_time / self.total_time) * 100

        self.total_fwd_attn_time_pct = (self.fwd_attn_time / self.total_time) * 100
        self.total_fwd_matmul_time_pct = (self.fwd_matmul_time / self.total_time) * 100
        self.total_bwd_x_attn_time_pct = (self.bwd_x_attn_time / self.total_time) * 100
        self.total_bwd_x_matmul_time_pct = (self.bwd_x_matmul_time / self.total_time) * 100
        self.total_head_time_pct = (self.head_time / self.total_time) * 100


        # --- Calculate Transfer Times (Keep logic) ---
        safe_home_bw_bps = self.home_bw_gbit_sec * 1e9 if self.home_bw_gbit_sec > 0 else 1
        safe_peer_bw_bps = self.peer_bw_gbit_sec * 1e9 if self.peer_bw_gbit_sec > 0 else 1

        self.round_frames_to_zero_cutoff = 0.1



        layer_transfer_time_sec = (self.layer_size_bytes * 8) / safe_home_bw_bps
        self.layerTransferFrames = max(1,round(layer_transfer_time_sec * self.cycles_per_second))
        self.trueLayerTransferCycles = layer_transfer_time_sec * self.cycles_per_second

        grad_layer_transfer_time_sec = (self.grad_layer_size_bytes * 8) / safe_home_bw_bps
        self.gradLayerTransferFrames = max(1,round(grad_layer_transfer_time_sec * self.cycles_per_second))
        self.trueGradLayerTransferCycles = grad_layer_transfer_time_sec * self.cycles_per_second


        head_layer_transfer_time_sec = (self.head_layer_size_bytes * 8) / safe_home_bw_bps
        self.headTransferFrames = max(1,round(head_layer_transfer_time_sec * self.cycles_per_second))
        self.trueHeadTransferCycles = head_layer_transfer_time_sec * self.cycles_per_second
        activation_transfer_time_sec = (self.activation_size_bytes * 8) / safe_home_bw_bps
        
        self.savedActivationsFrames = max(1,round(activation_transfer_time_sec * self.cycles_per_second))
        self.trueSavedActivationsCycles = activation_transfer_time_sec * self.cycles_per_second
        chunk_context_transfer_time_sec = (self.chunk_context_size_bytes * 8) / safe_home_bw_bps

       
        self.trueContextTransferCycles = chunk_context_transfer_time_sec * self.cycles_per_second
        if self.trueContextTransferCycles < self.round_frames_to_zero_cutoff:
            self.contextTransferFrames = 0
        else:
            self.contextTransferFrames = max(1,round(chunk_context_transfer_time_sec * self.cycles_per_second))
        
        transition_transfer_time_sec = (self.output_size_bytes * 8) / safe_peer_bw_bps
        self.trueActivationTransitionCycles = transition_transfer_time_sec * self.cycles_per_second if self.N > 1 else 0
        if self.trueActivationTransitionCycles < self.round_frames_to_zero_cutoff:
            self.activationTransitionFrames = 0
        else:
            self.activationTransitionFrames = max(1,round(transition_transfer_time_sec * self.cycles_per_second))


        # --- Simulation State ---
        self.all_devices = {}
        self.current_frame_index = 0
        self.animation_paused = True
        self.simulation_complete = False
        self.completion_stats = {}
        self.target_cycle = None
        self.max_frames = params.get('max_frames', 1000000)


    # --- Legend Text Creation (Keep for stats calculation, text can be sent to frontend) ---
    def _create_memory_legend_text(self):
        # Keep calculation logic, return the text string
        train_model_size = (self.vocab_size * self.model_dim * self.dtype_bytes) + (self.layer_size_bytes * self.total_layers) + self.head_layer_size_bytes
        train_activation_size = (self.total_chunks * self.activation_size_bytes * self.total_layers) # Based on non-head layers
        train_context_size = self.chunks_per_seq * self.chunk_context_size_bytes * self.total_layers
        train_block_inputs = self.total_chunks * self.chunk_size * self.model_dim * self.total_layers * self.dtype_bytes
        train_attn_output_size = self.total_chunks * self.chunk_size * self.model_dim * self.total_layers * self.dtype_bytes
        train_remain_activation_size = train_activation_size - train_context_size - train_attn_output_size - train_block_inputs
        train_gradient_size = train_model_size # Approximation
        train_optimizer_state_size = (2 * train_model_size) # Approximation
        aggregate_memory_size = train_model_size  + train_gradient_size + train_optimizer_state_size + train_activation_size
        chunk_workspace_size = (self.chunk_size * self.expert_dim * self.active_experts * self.dtype_bytes) # Review if needed

        layer_allocation = (self.layer_capacity * self.layer_size_bytes)
        grad_layer_allocation = (self.grad_layer_capacity * self.layer_size_bytes)
        context_buffer_allocation = (self.context_buffer_capacity * self.per_layer_full_context_size)
        grad_context_buffer_allocation = (self.grad_context_buffer_capacity * self.per_layer_grad_context_size)
        activation_allocation = (self.activations_capacity * self.activation_size_bytes)
        transition_allocation = (self.transitions_capacity * self.output_size_bytes) # Use head cap estimate
        typical_device_memory_size = (layer_allocation + grad_layer_allocation +
                                      context_buffer_allocation + grad_context_buffer_allocation +
                                      activation_allocation + transition_allocation)

        per_home_layers_base = self.total_layers // self.N if self.N > 0 else self.total_layers

        typical_home_layer_sizes = per_home_layers_base * self.layer_size_bytes

        remain_layers = self.total_layers % self.N if self.N > 0 else 0
        head_id = self.total_layers % self.N if self.N > 0 else 0 # Head ID based on total layers *including* head

        home_0_num_blocks = per_home_layers_base + (1 if 0 < remain_layers else 0)
        home_0_layer_size = home_0_num_blocks * self.layer_size_bytes
        if 0 == head_id: # Device 0 has the head
             # It holds its regular non-head blocks + the head layer
             home_0_layer_size = (per_home_layers_base * self.layer_size_bytes) + self.head_layer_size_bytes
             # Need to adjust num_blocks calculation if device 0 has head AND remainder layers?
             # Simpler: just calculate size directly. Assume head replaces one non-head block conceptually for activation count?
             # Let's stick to the size calculation. Activation count needs care.
             # Assume head layer requires similar activation storage as a non-head for estimate.
             home_0_num_blocks_for_act_est = per_home_layers_base + (1 if 0 < remain_layers else 0) # Base estimate
        else:
             home_0_num_blocks_for_act_est = home_0_num_blocks # If Dev 0 doesn't have head

        total_activation_cnt_dev0 = int(self.total_chunks * home_0_num_blocks_for_act_est)
        home_activation_stack_0 = max(0, total_activation_cnt_dev0 - self.activations_capacity)
        typical_home_activation_size = home_activation_stack_0 * self.activation_size_bytes

        model_bits = "32"
        grad_bits = str(self.grad_bitwidth)
        home_grad_size = home_0_layer_size * (self.grad_bitwidth / home_0_layer_size)
        opt_bits = "16"

       

        if self.dtype_bytes == 1:
            ## store optimizer state and gradients in bf16, and model in fp16 in home.
            home_model_size = 4 * home_0_layer_size
            grad_bits = "8"
            home_opt_size = 2 * 2 * home_0_layer_size
            ind_ring_model_shard_size = home_model_size + home_grad_size + home_opt_size
        else:
            ## store model in fp32 in home...
            if self.dtype_bytes == 2:
                home_model_size = 2 * home_0_layer_size
                home_opt_size = 2 * home_0_layer_size
                ind_ring_model_shard_size = home_model_size + home_grad_size + home_opt_size
            else:
                home_model_size = home_0_layer_size
                home_opt_size = home_0_layer_size
                opt_bits = self.bitwidth
                ind_ring_model_shard_size = home_model_size + home_grad_size + home_opt_size
                

        typical_home_total_size = typical_home_activation_size + ind_ring_model_shard_size

        typical_home_total_size_with_shared_rings = typical_home_activation_size + (ind_ring_model_shard_size / self.num_rings)

        gb = (1 << 30)
        mb = (1 << 20)
        
        embed_params = self.model_dim * self.vocab_size
        attn_block_params = self.model_dim + 2 * (self.model_dim * self.model_dim) + 2 * (self.model_dim * self.kv_dim)
        ffn_block_params = self.model_dim + 3 * self.num_experts * (self.model_dim * self.expert_dim)
        head_params = self.model_dim + self.model_dim * self.vocab_size
        total_model_params = embed_params + self.total_layers * (attn_block_params + ffn_block_params) + head_params

        ffn_active_params =  self.model_dim + 3 * self.active_experts * (self.model_dim * self.expert_dim)
        total_active_params = embed_params + self.total_layers * (attn_block_params + ffn_active_params) + head_params

        if self.chunk_size > self.min_chunk_size:
            chunk_size_str = f" (Min to Reach Compute-Bound)\n"
        else:
            chunk_size_str = f""

        if self.chunks_per_seq > 1:
            chunk_seq_str = f"- Chunks Per Seq: {self.chunks_per_seq}\n"
        else:
            if self.seqs_per_chunk > 1:
                chunk_seq_str = f"- Seqs Per Chunk: {self.seqs_per_chunk}\n"
            else:
                chunk_seq_str = f"- Seqs == Chunks\n"

        if self.num_experts > 1 and (self.active_experts % 2 == 1):
            avg_tokens_per_expert = math.ceil(self.chunk_size * (self.active_experts - 1) / self.num_experts)
        else:
            avg_tokens_per_expert = math.ceil(self.chunk_size * self.active_experts / self.num_experts)

        if self.num_experts > 1:
            avg_tokens_per_expert_str = f" - Avg Tokens Per Expert: {avg_tokens_per_expert}\n"
        else:
            avg_tokens_per_expert_str = f""

        text = (
          f"# Model Parameters: {total_model_params / (1e9):0.2f} B\n"
          f"     # Active: {total_active_params / (1e9):0.2f} B\n\n"
          f"--- FULL MEMORY OVERVIEW ---\n\n"
          f"Model: \n"
          f"- Parameters: {train_model_size / (1 << 30):.2f} GB\n"
          f"- Gradients: {train_gradient_size / (1 << 30):.2f} GB\n"
          f"- Opt. State: {(2 * train_model_size) / (1 << 30):.2f} GB\n\n"
          f"Data (Generated from Fwd):\n"
          f"- Block Inputs ({self.bitwidth} bits): {(train_block_inputs) / (1 << 30):.2f} GB\n"
          f"- Seq. Context ({self.attn_bitwidth} bits): {(train_context_size) / (1 << 30):.2f} GB\n"
          f"- Attn Outputs ({self.attn_bitwidth} bits): {(train_attn_output_size) / (1 << 30):.2f} GB\n"
          f"- Other Activs ({self.bitwidth} bits): {(train_remain_activation_size) / (1 << 30):.2f} GB\n\n"
          f"TOTAL:\n"
          f"- Attn Critical: {(train_block_inputs + train_context_size + train_attn_output_size) / (1 << 30):.2f} GB\n"
          f"- All: {aggregate_memory_size / (1 << 30):.2f} GB\n\n\n"
          f"--- DATAFLOW CONFIG ---\n\n"
          f"Chunk Size: {self.chunk_size}\n"
          f"{avg_tokens_per_expert_str}"
          f"{chunk_size_str}"
          f"{chunk_seq_str}"
          f"- Total Chunks: {self.total_chunks}\n\n"
          f"* TOTAL PER-DEVICE MEMORY *\n"
          f" - {typical_device_memory_size / (1 << 30):.2f} GB\n\n\n"
          f"* TOTAL PER-HOME SPACE *\n"
          f" - {typical_home_total_size / (1 << 30):.2f} GB\n"
          f"**Amortizing Model Shard over {self.num_rings} Rings...\n"
          f"=> Per Home: {(typical_home_total_size_with_shared_rings) / (1 << 30):.2f} GB\n\n\n\n"
          f"--- MEMORY PARTITIONS ---\n\n"
          f"Device Partitions (Typical):\n"
          f" - Activation Cap.: {self.activations_capacity}\n"
          f"   {(self.activations_capacity * self.activation_size_bytes)/ (1 << 30):.3f} GB\n"
          f" - Layer Cap.: {self.layer_capacity}\n"
          f"   {(self.layer_capacity * self.layer_size_bytes)/ (1 << 30):.3f} GB\n"
          f" - Grad Layer Cap.: {self.grad_layer_capacity}\n"
          f"   {(self.grad_layer_capacity * self.grad_layer_size_bytes)/ (1 << 30):.3f} GB\n"
          f" - Ctx Buffer Cap.: {self.context_buffer_capacity}\n"
          f"   {(self.context_buffer_capacity * self.per_layer_full_context_size)/ (1 << 30):.3f} GB\n"
          f" - Grad Ctx Buffer Cap.: {self.grad_context_buffer_capacity}\n"
          f"   {(self.grad_context_buffer_capacity * self.per_layer_grad_context_size)/ (1 << 30):.3f} GB\n"
          f" - Trans. Cap.: {self.transitions_capacity}\n"
          f"   {(self.transitions_capacity * self.output_size_bytes)/ (1 << 30):.3f} GB\n\n"     
          f"Home Partitions (Typical):\n"
          f" - Home # Act. Saved ({self.bitwidth} bits): {home_activation_stack_0}\n"
          f"    {typical_home_activation_size / (1 << 30):.2f} GB\n"
          f" - Model ({model_bits} bits): {home_model_size / (1 << 30):.2f} GB\n"
          f"   - Amortizing over Rings: {(home_model_size / self.num_rings) / (1 << 30):.2f} GB\n"
          f" - Model Grads ({grad_bits} bits): {home_grad_size / (1 << 30):.2f} GB\n"
          f"   - Amortizing over Rings: {(home_grad_size / self.num_rings) / (1 << 30):.2f} GB\n"
          f" - Opt. State ({opt_bits} bits): {home_opt_size / (1 << 30):.2f} GB\n"
          f"   - Amortizing over Rings: {(home_opt_size / self.num_rings) / (1 << 30):.2f} GB\n"
          f"Chunk Memory Usage:\n"
          f" - Saved Act.: {(self.activation_size_bytes)/ 1e6:.2f} MB\n" 
          f" - Context: {(self.chunk_context_size_bytes)/ 1e6:.2f} MB\n"
          f" - Transition: {(self.output_size_bytes)/ 1e6:.2f} MB\n"
          f" - Workspace: {(chunk_workspace_size)/ 1e6:.2f} MB"
        )
            
        return textwrap.dedent(text)
    
    def format_time_from_sec(self, seconds):

        # Thresholds for conversion
        MINUTE = 60
        HOUR = 60 * MINUTE
        DAY = 24 * HOUR

        if seconds < 1:
            milliseconds = seconds * 1000
            if milliseconds < 0.01 and milliseconds != 0: # handle very small non-zero values
                return f"{milliseconds:.2e} Millis" # use scientific notation for very small ms
            return f"{milliseconds:.2f} Millis"
        elif seconds < MINUTE:
            return f"{seconds:.2f} Secs"
        elif seconds < HOUR:
            minutes = seconds / MINUTE
            return f"{minutes:.2f} Mins"
        elif seconds < DAY:
            hours = seconds / HOUR
            return f"{hours:.2f} Hrs"
        else:
            days = seconds / DAY
            return f"{days:.2f} Days"

    def _create_compute_legend_text(self):
        # Keep calculation logic, return the text string
        safe_total_flops = self.total_flops if self.total_flops > 0 else 1.0
        matmul_pct = 100 * (self.total_matmul_flops / safe_total_flops)
        attn_pct = 100 * (self.total_attn_flops / safe_total_flops)
        fwd_pct = 100 * self.total_fwd_flops / safe_total_flops
        head_pct = 100 * self.total_head_flops / safe_total_flops
        bwd_pct = 100 * self.total_bwd_flops / safe_total_flops

        min_runtime_cycles = self.total_compute_cycles / self.N if self.N > 0 else self.total_compute_cycles # Idealized lower bound
        max_throughput_tflops = 0
        if min_runtime_cycles > 0 and self.cycles_per_second > 0:
             max_throughput_tflops = (self.total_flops / (min_runtime_cycles / self.cycles_per_second)) / 1e12

        tflops = 1e12
        gbps = 1e9
        gb = (1 << 30)

        if self.trueContextTransferCycles < 1:
            contextTransferCycleText = f"{self.trueContextTransferCycles:.2f}; set to {self.contextTransferFrames}"
        else:
            contextTransferCycleText = f"{self.contextTransferFrames}"

        if self.trueActivationTransitionCycles < 1:
            blockTransitionCyclesText = f"{self.trueActivationTransitionCycles:.2f}; set to {self.activationTransitionFrames}"
        else:
            blockTransitionCyclesText = f"{self.activationTransitionFrames}"

        ## override for single device case
        if self.N == 1:
            blockTransitionCyclesText = f"N/A"

        lower_bound_cycles = math.ceil(self.total_compute_cycles / self.N)
        lower_bound_seconds = lower_bound_cycles / self.cycles_per_second

        lower_bounds_time_str = self.format_time_from_sec(lower_bound_seconds)

        self.total_time_str = self.format_time_from_sec(self.total_time)

        if self.chunks_per_seq > 1:
            first_chunk_text = f"- C0 => [0, {self.chunk_size})\n"
        else:
            if self.seqs_per_chunk > 1:
                first_chunk_text = f"- C0 => Seqs: [0, {self.seqs_per_chunk})\n"
            else:
                first_chunk_text = f"- C0 => [0, {self.seqlen})\n"


        first_seq_last_chunk_text = f"- C{self.chunks_per_seq-1} => [{(self.chunks_per_seq-1) * self.chunk_size}, {self.seqlen})\n"
        if self.chunks_per_seq > 1:
            first_seq_text = f"First Seq:\n{first_chunk_text}{first_seq_last_chunk_text}\n"
        else:
            first_seq_text = f"First Seq:\n{first_chunk_text}\n"

        if self.chunks_per_seq > 1:
            last_seq_first_chunk_text = f"- C{self.total_chunks-self.chunks_per_seq} => [{(self.num_seqs-1) * self.seqlen}, {(self.num_seqs - 1) * self.seqlen + self.chunk_size})\n"
        else:
            if self.seqs_per_chunk > 1:
                last_seq_first_chunk_text = f"- C{self.total_chunks-self.chunks_per_seq} => Seqs: [{(self.total_chunks - 1) * self.seqs_per_chunk}, {self.num_seqs})\n"
            else:
                last_seq_first_chunk_text = f"- C{self.total_chunks-self.chunks_per_seq} => [{(self.num_seqs-1) * self.seqlen}, {self.num_seqs * self.seqlen})\n"
    
        
        final_chunk_text = f" - C{self.total_chunks-1} => [{(self.num_seqs-1) * self.seqlen + self.chunk_size * (self.chunks_per_seq-1)}, {(self.num_seqs) * self.seqlen})\n"

        if self.num_seqs > 1:
            if self.chunks_per_seq > 1:
                last_seq_text = f"Last Seq:\n{last_seq_first_chunk_text}{final_chunk_text}\n"
            else:
                last_seq_text = f"Last Seq:\n{last_seq_first_chunk_text}\n"
        else:
            last_seq_text = f""

        if self.chunks_per_seq > 1:
            fwd_last_chunk_in_seq_text = f" - C{self.chunks_per_seq-1} => [{(self.num_seqs-1) * self.seqlen + self.chunk_size * (self.chunks_per_seq-1)}, {(self.num_seqs) * self.seqlen})\n"
            bwd_last_chunk_in_seq_text = f" - C{self.chunks_per_seq-1}: {self.computation_times_frames_bwd.get(self.chunks_per_seq-1,0)}\n"
        else:
            fwd_last_chunk_in_seq_text = f""
            bwd_last_chunk_in_seq_text = f""

        text = (
            f"--- DISCOVERED CONSTANTS ---\n\n"
            f"Compute Constants:\n"
            f" - Theo. Compute: {int(self.hardware_max_flops / 1e12)} TFLOPs\n"
            f" - Memory BW: {int(self.hardware_mem_bw_bytes_sec / 1e9)} GB/s\n"
            f" - Peak Matmul Eff.: {self.matmul_efficiency}\n"
            f" - Peak Attention Fwd Eff.: {self.attn_fwd_efficiency}\n"
            f" - Peak Attention Bwd Eff.: {self.attn_bwd_efficiency}\n"
            f" - Peak Head Eff.: {self.head_efficiency}\n\n"
            f"Communication Constants:\n"
            f" - D2H BW: {self.home_bw_gbit_sec} (Gb/s)\n"
            f" - P2P BW: {self.peer_bw_gbit_sec} (Gb/s)\n\n\n"
            f"--- FULL FLOP OVERVIEW ---\n\n"    
            f"Total FLOPs: {self.total_flops:.2e}\n"
            f" - FWD: {self.total_fwd_flops:.2e} ({fwd_pct:.1f}%)\n"
            f"   - Attn: {self.total_fwd_attn_pct:.1f}%\n"
            f"   - Matmul: {self.total_fwd_matmul_pct:.1f}%\n"
            f" - Head: {self.total_head_flops:.2e} ({head_pct:.1f}%)\n"
            f" - BWD: {self.total_bwd_flops:.2e} ({bwd_pct:.1f}%)\n"
            f"   - Attn: {self.total_bwd_attn_pct:.1f}%\n"
            f"   - Matmul: {self.total_bwd_matmul_pct:.1f}%\n\n\n"
            f"Total Serial Time: {self.total_time_str}\n"
            f" - FWD: {self.total_fwd_time_pct:.1f}%\n"
            f"  - Attn: {self.total_fwd_attn_time_pct:.1f}%\n"
            f"  - Matmul: {self.total_fwd_matmul_time_pct:.1f}%\n"
            f" - Head: {self.total_head_time_pct:.1f}%\n"
            f" - BWD X: {self.total_bwd_x_time_pct:.1f}%\n"
            f"  - Attn: {self.total_bwd_x_attn_time_pct:.1f}%\n"
            f"  - Matmul: {self.total_bwd_x_matmul_time_pct:.1f}%\n"
            f" - BWD W: {self.total_bwd_w_time_pct:.1f}%\n\n\n"
            f"--- DERIVED SIM CONFIG ---\n\n"
            f"Cycle Rate:\n"
            f" - {1e6/self.cycles_per_second:.1f} us/cycle\n\n"
            f"* RUNTIME LOWER-BOUND * \n"
            f" - {lower_bound_cycles} Cycles\n"
            f" - {lower_bounds_time_str}\n\n"
            f"* THROUGHPUT UPPER-BOUND * \n" 
            f" - {math.ceil(self.total_flops / (self.total_compute_cycles / self.cycles_per_second) / 1e12)} TFLOPS\n\n\n\n"
            f"--- COMPUTE CYCLES ---\n\n"
            f"{first_seq_text}"
            f"{last_seq_text}"
            f"Fwd:\n"
            f" - C0: {self.computation_times_frames.get(0,0)}\n"
            f"{fwd_last_chunk_in_seq_text}"
            f"Bwd X:\n"
            f" - C{self.final_train_chunk}: {self.computation_times_frames_bwd.get(self.final_train_chunk,0)}\n"
            f"{bwd_last_chunk_in_seq_text}"
            f"Bwd W:\n"
            f" - All: {self.bwdWFrames}\n"
            f"Head:\n"
            f" - All: {self.headFrames}\n\n"
            f"--- COMMUNICATION CYCLES ---\n\n"
            f"Layers\n"
            f" - Block ({self.bitwidth} bits): {self.layerTransferFrames}\n"
            f" - Block Grads ({self.grad_bitwidth} bits): {self.gradLayerTransferFrames}\n"
            f" - Head ({self.head_bitwidth} bits): {self.headTransferFrames}\n"
            f"Chunk Info\n"
            f" - Activation Save/Fetch: {self.savedActivationsFrames}\n"
            f" - Block Transitions: {blockTransitionCyclesText}\n"
            f" - Context Transfers: {contextTransferCycleText}"
        )

        return textwrap.dedent(text)


    def reset_simulation_state(self):
         """ Resets the simulation state and creates/resets device objects. """
         self.current_frame_index = 0
         self.animation_paused = True
         self.simulation_complete = False
         self.completion_stats = {}
         self.target_cycle = None

         self.all_devices = {} # Clear or create dict

         if self.N > 0:
             for i in range(self.N):
                 # Create NEW Device instances on reset
                 self.all_devices[i] = Device(
                     device_id=i,
                     # ... pass all necessary params ...
                     layer_capacity=self.layer_capacity,
                     activations_capacity=self.activations_capacity,
                     transitions_capacity=self.transitions_capacity,
                     head_transitions_capacity=self.head_transitions_capacity,
                     total_devices=self.N,
                     total_layers=self.total_layers, # Pass total including head
                     num_seqs=self.num_seqs,
                     seqlen=self.seqlen,
                     chunks_per_seq=self.chunks_per_seq,
                     seqs_per_chunk=self.seqs_per_chunk,
                     total_chunks=self.total_chunks,
                     final_chunk_in_seq_data_frac=self.final_chunk_in_seq_data_frac,
                     computation_times_frames=self.computation_times_frames,
                     computation_times_frames_last_block=self.computation_times_frames_last_block,
                     computation_times_frames_bwd=self.computation_times_frames_bwd,
                     headFrames=self.headFrames,
                     bwdWFrames=self.bwdWFrames,
                     layerTransferFrames=self.layerTransferFrames,
                     gradLayerTransferFrames=self.gradLayerTransferFrames,
                     headTransferFrames=self.headTransferFrames,
                     savedActivationsFrames=self.savedActivationsFrames,
                     activationTransitionFrames=self.activationTransitionFrames,
                     contextTransferFrames=self.contextTransferFrames
                 )
                 # Note: _initialize_tasks_and_state is called within Device.__init__
         
         self.total_tasks = 0
         self.total_computation_time = 0
         for i in range(self.N):
            self.total_tasks += len(self.all_devices[i].fwd_computation_queue)
            for fwd_task in self.all_devices[i].fwd_computation_queue:
                self.total_computation_time += fwd_task[-1]
            self.total_tasks += len(self.all_devices[i].bwd_computation_queue)
            for bwd_task in self.all_devices[i].bwd_computation_queue:
                self.total_computation_time += bwd_task[-1]
              
               
    def get_serializable_state(self):
        """ Returns ALL internal state needed to resume the simulation. """
        # Serialize runner's own state
        runner_state = {
            "params": self.params, # Include original params
            "current_frame_index": self.current_frame_index,
            "animation_paused": self.animation_paused,
            "simulation_complete": self.simulation_complete,
            "completion_stats": self.completion_stats,
            "target_cycle": self.target_cycle,
            "total_tasks": self.total_tasks, # Save calculated totals maybe?
            "total_computation_time": self.total_computation_time,
        }
        # Serialize state of each device
        devices_internal_state = {}
        if self.N > 0:
            for i, device in self.all_devices.items():
                devices_internal_state[i] = device.get_serializable_state()

        runner_state["all_devices_state"] = devices_internal_state
        return runner_state

    def load_from_serializable_state(self, full_state_dict):
        """ Restores the simulation runner from a complete state dictionary. """
        try:
            # --- Step 1: Reset state and CREATE default Device objects ---
            # This ensures self.all_devices exists and is populated based on self.N (from params)
            # It will temporarily reset frame index etc., but we restore them next.
            self.reset_simulation_state()

            # --- Step 2: Restore basic runner attributes from the loaded state dict ---
            # This overwrites the defaults set by reset_simulation_state
            self.current_frame_index = full_state_dict.get('current_frame_index', 0)
            self.animation_paused = full_state_dict.get('animation_paused', True)
            self.simulation_complete = full_state_dict.get('simulation_complete', False)
            self.completion_stats = full_state_dict.get('completion_stats', {})
            self.target_cycle = full_state_dict.get('target_cycle', None)
            self.total_tasks = full_state_dict.get('total_tasks', self.total_tasks) # Keep recalculated if not saved
            self.total_computation_time = full_state_dict.get('total_computation_time', self.total_computation_time) # Keep recalculated if not saved


            # --- Step 3: Restore individual device states ---
            devices_internal_state = full_state_dict.get("all_devices_state", {})
            if len(devices_internal_state) == self.N:
                # Now self.all_devices[i] exists because reset_simulation_state created it
                for i_str, device_state_dict in devices_internal_state.items():
                     i = int(i_str) # Dictionary keys might be strings after JSON conversion
                     if i in self.all_devices:
                         self.all_devices[i].load_from_serializable_state(device_state_dict)
                     else:
                         print(f"  - Warning: Device key {i} from saved state not found in newly created self.all_devices.")
            elif self.N > 0: # Only warn if N > 0 and counts mismatch
                 print(f"  - Warning: Mismatch N ({self.N}) and loaded device state count ({len(devices_internal_state)}). Device states not fully loaded.")
            else:
                 print(f"  - N=0, no device states to load.")


        except Exception as e:
            print(f"ERROR loading SimulationRunner state: {e}")
            traceback.print_exc()
            # Reset to initial state on error? Safest bet.
            print("Resetting state due to load error.")
            self.reset_simulation_state()

    def step(self):
        """ Advances the simulation by one time step (frame). Returns True if simulation should continue."""
        if self.simulation_complete or self.current_frame_index >= self.max_frames:
            self.animation_paused = True # Ensure pause if finished
            return False

        T = self.current_frame_index

        if self.target_cycle is not None and T == self.target_cycle:
            self.animation_paused = True
            self.target_cycle = None
            return False # Stop stepping

        # --- Simulation Logic Order (Keep) ---
        if self.N > 0:
            for i in range(self.N):
               to_repeat = True
               ## incase we completed a 0 cycle transfer...
               while (to_repeat):
                   to_repeat = self.all_devices[i].handle_completed_transfers(T, self.all_devices)
            newly_completed_this_cycle = 0
            all_devices_finished_compute = True
            for i in range(self.N):
                newly_completed_this_cycle += self.all_devices[i].handle_computation(T)
                if not self.all_devices[i].device_has_finished: all_devices_finished_compute = False
            for i in range(self.N): self.all_devices[i].handle_new_transfers(T, self.all_devices)
        else: # Handle N=0 case
            all_devices_finished_compute = True # No devices, so trivially finished

        # --- Check for Simulation Completion ---
        if all_devices_finished_compute and not self.simulation_complete:
            self.simulation_complete = True
            self.animation_paused = True
            self.completion_stats = self._calculate_completion_stats(T)
            #print(f"Simulation Complete at Cycle {T}")
            # print(self.completion_stats.get("text", "")) # Optional server log
            return False # Simulation just completed, stop stepping for now

        # --- Increment Frame Index ---
        if not self.animation_paused:
            self.current_frame_index += 1
            if self.current_frame_index >= self.max_frames:
                #print(f"Max frames ({self.max_frames}) reached, stopping.")
                self.animation_paused = True
                self.simulation_complete = True
                if not self.completion_stats:
                    self.completion_stats = self._calculate_completion_stats(T)
                    self.completion_stats["text"] += "\n(Stopped at Max Frames)"
                return False # Max frames reached, stop

        return not self.animation_paused # Return True if sim should continue

    def _calculate_completion_stats(self, final_cycle):
        # Keep logic the same
        if self.N == 0:
             return {"text": "Simulation Complete (N=0)\nFinal Cycle Count: 0", "final_cycle": 0, "achieved_throughput": 0}

        T = final_cycle
        start_bubble = sum(d.device_start_time for d in self.all_devices.values() if d.device_has_started)
        stop_bubble = sum(max(0, T - (d.device_finish_time if d.device_has_finished else T)) for d in self.all_devices.values())
        total_dev_time = T * self.N if T > 0 else 0
        steady_time = max(0, total_dev_time - stop_bubble - start_bubble)

        overall_eff = (self.total_computation_time / total_dev_time * 100) if total_dev_time > 0 else 0
        steady_eff = (self.total_computation_time / steady_time * 100) if steady_time > 0 else 0
        runtime_in_seconds = T / self.cycles_per_second if self.cycles_per_second > 0 else 0

        ideal_compute_cycles = self.total_compute_cycles / self.N if self.N > 0 else 0
        ideal_compute_time_sec = ideal_compute_cycles / self.cycles_per_second if self.cycles_per_second > 0 else 0
        total_throughput_upper_bound_tflops = (self.total_flops / ideal_compute_time_sec / 1e12) if ideal_compute_time_sec > 0 else 0
        achieved_throughput_tflops = (self.total_flops / runtime_in_seconds / 1e12) if runtime_in_seconds > 0 else 0

        completion_text = (
            f"Simulation Complete!\n\nFinal Cycle Count: {T}\nRuntime: {runtime_in_seconds:.3f} seconds\n\n"
            f"--- COMPUTE THROUGHPUT ---\n\n"
            f"  Ideal Upper-Bound: {math.ceil(total_throughput_upper_bound_tflops / self.N)} TFLOPS\n"
            f"  Achieved Throughput: {math.ceil(achieved_throughput_tflops / self.N)} TFLOPS\n\n\n"
            f"--- PIPELINE STATS --- \n\n"
            f"Raw Task Compute Cycles: {self.total_compute_cycles}\n"
            f"Total Occupied Cycles: {self.N * T}\n\n"
            f"Fill Bubble: {start_bubble} Total Cycles\n"
            f"Flush Bubble: {stop_bubble} Total Cycles\n\n\n"
            f"EFFICIENCY:\n"
            f"Steady-State % Active: {steady_eff:.2f}%\n"
            f"Overall % Active: {overall_eff:.2f}%\n\n"
        )
        return {
            "text": completion_text, "final_cycle": T, "runtime_sec": runtime_in_seconds,
            "overall_eff_pct": overall_eff, "steady_eff_pct": steady_eff,
            "achieved_throughput": achieved_throughput_tflops,
             # Include legend text here if frontend will display it
             "memory_legend": self._create_memory_legend_text(),
             "compute_legend": self._create_compute_legend_text()
        }

    # --- REMOVE PNG Generation ---
    # def get_frame_png_data(self): DELETE THIS METHOD
    #     pass # DELETE

    # --- Control Methods (Keep as is) ---
    def pause(self):
        self.animation_paused = True
        self.target_cycle = None
        print("Simulation paused by controller.")

    def play(self):
        if not self.simulation_complete and self.current_frame_index < self.max_frames:
            self.animation_paused = False
            self.target_cycle = None
            print("Simulation playing by controller.")
        else:
            print("Simulation cannot play (already complete or max frames).")

    def set_target_cycle(self, cycle):
        if cycle < 0: return
        cycle = min(cycle, self.max_frames - 1)

        print(f"Setting target cycle: {cycle}")
        needs_reset = False
        if self.simulation_complete or cycle <= self.current_frame_index:
            print("Target in past or simulation complete. Resetting.")
            self.reset_simulation_state()
            needs_reset = True

        self.target_cycle = cycle
        if self.target_cycle > self.current_frame_index or needs_reset:
            self.animation_paused = False
            print(f"Simulation running towards cycle {self.target_cycle}")
        elif self.target_cycle == self.current_frame_index and not needs_reset:
            self.animation_paused = True
            self.target_cycle = None
            print(f"Already at target cycle {self.current_frame_index}. Pausing.")

    def get_state_summary(self):
        """ Returns the general simulation state (pause, complete,  etc.). """
        return {
            "current_frame": self.current_frame_index,
            "is_paused": self.animation_paused,
            "is_complete": self.simulation_complete,
            "target_cycle": self.target_cycle,
            "max_frames": self.max_frames,
            "completion_stats": self.completion_stats if self.simulation_complete else {}
        }

    # --- NEW: get_render_state ---
    def get_render_state(self):
        """ Returns the detailed state needed for frontend rendering. """
        T = self.current_frame_index
        devices_state = []
        if self.N > 0:
            for i in range(self.N):
                dev = self.all_devices[i]
                dev_state = {"id": i}

                # Status and Stall
                if dev.device_has_finished: dev_state["status"] = "Finished"
                elif dev.is_stalled: dev_state["status"] = "Stalled"
                elif dev.is_computing: dev_state["status"] = "Computing"
                else: dev_state["status"] = "Idle"
                dev_state["status_text"] = dev.computing_status_text # Use the formatted text
                dev_state["stall_reason"] = dev.stall_reason if dev.is_stalled else None

                # Compute Progress
                if dev.is_computing and dev.cur_computation_duration > 0:
                    progress = min(1.0, max(0.0, (T - dev.cur_computation_start_time) / dev.cur_computation_duration))
                    dev_state["compute"] = {
                        "type": dev.current_computation_type,
                        "layer": dev.current_computation_layer_id,
                        "chunk": dev.current_computation_chunk_id,
                        "progress": progress,
                         # Map type to color name (can be used by frontend)
                         "color": { "Fwd": COLOR_COMPUTE_FWD, "Bwd X": COLOR_COMPUTE_BWD_X,
                                     "Bwd W": COLOR_COMPUTE_BWD_W, "Head": COLOR_COMPUTE_HEAD
                                  }.get(dev.current_computation_type, COLOR_COMPUTE_DEFAULT)
                    }
                else: dev_state["compute"] = None

                # Inbound Transfer Progress
                if dev.is_inbound_transferring and dev.cur_inbound_duration > 0 and dev.cur_inbound_details:
                    progress = min(1.0, max(0.0, (T - dev.cur_inbound_start_time) / dev.cur_inbound_duration))
                    cid, lid, isg, isctx = dev.cur_inbound_details
                    transfer_type = "Wgt" if cid == -1 else ("Ctx" if isctx else "Act")
                    color = COLOR_INBOUND_WEIGHT if transfer_type == "Wgt" else (COLOR_INBOUND_BWD_FETCHED_CTX if transfer_type == "Ctx" else COLOR_INBOUND_BWD_FETCHED_ACTIVATION)
                    label = f"{transfer_type}:\n{'Head' if lid == self.total_layers else f'L{lid}'}" if transfer_type=="Wgt" else f"{transfer_type}:\nC{cid},L{lid}"
                    dev_state["inbound"] = {
                        "type": transfer_type, "layer": lid, "chunk": cid if cid!=-1 else None,
                        "progress": progress, "color": color, "label": label
                    }
                else: dev_state["inbound"] = None

                # Outbound Transfer Progress
                if dev.is_outbound_transferring and dev.cur_outbound_duration > 0 and dev.cur_outbound_details:
                    progress = min(1.0, max(0.0, (T - dev.cur_outbound_start_time) / dev.cur_outbound_duration))
                    cid, lid, isg, isctx = dev.cur_outbound_details
                    transfer_type = "WgtGrad" if (cid == -1 and isg) else ("Ctx" if (cid >= 0 and isctx) else "Act")
                    color = COLOR_OUTBOUND_WGT_GRAD if transfer_type == "WgtGrad" else (COLOR_OUTBOUND_FWD_CTX if transfer_type == "Ctx" else COLOR_OUTBOUND_FWD_ACTIVATION)
                    label = f"Grad:\n{'Head' if lid == self.total_layers else f'L{lid}'}" if transfer_type=="WgtGrad" else f"{transfer_type}:\nC{cid},L{lid}"
                    dev_state["outbound"] = {
                         "type": transfer_type, "layer": lid, "chunk": cid if cid!=-1 else None,
                         "progress": progress, "color": color, "label": label
                    }
                else: dev_state["outbound"] = None

                 # Peer Transfer Progress
                if dev.is_peer_transferring and dev.cur_peer_transfer_duration > 0 and dev.cur_peer_transfer_details:
                    progress = min(1.0, max(0.0, (T - dev.cur_peer_transfer_start_time) / dev.cur_peer_transfer_duration))
                    peer_id, cid, lid, isg = dev.cur_peer_transfer_details
                    transfer_type = "Grad" if isg else "Out" # Gradient or Activation/Output
                    direction = -1 if isg else 1 # CW for Grad, CCW for Out
                    color = COLOR_RING_CW if isg else COLOR_RING_CCW
                    label = f"{transfer_type}:\nC{cid},{'Head' if lid==self.total_layers else f'L{lid}'}"
                    dev_state["peer"] = {
                         "type": transfer_type, "layer": lid, "chunk": cid, "target_peer": peer_id,
                         "progress": progress, "direction": direction, "color": color, "label": label
                    }
                else: dev_state["peer"] = None

                devices_state.append(dev_state)

        # Combine with overall state
        state = self.get_state_summary()
        state["devices"] = devices_state
        # Add static config only if needed (e.g., on first call) - simpler to send always?
        # state["config"] = {"N": self.N, "total_layers": self.total_layers} # Example
        return state
