# run.py (Corrected for Full State Serialization via Session)
from flask import Flask, render_template, request, jsonify, session # Import session
from flask_session import Session # Import Session extension
import time
import sys
import os # For secret key
import traceback # Keep for detailed error logging

try:
    # Ensure these methods exist in SimulationRunner now!
    from simulation import SimulationRunner
    # Check if methods exist (optional defensive check)
    if not hasattr(SimulationRunner, 'get_serializable_state') or \
       not hasattr(SimulationRunner, 'load_from_serializable_state') or \
       not hasattr(SimulationRunner, 'get_render_state'):
           raise AttributeError("SimulationRunner missing required state methods.")

except ImportError:
    print("Error: simulation.py not found or contains import errors.")
    sys.exit(1)
except AttributeError as e:
    print(f"Error: {e}")
    print("Please ensure get_serializable_state, load_from_serializable_state, and get_render_state methods are defined in SimulationRunner class in simulation.py.")
    sys.exit(1)

app = Flask(__name__)

# --- Session Configuration ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-replace-in-prod-AGAIN') # CHANGE THIS KEY!
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = True # Assuming HTTPS via Nginx

server_session = Session(app)
# --- End Session Configuration ---

def parse_parameters(form_data):
    """ Safely parses form data into simulation parameters. (Unchanged) """
    params = {}
    # ... (keep existing parsing logic - ensure it matches SimulationRunner needs) ...
    params['N'] = form_data.get('N', default=8, type=int)
    params['seqlen'] = form_data.get('seqlen', default=32, type=int) # K tokens
    params['train_token_ratio'] = form_data.get('train_token_ratio', default=1, type=float)
    params['min_chunk_size'] = form_data.get('min_chunk_size', default=1536, type=int)
    params['chunk_type'] = form_data.get('chunk_type', default="Equal Data", type=str)
    params['bitwidth'] = form_data.get('bitwidth', default=16, type=int)
    params['total_layers'] = form_data.get('total_layers', default=64, type=int) # Non-head
    params['vocab_size'] = form_data.get('vocab_size', default=151646, type=int)
    params['model_dim'] = form_data.get('model_dim', default=5120, type=int)
    params['kv_dim'] = form_data.get('kv_dim', default=640, type=int) # REMOVE this, use factor
    params['num_experts'] = form_data.get('num_experts', default=1, type=int)
    params['active_experts'] = form_data.get('active_experts', default=1, type=int)
    params['expert_dim'] = form_data.get('expert_dim', default=27648, type=int)
    params['attn_type'] = form_data.get('attn_type', default="Exact", type=str)
    params['max_device_memory_bytes'] = form_data.get('max_device_memory_gb', default=80, type=float) * (1 << 30)
    params['hardware_max_flops'] = form_data.get('hardware_max_tflops', default=989, type=float) * 1e12
    params['hardware_mem_bw_bytes_sec'] = form_data.get('hardware_mem_bw_gbs', default=3350, type=float) * (1 << 30) # Input GB/s
    params['matmul_efficiency'] = form_data.get('matmul_efficiency', default=0.7, type=float)
    params['attn_efficiency'] = form_data.get('attn_efficiency', default=0.55, type=float)
    params['home_bw_gbit_sec'] = form_data.get('home_bw_gbit_sec', default=400, type=int)
    params['peer_bw_gbit_sec'] = form_data.get('peer_bw_gbit_sec', default=100, type=int)
    if params['N'] <= 0: raise ValueError("N must be positive.")
    return params

# --- Helper Function to Load/Recreate Runner from Session ---
def get_runner_from_session():
    """Retrieves FULL state from session, returns initialized SimulationRunner."""
    if not session.get('simulation_active'):
        print(f"Session {session.sid if session.sid else 'None'}: No active simulation found in session.")
        return None

    # *** CORRECTED: Load the FULL state dictionary ***
    full_state = session.get('simulation_full_state')

    # Check if state exists and seems valid (has 'params' at least)
    if not full_state or not isinstance(full_state, dict) or 'params' not in full_state:
        print(f"Session {session.sid if session.sid else 'None'}: Invalid or missing 'simulation_full_state' in session.")
        session.clear() # Clear corrupted session state
        session.modified = True
        return None

    try:
        # Recreate the runner instance with original parameters from state
        runner = SimulationRunner(full_state['params'])
        # *** CORRECTED: Load using the correct method and full state ***
        runner.load_from_serializable_state(full_state)
        print(f"Session {session.sid}: Successfully loaded runner state (Frame: {runner.current_frame_index}, Paused: {runner.animation_paused}).")
        return runner
    except Exception as e:
        print(f"Session {session.sid}: Error recreating SimulationRunner from session state: {e}")
        traceback.print_exc()
        session.clear() # Clear potentially problematic state
        session.modified = True
        return None

@app.route('/')
def index():
    """ Serves the main HTML page. """
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """ Initializes or restarts the simulation FOR THIS SESSION using full state. """
    print(f"Received /start_simulation request for session ID: {session.sid if session.sid else 'NEW'}")
    try:
        params = parse_parameters(request.form)
        print(f"Parsed parameters: {params}")

        # Create runner instance, reset it, get initial states
        try:
            runner = SimulationRunner(params)
            runner.reset_simulation_state() # Explicitly reset
            # *** CORRECTED: Get the FULL state for saving ***
            initial_full_state = runner.get_serializable_state()
            # *** Get the RENDER state separately for returning ***
            initial_render_state = runner.get_render_state()
            config = {
                'N': runner.N,
                'total_layers': runner.total_layers,
                'memory_legend': runner._create_memory_legend_text(),
                'compute_legend': runner._create_compute_legend_text()
            }
            interval_sec = runner.current_interval_sec
            # No need to keep runner instance in this request scope anymore

        except Exception as e:
            print(f"Error initializing SimulationRunner or getting initial state: {e}")
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Error initializing simulation: {e}"}), 500

        # --- Store FULL state in the user's session ---
        session.clear() # Clear any previous state first
        # *** CORRECTED: Store the FULL state dict ***
        session['simulation_full_state'] = initial_full_state
        session['simulation_active'] = True
        session.modified = True
        print(f"Session {session.sid}: Stored initial full state.")

        # --- Return RENDER state and config to frontend ---
        return jsonify({
            "success": True,
            "state": initial_render_state, # Send only render state needed by JS
            "config": config,
            "interval_sec": interval_sec
         })

    except ValueError as e: # Catch validation errors from parse_parameters
        print(f"Parameter validation error: {e}")
        session.clear()
        session.modified = True
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"Error in /start_simulation: {e}")
        traceback.print_exc()
        session.clear()
        session.modified = True
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


@app.route('/get_state_update')
def get_state_update():
    """ Steps the session's simulation and returns the current render state. """
    runner = get_runner_from_session() # Load runner with full state restored
    if not runner:
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    current_render_state = None
    interval = runner.current_interval_sec
    step_success = True

    should_step = not runner.animation_paused and not runner.simulation_complete

    if should_step:
        try:
            keep_running = runner.step() # Advances internal state of runner
        except Exception as e:
             print(f"Error during simulation step for session {session.sid}: {e}")
             traceback.print_exc()
             runner.pause() # Pause this instance
             step_success = False
             # Get render state even after error to show current situation
             current_render_state = runner.get_render_state()
             if isinstance(current_render_state, dict):
                 current_render_state["error_message"] = f"Error during step: {e}"

    # Get the latest render state AFTER potentially stepping (or if paused/complete/error)
    if current_render_state is None:
        current_render_state = runner.get_render_state()
    interval = runner.current_interval_sec # Get potentially updated interval

    # --- Store updated FULL state back into session ---
    if current_render_state: # Check if we have a valid state to save
        # *** CORRECTED: Get the FULL state for saving ***
        updated_full_state = runner.get_serializable_state()
        session['simulation_full_state'] = updated_full_state
        session.modified = True
        # print(f"Session {session.sid}: Saved updated full state frame {updated_full_state.get('current_frame_index')}") # Verbose Debug log

    # --- Return only RENDER state to frontend ---
    return jsonify({
        "success": step_success,
        "state": current_render_state, # Send render state
        "interval_sec": interval
    })


@app.route('/control', methods=['POST'])
def control():
    """ Handles control commands for the session's simulation. """
    runner = get_runner_from_session() # Load runner with full state restored
    if not runner:
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    command = request.json.get('command')
    value = request.json.get('value')
    print(f"Received control command for session {session.sid}: {command}, value: {value}")

    response_state_summary = {}
    interval = runner.current_interval_sec
    success = True
    error_msg = None

    try:
        # Apply command to the loaded runner instance
        if command == 'play':
            runner.play()
        elif command == 'pause':
            runner.pause()
        elif command == 'restart':
            # Get original params before resetting the runner instance
            original_params = runner.params
            # Create a completely new runner and reset it
            # (This ensures clean state, assuming params are stored correctly)
            runner = SimulationRunner(original_params)
            runner.reset_simulation_state()
            print(f"Session {session.sid}: Restarted simulation instance.")
        elif command == 'set_speed':
            if value is not None:
                runner.set_speed(int(value))
                interval = runner.current_interval_sec # Update interval after speed change
            else:
                success = False
                error_msg = "Speed value missing"
        elif command == 'run_to_cycle':
            if value is not None:
                # Run_to_cycle logic might need adjustment if it relies on stepping
                # multiple times within one request - this structure assumes
                # control commands apply instantly or set a target for the *next* step.
                # If run_to_cycle needs to run many steps, that logic
                # should ideally be in the background or handled differently.
                # For now, assume it just sets the target.
                runner.set_target_cycle(int(value))
            else:
                 success = False
                 error_msg = "Target cycle value missing"
        else:
            success = False
            error_msg = "Unknown command"

        # Get updated summary state AFTER applying command
        response_state_summary = runner.get_state_summary()
        # interval may have been updated by set_speed

        # --- Store updated FULL state back into session ---
        # *** CORRECTED: Get the FULL state for saving ***
        updated_full_state = runner.get_serializable_state()
        session['simulation_full_state'] = updated_full_state
        session.modified = True
        # print(f"Session {session.sid}: Saved state after control cmd {command}, frame {updated_full_state.get('current_frame_index')}") # Verbose Debug Log

    except Exception as e:
        print(f"Error executing control command '{command}' for session {session.sid}: {e}")
        traceback.print_exc()
        success = False
        error_msg = f"Error executing command: {e}"
        try: # Try to get summary state even after error
             response_state_summary = runner.get_state_summary()
             interval = runner.current_interval_sec
        except: pass # Ignore errors getting state after another error

    response = {
        "success": success,
        "state_summary": response_state_summary, # Send summary state
        "interval_sec": interval
    }
    if error_msg:
        response["error"] = error_msg

    # --- Return summary state to frontend ---
    return jsonify(response), 200 if success else (400 if error_msg else 500)


if __name__ == '__main__':
    print("Starting Flask dev server (for local testing ONLY)...")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)