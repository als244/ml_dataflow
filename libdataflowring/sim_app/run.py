# run.py (Modified for Per-User Sessions)
from flask import Flask, render_template, request, jsonify, session # Import session
from flask_session import Session # Import Session extension
import time
# import threading # No longer needed for global lock
import sys
import os # For secret key
import traceback # Keep for detailed error logging

try:
    from simulation import SimulationRunner
except ImportError:
    print("Error: simulation.py not found or contains import errors.")
    sys.exit(1)

app = Flask(__name__)

# --- Session Configuration ---
# IMPORTANT: Set a strong, random secret key in production!
# You can generate one using: python -c 'import os; print(os.urandom(24))'
# Store it as an environment variable (e.g., FLASK_SECRET_KEY)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-replace-in-prod')

# Configure session type to filesystem (stores sessions in a folder)
app.config['SESSION_TYPE'] = 'filesystem'
# Optional: Define where session files are stored (defaults to ./flask_session)
app.config['SESSION_FILE_DIR'] = './.flask_session/'
# Ensure the session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Configure session cookie settings (recommended for production)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' # Protects against CSRF
# Set SECURE to True if (and only if) your site is served over HTTPS
# For local testing over HTTP, keep it False or comment it out.
# For deployment behind Nginx with SSL termination, set to True.
app.config['SESSION_COOKIE_SECURE'] = True # ASSUMING you are using HTTPS via Nginx/Certbot

# Initialize the session extension AFTER setting configurations
server_session = Session(app)
# --- End Session Configuration ---


# --- REMOVED Global Variables ---
# simulation_instance = None
# simulation_lock = threading.Lock()


def parse_parameters(form_data):
    """ Safely parses form data into simulation parameters. (Unchanged) """
    params = {}
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
    # Corrected TB/s -> GB/s -> B/s
    params['hardware_mem_bw_bytes_sec'] = form_data.get('hardware_mem_bw_gbs', default=3350, type=float) * (1 << 30) # Input GB/s
    params['matmul_efficiency'] = form_data.get('matmul_efficiency', default=0.7, type=float)
    params['attn_efficiency'] = form_data.get('attn_efficiency', default=0.55, type=float)
    params['home_bw_gbit_sec'] = form_data.get('home_bw_gbit_sec', default=400, type=int)
    params['peer_bw_gbit_sec'] = form_data.get('peer_bw_gbit_sec', default=100, type=int)

    # Basic validation
    if params['N'] <= 0: raise ValueError("N must be positive.")
    # Add more validation as needed...
    return params

# --- Helper Function to Load/Recreate Runner from Session ---
# IMPORTANT: This assumes you add a method like 'load_internal_state'
# to your SimulationRunner class in simulation.py
def get_runner_from_session():
    """Retrieves params and state from session, returns initialized SimulationRunner."""
    if not session.get('simulation_active'):
        return None # No active simulation for this session

    params = session.get('simulation_params')
    current_state_dict = session.get('simulation_state') # This is the dict returned by get_render_state()

    if not params or not current_state_dict:
        print("Error: Params or State missing from session.")
        session.clear() # Clear corrupted session state
        session.modified = True
        return None

    try:
        # Recreate the runner instance with original parameters
        runner = SimulationRunner(params)
        # *** YOU NEED TO IMPLEMENT THIS METHOD in simulation.py ***
        # It should take the state dictionary and set the internal attributes
        # (like current_frame_index, animation_paused, device states, queues etc.)
        # back to match the stored state.
        runner.load_from_serializable_state(current_state_dict)
        return runner
    except Exception as e:
        print(f"Error recreating SimulationRunner from session state: {e}")
        traceback.print_exc()
        session.clear() # Clear potentially problematic state
        session.modified = True
        return None

@app.route('/')
def index():
    """ Serves the main HTML page. """
    # Clear any old simulation state when loading the main page? Optional.
    # session.clear()
    # session.modified = True
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """ Initializes or restarts the simulation FOR THIS SESSION. """
    print(f"Received /start_simulation request for session ID: {session.sid if session.sid else 'NEW'}")
    try:
        params = parse_parameters(request.form)
        print(f"Parsed parameters: {params}")

        # Create a temporary runner just to get initial state and config
        try:
            temp_runner = SimulationRunner(params)
            temp_runner.reset_simulation_state()
            initial_render_state = temp_runner.get_render_state()
            config = {
                'N': temp_runner.N,
                'total_layers': temp_runner.total_layers,
                'memory_legend': temp_runner._create_memory_legend_text(),
                'compute_legend': temp_runner._create_compute_legend_text()
            }
            interval_sec = temp_runner.current_interval_sec
            del temp_runner # Clean up temporary instance

        except Exception as e:
            print(f"Error initializing SimulationRunner: {e}")
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Error initializing simulation: {e}"}), 500

        # --- Store state in the user's session ---
        session['simulation_params'] = params
        session['simulation_state'] = initial_render_state # Store the state dictionary
        session['simulation_active'] = True
        session.modified = True # IMPORTANT: Mark session as modified to ensure it's saved
        print("Stored initial state in session.")

        return jsonify({
            "success": True,
            "state": initial_render_state,
            "config": config,
            "interval_sec": interval_sec
         })

    except ValueError as e: # Catch validation errors from parse_parameters
        print(f"Parameter validation error: {e}")
        # Clear potentially partial session state
        session.pop('simulation_active', None)
        session.pop('simulation_params', None)
        session.pop('simulation_state', None)
        session.modified = True
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"Error in /start_simulation: {e}")
        traceback.print_exc()
        # Clear potentially partial session state
        session.pop('simulation_active', None)
        session.pop('simulation_params', None)
        session.pop('simulation_state', None)
        session.modified = True
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


@app.route('/get_state_update')
def get_state_update():
    """ Steps the session's simulation (if running) and returns the current render state. """
    runner = get_runner_from_session() # Load runner based on session data
    if not runner:
        # If get_runner_from_session returned None, it means no active sim or error loading state
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    current_state = None
    interval = runner.current_interval_sec # Get interval from loaded runner
    step_success = True

    # No lock needed here as the 'runner' object is local to this request
    should_step = not runner.animation_paused and not runner.simulation_complete

    if should_step:
        try:
            keep_running = runner.step() # Advances frame index internally
        except Exception as e:
             print(f"Error during simulation step for session {session.sid}: {e}")
             traceback.print_exc()
             runner.pause() # Pause this instance
             step_success = False
             # Add error info to the state sent to frontend
             current_state = runner.get_render_state() # Get state even after error
             if isinstance(current_state, dict):
                 current_state["error_message"] = f"Error during step: {e}"

    # Get the render state AFTER potentially stepping (or if paused/complete)
    if step_success: # If step succeeded or wasn't needed
        current_state = runner.get_render_state()
        interval = runner.current_interval_sec # Get potentially updated interval

    # --- Store updated state back into session ---
    if current_state: # Make sure we have state to save
        session['simulation_state'] = current_state
        session.modified = True

    return jsonify({
        "success": step_success,
        "state": current_state,
        "interval_sec": interval
    })


@app.route('/control', methods=['POST'])
def control():
    """ Handles control commands for the session's simulation. """
    runner = get_runner_from_session() # Load runner based on session data
    if not runner:
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    command = request.json.get('command')
    value = request.json.get('value')

    print(f"Received control command for session {session.sid}: {command}, value: {value}")

    response_state_summary = {}
    interval = runner.current_interval_sec
    success = True
    error_msg = None

    # No lock needed, operate on the request-local 'runner' instance
    try:
        if command == 'play':
            runner.play()
        elif command == 'pause':
            runner.pause()
        elif command == 'restart':
            runner.reset_simulation_state() # Reset this instance
        elif command == 'set_speed':
            if value is not None:
                runner.set_speed(int(value))
            else:
                success = False
                error_msg = "Speed value missing"
        elif command == 'run_to_cycle':
            if value is not None:
                runner.set_target_cycle(int(value))
            else:
                 success = False
                 error_msg = "Target cycle value missing"
        else:
            success = False
            error_msg = "Unknown command"

        # Get updated states after applying command
        response_state_summary = runner.get_state_summary()
        updated_full_state_dict = runner.get_serializable_state()
        interval = runner.current_interval_sec

        # --- Store updated full state back into session ---
        session['simulation_state'] = updated_full_state_dict
        session.modified = True

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
        "state_summary": response_state_summary,
        "interval_sec": interval
    }
    if error_msg:
        response["error"] = error_msg

    return jsonify(response), 200 if success else (400 if error_msg else 500)


if __name__ == '__main__':
    # This block is ONLY for local development using 'python run.py'
    # It is NOT used when running with Gunicorn/WSGI.
    # The 'threaded=True' allows the dev server to handle multiple browser
    # requests concurrently to some extent, but session management handles
    # the state separation, not the threads directly. Debug mode should work fine here.
    print("Starting Flask dev server (for local testing ONLY)...")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)