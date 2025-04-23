# run.py (With Enhanced Debug Logging)
from flask import Flask, render_template, request, jsonify, session # Import session
from flask_session import Session # Import Session extension
import time
import sys
import os # For secret key
import traceback # Keep for detailed error logging

try:
    from simulation import SimulationRunner
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
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-replace-in-prod-AGAIN')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False # MUST be False for http://localhost development
# Make session permanent to ensure it lasts longer (helps debugging)
app.config['SESSION_PERMANENT'] = True
app.permanent_session_lifetime = 3600 # e.g., 1 hour

print(f"--- Flask App Config ---")
print(f"SECRET_KEY: {'*' * len(app.config['SECRET_KEY'])}")
print(f"SESSION_TYPE: {app.config['SESSION_TYPE']}")
print(f"SESSION_FILE_DIR: {os.path.abspath(app.config['SESSION_FILE_DIR'])}")
print(f"SESSION_COOKIE_SAMESITE: {app.config['SESSION_COOKIE_SAMESITE']}")
print(f"SESSION_COOKIE_SECURE: {app.config['SESSION_COOKIE_SECURE']}")
print(f"SESSION_PERMANENT: {app.config['SESSION_PERMANENT']}")
print(f"------------------------")


server_session = Session(app)
# --- End Session Configuration ---

def parse_parameters(form_data):
    # ... (parameter parsing code - unchanged) ...
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
    session_id = session.sid if session.sid else 'None'# Print entire session dict

    is_active = session.get('simulation_active')
    if not is_active:
        return None

    full_state = session.get('simulation_full_state')

    if not full_state or not isinstance(full_state, dict) or 'params' not in full_state:
        print(f"[get_runner_from_session] Invalid or missing 'simulation_full_state' or 'params' key.")
        print(f"[get_runner_from_session] Clearing potentially corrupted session and returning None.")
        session.clear()
        session.modified = True
        return None

    try:
        runner = SimulationRunner(full_state['params'])
        runner.load_from_serializable_state(full_state)
        return runner
    except Exception as e:
        print(f"[get_runner_from_session] !!! EXCEPTION OCCURRED while recreating/loading runner !!!")
        print(f"[get_runner_from_session] Error type: {type(e).__name__}, Error message: {e}")
        traceback.print_exc() # Print the full traceback to the console
        print(f"[get_runner_from_session] Clearing session due to exception and returning None.")
        session.clear()
        session.modified = True
        return None

@app.route('/')
def index():
    """ Serves the main HTML page. """
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """ Initializes or restarts the simulation FOR THIS SESSION using full state. """
    print(f"\n--- Request: /start_simulation ---")
    session.clear() # Clear previous state first

    try:
        params = parse_parameters(request.form)
        print(f"Parsed parameters: {params}")

        try:
            runner = SimulationRunner(params)
            runner.reset_simulation_state()
            initial_full_state = runner.get_serializable_state()
            initial_render_state = runner.get_render_state()
            config = {
                'N': runner.N,
                'total_layers': runner.total_layers,
                'memory_legend': runner._create_memory_legend_text(), # Ensure these methods exist
                'compute_legend': runner._create_compute_legend_text() # Ensure these methods exist
            }
            interval_sec = runner.current_interval_sec

        except Exception as e:
            print(f"!!! EXCEPTION during SimulationRunner initialization !!!")
            print(f"Error type: {type(e).__name__}, Error message: {e}")
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Error initializing simulation: {e}"}), 500

        # --- Store FULL state in the user's session ---
        session['simulation_full_state'] = initial_full_state
        session['simulation_active'] = True
        session.modified = True # Mark session as modified

        return jsonify({
            "success": True,
            "state": initial_render_state,
            "config": config,
            "interval_sec": interval_sec
         })

    except ValueError as e:
        print(f"!!! EXCEPTION during parameter parsing !!!")
        print(f"Error type: {type(e).__name__}, Error message: {e}")
        traceback.print_exc()
        session.clear() # Ensure session is clear on error
        session.modified = True
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"!!! UNEXPECTED EXCEPTION in /start_simulation !!!")
        print(f"Error type: {type(e).__name__}, Error message: {e}")
        traceback.print_exc()
        session.clear() # Ensure session is clear on error
        session.modified = True
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


@app.route('/get_state_update')
def get_state_update():
    """ Steps the session's simulation and returns the current render state. """
    runner = get_runner_from_session() # This function now has detailed logging

    if not runner:
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    # ... (rest of the logic is likely okay, but add prints if needed) ...
    current_render_state = None
    interval = runner.current_interval_sec
    step_success = True
    should_step = not runner.animation_paused and not runner.simulation_complete

    if should_step:
        try:
            keep_running = runner.step()
        except Exception as e:
             print(f"!!! EXCEPTION during simulation step !!!")
             print(f"Error type: {type(e).__name__}, Error message: {e}")
             traceback.print_exc()
             runner.pause()
             step_success = False
             current_render_state = runner.get_render_state()
             if isinstance(current_render_state, dict):
                 current_render_state["error_message"] = f"Error during step: {e}"

    if current_render_state is None:
        current_render_state = runner.get_render_state()
    interval = runner.current_interval_sec

    if current_render_state:
        updated_full_state = runner.get_serializable_state()
        session['simulation_full_state'] = updated_full_state
        session.modified = True
        # print(f"[/get_state_update] Session content AFTER update: {dict(session)}") # Can be verbose

    return jsonify({
        "success": step_success,
        "state": current_render_state,
        "interval_sec": interval
    })


@app.route('/control', methods=['POST'])
def control():
    """ Handles control commands for the session's simulation. """
    runner = get_runner_from_session() # This function now has detailed logging

    if not runner:
         print("[/control] Runner not found, returning error.")
         return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    # ... (rest of the logic is likely okay, but add prints if needed) ...
    command = request.json.get('command')
    value = request.json.get('value')

    response_state_summary = {}
    interval = runner.current_interval_sec
    success = True
    error_msg = None

    try:
        if command == 'play': runner.play()
        elif command == 'pause': runner.pause()
        elif command == 'restart':
            original_params = runner.params
            runner = SimulationRunner(original_params)
            runner.reset_simulation_state()
        elif command == 'set_speed':
            if value is not None: runner.set_speed(int(value)); interval = runner.current_interval_sec
            else: success = False; error_msg = "Speed value missing"
        elif command == 'run_to_cycle':
            if value is not None: runner.set_target_cycle(int(value))
            else: success = False; error_msg = "Target cycle value missing"
        else:
            success = False; error_msg = "Unknown command"

        response_state_summary = runner.get_state_summary()

        updated_full_state = runner.get_serializable_state()
        session['simulation_full_state'] = updated_full_state
        session.modified = True
        # print(f"[/control] Session content AFTER control: {dict(session)}") # Can be verbose

    except Exception as e:
        print(f"!!! EXCEPTION during control command '{command}' execution !!!")
        print(f"Error type: {type(e).__name__}, Error message: {e}")
        traceback.print_exc()
        success = False
        error_msg = f"Error executing command: {e}"
        try: response_state_summary = runner.get_state_summary(); interval = runner.current_interval_sec
        except: pass

    response = {"success": success, "state_summary": response_state_summary, "interval_sec": interval}
    if error_msg: response["error"] = error_msg

    return jsonify(response), 200 if success else (400 if error_msg else 500)


if __name__ == '__main__':
    print("Starting Flask dev server (for local testing ONLY)...")
    # Keep reloader disabled for now while debugging session issues
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True, use_reloader=False)