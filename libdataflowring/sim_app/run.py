# run.py (Corrected for Full State Serialization via Session)
from flask import Flask, render_template, request, jsonify, session # Import session
from flask_session import Session # Import Session extension
import time
import sys
import os # For secret key
import traceback # Keep for detailed error logging
from simulation import SimulationRunner

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
    params['cycle_rate_micros'] = form_data.get('cycle_rate_micros', default=2000, type=int)
    params['min_chunk_size'] = form_data.get('min_chunk_size', default=4096, type=int)
    params['N'] = form_data.get('N', default=8, type=int)
    params['seqlen'] = form_data.get('seqlen', default=256, type=int) # K tokens
    params['max_attended_tokens'] = form_data.get('max_attended_tokens', default=100, type=int) # K tokens
    params['train_token_ratio'] = form_data.get('train_token_ratio', default=1, type=float)
    params['chunk_type'] = form_data.get('chunk_type', default="Equal Data", type=str)
    params['train_chunk_distribution'] = form_data.get('train_chunk_distribution', default="Uniform", type=str)
    params['bitwidth'] = form_data.get('bitwidth', default=16, type=int)
    params['total_layers'] = form_data.get('total_layers', default=64, type=int) # Non-head
    params['vocab_size'] = form_data.get('vocab_size', default=151646, type=int)
    params['model_dim'] = form_data.get('model_dim', default=5120, type=int)
    params['kv_dim'] = form_data.get('kv_dim', default=640, type=int) # REMOVE this, use factor
    params['num_experts'] = form_data.get('num_experts', default=1, type=int)
    params['active_experts'] = form_data.get('active_experts', default=1, type=int)
    params['expert_dim'] = form_data.get('expert_dim', default=27648, type=int)
    params['attn_type'] = form_data.get('attn_type', default="Exact", type=str)
    params['max_device_memory_bytes'] = form_data.get('max_device_memory_gb', default=12, type=float) * (1 << 30)
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
def start_simulation_route(): # Renamed to avoid conflict with any potential import
    print(f"Received /start_simulation request for session ID: {session.sid if session.sid else 'NEW'}")
    try:
        params = parse_parameters(request.form)
        runner = SimulationRunner(params) # Initializes with default speed_level internally
        runner.reset_simulation_state()

        initial_full_state = runner.get_serializable_state()
        initial_render_state = runner.get_render_state() # Should include the initial speed_level
        config = {
            'N': runner.N,
            'total_layers': runner.total_layers,
            'memory_legend': runner._create_memory_legend_text(),
            'compute_legend': runner._create_compute_legend_text()
        }

        session.clear()
        session['simulation_full_state'] = initial_full_state
        session['simulation_active'] = True
        session.modified = True


        return jsonify({
            "success": True,
            "state": initial_render_state, # This state should contain the initial speed_level
            "config": config
         })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


@app.route('/get_state_update')
def get_state_update_route(): # Renamed
    runner = get_runner_from_session()
    if not runner:
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    current_render_state = None
    step_success = True
    should_step = not runner.animation_paused and not runner.simulation_complete

    if should_step:
        try:
            runner.step()
        except Exception as e:
             traceback.print_exc()
             runner.pause()
             step_success = False
             current_render_state = runner.get_render_state()
             if isinstance(current_render_state, dict):
                 current_render_state["error_message"] = f"Error during step: {e}"

    if current_render_state is None:
        current_render_state = runner.get_render_state() # speed_level is part of this state

    if current_render_state:
        updated_full_state = runner.get_serializable_state()
        session['simulation_full_state'] = updated_full_state
        session.modified = True

    return jsonify({
        "success": step_success,
        "state": current_render_state
    })


@app.route('/control', methods=['POST'])
def control_route(): # Renamed
    runner = get_runner_from_session()
    if not runner:
        return jsonify({"success": False, "error": "Simulation not started or session expired"}), 400

    command = request.json.get('command')
    value = request.json.get('value')
    success = True
    error_msg = None

    try:
        if command == 'play':
            runner.play()
        elif command == 'pause':
            runner.pause()
        elif command == 'restart':
            original_params = runner.params
            runner = SimulationRunner(original_params) # Re-initializes with default speed_level
            runner.reset_simulation_state()
        elif command == 'run_to_cycle':
            if value is not None:
                runner.set_target_cycle(int(value)) # This might also unpause the simulation
            else:
                 success = False
                 error_msg = "Target cycle value missing"
        else:
            success = False
            error_msg = "Unknown command"

        response_state_summary = runner.get_state_summary() # Ensure this includes speed_level

        updated_full_state = runner.get_serializable_state()
        session['simulation_full_state'] = updated_full_state
        session.modified = True

    except Exception as e:
        traceback.print_exc()
        success = False
        error_msg = f"Error executing command: {e}"
        try:
             response_state_summary = runner.get_state_summary()
        except: response_state_summary = {}


    response = {
        "success": success,
        "state_summary": response_state_summary
    }
    if error_msg:
        response["error"] = error_msg
    return jsonify(response), 200 if success else (400 if error_msg else 500)
