# app.py
from flask import Flask, render_template, request, jsonify
import time
import threading
import sys

try:
    from simulation import SimulationRunner
except ImportError:
    print("Error: simulation.py not found or contains import errors.")
    sys.exit(1)

app = Flask(__name__)

simulation_instance = None
simulation_lock = threading.Lock()

def parse_parameters(form_data):
    """ Safely parses form data into simulation parameters. """
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

@app.route('/')
def index():
    """ Serves the main HTML page. """
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """ Initializes or restarts the simulation with new parameters. """
    global simulation_instance
    print("Received /start_simulation request")
    try:
        params = parse_parameters(request.form)
        print(f"Parsed parameters: {params}")

        with simulation_lock:
            # Clean up old instance if it exists
            if simulation_instance:
                # simulation_instance.close_figure() # REMOVED
                simulation_instance = None
                print("Cleaned up previous simulation instance.")

            try:
                simulation_instance = SimulationRunner(params)
                simulation_instance.reset_simulation_state() # Already called in init, but good for clarity
                print("Created new SimulationRunner instance.")
                # Get initial state for rendering
                initial_render_state = simulation_instance.get_render_state()
                # Include static config needed by frontend
                config = {
                    'N': simulation_instance.N,
                    'total_layers': simulation_instance.total_layers,
                    'memory_legend': simulation_instance._create_memory_legend_text(), # Send initial legends
                    'compute_legend': simulation_instance._create_compute_legend_text()
                }
                return jsonify({
                    "success": True,
                    "state": initial_render_state, # Send full render state
                    "config": config, # Send static config
                    "interval_sec": simulation_instance.current_interval_sec
                 })
            except Exception as e:
                 print(f"Error initializing SimulationRunner: {e}")
                 import traceback
                 traceback.print_exc()
                 simulation_instance = None
                 return jsonify({"success": False, "error": f"Error initializing simulation: {e}"}), 500

    except ValueError as e: # Catch validation errors from parse_parameters
        print(f"Parameter validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"Error in /start_simulation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

# Renamed route for clarity
@app.route('/get_state_update')
def get_state_update():
    """ Steps the simulation (if running) and returns the current render state. """
    global simulation_instance
    if not simulation_instance:
        return jsonify({"success": False, "error": "Simulation not started"}), 400

    start_time = time.perf_counter()

    current_state = None
    interval = 1.0 # Default interval
    step_success = True

    with simulation_lock:
        interval = simulation_instance.current_interval_sec # Get current interval
        should_step = not simulation_instance.animation_paused and not simulation_instance.simulation_complete

        if should_step:
            try:
                keep_running = simulation_instance.step() # Advances frame index internally
                # If step caused completion, update state reflects this
            except Exception as e:
                 print(f"Error during simulation step: {e}")
                 import traceback
                 traceback.print_exc()
                 simulation_instance.pause() # Pause on error
                 step_success = False
                 # Try to get state even after error
                 current_state = simulation_instance.get_render_state()
                 # Add error info to the state sent to frontend
                 if isinstance(current_state, dict): # Ensure it's a dict before modifying
                    current_state["error_message"] = f"Error during step: {e}"

        # Get the render state AFTER potentially stepping (or if paused/complete)
        if step_success:
            current_state = simulation_instance.get_render_state()
            interval = simulation_instance.current_interval_sec # Get potentially updated interval

    end_time = time.perf_counter() # End timer
    processing_time_ms = (end_time - start_time) * 1000
    print(f"[/get_state_update] Processing time: {processing_time_ms:.2f} ms") # Log tim
    
    return jsonify({
        "success": step_success,
        "state": current_state,
        "interval_sec": interval
    })


@app.route('/control', methods=['POST'])
def control():
    """ Handles control commands from the frontend. """
    global simulation_instance
    if not simulation_instance:
        return jsonify({"success": False, "error": "Simulation not started"}), 400

    command = request.json.get('command')
    value = request.json.get('value')

    print(f"Received control command: {command}, value: {value}")

    response_state_summary = {} # Return summary state for basic updates
    interval = 1.0
    success = True
    error_msg = None

    with simulation_lock:
        try:
            if command == 'play':
                simulation_instance.play()
            elif command == 'pause':
                simulation_instance.pause()
            elif command == 'restart':
                # Re-use initial parameters implicitly stored in the instance
                simulation_instance.reset_simulation_state()
            elif command == 'set_speed':
                if value is not None:
                    simulation_instance.set_speed(int(value))
                else:
                    success = False
                    error_msg = "Speed value missing"
            elif command == 'run_to_cycle':
                if value is not None:
                    simulation_instance.set_target_cycle(int(value))
                else:
                     success = False
                     error_msg = "Target cycle value missing"
            else:
                success = False
                error_msg = "Unknown command"

            response_state_summary = simulation_instance.get_state_summary() # Get summary
            interval = simulation_instance.current_interval_sec

        except Exception as e:
            print(f"Error executing control command '{command}': {e}")
            import traceback
            traceback.print_exc()
            success = False
            error_msg = f"Error executing command: {e}"
            try: # Try to get summary state even after error
                 response_state_summary = simulation_instance.get_state_summary()
                 interval = simulation_instance.current_interval_sec
            except: pass # Ignore errors getting state after another error

    response = {
        "success": success,
        "state_summary": response_state_summary, # Send summary state
        "interval_sec": interval
    }
    if error_msg:
        response["error"] = error_msg

    return jsonify(response), 200 if success else (400 if error_msg else 500)


if __name__ == '__main__':
    print("Starting Flask server...")
    # Use threaded=True for basic concurrency handling with the global instance
    # Still NOT recommended for production with multiple users.
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)