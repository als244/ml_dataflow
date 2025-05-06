// static/simulationWorker.js

let workerSimulationState = { 
    current_frame: 0, 
    is_paused: true, 
    is_complete: false, 
    speed_level: 50, // Definitively 50 as the starting logical speed
    target_cycle: null, 
    max_frames: 30000, 
    completion_stats: {}, 
    devices: [] 
};


// Calculate initial interval RIGHT AWAY based on the defined default speed_level
let workerCurrentIntervalSec = calculateIntervalFromSpeed(workerSimulationState.speed_level); 

let workerInitialized = false;
let workerSimulationConfig = null;
let workerUpdateTimer = null;
let isWorkerFetching = false;

console.log(`Worker: Script loaded. Initial default speed_level: ${workerSimulationState.speed_level}, calculated initial workerCurrentIntervalSec: ${workerCurrentIntervalSec}s. workerInitialized: ${workerInitialized}`);

// --- Helper function to map speed level to an interval ---
// Adjust min/max interval and linearity/non-linearity as needed for your desired feel
function calculateIntervalFromSpeed(speedLevel) {
    const MIN_SPEED_SLIDER = 1;    // Corresponds to slider's min value
    const MAX_SPEED_SLIDER = 100;  // Corresponds to slider's max value

    // Define the fastest and slowest intervals for your simulation (in milliseconds)
    const FASTEST_INTERVAL_MS = 1;   // e.g., for speed_level 100
    const SLOWEST_INTERVAL_MS = 500; // e.g., for speed_level 1

    // Ensure speedLevel is an integer and clamped within expected bounds
    const clampedSpeed = Math.max(MIN_SPEED_SLIDER, Math.min(MAX_SPEED_SLIDER, parseInt(speedLevel, 10)));

    // Linear interpolation: Higher speed -> shorter interval
    // Calculate the proportion of speed: (clampedSpeed - MIN_SPEED_SLIDER) / (MAX_SPEED_SLIDER - MIN_SPEED_SLIDER)
    // This gives a value from 0 (slowest) to 1 (fastest)
    const speedRatio = (clampedSpeed - MIN_SPEED_SLIDER) / (MAX_SPEED_SLIDER - MIN_SPEED_SLIDER);

    // Calculate interval: Start with slowest, subtract based on speedRatio
    const intervalMs = SLOWEST_INTERVAL_MS - (speedRatio * (SLOWEST_INTERVAL_MS - FASTEST_INTERVAL_MS));
    
    // console.log(`Worker: Speed ${clampedSpeed} -> Interval ${intervalMs}ms`);
    return intervalMs / 1000.0; // Convert to seconds
}

async function fetchUpdateFromAPI() {
    if (!workerInitialized || workerSimulationState.is_paused || workerSimulationState.is_complete || isWorkerFetching) {
        return;
    }
    isWorkerFetching = true;
    try {
        const response = await fetch('/get_state_update');
        const data = await response.json();

        if (response.ok && data.success) {
            if (data.state && typeof data.state === 'object') {
                let previousClientSpeedLevel = workerSimulationState.speed_level; // Store client's current speed

                workerSimulationState = data.state; // Update with full state from server

                // Restore the client's authoritative speed_level over whatever the server sent
                workerSimulationState.speed_level = previousClientSpeedLevel; 
                
                // Ensure other critical booleans are valid
                if (typeof workerSimulationState.is_paused !== 'boolean') workerSimulationState.is_paused = true;
                if (typeof workerSimulationState.is_complete !== 'boolean') workerSimulationState.is_complete = false;

            } else {
                console.error("Worker [fetchUpdateFromAPI]: Server success but data.state is missing or not an object. Keeping previous worker state.");
            }
        } else {
            console.error("Worker: Error fetching update from API:", data.error || response.statusText);
            self.postMessage({ type: 'fetchError', error: data.error || response.statusText });
        }
    } catch (error) {
        console.error("Worker: Network error during fetchUpdateFromAPI:", error);
        self.postMessage({ type: 'fetchError', error: error.message });
    } finally {
        isWorkerFetching = false;
    }

    // Always post message, even if state wasn't updated from server due to bad data from server
    self.postMessage({ type: 'stateUpdate', state: workerSimulationState });

    if (workerSimulationState.is_complete || workerSimulationState.is_paused) {
        stopWorkerLoop();
    }
}

async function sendControlToAPI(command, value = null) {
    console.log(`Worker: Sending command to API: ${command}`, value !== null ? `Value: ${value}` : '');
    isWorkerFetching = true;
    try {
        const body = { command };
        if (value !== null) { body.value = value; }

        const response = await fetch('/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await response.json();

        if (response.ok && data.success) {

            let previousClientSpeedLevel = workerSimulationState.speed_level;

            if (data.state_summary && typeof data.state_summary === 'object') {
                console.log("Worker [sendControlToAPI]: Updating from state_summary. Server speed_level (will be ignored):", data.state_summary.speed_level);
                // Selectively update from summary, DO NOT take speed_level from here
                if (typeof data.state_summary.is_paused === 'boolean') workerSimulationState.is_paused = data.state_summary.is_paused;
                if (typeof data.state_summary.is_complete === 'boolean') workerSimulationState.is_complete = data.state_summary.is_complete;
                if (typeof data.state_summary.current_frame !== 'undefined') workerSimulationState.current_frame = data.state_summary.current_frame;
                workerSimulationState.target_cycle = data.state_summary.target_cycle; // Can be null
            } else if (data.state && typeof data.state === 'object') {
                // If server sends full state for a control command (less common but possible)
                console.log("Worker [sendControlToAPI]: Updating from full data.state. Server speed_level (will be ignored/overridden):", data.state.speed_level);
                workerSimulationState = data.state; // Full overwrite
            } else {
                console.warn(`Worker [sendControlToAPI]: Command '${command}' response lacked valid state_summary or state.`);
            }
    
            // Restore/ensure client's authoritative speed_level after any potential overwrite
            workerSimulationState.speed_level = previousClientSpeedLevel;
    
            // Apply immediate effects of the command for loop control
            if (command === 'play') workerSimulationState.is_paused = false;
            if (command === 'pause') workerSimulationState.is_paused = true;
            if (command === 'run_to_cycle') workerSimulationState.is_paused = false;


            // For play/pause, update local state immediately for responsiveness
            if (command === 'play') workerSimulationState.is_paused = false;
            if (command === 'pause') workerSimulationState.is_paused = true;
            if (command === 'run_to_cycle' && data.state_summary) { // or data.state
                 workerSimulationState.target_cycle = data.state_summary.target_cycle; // or data.state.target_cycle
                 workerSimulationState.is_paused = false; // Run_to implies unpausing
            }

            // Safety checks for booleans
            if (typeof workerSimulationState.is_paused !== 'boolean') workerSimulationState.is_paused = true;
            if (typeof workerSimulationState.is_complete !== 'boolean') workerSimulationState.is_complete = false;
    
            self.postMessage({ type: 'stateUpdate', state: workerSimulationState });
    
            if (workerSimulationState.is_complete || workerSimulationState.is_paused) {
                stopWorkerLoop();
            } else {
                startWorkerLoop(); // If command made it runnable (e.g., play, run_to_cycle)
            }

        } else {
            console.error(`Worker: Backend error for command '${command}':`, data.error || response.statusText);
            self.postMessage({ type: 'controlError', command: command, error: data.error || response.statusText });
        }
    } catch (error) {
        console.error(`Worker: Network error sending command '${command}':`, error);
        self.postMessage({ type: 'controlError', command: command, error: error.message });
    } finally {
        isWorkerFetching = false;
    }
}

function startWorkerLoop() {
    if (workerUpdateTimer) clearInterval(workerUpdateTimer);
    if (workerInitialized && !workerSimulationState.is_paused && !workerSimulationState.is_complete) {
        // workerCurrentIntervalSec is now managed internally
        const delay = Math.max(20, workerCurrentIntervalSec * 1000);
        workerUpdateTimer = setInterval(fetchUpdateFromAPI, delay);
        console.log(`Worker: Loop started/restarted with interval ${delay}ms (derived from speed_level ${workerSimulationState.speed_level}).`);
    } else {
        console.log(`Worker: Loop not started (initialized: ${workerInitialized}, paused: ${workerSimulationState.is_paused}, complete: ${workerSimulationState.is_complete})`);
    }
}

function stopWorkerLoop() {
    if (workerUpdateTimer) {
        clearInterval(workerUpdateTimer);
        console.log("Worker: Cleared workerUpdateTimer ID:", workerUpdateTimer);
        workerUpdateTimer = null;
    }
}

self.onmessage = function(e) {
    const { command, value, initialState, initialInterval, config } = e.data; // initialInterval will be ignored
    console.log("Worker: Message received from main script:", JSON.parse(JSON.stringify(e.data)));

    switch (command) {
        case 'initialize':
            console.log("Worker [initialize]: Received 'initialize' command.");
            stopWorkerLoop(); // Kill any rogue loop

            if (!initialState) {
                console.error("Worker [initialize]: CRITICAL - initialState is missing! Cannot proceed.");
                workerInitialized = false; 
                return; 
            }
            
            // The server's initialState should be the source of truth for starting speed.
            // If server doesn't send speed_level, we might default, but it's better if server defines it.
            // For now, assume initialState CONTAINS a speed_level.
            workerSimulationState = initialState; 
            
            // If server *might not* send speed_level (e.g., you removed all server-side speed_level logic)
            // then ensure workerSimulationState gets a sensible default if initialState lacks it.
            if (typeof workerSimulationState.speed_level === 'undefined') {
                console.warn("Worker [initialize]: initialState from server did not contain speed_level. Defaulting to worker's initial 50.");
                workerSimulationState.speed_level = 50; // Fallback to worker's default
            }

            workerSimulationConfig = config;
            workerCurrentIntervalSec = calculateIntervalFromSpeed(workerSimulationState.speed_level); // Recalculate based on potentially new initialState.speed_level
            
            console.log(`Worker [initialize]: State processed. Final speed_level for init: ${workerSimulationState.speed_level}, Interval=${workerCurrentIntervalSec}s. is_paused=${workerSimulationState.is_paused}`);
            
            workerInitialized = true; 

            self.postMessage({ type: 'stateUpdate', state: workerSimulationState });
            startWorkerLoop(); // Will respect is_paused from initialState
            break;
        case 'play':
            workerSimulationState.is_paused = false;
        case 'pause':
            workerSimulationState.is_paused = false;
        case 'run_to_cycle':
            if (!workerInitialized) {
                 console.warn("Worker: Control command received before initialization.");
                 return;
            }
            // Update local state for immediate UI feedback before API call for play/pause
            if (command === 'play') workerSimulationState.is_paused = false;
            if (command === 'pause') workerSimulationState.is_paused = true;

            sendControlToAPI(command, value).then(() => {
                // State update and loop management is handled within sendControlToAPI
                // For play/pause, start/stopWorkerLoop is called there.
            });
            break;
        case 'set_speed':
            if (!workerInitialized) { /* ... */ return; }
            const newSpeed = parseInt(value, 10);
            if (!isNaN(newSpeed)) {
                workerSimulationState.speed_level = newSpeed; // This is the authoritative client-side update
                workerCurrentIntervalSec = calculateIntervalFromSpeed(newSpeed);
                console.log(`Worker: Speed explicitly set to ${newSpeed}, new interval ${workerCurrentIntervalSec}s.`);
                self.postMessage({ type: 'stateUpdate', state: workerSimulationState });
                startWorkerLoop(); // This will use the new workerCurrentIntervalSec
            }
            break;
        case 'resetWorker':
            console.log("Worker: Reset command received.");
            stopWorkerLoop();
            workerInitialized = false;
            workerSimulationState = { current_frame: 0, is_paused: true, is_complete: false, speed_level: 50, target_cycle: null, max_frames: 30000, devices: [] };
            workerCurrentIntervalSec = calculateIntervalFromSpeed(workerSimulationState.speed_level); // Reset to default speed interval
            workerSimulationConfig = null;
            self.postMessage({ type: 'workerResetComplete' });
            break;
        default:
            console.warn("Worker: Unknown command received:", command);
    }
};

console.log("Worker: simulationWorker.js loaded and ready.");