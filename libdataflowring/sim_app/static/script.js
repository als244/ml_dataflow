// static/script.js

// static/script.js

// Debounce function to limit resize calls
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
};

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const paramsForm = document.getElementById('paramsForm');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');
    const runToCycleInput = document.getElementById('runToCycleInput');
    const runToBtn = document.getElementById('runToBtn');
    const statusArea = document.getElementById('status-area');
    const svgContainer = document.getElementById('simulation-svg-container');
    const svg = document.getElementById('simulation-svg');
    const completionArea = document.getElementById('completion-area'); // Content area inside popup
    const memoryLegendArea = document.getElementById('memory-legend-area');
    const computeLegendArea = document.getElementById('compute-legend-area');
    const errorMessageDiv = document.getElementById('error-message');
    const submitButton = paramsForm.querySelector('button[type="submit"]');
    const controlElements = document.querySelectorAll('.controls button, .controls input, .controls span');
    
    const torusPlotContainer = document.getElementById('torus-plot-container');
    const horizontalResizer = document.getElementById('horizontal-resizer');
    const completionPopup = document.getElementById('completion-popup');
    const closeCompletionPopupBtn = document.getElementById('closeCompletionPopupBtn');


    // --- Simulation State ---
    let simulationState = { current_frame: 0, is_paused: true, is_complete: false, speed_level: 50, target_cycle: null, max_frames: 30000, completion_stats: {}, devices: [] };
    let simulationConfig = { N: 0, total_layers: 0, total_layers_non_head: 0, memory_legend: "", compute_legend: "" };
    let currentIntervalSec = 1.0;
    let animationTimer = null;
    let simulationInitialized = false;
    let torusPlotInitialized = false;
    let isFetching = false;
    let simulationActive = false;
    let isResetting = false; // Flag to identify reset-triggered updates

    // --- SVG Rendering Constants ---
    const svgNS = "http://www.w3.org/2000/svg";
    const viewBoxWidth = 50;
    const centerX = viewBoxWidth / 2;
    const effectiveCenterY = viewBoxWidth / 2;
    const totalDistance           = effectiveCenterY * 0.98;
    const innerRadius             = effectiveCenterY * 0.35;
    const innerNodeRadius         = effectiveCenterY * 0.12;
    const outerNodeRadius         = effectiveCenterY * 0.15;
    const stallNodeRadius         = outerNodeRadius * 1.3;
    const desiredStallGap         = viewBoxWidth * 0.03;
    const stallNodeCenterOffset   = outerNodeRadius + desiredStallGap + stallNodeRadius;
    const computeArcRadiusScale   = 1.15;
    const arrowOffsetDist         = innerNodeRadius * 0.4;
    const labelOffsetDistance     = effectiveCenterY * 0.04;
    const baseStrokeWidth         = viewBoxWidth * 0.005;
    const outerLabelFontSize    = viewBoxWidth * 0.024;
    const innerLabelFontSize    = viewBoxWidth * 0.024;
    const stallLabelFontSize    = viewBoxWidth * 0.020;
    const transferLabelFontSize = viewBoxWidth * 0.022;
    const deviceOpacity           = 0.7;
    const innerNodeOpacity        = 0.8;
    const stallNodeOpacity        = 0.9;
    let svgElements = {};
    let nodePositions = {};
    let drawingBounds = { minY: Infinity, maxY: -Infinity, minX: Infinity, maxX: -Infinity };

    let baseAnimationHue = 240;

    let isResizing = false;
    let startY, initialTopHeight, initialBottomHeight;
    const minPaneHeight = 50;

    // --- Event Listeners ---
    paramsForm.addEventListener('submit', startSimulation);
    playBtn.addEventListener('click', () => sendControlCommand('play'));
    pauseBtn.addEventListener('click', () => sendControlCommand('pause'));
    speedSlider.addEventListener('change', () => sendControlCommand('set_speed', speedSlider.value));
    speedSlider.addEventListener('input', () => { speedValue.textContent = speedSlider.value; });
    runToBtn.addEventListener('click', () => {
        const cycle = parseInt(runToCycleInput.value, 10);
        if (!isNaN(cycle) && cycle >= 0) { sendControlCommand('run_to_cycle', cycle); }
        else { displayError("Invalid target cycle number."); }
    });
    // **NEW** Listener for closing the completion popup
    if (closeCompletionPopupBtn) {
        closeCompletionPopupBtn.addEventListener('click', () => {
            if (completionPopup) {
                completionPopup.style.display = 'none';
            }
        });
    }

    if (horizontalResizer) {
        horizontalResizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY; // Initial mouse Y position
            // Get computed heights in pixels
            initialTopHeight = svgContainer.offsetHeight;
            initialBottomHeight = torusPlotContainer.offsetHeight;

            // Add listeners to the document to track mouse movement everywhere
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);

            // Prevent text selection during drag
            e.preventDefault();
            // Optional: Change cursor for the whole body during resize
            document.body.style.cursor = 'row-resize';
            svgContainer.style.pointerEvents = 'none'; // Prevent interaction with panes during resize
            torusPlotContainer.style.pointerEvents = 'none';
        });
    }

    function handleMouseMove(e) {
        if (!isResizing) return;

        const currentY = e.clientY;
        const deltaY = currentY - startY;
        let newTopHeight = initialTopHeight + deltaY;
        let newBottomHeight = initialBottomHeight - deltaY;
        const totalAvailableHeight = initialTopHeight + initialBottomHeight; // Keep total height constant

        // Enforce minimum heights and adjust the other pane
        if (newTopHeight < minPaneHeight) {
            newTopHeight = minPaneHeight;
            newBottomHeight = totalAvailableHeight - newTopHeight;
        }
        if (newBottomHeight < minPaneHeight) {
            newBottomHeight = minPaneHeight;
            newTopHeight = totalAvailableHeight - newBottomHeight;
        }

        // Ensure we don't exceed original total height if mins push boundaries
        if (newTopHeight + newBottomHeight > totalAvailableHeight) {
             // This case should ideally be handled by the above logic, but as a fallback:
             // Prioritize the pane being expanded, adjust the other
            if (deltaY > 0) { // Increasing top height
                newBottomHeight = totalAvailableHeight - newTopHeight;
            } else { // Increasing bottom height
                 newTopHeight = totalAvailableHeight - newBottomHeight;
            }
        }


        svgContainer.style.height = `${newTopHeight}px`;
        torusPlotContainer.style.height = `${newBottomHeight}px`;

        // Resize Plotly plot during vertical drag
        if (torusPlotInitialized && torusPlotContainer) {
            try {
                Plotly.Plots.resize(torusPlotContainer);
            } catch (resizeError) {
                console.warn("Error resizing Plotly plot during drag:", resizeError);
            }
        }
        // Optional: Resize SVG viewbox if needed (might not be necessary with grid centering)
    }

    function handleMouseUp() {
        if (isResizing) {
            isResizing = false;
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
            svgContainer.style.pointerEvents = '';
            torusPlotContainer.style.pointerEvents = '';

            // Final resize call after mouse up
            if (torusPlotInitialized && torusPlotContainer) {
                try {
                    Plotly.Plots.resize(torusPlotContainer);
                } catch (resizeError) {
                    console.warn("Error resizing Plotly plot on mouseup:", resizeError);
                }
            }
        }
    }

    const handleWindowResize = debounce(() => {
        console.log("Window resized, attempting Plotly resize...");
        if (torusPlotInitialized && torusPlotContainer && torusPlotContainer.style.display !== 'none') {
            try {
                // Adjust heights based on new parent size before resizing Plotly
                const parentHeight = torusPlotContainer.parentElement.clientHeight;
                const resizerHeight = horizontalResizer.offsetHeight || 8;
                const availableHeight = parentHeight - resizerHeight;

                // Maintain the current *proportion* if possible, respecting min heights
                const currentTopHeight = svgContainer.offsetHeight;
                const currentBottomHeight = torusPlotContainer.offsetHeight;
                const currentTotal = currentTopHeight + currentBottomHeight;
                let newTopPx, newBottomPx;

                if (currentTotal > 0 && availableHeight > (minPaneHeight * 2)) { // Check for valid current heights
                     const topRatio = currentTopHeight / currentTotal;
                     newTopPx = Math.max(minPaneHeight, Math.floor(availableHeight * topRatio));
                     newBottomPx = Math.max(minPaneHeight, availableHeight - newTopPx);
                      // Re-check and adjust if min height caused overshoot
                     if(newTopPx + newBottomPx > availableHeight) {
                         newTopPx = availableHeight - newBottomPx; // Adjust top based on clamped bottom
                     }
                } else {
                     // Fallback to default split if current heights are invalid (e.g., 0)
                     const initialTopPercent = 0.60;
                     newTopPx = Math.max(minPaneHeight, Math.floor(availableHeight * initialTopPercent));
                     newBottomPx = Math.max(minPaneHeight, availableHeight - newTopPx);
                     if(newTopPx + newBottomPx > availableHeight) {
                          newBottomPx = availableHeight - newTopPx;
                          if(newBottomPx < minPaneHeight) {
                              newBottomPx = minPaneHeight;
                              newTopPx = availableHeight - newBottomPx;
                          }
                     }
                }

                console.log(`Window resize: Setting heights Top=${newTopPx}px, Bottom=${newBottomPx}px`);
                svgContainer.style.height = `${newTopPx}px`;
                torusPlotContainer.style.height = `${newBottomPx}px`;

                // Now resize Plotly
                Plotly.Plots.resize(torusPlotContainer);
                console.log("Plotly resize triggered by window resize.");
            } catch (resizeError) {
                console.warn("Error resizing Plotly plot on window resize:", resizeError);
            }
        } else {
            console.log("Window resize skipped: Plot not initialized or hidden.");
        }
         // Optional: Resize SVG if needed
         // setupSVG(simulationConfig); // This might be too heavy, maybe just adjust viewbox
    }, 250); // Debounce for 250ms

    window.addEventListener('resize', handleWindowResize);


    // --- API Communication Functions ---

    // --- API Communication Functions ---

    // **** Make the function async ****
    // **** Fully Revised startSimulation Function ****
    async function startSimulation(event) {
        event.preventDefault(); // Prevent default form submission behavior
        displayError(''); // Clear any previous error messages
        hideCompletionPopup(); // Ensure the completion popup is hidden

        if (simulationInitialized) {
            console.log("Attempting to reset simulation via API...");
            isResetting = true; // SET FLAG to track reset process state
            simulationActive = false; // Signal intention to stop the frontend loop
            stopAnimationLoop(); // Clear any scheduled timer for fetchUpdate

            console.log("Hiding and clearing legends...");
            if (memoryLegendArea) {
                memoryLegendArea.style.display = 'none'; // Hide memory legend
                 if(memoryLegendArea.querySelector('pre')) memoryLegendArea.querySelector('pre').textContent = ''; // Clear text
            }
            if (computeLegendArea) {
                computeLegendArea.style.display = 'none'; // Hide compute legend
                if(computeLegendArea.querySelector('pre')) computeLegendArea.querySelector('pre').textContent = ''; // Clear text
            }

            hideTorusPlot();

            try {
                // sendControlCommand('restart') will now:
                // 1. Call backend '/control' with 'restart'
                // 2. Backend resets its state, saves it to session
                // 3. Frontend clears svg.innerHTML = '' (DOES NOT CALL setupSVG)
                // 4. Frontend calls fetchInitialStateAfterReset (gets state 0, updates status/controls, DOES NOT CALL updateSVG)
                await sendControlCommand('restart');

                // If control command was successful (didn't throw error):
                console.log("Backend restart command successful. Simulation is now inactive and reset.");
                setFormEnabled(true); // Re-enable form, disable controls, reset simulationConfig.N = 0
                simulationInitialized = false; // Mark as not initialized
                torusPlotInitialized = false;
                console.log("UI reset to prepare state. Ready for new simulation parameters.");

            } catch (error) {
                 // Error during the sendControlCommand('restart') process
                 console.error("Error during backend restart command execution:", error);
                 // Attempt to reset UI to a safe state even on error
                 setFormEnabled(true); // Re-enable form
                 simulationInitialized = false; // Assume reset state on error
                 torusPlotInitialized = false;
                 if (svg) svg.innerHTML = ''; // Ensure SVG is clear on error too
                 updateStatusDisplay({ current_frame: 0, is_paused: true, is_complete: false, target_cycle: null }); // Basic reset status
                 updateControls({ is_paused: true, is_complete: false }); // Basic reset controls (disabled)
                 displayError(`Failed to reset simulation properly: ${error.message}`);
            } finally {
                isResetting = false; // UNSET FLAG
            }
            return; // Exit after handling reset
        }

        // --- START New Simulation ---
        // (This part runs only when simulationInitialized is false)
        console.log("Attempting to start NEW simulation...");
        if (memoryLegendArea) memoryLegendArea.style.display = 'none';
        if (computeLegendArea) computeLegendArea.style.display = 'none';

        // Ensure SVG is clear before setting up a new one
        if (svg) {
            svg.innerHTML = '';
            console.log("Cleared SVG content before starting new simulation.");
        } else {
            console.error("startSimulation (New): svg element reference is null!");
            displayError("Critical Error: SVG container not found. Cannot start simulation.");
            return; // Cannot proceed without SVG element
        }

        stopAnimationLoop(); // Ensure no old animation loops are running
        simulationActive = false; // Assume starting paused unless state says otherwise
        const formData = new FormData(paramsForm);
        console.log("Starting simulation with params:", Object.fromEntries(formData));
        setFormEnabled(false); // Disable form elements while starting
        statusArea.textContent = "Status: Initializing...";

        try {
            const response = await fetch('/start_simulation', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok && data.success) {
                console.log("Simulation started successfully via API.");
                // Store configuration and initial state
                simulationConfig = data.config;
                simulationState = data.state;
                currentIntervalSec = data.interval_sec;
                simulationInitialized = true; // Mark simulation as ready

                // --- SETUP SVG STRUCTURE and PERFORM INITIAL DRAW ---
                console.log("Calling setupSVG with new config:", simulationConfig);
                setupSVG(simulationConfig); // Build the SVG structure based on NEW config (N)
                console.log("Requesting initial SVG draw with state:", simulationState);
                window.requestAnimationFrame(() => {
                    console.log("Executing initial updateSVG call via requestAnimationFrame.");
                    updateSVG(simulationState); // Perform the initial draw based on NEW state
                });

                // Update UI elements
                displayLegends(simulationConfig);
                updateStatusDisplay(simulationState);
                updateControls(simulationState); // Enable/disable controls based on initial state
                if (simulationState.max_frames) {
                    runToCycleInput.max = simulationState.max_frames;
                }
                runToCycleInput.value = simulationState.current_frame;

                // *** NEW: Draw and show the Torus plot ***
                // Hardcode parameters for now - replace with values derived
                // from formData or simulationConfig later if needed.
                const modelStages = simulationConfig.N; // Example value from screenshot
                const dataParallelismFactor = 8; // Example value from screenshot
                drawTorusPlot(modelStages, dataParallelismFactor);
                showTorusPlot();
                torusPlotInitialized = true;
                // *** END NEW ***

                // Start the update loop ONLY if the initial state is not paused
                if (!simulationState.is_paused && !simulationState.is_complete) {
                    simulationActive = true;
                    console.log("Simulation starting in non-paused state. Activating fetch loop.");
                    fetchUpdate(); // Start the periodic updates
                } else {
                     simulationActive = false; // Remains inactive if starting paused or already complete (edge case)
                     console.log("Simulation starting in paused or complete state. Loop inactive.");
                }
                console.log("Frontend initialization complete.");

            } else {
                // Handle API error from /start_simulation
                handleApiError(data.error || response.statusText || "Unknown error starting simulation", "starting simulation");
                setFormEnabled(true); // Re-enable form on failure
                simulationActive = false;
                simulationInitialized = false;
                // Attempt to reset local state and UI display
                simulationState = { ...simulationState, is_paused: true, is_complete: false, current_frame: 0 };
                if (svg) svg.innerHTML = ''; // Clear SVG on failure
                updateStatusDisplay(simulationState);
                updateControls(simulationState);
                hideTorusPlot();
            }
        } catch (error) {
            // Handle network/fetch error for /start_simulation
            handleApiError(error.message, "starting simulation (network/fetch error)");
            setFormEnabled(true); // Re-enable form on failure
            simulationActive = false;
            simulationInitialized = false;
            // Attempt to reset local state and UI display
            simulationState = { ...simulationState, is_paused: true, is_complete: false, current_frame: 0 };
            if (svg) svg.innerHTML = ''; // Clear SVG on failure
            updateStatusDisplay(simulationState);
            updateControls(simulationState);
            hideTorusPlot();
        }
    } // End of startSimulation function

    // Fetches state updates periodically if simulationActive is true
    async function fetchUpdate() {
        clearTimeout(animationTimer); // Clear any previously scheduled timer

        // GUARD 1: Check state before fetching
        if (!simulationActive || simulationState.is_paused || simulationState.is_complete || isFetching) {
            // Log why we are returning early if needed for debugging:
            // console.log(`WorkspaceUpdate returning early: active=${simulationActive}, paused=${simulationState.is_paused}, complete=${simulationState.is_complete}, fetching=${isFetching}`);
            return;
        }

        isFetching = true;
        try {
            const response = await fetch('/get_state_update');

            // GUARD 2: Check state *immediately* after await returns
            // Catches cases where Reset was clicked during the network request
            if (!simulationActive) {
                console.log("fetchUpdate: Simulation became inactive during fetch. Aborting processing.");
                isFetching = false; // Ensure flag is reset
                return; // Exit before processing the potentially stale response
            }

            const data = await response.json();

            // Check response status and data content
            if (response.ok && data.success) {
                // GUARD 3: Check state *again* just after getting data, before processing
                // Belt-and-suspenders check against race conditions
                 if (!simulationActive) {
                    console.log("fetchUpdate: Simulation became inactive just after fetch success. Aborting UI update.");
                    isFetching = false;
                    return;
                 }

                // Update local state from the received data
                simulationState = data.state;
                currentIntervalSec = data.interval_sec;

                // GUARD 4: Final check *RIGHT BEFORE* scheduling UI update
                // Ensure simulation wasn't stopped between getting data and scheduling this
                if (!simulationActive) {
                    console.log("fetchUpdate: Simulation became inactive JUST before UI update. Aborting.");
                    isFetching = false;
                    return;
                }

                // Schedule UI updates for the next available animation frame
                window.requestAnimationFrame(() => {
                    // We assume if Guard 4 passed, it's safe to update here.
                    // Adding another check inside rAF might be possible but likely adds complexity.
                    // if (!simulationActive) { console.log("fetchUpdate/rAF: Inactive, skipping updateSVG."); return; }
                    updateSVG(simulationState); // Update the main SVG visualization
                });
                updateStatusDisplay(simulationState); // Update cycle count / status text
                updateControls(simulationState);    // Update button enabled/disabled states

                // Handle completion or schedule the next update
                if (simulationState.is_complete) {
                     displayCompletionStats(simulationState.completion_stats);
                     console.log("fetchUpdate: Simulation complete.");
                     simulationActive = false; // Stop the loop
                } else if (!simulationState.is_paused) {
                    // GUARD 5: Check state one last time before scheduling the next timer
                    if (simulationActive) {
                        const delay = Math.max(1, currentIntervalSec * 1000);
                        animationTimer = setTimeout(fetchUpdate, delay); // Schedule next fetch
                    } else {
                         console.log("fetchUpdate: Not scheduling next update - simulation became inactive.");
                    }
                } else {
                    // If state is paused, the loop stops naturally here
                    console.log("fetchUpdate: Simulation is paused. Loop stopping.");
                    simulationActive = false; // Ensure inactive flag reflects paused state
                }
            } else {
                // Handle backend API error (e.g., simulation failed on server)
                handleApiError(data.error || response.statusText || `Backend error during update`, "fetching update");
                simulationState.is_paused = true; // Assume pause on error
                simulationActive = false; // Stop the loop
                updateControls(simulationState); // Update controls to reflect paused state
            }
        } catch (error) {
            // Handle network errors or errors parsing JSON response
            handleApiError(error.message, "fetching update (network error)");
            simulationState.is_paused = true; // Assume pause on error
            simulationActive = false; // Stop the loop
            updateControls(simulationState); // Update controls
        } finally {
            isFetching = false; // Ensure fetching flag is reset
        }
    }

    async function sendControlCommand(command, value = null) {
        displayError('');
        console.log(`Sending command: ${command}`, value !== null ? `Value: ${value}` : '');

        // --- Stop any ongoing/pending updates ---
        stopAnimationLoop(); // Clear setTimeout for fetchUpdate
        simulationActive = false; // Tentatively mark inactive - fetchFullState will confirm final state
                                  // Guard in updateSVG handles lingering requestAnimationFrames

        hideCompletionPopup();

        try {
            const body = { command }; if (value !== null) { body.value = value; }
            const response = await fetch('/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            // data includes {success: bool, state_summary: {...}, interval_sec: float}
            // We will largely ignore state_summary now and rely on a subsequent fetch
            const data = await response.json();

            if (response.ok && data.success) {
                console.log(`Command '${command}' successful on backend.`);

                // --- Backend command succeeded, now fetch the definitive resulting state ---
                console.log(`Workspaceing full state after '${command}' command.`);
                if (command === 'restart') {
                    // Special handling for restart: clear SVG immediately, then fetch state 0
                    console.log("[sendControlCommand/restart] Clearing SVG content directly.");
                    if (svg) { svg.innerHTML = ''; }
                    else { console.error("[sendControlCommand/restart] svg element reference is null!"); }
                    
                    hideTorusPlot();
                    // Fetches state 0, updates local state, updates UI (status/controls), NO updateSVG
                    await fetchInitialStateAfterReset();

                     // Reset specific frontend flags AFTER state is confirmed reset
                     setFormEnabled(true);
                     simulationInitialized = false;
                     torusPlotInitialized = false;
                     console.log("Frontend reset complete after backend restart confirmed.");
                } else {
                    // For Play, Pause, RunTo, SetSpeed: Fetch the resulting full state
                    // This fetches state, updates local state, interval, status, controls, AND schedules updateSVG
                    await fetchFullStateAfterCommand();
                }

                // --- Logic based on the definitive simulationState updated by fetch functions ---
                const newState = simulationState; // Use the state updated by fetch functions

                // Determine if the simulation loop should run based on the fetched state
                simulationActive = !newState.is_paused && !newState.is_complete;

                if (simulationActive) {
                    console.log(`Command '${command}' resulted in active state. Ensuring fetch loop is running.`);
                    fetchUpdate(); // Start or continue the loop
                } else if (newState.is_complete) {
                    console.log(`Command '${command}' resulted in complete state.`);
                    displayCompletionStats(newState.completion_stats); // Ensure popup shows if completed via command
                    simulationActive = false; // Ensure inactive
                } else if (newState.is_paused) {
                    console.log(`Command '${command}' resulted in paused state.`);
                    simulationActive = false; // Ensure inactive
                } else {
                    console.log(`Command '${command}' resulted in inactive state (reason unknown/edge case).`);
                    simulationActive = false;
                }

                // Update the run-to cycle input field value based on the fetched state
                if (runToCycleInput) { runToCycleInput.value = newState.current_frame; }

            } else {
                // Backend command failed
                handleApiError(data.error || response.statusText || `Backend error for command '${command}'`, `sending command '${command}'`);
                simulationActive = false; // Stop on error
                // Attempt to fetch current state to sync UI, might show error state
                console.log("Attempting to fetch state after command failure to sync UI.");
                await fetchFullStateAfterCommand();
            }
        } catch (error) {
            // Network error
            handleApiError(error.message, `sending command '${command}' (network error)`);
            simulationActive = false; // Stop on error
            // Optionally try to resync UI after network error
            // console.log("Attempting to fetch state after network error to sync UI.");
            // await fetchFullStateAfterCommand();
        }
    }

    async function fetchFullStateAfterCommand() {
         try {
             const response = await fetch('/get_state_update'); const data = await response.json();
             if (response.ok && data.success) {
                 simulationState = data.state; currentIntervalSec = data.interval_sec;
                 window.requestAnimationFrame(() => { updateSVG(simulationState); });
                 updateStatusDisplay(simulationState); updateControls(simulationState);
             }
         } catch(e) { console.error("Error fetching state after command:", e)}
    }

     // This function is called by sendControlCommand after a successful 'restart' command
     // This function is called by sendControlCommand after a successful 'restart' command
     async function fetchInitialStateAfterReset() {
        console.log("[fetchInitialStateAfterReset] Attempting to fetch state after reset..."); // Log entry
        try {
            const response = await fetch('/get_state_update');
            // Log basic response info immediately
            console.log(`[fetchInitialStateAfterReset] Fetch response status: ${response.status}, OK: ${response.ok}`);

            // Try to parse JSON regardless of status, but handle potential errors
            let data = null;
            try {
                data = await response.json();
                console.log("[fetchInitialStateAfterReset] Received data:", data); // Log raw data
            } catch (jsonError) {
                console.error("[fetchInitialStateAfterReset] Error parsing JSON response:", jsonError);
                // Try to get response text if JSON parsing fails
                try {
                    const textResponse = await response.text();
                    console.error("[fetchInitialStateAfterReset] Response text:", textResponse);
                } catch (textError) {
                    console.error("[fetchInitialStateAfterReset] Could not get response text either:", textError);
                }
                handleApiError(`JSON Parse Error: ${jsonError.message}`, "fetching state after reset (JSON error)");
                return; // Stop processing if JSON is bad
            }

            // Now check if the fetch was logically successful according to backend and data exists
            if (response.ok && data && data.success) {
                console.log("[fetchInitialStateAfterReset] Fetch successful. Updating state and UI."); // Log success path
                simulationState = data.state; // Update local state with the fresh reset state
                currentIntervalSec = data.interval_sec;

                // --- SVG Clearing Fix ---
                // DO NOT CALL updateSVG here.
                // setupSVG already cleared the SVG in sendControlCommand('restart').
                // We want the SVG to REMAIN empty after a reset until a NEW simulation starts.
                // The updateSVG call below would redraw the base elements using the old N.
                /*
                window.requestAnimationFrame(() => {
                     console.log("[fetchInitialStateAfterReset] Calling updateSVG via requestAnimationFrame with fetched state."); // Log call schedule
                     updateSVG(simulationState); // <<< REMOVE OR COMMENT OUT THIS CALL
                });
                */
                console.log("[fetchInitialStateAfterReset] SKIPPING updateSVG call to keep SVG clear after reset.");

                updateStatusDisplay(simulationState); // Update status text (Cycle 0, Paused) - OK
                updateControls(simulationState);    // Update control buttons (should reflect paused state) - OK
            } else {
                // Log failure details clearly if response not ok OR data indicates failure
                console.error("[fetchInitialStateAfterReset] Fetch failed or backend indicated error.", { ok: response.ok, data_success: data ? data.success : 'N/A', error: data ? data.error : 'N/A' });
                handleApiError(data ? data.error : `HTTP ${response.status}`, "fetching state after reset");
                // updateSVG is NOT called in this failure case
            }
        } catch (error) {
            // Log network errors (e.g., server unreachable)
            console.error("[fetchInitialStateAfterReset] Network error during fetch:", error);
            handleApiError(error.message, "fetching state after reset (network error)");
            // updateSVG is NOT called in this failure case
        }
    }


    // --- SVG Setup Function ---
    // setupSVG remains unchanged
    function setupSVG(config) {

        console.log("[setupSVG] Checking svg variable before clearing:", svg);

        // Add a check to prevent errors if svg is null
        if (!svg) {
            console.error("[setupSVG] Cannot proceed: svg element reference is null!");
            // Maybe display an error to the user?
            displayError("Critical Error: Simulation SVG container not found.");
            return; // Stop execution of setupSVG if the element doesn't exist
        }

        console.log("[setupSVG] Attempting to clear SVG content...");
        try {
            svg.innerHTML = ''; // Clear existing content first
            console.log("[setupSVG] svg.innerHTML = '' executed.");
        } catch (e) {
            console.error("[setupSVG] Error setting svg.innerHTML:", e);
            // Depending on the error, might want to return or try to continue
        }

        svg.innerHTML = ''; svgElements = {};
        nodePositions = { outer: [], inner: [], stall: [], unit: [], angleRad: [], angleDeg: [], angleToPrev: [], angleToNext: [] };
        drawingBounds = { minY: Infinity, maxY: -Infinity, minX: Infinity, maxX: -Infinity };
        const N = config.N; if (N <= 0) { console.error("Invalid number of devices (N):", N); return; }
        for (let i = 0; i < N; i++) { const angleRad = (2 * Math.PI * i) / N - Math.PI / 2; const angleDeg = angleRad * 180 / Math.PI; const unitDir = { x: Math.cos(angleRad), y: Math.sin(angleRad) }; nodePositions.unit.push(unitDir); nodePositions.angleRad.push(angleRad); nodePositions.angleDeg.push(angleDeg); const outerPos = { x: centerX + totalDistance * unitDir.x, y: effectiveCenterY + totalDistance * unitDir.y }; const innerPos = { x: centerX + innerRadius * unitDir.x, y: effectiveCenterY + innerRadius * unitDir.y }; const stallPos = { x: outerPos.x + stallNodeCenterOffset * unitDir.x, y: outerPos.y + stallNodeCenterOffset * unitDir.y }; nodePositions.outer.push(outerPos); nodePositions.inner.push(innerPos); nodePositions.stall.push(stallPos); drawingBounds.minX = Math.min(drawingBounds.minX, innerPos.x - innerNodeRadius, outerPos.x - outerNodeRadius, stallPos.x - stallNodeRadius); drawingBounds.maxX = Math.max(drawingBounds.maxX, innerPos.x + innerNodeRadius, outerPos.x + outerNodeRadius, stallPos.x + stallNodeRadius); drawingBounds.minY = Math.min(drawingBounds.minY, innerPos.y - innerNodeRadius, outerPos.y - outerNodeRadius, stallPos.y - stallNodeRadius); drawingBounds.maxY = Math.max(drawingBounds.maxY, innerPos.y + innerNodeRadius, outerPos.y + outerNodeRadius, stallPos.y + stallNodeRadius); }
        for (let i = 0; i < N; i++) { if (N > 1) { const prevIdx = (i - 1 + N) % N; const nextIdx = (i + 1) % N; const vecToPrev = { x: nodePositions.outer[prevIdx].x - nodePositions.outer[i].x, y: nodePositions.outer[prevIdx].y - nodePositions.outer[i].y }; const vecToNext = { x: nodePositions.outer[nextIdx].x - nodePositions.outer[i].x, y: nodePositions.outer[nextIdx].y - nodePositions.outer[i].y }; nodePositions.angleToPrev.push(Math.atan2(vecToPrev.y, vecToPrev.x) * 180 / Math.PI); nodePositions.angleToNext.push(Math.atan2(vecToNext.y, vecToNext.x) * 180 / Math.PI); } else { nodePositions.angleToPrev.push(180); nodePositions.angleToNext.push(0); } }
        const padding = viewBoxWidth * 0.05; const vbMinX = drawingBounds.minX - padding; const vbMinY = drawingBounds.minY - padding; const vbWidth = (drawingBounds.maxX + padding) - vbMinX; const vbHeight = (drawingBounds.maxY + padding) - vbMinY; svg.setAttribute('viewBox', `${vbMinX} ${vbMinY} ${vbWidth} ${vbHeight}`);
        
        if (svgContainer && vbHeight > 0) { // Check container exists & avoid division by zero
            const aspectRatio = vbWidth / vbHeight;
            svgContainer.style.setProperty('--svg-aspect-ratio', aspectRatio);
            console.log(`Set --svg-aspect-ratio: ${aspectRatio}`); // For debugging
        } else if (svgContainer) {
             svgContainer.style.removeProperty('--svg-aspect-ratio');
        }
        
        const defs = document.createElementNS(svgNS, 'defs'); const markerSize = baseStrokeWidth * 10; const markerRefX = markerSize * 0.8; const marker = createMarker('arrowhead', 'black', markerSize, markerRefX, markerSize, markerSize); defs.appendChild(marker); svg.appendChild(defs);
        const g = addSvgElement(svg, 'g'); svgElements['main_group'] = g;
        for (let i = 0; i < N; i++) { const deviceId = `dev_${i}`; const outerPos = nodePositions.outer[i]; const innerPos = nodePositions.inner[i]; const stallPos = nodePositions.stall[i]; const unitDir = nodePositions.unit[i]; const baseColor = getColorForDevice(i, N, baseAnimationHue); const devGroup = addSvgElement(g, 'g', { id: deviceId }); svgElements[`${deviceId}_outer_circle`] = addSvgCircle(devGroup, outerPos.x, outerPos.y, outerNodeRadius, { fill: baseColor, stroke: 'black', 'stroke-width': baseStrokeWidth, opacity: deviceOpacity, id: `${deviceId}_outer_circle` }); const innerSquareSide = innerNodeRadius * 1.414; svgElements[`${deviceId}_inner_square`] = addSvgRect(devGroup, innerPos.x - innerSquareSide / 2, innerPos.y - innerSquareSide / 2, innerSquareSide, innerSquareSide, { fill: baseColor, stroke: 'black', 'stroke-width': baseStrokeWidth, opacity: innerNodeOpacity, id: `${deviceId}_inner_square` }); svgElements[`${deviceId}_stall_node`] = addSvgCircle(devGroup, stallPos.x, stallPos.y, stallNodeRadius, { fill: 'red', stroke: baseColor, 'stroke-width': baseStrokeWidth * 2, opacity: stallNodeOpacity, visibility: 'hidden', id: `${deviceId}_stall_node` }); svgElements[`${deviceId}_finish_indicator`] = addSvgCircle(devGroup, stallPos.x, stallPos.y, stallNodeRadius, { fill: 'lime', stroke: baseColor, 'stroke-width': baseStrokeWidth * 2, opacity: stallNodeOpacity, visibility: 'hidden', id: `${deviceId}_finish_indicator` }); svgElements[`${deviceId}_outer_label`] = addSvgText(devGroup, outerPos.x, outerPos.y, `D${i}`, { 'font-size': outerLabelFontSize, fill: 'black', id: `${deviceId}_outer_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_inner_label`] = addSvgText(devGroup, innerPos.x, innerPos.y, `Home`, { 'font-size': innerLabelFontSize, fill: 'black', id: `${deviceId}_inner_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_stall_label`] = addSvgText(devGroup, stallPos.x, stallPos.y, "", { 'font-size': stallLabelFontSize, fill: 'white', 'font-weight': 'bold', visibility: 'hidden', id: `${deviceId}_stall_label`, 'pointer-events': 'none' }); const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const inStartX = innerPos.x + unitDir.x * innerNodeRadius + offset.x; const inStartY = innerPos.y + unitDir.y * innerNodeRadius + offset.y; const outStartX = outerPos.x - unitDir.x * outerNodeRadius - offset.x; const outStartY = outerPos.y - unitDir.y * outerNodeRadius - offset.y; svgElements[`${deviceId}_in_arrow`] = addSvgPath(devGroup, `M ${inStartX} ${inStartY}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_in_arrow` }); svgElements[`${deviceId}_in_label`] = addSvgText(devGroup, inStartX, inStartY, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_in_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_out_arrow`] = addSvgPath(devGroup, `M ${outStartX} ${outStartY}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_out_arrow` }); svgElements[`${deviceId}_out_label`] = addSvgText(devGroup, outStartX, outStartY, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_out_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_ring_arrow`] = addSvgPath(devGroup, `M ${outerPos.x} ${outerPos.y}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_ring_arrow` }); svgElements[`${deviceId}_ring_label`] = addSvgText(devGroup, outerPos.x, outerPos.y, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_ring_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_compute_arc`] = addSvgPath(devGroup, "", { stroke: 'gray', 'stroke-width': baseStrokeWidth * 3, fill: 'none', visibility: 'hidden', id: `${deviceId}_compute_arc` }); }
    }

    // --- SVG Update Function ---
    function updateSVG(state) {
        // --- GUARD AGAINST DRAWING WHEN RESETTING OR NOT INITIALIZED ---
        // Check if the simulation is actually running/initialized.
        // After a reset, simulationInitialized will be false.
        // Also check the isResetting flag just in case.
        if (!simulationInitialized || isResetting) {
             console.warn(`[updateSVG] Drawing aborted. simulationInitialized=${simulationInitialized}, isResetting=${isResetting}`);
             // Optionally, force clear the SVG again here as a safety measure,
             // though setupSVG should have handled the primary clearing during reset.
             // if (svg) { svg.innerHTML = ''; }
             return; // <<< Stop execution of updateSVG if not initialized or resetting
        }
        // --- End Guard ---

        // Conditional logging during reset (using the global isResetting flag) - This log might now be less relevant due to the guard above
        // if (isResetting) {
        //     console.log("[updateSVG during reset] Received state object:", JSON.parse(JSON.stringify(state)));
        //     if (state.devices && state.devices.length > 0) {
        //          console.log("[updateSVG during reset] Details for Device 0:", JSON.parse(JSON.stringify(state.devices[0])));
        //     } else {
        //          console.log("[updateSVG during reset] No devices found in state.");
        //     }
        // }

        const N = simulationConfig.N; // N from the config used to build the SVG structure

        // --- Guard Clauses ---
        if (!state || !state.devices) {
            console.error("[updateSVG] Error: Received invalid state object.", state);
            return;
        }

        // --- This log should now only appear when drawing is allowed ---
        const numDevicesInState = state.devices.length;
        console.log(`[updateSVG] Processing ${numDevicesInState} devices from received state.`); // Log device count from state

        if (numDevicesInState === 0 && N > 0) { // Adjusted logic slightly: only warn if setup N > 0 but state has 0
             console.warn("[updateSVG] State has 0 devices, but config N is > 0. Clearing might be needed?");
            // Consider explicitly clearing SVG here if state is valid but has 0 devices?
             // if (svg) svg.innerHTML = '';
             return;
        }
        // --- End Guard Clauses ---

        // --- Main Update Loop ---
        for (let i = 0; i < N; i++) { // Loop based on N used during setupSVG. The guard above should prevent this loop from running inappropriately after a reset (when N is 0)
             const device = state.devices[i];
             if (!device || typeof device.id === 'undefined') {
                 // Log if state data for a device expected by N is missing
                 console.warn(`[updateSVG] Invalid device data at index ${i}`, device);
                 // Optionally hide all elements for this device?
                 // Or just continue, leaving elements as they were (potentially hidden)?
                 continue;
             }

             const deviceId = `dev_${i}`;
             const outerPos = nodePositions.outer[i]; // Assumes nodePositions is populated correctly by setupSVG
             const innerPos = nodePositions.inner[i];
             const stallPos = nodePositions.stall[i];
             const unitDir = nodePositions.unit[i];
             // Ensure calculation happens only if nodes exist
             const transferDist = (outerPos && innerPos) ? Math.max(0, Math.hypot(outerPos.x - innerPos.x, outerPos.y - innerPos.y) - outerNodeRadius - innerNodeRadius) : 0;

             // Update Outer Node Label (Device Status Text) - Assumes svgElements[...] exists from setupSVG
             if (svgElements[`${deviceId}_outer_label`]) {
                 updateMultiLineText(svgElements[`${deviceId}_outer_label`], device.status_text || `D${i}`, outerPos.x, outerPos.y, outerLabelFontSize);
             }

             // Update Stall Node Visibility & Log
             const stallNode = svgElements[`${deviceId}_stall_node`];
             const stallLabel = svgElements[`${deviceId}_stall_label`];
             if (stallNode && stallLabel) {
                 const isStalled = device.status === "Stalled";
                 const stallVisibility = isStalled ? 'visible' : 'hidden';
                 stallNode.setAttribute('visibility', stallVisibility);
                 if (isStalled) {
                     updateMultiLineText(stallLabel, device.stall_reason || "Stalled", stallPos.x, stallPos.y, stallLabelFontSize);
                     stallLabel.setAttribute('visibility', 'visible');
                 } else {
                     stallLabel.setAttribute('visibility', 'hidden');
                 }
                 // Unconditional Log Example (Uncomment if needed):
                 // console.log(`[updateSVG] Device ${i} stall node visibility set to: ${stallVisibility}`);
             }


             // Update Finish Indicator Visibility & Log
             const finishIndicator = svgElements[`${deviceId}_finish_indicator`];
             if (finishIndicator) {
                 const isFinished = device.status === "Finished";
                 const finishVisibility = isFinished ? 'visible' : 'hidden';
                 finishIndicator.setAttribute('visibility', finishVisibility);
                 // Unconditional Log Example (Uncomment if needed):
                 // console.log(`[updateSVG] Device ${i} finish indicator visibility set to: ${finishVisibility}`);
             }


             // Update Inner Node Label (Static "Home")
             if (svgElements[`${deviceId}_inner_label`]) {
                updateMultiLineText(svgElements[`${deviceId}_inner_label`], `Home`, innerPos.x, innerPos.y, innerLabelFontSize);
             }


             // Update Inbound Transfer Arrow and Label & Log
             const inArrow = svgElements[`${deviceId}_in_arrow`];
             const inLabel = svgElements[`${deviceId}_in_label`];
             let setInboundVisible = false;
             if (inArrow && inLabel && device.inbound && device.inbound.progress > 1e-6 && transferDist > 1e-6) {
                 setInboundVisible = true;
                 // Arrow/label drawing logic
                 const progress = Math.min(1.0, device.inbound.progress); const lenProg = progress * transferDist; const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const startX = innerPos.x + unitDir.x * innerNodeRadius + offset.x; const startY = innerPos.y + unitDir.y * innerNodeRadius + offset.y; const endX = startX + unitDir.x * lenProg; const endY = startY + unitDir.y * lenProg; const midX = (startX + endX) / 2; const midY = (startY + endY) / 2; const labelOffsetX = perpVec.x * labelOffsetDistance; const labelOffsetY = perpVec.y * labelOffsetDistance; inArrow.setAttribute('d', `M ${startX} ${startY} L ${endX} ${endY}`); inArrow.setAttribute('stroke', device.inbound.color || 'gray');
                 inArrow.setAttribute('visibility', 'visible');
                 updateMultiLineText(inLabel, device.inbound.label || '', midX + labelOffsetX, midY + labelOffsetY, transferLabelFontSize); inLabel.setAttribute('fill', device.inbound.color || 'gray');
                 inLabel.setAttribute('visibility', 'visible');
             } else if (inArrow && inLabel) { // Ensure elements exist before hiding
                 setInboundVisible = false;
                 inArrow.setAttribute('visibility', 'hidden');
                 inLabel.setAttribute('visibility', 'hidden');
             }
             // Unconditional Log:
             // console.log(`[updateSVG] Device ${i} inbound arrow visibility set to: ${setInboundVisible ? 'visible' : 'hidden'}`); // Kept commented as it's verbose


             // Update Outbound Transfer Arrow and Label & Log
             const outArrow = svgElements[`${deviceId}_out_arrow`];
             const outLabel = svgElements[`${deviceId}_out_label`];
             let setOutboundVisible = false;
             if (outArrow && outLabel && device.outbound && device.outbound.progress > 1e-6 && transferDist > 1e-6) {
                  setOutboundVisible = true;
                  // Arrow/label drawing logic
                  const progress = Math.min(1.0, device.outbound.progress); const lenProg = progress * transferDist; const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const startX = outerPos.x - unitDir.x * outerNodeRadius - offset.x; const startY = outerPos.y - unitDir.y * outerNodeRadius - offset.y; const endX = startX - unitDir.x * lenProg; const endY = startY - unitDir.y * lenProg; const midX = (startX + endX) / 2; const midY = (startY + endY) / 2; const labelOffsetX = -perpVec.x * labelOffsetDistance; const labelOffsetY = -perpVec.y * labelOffsetDistance; outArrow.setAttribute('d', `M ${startX} ${startY} L ${endX} ${endY}`); outArrow.setAttribute('stroke', device.outbound.color || 'gray');
                  outArrow.setAttribute('visibility', 'visible');
                  updateMultiLineText(outLabel, device.outbound.label || '', midX + labelOffsetX, midY + labelOffsetY, transferLabelFontSize); outLabel.setAttribute('fill', device.outbound.color || 'gray');
                  outLabel.setAttribute('visibility', 'visible');
             } else if (outArrow && outLabel) { // Ensure elements exist before hiding
                 setOutboundVisible = false;
                 outArrow.setAttribute('visibility', 'hidden');
                 outLabel.setAttribute('visibility', 'hidden');
             }
             // Unconditional Log:
             // console.log(`[updateSVG] Device ${i} outbound arrow visibility set to: ${setOutboundVisible ? 'visible' : 'hidden'}`); // Kept commented


            // Update Peer Transfer Arrow and Label & Log
            const ringArrow = svgElements[`${deviceId}_ring_arrow`];
            const ringLabel = svgElements[`${deviceId}_ring_label`];
            let setPeerVisible = false;
             if (ringArrow && ringLabel && device.peer && device.peer.progress > 1e-6 && N > 1) {
                const progress = Math.min(1.0, device.peer.progress); const peerId = device.peer.target_peer;
                 if (peerId >= 0 && peerId < N && peerId !== i && nodePositions.outer[peerId]) { // Check peerId validity
                    // Arrow/label drawing logic
                    const targetPos = nodePositions.outer[peerId]; const startPos = outerPos; const dx = targetPos.x - startPos.x; const dy = targetPos.y - startPos.y; const dist = Math.hypot(dx, dy); const startAngleRad = Math.atan2(dy, dx); const startEdge = pointOnCircle(startPos.x, startPos.y, outerNodeRadius, startAngleRad * 180 / Math.PI); const endAngleRad = Math.atan2(-dy, -dx); const endEdge = pointOnCircle(targetPos.x, targetPos.y, outerNodeRadius, endAngleRad * 180 / Math.PI); const currentEdgeX = startEdge.x + (endEdge.x - startEdge.x) * progress; const currentEdgeY = startEdge.y + (endEdge.y - startEdge.y) * progress; const sweepFlag = device.peer.direction > 0 ? 1 : 0; let arcRadius = dist * 0.6;
                    if (Math.hypot(currentEdgeX - startEdge.x, currentEdgeY - startEdge.y) > 1e-3) { // Check if progress is significant enough to draw
                        setPeerVisible = true;
                        const pathData = `M ${startEdge.x} ${startEdge.y} A ${arcRadius} ${arcRadius} 0 0 ${sweepFlag} ${currentEdgeX} ${currentEdgeY}`; ringArrow.setAttribute('d', pathData); ringArrow.setAttribute('stroke', device.peer.color || 'indigo');
                        ringArrow.setAttribute('visibility', 'visible');
                        const midX = (startEdge.x + currentEdgeX) / 2; const midY = (startEdge.y + currentEdgeY) / 2; const vecCenterX = midX - centerX; const vecCenterY = midY - effectiveCenterY; const vecCenterMag = Math.hypot(vecCenterX, vecCenterY); const outX = vecCenterMag > 1e-6 ? vecCenterX / vecCenterMag : 0; const outY = vecCenterMag > 1e-6 ? vecCenterY / vecCenterMag : 0; const labelOffsetMag = labelOffsetDistance * 2.0; const labelX = midX + outX * labelOffsetMag; const labelY = midY + outY * labelOffsetMag; updateMultiLineText(ringLabel, device.peer.label || '', labelX, labelY, transferLabelFontSize); ringLabel.setAttribute('fill', device.peer.color || 'indigo');
                        ringLabel.setAttribute('visibility', 'visible');
                    } else {
                        setPeerVisible = false; // Explicitly hide if progress is too small
                        ringArrow.setAttribute('visibility', 'hidden');
                        ringLabel.setAttribute('visibility', 'hidden');
                    }
                 } else {
                     setPeerVisible = false; // Hide if peerId invalid or target node missing
                     ringArrow.setAttribute('visibility', 'hidden');
                     ringLabel.setAttribute('visibility', 'hidden');
                 }
             } else if (ringArrow && ringLabel) { // Ensure elements exist before hiding if no peer data
                 setPeerVisible = false;
                 ringArrow.setAttribute('visibility', 'hidden');
                 ringLabel.setAttribute('visibility', 'hidden');
             }
             // Unconditional Log:
             // console.log(`[updateSVG] Device ${i} peer arrow visibility set to: ${setPeerVisible ? 'visible' : 'hidden'}`); // Kept commented


            // Update Compute Arc & Log
            const computeArc = svgElements[`${deviceId}_compute_arc`];
            let setComputeVisible = false;
            if (computeArc && device.compute && device.compute.progress > 1e-6) {
                 // Angle calculation logic
                 const progress = Math.min(1.0, device.compute.progress); const type = device.compute.type; const arcOuterRadius = outerNodeRadius * computeArcRadiusScale; const anglePrev = nodePositions.angleToPrev[i]; const angleNext = nodePositions.angleToNext[i]; let theta1Deg = 0, theta2Deg = 0, totalSweepDeg = 0; if (N > 1) { if (type === "Fwd") { const angleStartDeg = (device.compute.layer > 0 || N <= 1) ? anglePrev : (angleNext - 180); const angleEndTargetDeg = angleNext; totalSweepDeg = (angleEndTargetDeg - angleStartDeg + 360) % 360; if (N === 1) totalSweepDeg = 360; theta1Deg = angleStartDeg; theta2Deg = angleStartDeg + progress * totalSweepDeg; } else if (type === "Head") { theta1Deg = anglePrev; totalSweepDeg = 360; theta2Deg = theta1Deg + progress * totalSweepDeg; } else { const angleStartDeg = angleNext; const angleEndTargetDeg = anglePrev; totalSweepDeg = (angleStartDeg - angleEndTargetDeg + 360) % 360; if (N === 1) totalSweepDeg = 360; const currentSweepDeg = progress * totalSweepDeg; theta1Deg = angleStartDeg - currentSweepDeg; theta2Deg = angleStartDeg; } } else { totalSweepDeg = 360; theta1Deg = -90; theta2Deg = theta1Deg + progress * totalSweepDeg; }

                 if (Math.abs(progress * totalSweepDeg) > 1e-3) { // Check if arc angle is significant enough to draw
                    setComputeVisible = true;
                    // Path drawing logic
                    const startPoint = pointOnCircle(outerPos.x, outerPos.y, arcOuterRadius, theta1Deg); const endPoint = pointOnCircle(outerPos.x, outerPos.y, arcOuterRadius, theta2Deg); const largeArcFlag = Math.abs(progress * totalSweepDeg) >= 360 ? 1 : (Math.abs(progress * totalSweepDeg) > 180 ? 1 : 0); const sweepFlagArc = 1; // Assuming clockwise/consistent sweep direction
                    const pathData = `M ${startPoint.x} ${startPoint.y} A ${arcOuterRadius} ${arcOuterRadius} 0 ${largeArcFlag} ${sweepFlagArc} ${endPoint.x} ${endPoint.y}`;
                    computeArc.setAttribute('d', pathData);
                    computeArc.setAttribute('stroke', device.compute.color || 'gray');
                    computeArc.setAttribute('visibility', 'visible');
                 } else {
                     setComputeVisible = false; // Explicitly hide if arc angle too small
                     computeArc.setAttribute('visibility', 'hidden');
                 }
            } else if (computeArc) { // Ensure element exists before hiding if no compute data
                 setComputeVisible = false;
                 computeArc.setAttribute('visibility', 'hidden');
            }
             // Unconditional Log:
             // console.log(`[updateSVG] Device ${i} compute arc visibility set to: ${setComputeVisible ? 'visible' : 'hidden'}`); // Kept commented

         } // End of device loop
    } // End of updateSVG function

    // --- UI Update Functions ---
    function updateStatusDisplay(state) { let statusText = `Cycle: ${state.current_frame}`; if (state.is_complete) { statusText += " | Status: Complete"; } else if (state.is_paused) { statusText += " | Status: Paused"; } else { statusText += " | Status: Running"; } if (state.target_cycle !== null && state.target_cycle > state.current_frame && !state.is_complete) { statusText += ` (Target: ${state.target_cycle})`; } statusArea.textContent = statusText; runToCycleInput.value = state.current_frame; }
    function updateControls(state) { const canControl = simulationConfig.N > 0; const isRunning = !state.is_paused && !state.is_complete; playBtn.disabled = !canControl || isRunning || state.is_complete; pauseBtn.disabled = !canControl || !isRunning; speedSlider.disabled = !canControl; runToBtn.disabled = !canControl; runToCycleInput.disabled = !canControl; if (speedSlider.value != state.speed_level) { speedSlider.value = state.speed_level; speedValue.textContent = state.speed_level; } }
    function displayLegends(config) { if (memoryLegendArea) { memoryLegendArea.style.display = config.memory_legend ? 'block' : 'none'; if (config.memory_legend) memoryLegendArea.querySelector('pre').textContent = config.memory_legend; } if (computeLegendArea) { computeLegendArea.style.display = config.compute_legend ? 'block' : 'none'; if (config.compute_legend) computeLegendArea.querySelector('pre').textContent = config.compute_legend; } }
    // **MODIFIED** displayCompletionStats to show popup
    function displayCompletionStats(stats) {
        if (completionPopup && completionArea && stats && stats.text) {
            completionArea.textContent = stats.text; // Set text content
            completionPopup.style.display = 'block'; // Show the popup
        } else {
            hideCompletionPopup(); // Hide if no stats or elements missing
        }
    }
    // **NEW** Helper function to hide completion popup
    function hideCompletionPopup() {
        if (completionPopup) {
            completionPopup.style.display = 'none';
        }
    }
    function displayError(message) { if (errorMessageDiv) { errorMessageDiv.textContent = message; errorMessageDiv.classList.toggle('visible', !!message); } }
    function handleApiError(errorMsg, context) { const fullMsg = `Error ${context}: ${errorMsg}`; console.error(fullMsg); displayError(fullMsg); }
    function setFormEnabled(isEnabled) {
        // --- Handle Form Parameter Inputs ---
        Array.from(paramsForm.elements).forEach(el => {
            // Example: Keep certain form elements always disabled in prepare state if needed
            if (isEnabled && (el.id === 'attn_type' || el.id == 'chunk_type' || el.id === 'min_chunk_size')) {
                 el.disabled = true;
            } else if (el !== submitButton) { // Don't disable the submit/reset button itself here
                // Disable form inputs when simulation is active (!isEnabled)
                // Enable form inputs when in prepare state (isEnabled)
                el.disabled = !isEnabled;
            }
        });

        // --- Handle Simulation Control Elements ---
        // Use the specific variables for clarity and robustness, especially for disabling
        // The controlElements loop is useful primarily when ENABLING controls (isEnabled = false),
        // deferring detailed enable/disable logic to updateControls.

        // --- Explicitly Set State for Prepare Mode (isEnabled = true) ---
        if (isEnabled) {
            console.log("Setting form/controls to ENABLED (Prepare State)...");

            // Disable all simulation controls
            if (playBtn) {
                playBtn.disabled = true;
                console.log(`playBtn (${playBtn.id}) disabled:`, playBtn.disabled);
            } else { console.error("setFormEnabled: playBtn reference is null"); }

            if (pauseBtn) {
                pauseBtn.disabled = true;
                console.log(`pauseBtn (${pauseBtn.id}) disabled:`, pauseBtn.disabled);
            } else { console.error("setFormEnabled: pauseBtn reference is null"); }

            if (speedSlider) {
                speedSlider.disabled = true;
                console.log(`speedSlider (${speedSlider.id}) disabled:`, speedSlider.disabled);
            } else { console.error("setFormEnabled: speedSlider reference is null"); }

            if (runToCycleInput) {
                runToCycleInput.disabled = true;
                console.log(`runToCycleInput (${runToCycleInput.id}) disabled:`, runToCycleInput.disabled);
            } else { console.error("setFormEnabled: runToCycleInput reference is null"); }

            if (runToBtn) {
                runToBtn.disabled = true;
                console.log(`runToBtn (${runToBtn.id}) disabled:`, runToBtn.disabled);
            } else { console.error("setFormEnabled: runToBtn reference is null"); }

            // Also adjust visual cues like opacity if needed
            if (speedValue) {
                speedValue.style.opacity = '0.7'; // Example: Dim the speed value display
            }

            hideTorusPlot(); // Hide plot when form is enabled (reset state)
            torusPlotInitialized = false;

            // Reset simulation config N value for correct canControl logic later
            simulationConfig.N = 0;
            console.log("simulationConfig.N reset to 0.");

            // Set Submit button text
            if (submitButton) {
                submitButton.textContent = 'Prepare Simulation';
                submitButton.disabled = false; // Ensure Prepare/Reset button is always usable
                console.log(`submitButton (${submitButton.id}) text set, enabled:`, !submitButton.disabled);
            } else { console.error("setFormEnabled: submitButton reference is null"); }

        } else {
            // --- Handle State for Simulation Active Mode (isEnabled = false) ---
            console.log("Setting form/controls to DISABLED (Simulation Active)...");
            // Form inputs were already disabled by the loop above.
            // For controls, enabling them here is complex because their state
            // depends on whether the simulation is running, paused, or complete.
            // It's better to let `updateControls` handle the specific state of each control button.
            // We just ensure the submit button is correctly labeled "Reset".
            if (submitButton) {
                submitButton.textContent = 'Reset Simulation';
                submitButton.disabled = false; // Ensure Prepare/Reset button is always usable
                console.log(`submitButton (${submitButton.id}) text set, enabled:`, !submitButton.disabled);
            } else { console.error("setFormEnabled: submitButton reference is null"); }

            if (torusPlotInitialized) {
                showTorusPlot();
            } else {
                hideTorusPlot(); // Ensure it's hidden if not yet initialized
            }
             // We rely on updateControls() being called shortly after this
             // (e.g., after fetching initial state or handling a control command)
             // to set the correct enabled/disabled state for Play, Pause, Slider etc.
             console.log("Control button states will be set by updateControls() based on simulation state.");
        }
    }

    // --- Animation Loop Control ---
    function startAnimationLoop() { stopAnimationLoop(); if (!simulationState.is_paused && !simulationState.is_complete) { const delay = Math.max(1, currentIntervalSec * 1000); animationTimer = setTimeout(fetchUpdate, delay); } }
    function stopAnimationLoop() { clearTimeout(animationTimer); animationTimer = null; }

    // --- SVG Helper Functions ---
    // SVG Helpers remain unchanged
    function addSvgElement(parent, tagName, attributes = {}) { const el = document.createElementNS(svgNS, tagName); for (const attr in attributes) { el.setAttribute(attr, attributes[attr]); } parent.appendChild(el); return el; }
    function addSvgCircle(parent, cx, cy, r, attributes = {}) { return addSvgElement(parent, 'circle', { cx, cy, r, ...attributes }); }
    function addSvgRect(parent, x, y, width, height, attributes = {}) { return addSvgElement(parent, 'rect', { x, y, width, height, ...attributes }); }
    function addSvgPath(parent, d, attributes = {}) { return addSvgElement(parent, 'path', { d, ...attributes }); }
    function addSvgText(parent, x, y, textContent, attributes = {}) { attributes['dominant-baseline'] = attributes['dominant-baseline'] || 'central'; attributes['text-anchor'] = attributes['text-anchor'] || 'middle'; const el = addSvgElement(parent, 'text', { x, y, ...attributes }); updateMultiLineText(el, textContent, x, y, parseFloat(attributes['font-size'] || 1)); return el; }
    function createMarker(id, color, size, refX, markerWidth, markerHeight) { const marker = document.createElementNS(svgNS, 'marker'); marker.setAttribute('id', id); marker.setAttribute('viewBox', `0 0 ${size*1.2} ${markerHeight*1.2}`); marker.setAttribute('markerWidth', markerWidth); marker.setAttribute('markerHeight', markerHeight); marker.setAttribute('refX', refX); marker.setAttribute('refY', markerHeight / 2); marker.setAttribute('orient', 'auto-start-reverse'); const path = addSvgElement(marker, 'path', { d: `M 0 0 L ${size} ${markerHeight / 2} L 0 ${markerHeight} z`, fill: color }); return marker; }
    function updateMultiLineText(textElement, text, x, y, fontSize) { textElement.textContent = ''; textElement.setAttribute('x', x); textElement.setAttribute('y', y); if (fontSize) { textElement.setAttribute('font-size', fontSize); } textElement.setAttribute('dominant-baseline', 'central'); textElement.setAttribute('text-anchor', 'middle'); const lines = String(text).split('\n'); const numLines = lines.length; const lineSpacingEm = 1.2; const initialDyEm = numLines > 1 ? - (numLines - 1) * 0.5 * lineSpacingEm : 0; lines.forEach((line, index) => { const tspan = document.createElementNS(svgNS, "tspan"); tspan.setAttribute("x", x); tspan.setAttribute("dy", index === 0 ? `${initialDyEm}em` : `${lineSpacingEm}em`); tspan.textContent = line; textElement.appendChild(tspan); }); }
    function pointOnCircle(cx, cy, radius, angleDegrees) { const angleRadians = angleDegrees * Math.PI / 180; return { x: cx + radius * Math.cos(angleRadians), y: cy + radius * Math.sin(angleRadians) }; }
    
    function getColorForDevice(index, total, baseHue) {
        // --- Customizable Parameters for the Gradient ---
        const startLightness = 90; // Lightest shade (%) - e.g., 75%
        const endLightness = 10;   // Darkest shade (%) - e.g., 30%
        const saturation = 95;     // Saturation (%) - keep reasonably high for color intensity
    
        // --- Input Validation ---
        if (total <= 0) {
            // Return a default fallback color if total is invalid
            console.warn("getColorForDevice called with invalid total:", total);
            return `hsl(${baseHue}, ${saturation}%, 50%)`; // Return mid-lightness default
        }
        // Ensure baseHue is within range
        const validatedHue = ((baseHue % 360) + 360) % 360; // Wraps hue to 0-359
    
        // --- Calculate Interpolation Factor ---
        // Factor goes from 0 for the first device (index 0) to 1 for the last device
        // Avoid division by zero if total is 1
        const factor = (total > 1) ? index / (total - 1) : 0;
    
        // --- Interpolate Lightness ---
        // Linear interpolation from startLightness to endLightness
        const currentLightness = startLightness + factor * (endLightness - startLightness);
    
        // --- Return the HSL Color String ---
        return `hsl(${validatedHue}, ${saturation}%, ${currentLightness}%)`;
    }



    // *** NEW: Torus Plot Functions ***
    function linspace(start, end, num) {
        if (num <= 1) return [start];
        const step = (end - start) / (num - 1);
        const arr = [];
        for (let i = 0; i < num; i++) {
            arr.push(start + i * step);
        }
        return arr;
    }

    function hslToRgbString(h, s, l) {
        // Simplified conversion, assumes s=0.75, l=0.65 as in getColorForDevice
        s = 0.75;
        l = 0.65;
        let r, g, b;
        if (s === 0) {
            r = g = b = l; // achromatic
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1 / 6) return p + (q - p) * 6 * t;
                if (t < 1 / 2) return q;
                if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            h /= 360; // Convert hue to range [0, 1]
            r = hue2rgb(p, q, h + 1 / 3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1 / 3);
        }
        const to255 = x => Math.round(x * 255);
        return `rgb(${to255(r)},${to255(g)},${to255(b)})`;
    }

    function drawTorusPlot(N_slices, M_nodes_per_slice) {
        if (!torusPlotContainer) return;
        console.log(`Drawing Torus Plot: Slices=${N_slices}, Nodes/Slice=${M_nodes_per_slice}`);

        // --- Parameters (from Python) ---
        const R_major = 4.0;
        const r_minor = 1.0;
        const twist_factor = 1;

        // --- Calculate Angles ---
        // Need endpoint=false equivalent for linspace
        const phi_slices = linspace(0, 2 * Math.PI, N_slices + 1).slice(0, N_slices); // Exclude endpoint
        const theta_nodes_base = linspace(0, 2 * Math.PI, M_nodes_per_slice + 1).slice(0, M_nodes_per_slice); // Exclude endpoint

        // --- Calculate Node Coordinates ---
        const nodes = []; // Will be array of slices, each slice an array of nodes [x,y,z]
        const all_x = []; // Flat arrays for Plotly scatter
        const all_y = [];
        const all_z = [];
        const node_colors = []; // Color for each node
        const slice_labels = []; // For annotations

        const face_i = []; // Stores the index (in all_x/y/z) of the 1st vertex of each triangle
        const face_j = []; // Stores the index of the 2nd vertex of each triangle
        const face_k = []; // Stores the index of the 3rd vertex of each triangle

        // --- Define Colors (Simplified HSL) ---
        const node_wire_colors_hsl = []; // Store HSL [h, s, l]
        const node_wire_colors_rgb = []; // Store Plotly RGB strings
        for (let j = 0; j < M_nodes_per_slice; j++) {
             const hue = (j * 360 / M_nodes_per_slice) % 360;
             node_wire_colors_hsl.push([hue, 0.75, 0.65]); // Match getColorForDevice style
             node_wire_colors_rgb.push(hslToRgbString(hue, 0.75, 0.65));
        }

        // --- Main Loop to Calculate Coordinates and Face Indices ---
        for (let k = 0; k < N_slices; k++) { // Loop through each slice
            const phi = phi_slices[k];
            const slice_nodes = [];
            let sum_x = 0, sum_y = 0, sum_z = 0;
    
            // Calculate the starting index in the flat arrays (all_x/y/z) for this slice's vertices
            const baseVertexIndex = k * M_nodes_per_slice;
    
            for (let j = 0; j < M_nodes_per_slice; j++) { // Loop through nodes within the slice
                const theta_base = theta_nodes_base[j];
                const theta_twisted = theta_base + twist_factor * phi;
    
                const radius_factor = R_major + r_minor * Math.cos(theta_twisted);
                const x = radius_factor * Math.cos(phi);
                const y = radius_factor * Math.sin(phi);
                const z = r_minor * Math.sin(theta_twisted);
    
                slice_nodes.push([x, y, z]);
                // Add coordinates to flat arrays
                all_x.push(x);
                all_y.push(y);
                all_z.push(z);
                // Assign color based on node index j within the slice
                node_colors.push(node_wire_colors_rgb[j]);
    
                sum_x += x; sum_y += y; sum_z += z;
            }
            nodes.push(slice_nodes); // Store slice data if needed elsewhere
    
            // Calculate centroid for slice label
            const center_x = sum_x / M_nodes_per_slice;
            const center_y = sum_y / M_nodes_per_slice;
            const center_z = sum_z / M_nodes_per_slice;
            slice_labels.push({
                 x: center_x, y: center_y, z: center_z,
                 text: k.toString(),
                 font: { color: 'darkgreen', size: 9 }, // Removed weight for simplicity
                 showarrow: false,
            });
    
            // *** NEW: Generate Face Triangles for this Slice ***
            // We create triangles using a "fan" approach from the first vertex (j=0)
            // For a polygon v0, v1, v2, v3, ..., triangles are (v0,v1,v2), (v0,v2,v3), ...
            if (M_nodes_per_slice >= 3) { // Need at least 3 vertices to form a triangle
                const v0_idx = baseVertexIndex + 0; // Index of the first vertex in the flat arrays
                for (let m = 0; m < M_nodes_per_slice - 2; m++) {
                    // Indices of the other two vertices in the flat arrays
                    const v1_idx = baseVertexIndex + (m + 1);
                    const v2_idx = baseVertexIndex + (m + 2);
    
                    // Add the indices for this triangle to the face arrays
                    face_i.push(v0_idx);
                    face_j.push(v1_idx);
                    face_k.push(v2_idx);
                }
            }
            // *** END NEW ***
    
        } // End of slice loop (k)

        // --- Prepare Plotly Traces ---
        const traces = [];
        // --- Face Traces ---
        if (face_i.length > 0) { // Check if any faces were generated
            traces.push({
               type: 'mesh3d',
               x: all_x, // Use the flat arrays of all vertex coordinates
               y: all_y,
               z: all_z,
               i: face_i, // Use the calculated triangle indices
               j: face_j,
               k: face_k,
               color: 'black',      // Face color
               opacity: 0.3,        // Desired opacity
               flatshading: true,   // Use flat shading for uniform face color
               hoverinfo: 'none',   // Disable hover info for the mesh faces
               showlegend: false    // Do not show this mesh in the legend
           });
       }

        // 1. Nodes (Scatter3d)
        traces.push({
            x: all_x,
            y: all_y,
            z: all_z,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                color: node_colors, // Array of colors for each point
                size: 6, // Adjust size to match Matplotlib s=75 visually
                symbol: 'circle',
                line: { // Edge color
                    color: 'black',
                    width: 1.5
                },
                opacity: 1.0 // Approximate depthshade/alpha
            },
            showlegend: false,
            hoverinfo: 'none' // Disable hover text for nodes if desired
        });

        // 2. Wires (Line traces, one per color/sequence)
        for (let j = 0; j < M_nodes_per_slice; j++) { // This loop starts with j = 0
            const wire_x = [];
            const wire_y = [];
            const wire_z = [];
            // Collect points for this specific 'j' across all slices 'k'
            for (let k = 0; k < N_slices; k++) {
                // Access node data: nodes[slice_index][node_index_within_slice]
                if (nodes && nodes[k] && nodes[k][j] && nodes[k][j].length === 3) {
                    wire_x.push(nodes[k][j][0]);
                    wire_y.push(nodes[k][j][1]);
                    wire_z.push(nodes[k][j][2]);
                } else {
                    console.error(`Error accessing node data for k=${k}, j=${j}`);
                }
            }
            // Close the loop by adding the first point (k=0 for this j) again
            if (nodes && nodes[0] && nodes[0][j] && nodes[0][j].length === 3) {
                wire_x.push(nodes[0][j][0]);
                wire_y.push(nodes[0][j][1]);
                wire_z.push(nodes[0][j][2]);
            } else {
                 console.error(`Error accessing node data for closing loop at j=${j}`);
            }


            // Create the trace object for this wire
            const trace = {
                x: wire_x,
                y: wire_y,
                z: wire_z,
                mode: 'lines',
                type: 'scatter3d',
                name: `Seq. ${j}`, // Name for legend (e.g., "Seq. 0", "Seq. 1", ...)
                line: {
                    color: node_wire_colors_rgb[j], // Get color for index j
                    width: 2
                },
                 hoverinfo: 'none',
                 showlegend: true
            };

            traces.push(trace); // Add trace for this 'j'
        }


        // --- Define Layout ---
        // Calculate plot bounds roughly like in Python
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
        all_x.forEach(v => { minX = Math.min(minX, v); maxX = Math.max(maxX, v); });
        all_y.forEach(v => { minY = Math.min(minY, v); maxY = Math.max(maxY, v); });
        all_z.forEach(v => { minZ = Math.min(minZ, v); maxZ = Math.max(maxZ, v); });
        const buffer = 0.1 * (R_major + r_minor);
        minX -= buffer; maxX += buffer; minY -= buffer; maxY += buffer; minZ -= buffer; maxZ += buffer;

        const layout = {
            // *** MODIFIED: Title Object for Centering ***
            title: {
                text: `Model Stages: ${N_slices}, Data Parallelism Factor: ${M_nodes_per_slice}`,
                x: 0.5, // Center horizontally
                xanchor: 'center', // Anchor the center of the text at x=0.5
                y: 0.95, // Adjust vertical position if needed
                yanchor: 'top'
            },
           showlegend: true,
            legend: {
                title: { text: '' },
                x: 1, // Position legend slightly to the right
                xanchor: 'left',
                y: 0.5,
                yanchor: 'middle'
             },
            // *** MODIFIED: Margins (Balanced L/R) ***
            margin: { l: 20, r: 20, b: 20, t: 40 }, // Balanced left/right, added small bottom margin
           scene: {
               xaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
               yaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
               zaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
               aspectmode: 'data', // Use 'auto' or 'cube' if 'data' looks distorted
               camera: { eye: { x: 1.25, y: 1.25, z: 1.25 } },
               bgcolor: 'rgba(0,0,0,0)', // Transparent background
               annotations: slice_labels
           },
            paper_bgcolor: 'rgba(0,0,0,0)', // Make background behind plot transparent too
            plot_bgcolor: 'rgba(0,0,0,0)'
       };
      

        // --- Render the Plot ---
        Plotly.newPlot(torusPlotContainer, traces, layout, { responsive: true });
    }

    function showTorusPlot() {
        if (torusPlotContainer && horizontalResizer) {
            // *** NEW: Set initial heights and show resizer ***
            const parentHeight = torusPlotContainer.parentElement.clientHeight; // Get height of .simulation-area
            const resizerHeight = horizontalResizer.offsetHeight || 8; // Use defined or default height
            const availableHeight = parentHeight - resizerHeight;

            // Example initial split (adjust percentage as needed)
            const initialTopPercent = 0.60; // 60% for SVG container
            let initialTopPx = Math.max(minPaneHeight, Math.floor(availableHeight * initialTopPercent));
            let initialBottomPx = Math.max(minPaneHeight, availableHeight - initialTopPx);

             // Adjust if total exceeds available due to min height constraints
             if(initialTopPx + initialBottomPx > availableHeight) {
                // Prioritize top pane perhaps? Or split remaining space?
                // Simple approach: Reduce bottom pane if possible
                initialBottomPx = availableHeight - initialTopPx;
                if (initialBottomPx < minPaneHeight) {
                    initialBottomPx = minPaneHeight;
                    initialTopPx = availableHeight - initialBottomPx; // Adjust top accordingly
                }
             }


            console.log(`Setting initial heights: Top=${initialTopPx}px, Bottom=${initialBottomPx}px`);
            svgContainer.style.height = `${initialTopPx}px`;
            torusPlotContainer.style.height = `${initialBottomPx}px`;

            torusPlotContainer.style.display = 'block';
            horizontalResizer.style.display = 'block'; // Show the resizer

            // Tell Plotly to resize after container is visible and sized
             if (torusPlotInitialized) {
                 try {
                    Plotly.Plots.resize(torusPlotContainer);
                 } catch(e) { console.warn("Plotly resize failed on show:", e); }
             }

        }
    }

    function hideTorusPlot() {
        if (torusPlotContainer && horizontalResizer) {
            torusPlotContainer.style.display = 'none';
            horizontalResizer.style.display = 'none'; // Hide the resizer too
            // Reset heights to default or remove them if relying on CSS initial state
            svgContainer.style.height = ''; // Let CSS take over again? Or set default %
            torusPlotContainer.style.height = '';

            // Optional: Purge Plotly plot
            try {
                 Plotly.purge(torusPlotContainer);
                 console.log("Purged Torus plot.");
            } catch (e) {
                 console.warn("Could not purge torus plot.", e);
            }
        }
    }

    // --- Initial Setup ---
    // Hide legends initially - displayLegends will show them when simulation starts
    if (memoryLegendArea) memoryLegendArea.style.display = 'none';
    if (computeLegendArea) computeLegendArea.style.display = 'none';

    setFormEnabled(true);
    updateControls(simulationState);
    updateStatusDisplay(simulationState);
    hideCompletionPopup(); // Ensure popup is hidden initially


    

}); // End DOMContentLoaded