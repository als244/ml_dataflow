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
    const completionArea = document.getElementById('completion-area');
    const memoryLegendArea = document.getElementById('memory-legend-area');
    const computeLegendArea = document.getElementById('compute-legend-area');
    const simulationArea = document.querySelector('.simulation-area');
    const leftLegendResizer = document.getElementById('left-legend-resizer');
    const rightLegendResizer = document.getElementById('right-legend-resizer');
    const errorMessageDiv = document.getElementById('error-message');
    const submitButton = paramsForm.querySelector('button[type="submit"]');
    // const controlElements = document.querySelectorAll('.controls button, .controls input, .controls span'); // Less used now

    const torusPlotContainer = document.getElementById('torus-plot-container');
    const horizontalResizer = document.getElementById('horizontal-resizer');
    const completionPopup = document.getElementById('completion-popup');
    const closeCompletionPopupBtn = document.getElementById('closeCompletionPopupBtn');
    const animationHeader = document.getElementById('animation-header');

    // --- Simulation State (Main Thread - primarily for UI and config) ---

    const initialSpeedDisplay = 80;
    
    let simulationState = { current_frame: 0, is_paused: true, is_complete: false, speed_level: 80, target_cycle: null, max_frames: 30000, completion_stats: {}, devices: [] };
    let simulationConfig = { N: 0, total_layers: 0, total_layers_non_head: 0, memory_legend: "", compute_legend: "" };
    // currentIntervalSec is now primarily managed by worker, main thread receives it via messages
    let simulationInitialized = false; // Tracks if the main UI and SVG structure are set up for a simulation
    let torusPlotInitialized = false;
    let isResetting = false; // Flag to identify reset-triggered updates

    // --- SVG Rendering Constants (assuming these are unchanged) ---
    const svgNS = "http://www.w3.org/2000/svg";
    const viewBoxWidth = 50; // Base for relative calculations
    const centerX = viewBoxWidth / 2;
    const effectiveCenterY = viewBoxWidth / 2; // Keep Y consistent if aspect ratio is 1:1
    const totalDistance = effectiveCenterY * 0.98;
    const innerRadius = effectiveCenterY * 0.35;
    const innerNodeRadius = effectiveCenterY * 0.12;
    const outerNodeRadius = effectiveCenterY * 0.15;
    const stallNodeRadius = outerNodeRadius * 1.3;
    const desiredStallGap = viewBoxWidth * 0.03;
    const stallNodeCenterOffset = outerNodeRadius + desiredStallGap + stallNodeRadius;
    const computeArcRadiusScale = 1.15;
    const arrowOffsetDist = innerNodeRadius * 0.4;
    const labelOffsetDistance = effectiveCenterY * 0.04;
    const baseStrokeWidth = viewBoxWidth * 0.005;
    const outerLabelFontSize = viewBoxWidth * 0.024;
    const innerLabelFontSize = viewBoxWidth * 0.024;
    const stallLabelFontSize = viewBoxWidth * 0.020;
    const transferLabelFontSize = viewBoxWidth * 0.022;
    const deviceOpacity = 0.55;
    const innerNodeOpacity = 0.8;
    const stallNodeOpacity = 0.9;
    let svgElements = {};
    let nodePositions = {};
    let drawingBounds = { minY: Infinity, maxY: -Infinity, minX: Infinity, maxX: -Infinity };
    let baseAnimationHue = 240;

    // --- Resizer State ---
    let isResizing = false; // Vertical resizer
    let startY_resize, initialTopHeight, initialBottomHeight;
    const minPaneHeight = 50;

    let isResizingHorizontal = false; // Horizontal resizers
    let currentHorizontalResizer = null;
    let startX_resize, initialLeftWidth_resize, initialMainWidth_resize, initialRightWidth_resize;
    const minPaneWidth = 100;


    // --- Web Worker ---
    let simulationWorker = null;

    if (window.Worker) {
        try {
            simulationWorker = new Worker('/static/simulationWorker.js');

            simulationWorker.onmessage = function(e) {
                const { type, state: workerState, interval_sec: workerInterval, error: workerError, command: workerCommand } = e.data;
                // console.log("Main: Message received from worker:", JSON.parse(JSON.stringify(e.data)));


                if (type === 'stateUpdate') {
                    // console.log("Main: Received stateUpdate. Main simulationInitialized:", simulationInitialized, "Main simulationConfig.N:", simulationConfig ? simulationConfig.N : 'N/A');
                    simulationState = workerState;
                    updateControls(simulationState);
                    updateStatusDisplay(simulationState);

                    if (document.visibilityState === 'visible') {
                        window.requestAnimationFrame(() => {
                            if (simulationInitialized && !isResetting) {
                                updateSVG(simulationState);
                            }
                            syncAnimationHeaderVisibility();
                        });
                    }
    

                    if (simulationState.is_complete && simulationInitialized) {
                        displayCompletionStats(simulationState.completion_stats);
                    }
                     // console.log("Main: UI updated. Status text should change from 'Initializing...'. Play button disabled state:", playBtn ? playBtn.disabled : 'N/A');
                } else if (type === 'fetchError' || type === 'controlError') {
                    const context = type === 'controlError' ? `controlling command '${workerCommand}' via worker` : "worker fetching update";
                    handleApiError(workerError, context);
                    simulationState.is_paused = true;
                    updateControls(simulationState);
                } else if (type === 'workerResetComplete') {
                    console.log("Main: Worker has confirmed its internal reset.");
                }
            };

            simulationWorker.onerror = function(errorEvent) {
                console.error("Main: Full critical error event in Web Worker:", errorEvent);
                let errorMessageText = "Unknown worker error occurred";

                if (errorEvent && errorEvent.message) {
                    errorMessageText = errorEvent.message;
                } else if (errorEvent && (errorEvent.filename || errorEvent.lineno)) {
                    errorMessageText = `Error in ${errorEvent.filename || 'worker script'} at line ${errorEvent.lineno || 'N/A'}`;
                    if (errorEvent.colno) {
                        errorMessageText += ` column ${errorEvent.colno}`;
                    }
                } else if (typeof errorEvent === 'string') {
                    errorMessageText = errorEvent;
                }

                displayError(`Critical Worker Error: ${errorMessageText}. The simulation may not function correctly. Please refresh.`);
                setFormEnabled(true);
                simulationInitialized = false;
                if (svg) svg.innerHTML = '';
                hideTorusPlot();
                const errorDisplayState = { current_frame: 0, is_paused: true, is_complete: true, target_cycle: null };
                updateStatusDisplay(errorDisplayState);
                updateControls(errorDisplayState); // Update controls to reflect error/stopped state
            };
        } catch (e) {
            console.error("Main: Failed to create Web Worker:", e);
            displayError("Failed to initialize the simulation worker. Your browser might not support it or a script error occurred. Please refresh or try a different browser.");
            if (submitButton) submitButton.disabled = true; // Disable starting the simulation
        }
    } else {
        console.error("Web Workers are not supported in this browser.");
        displayError("Your browser does not support Web Workers, which are needed for this simulation. Please try a modern browser.");
        if (submitButton) submitButton.disabled = true;
    }

    // --- Event Listeners ---
    paramsForm.addEventListener('submit', startSimulation); // Handles both start and reset

    // Control commands are now sent to the worker
    playBtn.addEventListener('click', () => {
        if (simulationWorker && simulationInitialized) {
            simulationWorker.postMessage({ command: 'play' });
        }
    });
    pauseBtn.addEventListener('click', () => {
        if (simulationWorker && simulationInitialized) {
            simulationWorker.postMessage({ command: 'pause' });
        }
    });
    speedSlider.addEventListener('input', () => {
        // 1. Update the numerical display immediately as the user drags
        if (speedValue) {
            speedValue.textContent = speedSlider.value;
        }
        // 2. Send the 'set_speed' command to the worker in real-time
        if (simulationWorker && simulationInitialized) {
            simulationWorker.postMessage({ command: 'set_speed', value: speedSlider.value });
        }
    });

    // The 'change' event fires after the user releases the slider.
    // It can act as a final confirmation or a fallback.
    // Given we're sending on 'input', this might be redundant but doesn't hurt.
    speedSlider.addEventListener('change', () => {
        if (simulationWorker && simulationInitialized) {
            simulationWorker.postMessage({ command: 'set_speed', value: speedSlider.value });
        }
    });

    runToBtn.addEventListener('click', () => {
        const cycle = parseInt(runToCycleInput.value, 10);
        if (!isNaN(cycle) && cycle >= 0) {
            if (simulationWorker && simulationInitialized) {
                simulationWorker.postMessage({ command: 'run_to_cycle', value: cycle });
            }
        } else {
            displayError("Invalid target cycle number.");
        }
    });

    

    if (closeCompletionPopupBtn) {
        closeCompletionPopupBtn.addEventListener('click', () => {
            if (completionPopup) completionPopup.style.display = 'none';
        });
    }

    // --- Resizer Event Listeners (Vertical and Horizontal) ---
    // These manage UI layout and should remain on the main thread.
    // Vertical Resizer (svgContainer vs torusPlotContainer)
    if (horizontalResizer) {
        horizontalResizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY_resize = e.clientY;
            initialTopHeight = svgContainer.offsetHeight;
            initialBottomHeight = torusPlotContainer.offsetHeight;
            document.addEventListener('mousemove', handleVerticalMouseMove);
            document.addEventListener('mouseup', handleVerticalMouseUp);
            e.preventDefault();
            document.body.style.cursor = 'row-resize';
            svgContainer.style.pointerEvents = 'none';
            torusPlotContainer.style.pointerEvents = 'none';
        });
    }

    function handleVerticalMouseMove(e) {
        if (!isResizing) return;
        const currentY = e.clientY;
        const deltaY = currentY - startY_resize;
        let newTopHeight = initialTopHeight + deltaY;
        let newBottomHeight = initialBottomHeight - deltaY;
        const totalAvailableHeight = initialTopHeight + initialBottomHeight;

        if (totalAvailableHeight >= 2 * minPaneHeight) {
            if (newTopHeight < minPaneHeight) {
                newTopHeight = minPaneHeight;
                newBottomHeight = totalAvailableHeight - newTopHeight;
            } else if (newBottomHeight < minPaneHeight) {
                newBottomHeight = minPaneHeight;
                newTopHeight = totalAvailableHeight - newBottomHeight;
            }
        } else if (totalAvailableHeight >= minPaneHeight) {
            if (deltaY > 0) {
                newTopHeight = Math.min(newTopHeight, totalAvailableHeight - minPaneHeight);
                newTopHeight = Math.max(newTopHeight, minPaneHeight);
                newBottomHeight = totalAvailableHeight - newTopHeight;
            } else {
                newBottomHeight = Math.min(newBottomHeight, totalAvailableHeight - minPaneHeight);
                newBottomHeight = Math.max(newBottomHeight, minPaneHeight);
                newTopHeight = totalAvailableHeight - newBottomHeight;
            }
        } else {
            const topRatio = (initialTopHeight / totalAvailableHeight) || 0.5;
            newTopHeight = totalAvailableHeight * topRatio;
            newBottomHeight = totalAvailableHeight - newTopHeight;
        }
        newTopHeight = Math.max(0, newTopHeight);
        newBottomHeight = Math.max(0, newBottomHeight);

        if (svgContainer) svgContainer.style.height = `${newTopHeight}px`;
        if (torusPlotContainer) torusPlotContainer.style.height = `${newBottomHeight}px`;

        if (torusPlotInitialized && torusPlotContainer) {
            try {
                Plotly.Plots.resize(torusPlotContainer);
            } catch (resizeError) {
                console.warn("Error resizing Plotly plot during vertical drag:", resizeError);
            }
        }
    }

    function handleVerticalMouseUp() {
        if (isResizing) {
            isResizing = false;
            document.removeEventListener('mousemove', handleVerticalMouseMove);
            document.removeEventListener('mouseup', handleVerticalMouseUp);
            if (document.body) document.body.style.cursor = '';
            if (svgContainer) svgContainer.style.pointerEvents = '';
            if (torusPlotContainer) torusPlotContainer.style.pointerEvents = '';

            if (torusPlotInitialized && torusPlotContainer) {
                try {
                    Plotly.Plots.resize(torusPlotContainer);
                } catch (resizeError) {
                    console.warn("Error resizing Plotly plot on vertical mouseup:", resizeError);
                }
            }
        }
    }

    // Horizontal Resizers (Legends vs Simulation Area)
    function initHorizontalResizer(resizer, leftPane, rightPane) {
        if (!resizer || !leftPane || !rightPane) return;
        resizer.addEventListener('mousedown', (e) => {
            isResizingHorizontal = true;
            currentHorizontalResizer = resizer;
            startX_resize = e.clientX;
            initialLeftWidth_resize = leftPane.offsetWidth;
            initialRightWidth_resize = rightPane.offsetWidth; // For right resizer, this is the simulationArea width

            document.addEventListener('mousemove', handleHorizontalMouseMove);
            document.addEventListener('mouseup', handleHorizontalMouseUp);
            e.preventDefault();
            document.body.style.cursor = 'col-resize';
            leftPane.style.pointerEvents = 'none';
            rightPane.style.pointerEvents = 'none';
            if (simulationArea && leftPane !== simulationArea && rightPane !== simulationArea) {
                simulationArea.style.pointerEvents = 'none';
            }
        });
    }

    const handleWindowResize = debounce(() => {
        if (torusPlotInitialized && torusPlotContainer && torusPlotContainer.style.display !== 'none') {
            try {
                const parentHeight = torusPlotContainer.parentElement.clientHeight;
                const resizerHeight = horizontalResizer ? horizontalResizer.offsetHeight : 8;
                const availableHeight = parentHeight - resizerHeight;
                const currentTopHeight = svgContainer.offsetHeight;
                const currentBottomHeight = torusPlotContainer.offsetHeight;
                const currentTotal = currentTopHeight + currentBottomHeight;
                let newTopPx, newBottomPx;

                if (currentTotal > 0 && availableHeight > (minPaneHeight * 2)) {
                    const topRatio = currentTopHeight / currentTotal;
                    newTopPx = Math.max(minPaneHeight, Math.floor(availableHeight * topRatio));
                    newBottomPx = Math.max(minPaneHeight, availableHeight - newTopPx);
                    if (newTopPx + newBottomPx > availableHeight) {
                        newTopPx = availableHeight - newBottomPx;
                    }
                } else {
                    const initialTopPercent = 0.65;
                    newTopPx = Math.max(minPaneHeight, Math.floor(availableHeight * initialTopPercent));
                    newBottomPx = Math.max(minPaneHeight, availableHeight - newTopPx);
                    if (newTopPx + newBottomPx > availableHeight) {
                        newBottomPx = availableHeight - newTopPx;
                        if (newBottomPx < minPaneHeight) {
                            newBottomPx = minPaneHeight;
                            newTopPx = availableHeight - newBottomPx;
                        }
                    }
                }
                svgContainer.style.height = `${newTopPx}px`;
                torusPlotContainer.style.height = `${newBottomPx}px`;
                Plotly.Plots.resize(torusPlotContainer);
            } catch (resizeError) {
                console.warn("Error resizing Plotly plot on window resize:", resizeError);
            }
        }
    }, 250);

    window.addEventListener('resize', handleWindowResize);

    function handleHorizontalMouseMove(e) {
        if (!isResizingHorizontal || !currentHorizontalResizer) return;
        const currentX = e.clientX;
        const deltaX = currentX - startX_resize;
        let leftPane, rightPane;

        if (currentHorizontalResizer.id === 'left-legend-resizer') {
            leftPane = memoryLegendArea;
            rightPane = simulationArea;
            let newLeftWidth = initialLeftWidth_resize + deltaX;
            newLeftWidth = Math.max(minPaneWidth, newLeftWidth);
            const mainAreaCurrentWidth = rightPane.offsetWidth;
            if (mainAreaCurrentWidth - (newLeftWidth - initialLeftWidth_resize) < minPaneWidth) {
                newLeftWidth = initialLeftWidth_resize + (mainAreaCurrentWidth - minPaneWidth);
            }
            leftPane.style.flexBasis = `${newLeftWidth}px`;
        } else if (currentHorizontalResizer.id === 'right-legend-resizer') {
            leftPane = simulationArea;
            rightPane = computeLegendArea;
            let newRightWidth = initialRightWidth_resize - deltaX;
            newRightWidth = Math.max(minPaneWidth, newRightWidth);
            const mainAreaCurrentWidth = leftPane.offsetWidth;
            if (mainAreaCurrentWidth - (newRightWidth - initialRightWidth_resize) < minPaneWidth) {
                newRightWidth = initialRightWidth_resize + (mainAreaCurrentWidth - minPaneWidth);
            }
            rightPane.style.flexBasis = `${newRightWidth}px`;
        } else { return; }

        if (torusPlotInitialized && torusPlotContainer && (leftPane === simulationArea || rightPane === simulationArea)) {
            try { Plotly.Plots.resize(torusPlotContainer); } catch (resizeError) { /* console.warn */ }
        }
    }

    function handleHorizontalMouseUp() {
        if (isResizingHorizontal) {
            isResizingHorizontal = false;
            document.removeEventListener('mousemove', handleHorizontalMouseMove);
            document.removeEventListener('mouseup', handleHorizontalMouseUp);
            document.body.style.cursor = '';
            if(memoryLegendArea) memoryLegendArea.style.pointerEvents = '';
            if(computeLegendArea) computeLegendArea.style.pointerEvents = '';
            if (simulationArea) simulationArea.style.pointerEvents = '';

            if (torusPlotInitialized && torusPlotContainer && currentHorizontalResizer) {
                if (currentHorizontalResizer.id === 'left-legend-resizer' || (currentHorizontalResizer.id === 'right-legend-resizer' && simulationArea)) {
                    try { Plotly.Plots.resize(torusPlotContainer); } catch (resizeError) { /* console.warn */ }
                }
            }
            currentHorizontalResizer = null;
        }
    }

    initHorizontalResizer(leftLegendResizer, memoryLegendArea, simulationArea);
    initHorizontalResizer(rightLegendResizer, simulationArea, computeLegendArea);


    function syncAnimationHeaderVisibility() {
        if (!animationHeader || !svg) return;
        const isSvgElementDisplayBlock = window.getComputedStyle(svg).display !== 'none';
        const shouldBeVisible = isSvgElementDisplayBlock && simulationInitialized && !errorMessageDiv.classList.contains('visible');

        if (shouldBeVisible) {
            animationHeader.classList.add('visible');

            /*
            let headerText = "Animation";
            if(simulationState) { // Check if simulationState is available
                if (simulationState.is_complete) headerText = "Animation Complete";
                else if (simulationState.is_paused) headerText = "Animation Paused";
                else headerText = "Animation Running";
                animationHeader.textContent = `${headerText} (Cycle: ${simulationState.current_frame || 0})`;
            } else {
                animationHeader.textContent = "Animation Area";
            }
            */
        } else {
            animationHeader.classList.remove('visible');
        }
    }
    // Initial call and observers for animation header
    syncAnimationHeaderVisibility();
    if (errorMessageDiv) {
        new MutationObserver(syncAnimationHeaderVisibility).observe(errorMessageDiv, { attributes: true, attributeFilter: ['class'] });
    }
    if (svg) {
        new MutationObserver(syncAnimationHeaderVisibility).observe(svg, { attributes: true, attributeFilter: ['style'] });
    }
    if (svgContainer) {
        new MutationObserver(syncAnimationHeaderVisibility).observe(svgContainer, { attributes: true, attributeFilter: ['style', 'class'] });
    }


    // --- API Communication Functions ---
    async function startSimulation(event) {
        event.preventDefault();
        displayError(''); // Clear previous errors on new action
        hideCompletionPopup();

        if (simulationInitialized) { // This means "Reset" button was clicked
            console.log("Main: Attempting to reset simulation...");
            isResetting = true;

            if (memoryLegendArea) { memoryLegendArea.style.display = 'none'; if(memoryLegendArea.querySelector('pre')) memoryLegendArea.querySelector('pre').textContent = ''; }
            if (computeLegendArea) { computeLegendArea.style.display = 'none'; if(computeLegendArea.querySelector('pre')) computeLegendArea.querySelector('pre').textContent = ''; }
            hideTorusPlot();

            try {
                const response = await fetch('/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command: 'restart' }) });
                const data = await response.json();

                if (response.ok && data.success) {
                    console.log("Main: Backend restart command successful.");
                    if (svg) { svg.innerHTML = ''; }
                    if (simulationWorker) simulationWorker.postMessage({ command: 'resetWorker' });
                    await fetchInitialStateAfterReset(); // Updates main `simulationState` and UI text/controls for frame 0
                    setFormEnabled(true); // This also sets simulationInitialized = false
                    torusPlotInitialized = false;
                    console.log("Main: UI reset complete. Ready for new simulation parameters.");
                } else {
                    handleApiError(data.error || response.statusText || "Unknown error from /control restart", "restarting simulation (API)");
                    setFormEnabled(true); simulationInitialized = false; // Attempt to restore UI
                }
            } catch (error) {
                handleApiError(error.message, "restarting simulation (network)");
                setFormEnabled(true); simulationInitialized = false;
            } finally {
                isResetting = false;
                syncAnimationHeaderVisibility();
            }
            return;
        }

        // --- START New Simulation ---
        console.log("Main: Attempting to start NEW simulation...");
        if (svg) { svg.innerHTML = ''; }
        if (memoryLegendArea) memoryLegendArea.style.display = 'none';
        if (computeLegendArea) computeLegendArea.style.display = 'none';

        const formData = new FormData(paramsForm);
        setFormEnabled(false); // Disables form, sets button to "Reset". Does NOT call updateControls for active state yet.
        statusArea.textContent = "Status: Initializing..."; // User sees this

        try {
            const response = await fetch('/start_simulation', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok && data.success) {
                console.log("Main: Initial simulation parameters accepted by server (/start_simulation).");
                simulationConfig = data.config;
                simulationInitialized = true; // MAIN THREAD is now initialized for this new sim

                setupSVG(simulationConfig);
                displayLegends(simulationConfig);
                if (data.state.max_frames) runToCycleInput.max = data.state.max_frames;
                runToCycleInput.value = data.state.current_frame;

                const modelStages = simulationConfig.N;
                const dpDegreeValue = formData.get('dp_degree');
                const dataParallelismFactor = dpDegreeValue ? parseInt(dpDegreeValue) : 16; // Default if not found or invalid
                drawTorusPlot(modelStages, dataParallelismFactor);
                torusPlotInitialized = true;
                showTorusPlot();

                if (simulationWorker) {
                    simulationWorker.postMessage({
                        command: 'initialize',
                        initialState: data.state,
                        config: simulationConfig
                    });
                }
                console.log("Main: Sent 'initialize' to worker. Waiting for first stateUpdate.");
                // UI (status, controls) will be fully updated when the first 'stateUpdate' is received from the worker.
            } else {
                handleApiError(data.error || response.statusText || "Unknown error from /start_simulation", "starting simulation (API)");
                setFormEnabled(true); simulationInitialized = false;
                if (svg) svg.innerHTML = '';
                updateStatusDisplay({ current_frame: 0, is_paused: true, is_complete: false, target_cycle: null });
                updateControls({ is_paused: true, is_complete: false });
                hideTorusPlot();
            }
        } catch (error) {
            handleApiError(error.message, "starting simulation (network)");
            setFormEnabled(true); simulationInitialized = false;
            if (svg) svg.innerHTML = '';
            updateStatusDisplay({ current_frame: 0, is_paused: true, is_complete: false, target_cycle: null });
            updateControls({ is_paused: true, is_complete: false });
            hideTorusPlot();
        } finally {
            // syncAnimationHeaderVisibility will be called by subsequent state updates or error displays
        }
    }

    // Fetches the clean "frame 0" state after a server-side reset.
    // This is crucial for displaying the correct "Cycle: 0, Paused" state.
    async function fetchInitialStateAfterReset() {
        console.log("Main [fetchInitialStateAfterReset]: Fetching state 0 after reset...");
        try {
            const response = await fetch('/get_state_update');
            const data = await response.json();
            if (response.ok && data.success) {
                simulationState = data.state;
                updateStatusDisplay(simulationState);
                updateControls(simulationState);
                console.log("Main [fetchInitialStateAfterReset]: UI (status, controls) updated to frame 0 state.");
            } else {
                handleApiError(data.error || "Failed to fetch state 0", "fetching initial state after reset (API)");
                updateStatusDisplay({ current_frame: 0, is_paused: true, is_complete: false, target_cycle: null });
                updateControls({ is_paused: true, is_complete: false, N: 0 });
            }
        } catch (error) {
            handleApiError(error.message, "fetching initial state after reset (network)");
            updateStatusDisplay({ current_frame: 0, is_paused: true, is_complete: false, target_cycle: null });
            updateControls({ is_paused: true, is_complete: false, N: 0 });
        }
        // syncAnimationHeaderVisibility(); // Caller should handle
    }


    // --- SVG Setup Function ---
    function setupSVG(config) {
        if (!svg) {
            console.error("[setupSVG] Cannot proceed: svg element reference is null!");
            displayError("Critical Error: Simulation SVG container not found.");
            return;
        }
        svg.innerHTML = ''; // Clear existing content first
        svgElements = {}; // Reset references
        nodePositions = { outer: [], inner: [], stall: [], unit: [], angleRad: [], angleDeg: [], angleToPrev: [], angleToNext: [] };
        drawingBounds = { minY: Infinity, maxY: -Infinity, minX: Infinity, maxX: -Infinity };

        const N = config.N;
        if (N <= 0) {
            console.warn("[setupSVG] Invalid number of devices (N):", N, ". SVG will be empty.");
            // Ensure viewbox is minimal or default if N=0
             svg.setAttribute('viewBox', `0 0 ${viewBoxWidth} ${viewBoxWidth}`);
             if (svgContainer) svgContainer.style.removeProperty('--svg-aspect-ratio');
            syncAnimationHeaderVisibility();
            return;
        }

        // ... (rest of your existing setupSVG logic to create static elements)
        // Calculate nodePositions, drawingBounds
        for (let i = 0; i < N; i++) { const angleRad = (2 * Math.PI * i) / N - Math.PI / 2; const angleDeg = angleRad * 180 / Math.PI; const unitDir = { x: Math.cos(angleRad), y: Math.sin(angleRad) }; nodePositions.unit.push(unitDir); nodePositions.angleRad.push(angleRad); nodePositions.angleDeg.push(angleDeg); const outerPos = { x: centerX + totalDistance * unitDir.x, y: effectiveCenterY + totalDistance * unitDir.y }; const innerPos = { x: centerX + innerRadius * unitDir.x, y: effectiveCenterY + innerRadius * unitDir.y }; const stallPos = { x: outerPos.x + stallNodeCenterOffset * unitDir.x, y: outerPos.y + stallNodeCenterOffset * unitDir.y }; nodePositions.outer.push(outerPos); nodePositions.inner.push(innerPos); nodePositions.stall.push(stallPos); drawingBounds.minX = Math.min(drawingBounds.minX, innerPos.x - innerNodeRadius, outerPos.x - outerNodeRadius, stallPos.x - stallNodeRadius); drawingBounds.maxX = Math.max(drawingBounds.maxX, innerPos.x + innerNodeRadius, outerPos.x + outerNodeRadius, stallPos.x + stallNodeRadius); drawingBounds.minY = Math.min(drawingBounds.minY, innerPos.y - innerNodeRadius, outerPos.y - outerNodeRadius, stallPos.y - stallNodeRadius); drawingBounds.maxY = Math.max(drawingBounds.maxY, innerPos.y + innerNodeRadius, outerPos.y + outerNodeRadius, stallPos.y + stallNodeRadius); }
        for (let i = 0; i < N; i++) { if (N > 1) { const prevIdx = (i - 1 + N) % N; const nextIdx = (i + 1) % N; const vecToPrev = { x: nodePositions.outer[prevIdx].x - nodePositions.outer[i].x, y: nodePositions.outer[prevIdx].y - nodePositions.outer[i].y }; const vecToNext = { x: nodePositions.outer[nextIdx].x - nodePositions.outer[i].x, y: nodePositions.outer[nextIdx].y - nodePositions.outer[i].y }; nodePositions.angleToPrev.push(Math.atan2(vecToPrev.y, vecToPrev.x) * 180 / Math.PI); nodePositions.angleToNext.push(Math.atan2(vecToNext.y, vecToNext.x) * 180 / Math.PI); } else { nodePositions.angleToPrev.push(180); nodePositions.angleToNext.push(0); } }
        const padding = viewBoxWidth * 0.05; const vbMinX = drawingBounds.minX - padding; const vbMinY = drawingBounds.minY - padding; const vbWidth = (drawingBounds.maxX + padding) - vbMinX; const vbHeight = (drawingBounds.maxY + padding) - vbMinY; svg.setAttribute('viewBox', `${vbMinX} ${vbMinY} ${vbWidth} ${vbHeight}`);
        if (svgContainer && vbHeight > 0) { const aspectRatio = vbWidth / vbHeight; svgContainer.style.setProperty('--svg-aspect-ratio', aspectRatio); } else if (svgContainer) { svgContainer.style.removeProperty('--svg-aspect-ratio');}
        const defs = document.createElementNS(svgNS, 'defs'); const markerSize = baseStrokeWidth * 10; const markerRefX = markerSize * 0.8; const marker = createMarker('arrowhead', 'black', markerSize, markerRefX, markerSize, markerSize); defs.appendChild(marker); svg.appendChild(defs);
        const g = addSvgElement(svg, 'g'); svgElements['main_group'] = g;
        for (let i = 0; i < N; i++) { const deviceId = `dev_${i}`; const outerPos = nodePositions.outer[i]; const innerPos = nodePositions.inner[i]; const stallPos = nodePositions.stall[i]; const unitDir = nodePositions.unit[i]; const baseColor = getColorForDevice(i, N, baseAnimationHue); const devGroup = addSvgElement(g, 'g', { id: deviceId }); svgElements[`${deviceId}_outer_circle`] = addSvgCircle(devGroup, outerPos.x, outerPos.y, outerNodeRadius, { fill: baseColor, stroke: 'black', 'stroke-width': baseStrokeWidth, opacity: deviceOpacity, id: `${deviceId}_outer_circle` }); const innerSquareSide = innerNodeRadius * 1.414; svgElements[`${deviceId}_inner_square`] = addSvgRect(devGroup, innerPos.x - innerSquareSide / 2, innerPos.y - innerSquareSide / 2, innerSquareSide, innerSquareSide, { fill: baseColor, stroke: 'black', 'stroke-width': baseStrokeWidth, opacity: innerNodeOpacity, id: `${deviceId}_inner_square` }); svgElements[`${deviceId}_stall_node`] = addSvgCircle(devGroup, stallPos.x, stallPos.y, stallNodeRadius, { fill: 'red', stroke: baseColor, 'stroke-width': baseStrokeWidth * 2, opacity: stallNodeOpacity, visibility: 'hidden', id: `${deviceId}_stall_node` }); svgElements[`${deviceId}_finish_indicator`] = addSvgCircle(devGroup, stallPos.x, stallPos.y, stallNodeRadius, { fill: 'lime', stroke: baseColor, 'stroke-width': baseStrokeWidth * 2, opacity: stallNodeOpacity, visibility: 'hidden', id: `${deviceId}_finish_indicator` }); svgElements[`${deviceId}_outer_label`] = addSvgText(devGroup, outerPos.x, outerPos.y, `D${i}`, { 'font-size': outerLabelFontSize, fill: 'black', id: `${deviceId}_outer_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_inner_label`] = addSvgText(devGroup, innerPos.x, innerPos.y, `Home`, { 'font-size': innerLabelFontSize, fill: 'black', id: `${deviceId}_inner_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_stall_label`] = addSvgText(devGroup, stallPos.x, stallPos.y, "", { 'font-size': stallLabelFontSize, fill: 'white', 'font-weight': 'bold', visibility: 'hidden', id: `${deviceId}_stall_label`, 'pointer-events': 'none' }); const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const inStartX = innerPos.x + unitDir.x * innerNodeRadius + offset.x; const inStartY = innerPos.y + unitDir.y * innerNodeRadius + offset.y; const outStartX = outerPos.x - unitDir.x * outerNodeRadius - offset.x; const outStartY = outerPos.y - unitDir.y * outerNodeRadius - offset.y; svgElements[`${deviceId}_in_arrow`] = addSvgPath(devGroup, `M ${inStartX} ${inStartY}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_in_arrow` }); svgElements[`${deviceId}_in_label`] = addSvgText(devGroup, inStartX, inStartY, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_in_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_out_arrow`] = addSvgPath(devGroup, `M ${outStartX} ${outStartY}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_out_arrow` }); svgElements[`${deviceId}_out_label`] = addSvgText(devGroup, outStartX, outStartY, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_out_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_ring_arrow`] = addSvgPath(devGroup, `M ${outerPos.x} ${outerPos.y}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_ring_arrow` }); svgElements[`${deviceId}_ring_label`] = addSvgText(devGroup, outerPos.x, outerPos.y, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_ring_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_compute_arc`] = addSvgPath(devGroup, "", { stroke: 'gray', 'stroke-width': baseStrokeWidth * 3, fill: 'none', visibility: 'hidden', id: `${deviceId}_compute_arc` }); }
        // ... end of existing setupSVG logic

        syncAnimationHeaderVisibility(); // Update header after SVG might have been structured
    }

    // --- SVG Update Function ---
    function updateSVG(state) {
        // Critical Guard: Do not attempt to update SVG if main UI isn't initialized for it,
        // or if it's in the process of resetting, or if SVG elements are missing.
        if (!simulationInitialized || isResetting || !svg || Object.keys(svgElements).length === 0 || !nodePositions.outer || nodePositions.outer.length === 0) {
            // console.warn(`[updateSVG] Drawing aborted. Conditions: simulationInitialized=${simulationInitialized}, isResetting=${isResetting}, svgElementsEmpty=${Object.keys(svgElements).length === 0}, nodePositionsEmpty=${!nodePositions.outer || nodePositions.outer.length === 0}`);
            if (svg && !simulationInitialized && !isResetting) {
                // If not initialized and not resetting (e.g. page load before first sim), ensure it's clear.
                // This helps if somehow worker sends an update before main UI is fully ready.
                // svg.innerHTML = ''; // This might be too aggressive if called frequently
            }
            syncAnimationHeaderVisibility(); // Still sync header state
            return;
        }

        const N = simulationConfig.N; // N from the config used to build the SVG structure
        if (!state || !state.devices) {
            console.error("[updateSVG] Error: Received invalid state object.", state);
            return;
        }
        if (state.devices.length !== N && N > 0) {
            // This can happen if worker sends an update from a PREVIOUS simulation session
            // just as a NEW simulation (with different N) is starting up, or after a reset.
            // The `simulationInitialized` flag and checks in `onmessage` should help,
            // but this is a final defense.
            console.warn(`[updateSVG] Mismatch: state.devices.length (${state.devices.length}) !== simulationConfig.N (${N}). Aborting draw. This might be a race condition during reset/restart.`);
            return;
        }
        if (N === 0 && state.devices.length > 0) {
            console.warn(`[updateSVG] N is 0 but received state with devices. Aborting draw.`);
            return;
        }
        if (N === 0 && state.devices.length === 0) { // If N is 0 (e.g. after reset), nothing to draw
             // svg.innerHTML = ''; // Ensure it's clear
             syncAnimationHeaderVisibility();
             return;
        }


        // ... (rest of your existing updateSVG logic to update dynamic elements based on state.devices[i])
        // Ensure all accesses to svgElements[`${deviceId}_...`] and nodePositions[i] are valid.
        for (let i = 0; i < N; i++) {
            const device = state.devices[i];
            if (!device || typeof device.id === 'undefined' || !nodePositions.outer[i] || !nodePositions.inner[i] || !nodePositions.stall[i] || !nodePositions.unit[i]) {
                console.warn(`[updateSVG] Invalid device data or missing node position at index ${i}`, device);
                continue; // Skip this device if data or base positions are bad
            }
            const deviceId = `dev_${i}`; const outerPos = nodePositions.outer[i]; const innerPos = nodePositions.inner[i]; const stallPos = nodePositions.stall[i]; const unitDir = nodePositions.unit[i]; const transferDist = Math.max(0, Math.hypot(outerPos.x - innerPos.x, outerPos.y - innerPos.y) - outerNodeRadius - innerNodeRadius);
            if (svgElements[`${deviceId}_outer_label`]) { updateMultiLineText(svgElements[`${deviceId}_outer_label`], device.status_text || `D${i}`, outerPos.x, outerPos.y, outerLabelFontSize); }
            const stallNode = svgElements[`${deviceId}_stall_node`]; const stallLabel = svgElements[`${deviceId}_stall_label`];
            if (stallNode && stallLabel) { const isStalled = device.status === "Stalled"; const stallVisibility = isStalled ? 'visible' : 'hidden'; stallNode.setAttribute('visibility', stallVisibility); if (isStalled) { updateMultiLineText(stallLabel, device.stall_reason || "Stalled", stallPos.x, stallPos.y, stallLabelFontSize); stallLabel.setAttribute('visibility', 'visible'); } else { stallLabel.setAttribute('visibility', 'hidden'); } }
            const finishIndicator = svgElements[`${deviceId}_finish_indicator`]; if (finishIndicator) { const isFinished = device.status === "Finished"; const finishVisibility = isFinished ? 'visible' : 'hidden'; finishIndicator.setAttribute('visibility', finishVisibility); }
            if (svgElements[`${deviceId}_inner_label`]) { updateMultiLineText(svgElements[`${deviceId}_inner_label`], `Home`, innerPos.x, innerPos.y, innerLabelFontSize); }
            const inArrow = svgElements[`${deviceId}_in_arrow`]; const inLabel = svgElements[`${deviceId}_in_label`];
            if (inArrow && inLabel && device.inbound && device.inbound.progress > 1e-6 && transferDist > 1e-6) { const progress = Math.min(1.0, device.inbound.progress); const lenProg = progress * transferDist; const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const startX = innerPos.x + unitDir.x * innerNodeRadius + offset.x; const startY = innerPos.y + unitDir.y * innerNodeRadius + offset.y; const endX = startX + unitDir.x * lenProg; const endY = startY + unitDir.y * lenProg; const midX = (startX + endX) / 2; const midY = (startY + endY) / 2; const labelOffsetX = perpVec.x * labelOffsetDistance; const labelOffsetY = perpVec.y * labelOffsetDistance; inArrow.setAttribute('d', `M ${startX} ${startY} L ${endX} ${endY}`); inArrow.setAttribute('stroke', device.inbound.color || 'gray'); inArrow.setAttribute('visibility', 'visible'); updateMultiLineText(inLabel, device.inbound.label || '', midX + labelOffsetX, midY + labelOffsetY, transferLabelFontSize); inLabel.setAttribute('fill', device.inbound.color || 'gray'); inLabel.setAttribute('visibility', 'visible'); } else if (inArrow && inLabel) { inArrow.setAttribute('visibility', 'hidden'); inLabel.setAttribute('visibility', 'hidden'); }
            const outArrow = svgElements[`${deviceId}_out_arrow`]; const outLabel = svgElements[`${deviceId}_out_label`];
            if (outArrow && outLabel && device.outbound && device.outbound.progress > 1e-6 && transferDist > 1e-6) { const progress = Math.min(1.0, device.outbound.progress); const lenProg = progress * transferDist; const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const startX = outerPos.x - unitDir.x * outerNodeRadius - offset.x; const startY = outerPos.y - unitDir.y * outerNodeRadius - offset.y; const endX = startX - unitDir.x * lenProg; const endY = startY - unitDir.y * lenProg; const midX = (startX + endX) / 2; const midY = (startY + endY) / 2; const labelOffsetX = -perpVec.x * labelOffsetDistance; const labelOffsetY = -perpVec.y * labelOffsetDistance; outArrow.setAttribute('d', `M ${startX} ${startY} L ${endX} ${endY}`); outArrow.setAttribute('stroke', device.outbound.color || 'gray'); outArrow.setAttribute('visibility', 'visible'); updateMultiLineText(outLabel, device.outbound.label || '', midX + labelOffsetX, midY + labelOffsetY, transferLabelFontSize); outLabel.setAttribute('fill', device.outbound.color || 'gray'); outLabel.setAttribute('visibility', 'visible'); } else if (outArrow && outLabel) { outArrow.setAttribute('visibility', 'hidden'); outLabel.setAttribute('visibility', 'hidden'); }
            const ringArrow = svgElements[`${deviceId}_ring_arrow`]; const ringLabel = svgElements[`${deviceId}_ring_label`];
            if (ringArrow && ringLabel && device.peer && device.peer.progress > 1e-6 && N > 1) { const progress = Math.min(1.0, device.peer.progress); const peerId = device.peer.target_peer; if (peerId >= 0 && peerId < N && peerId !== i && nodePositions.outer[peerId]) { const targetPos = nodePositions.outer[peerId]; const startPos = outerPos; const dx = targetPos.x - startPos.x; const dy = targetPos.y - startPos.y; const dist = Math.hypot(dx, dy); const startAngleRad = Math.atan2(dy, dx); const startEdge = pointOnCircle(startPos.x, startPos.y, outerNodeRadius, startAngleRad * 180 / Math.PI); const endAngleRad = Math.atan2(-dy, -dx); const endEdge = pointOnCircle(targetPos.x, targetPos.y, outerNodeRadius, endAngleRad * 180 / Math.PI); const currentEdgeX = startEdge.x + (endEdge.x - startEdge.x) * progress; const currentEdgeY = startEdge.y + (endEdge.y - startEdge.y) * progress; const sweepFlag = device.peer.direction > 0 ? 1 : 0; let arcRadius = dist * 0.6; if (Math.hypot(currentEdgeX - startEdge.x, currentEdgeY - startEdge.y) > 1e-3) { const pathData = `M ${startEdge.x} ${startEdge.y} A ${arcRadius} ${arcRadius} 0 0 ${sweepFlag} ${currentEdgeX} ${currentEdgeY}`; ringArrow.setAttribute('d', pathData); ringArrow.setAttribute('stroke', device.peer.color || 'indigo'); ringArrow.setAttribute('visibility', 'visible'); const midX = (startEdge.x + currentEdgeX) / 2; const midY = (startEdge.y + currentEdgeY) / 2; const vecCenterX = midX - centerX; const vecCenterY = midY - effectiveCenterY; const vecCenterMag = Math.hypot(vecCenterX, vecCenterY); const outX = vecCenterMag > 1e-6 ? vecCenterX / vecCenterMag : 0; const outY = vecCenterMag > 1e-6 ? vecCenterY / vecCenterMag : 0; const labelOffsetMag = labelOffsetDistance * 2.0; const labelX = midX + outX * labelOffsetMag; const labelY = midY + outY * labelOffsetMag; updateMultiLineText(ringLabel, device.peer.label || '', labelX, labelY, transferLabelFontSize); ringLabel.setAttribute('fill', device.peer.color || 'indigo'); ringLabel.setAttribute('visibility', 'visible'); } else { ringArrow.setAttribute('visibility', 'hidden'); ringLabel.setAttribute('visibility', 'hidden'); } } else { ringArrow.setAttribute('visibility', 'hidden'); ringLabel.setAttribute('visibility', 'hidden'); } } else if (ringArrow && ringLabel) { ringArrow.setAttribute('visibility', 'hidden'); ringLabel.setAttribute('visibility', 'hidden'); }
            const computeArc = svgElements[`${deviceId}_compute_arc`];
            if (computeArc && device.compute && device.compute.progress > 1e-6) {
                const progress = Math.min(1.0, device.compute.progress);
                const type = device.compute.type;
                const arcOuterRadius = outerNodeRadius * computeArcRadiusScale;

                // These are the angular positions of the previous/next nodes in the ring.
                // For N=2, anglePrev and angleNext will be the angle of the *other* node.
                const anglePrev = nodePositions.angleToPrev[i];
                const angleNext = nodePositions.angleToNext[i];

                let theta1Deg = 0, theta2Deg = 0, totalSweepDeg = 0;
                let angleStartDeg; // Keep angleStartDeg in a scope accessible for theta1Deg calculation

                const isInitialLayer = (device.compute.layer === 0);

                if (N > 1) {
                    if (type === "Fwd") {
                        if (N === 2) {
                            // For N=2, a "Fwd" compute sweeps 180 degrees.
                            // angleNext is the angle of the other node.
                            // Start the arc from the point "opposite" the current node's direct line to the other node.
                            angleStartDeg = angleNext - 180; // e.g., if other node is at 180 deg, this starts arc at 0 deg.
                            totalSweepDeg = 180;
                        } else { // N > 2 (original logic for more than 2 nodes)
                            angleStartDeg = isInitialLayer ? (angleNext - 180) : anglePrev;
                            totalSweepDeg = (angleNext - angleStartDeg + 360) % 360;
                            // Fallback: if calculated sweep is 0 but should be a full circle (e.g. prev/next align unexpectedly)
                            if (totalSweepDeg === 0 && angleNext === angleStartDeg && progress > 0) {
                                totalSweepDeg = 360;
                            }
                        }
                        theta1Deg = angleStartDeg;
                        theta2Deg = angleStartDeg + progress * totalSweepDeg;

                    } else if (type === "Head") {
                        // "Head" type always sweeps a full circle, starting from anglePrev.
                        angleStartDeg = anglePrev; // Defined for clarity, though only theta1Deg uses it directly here
                        theta1Deg = angleStartDeg;
                        totalSweepDeg = 360;
                        theta2Deg = theta1Deg + progress * totalSweepDeg;

                    } else { // Bwd type
                        if (N === 2) {
                            // For N=2, a "Bwd" compute sweeps 180 degrees "backwards" from the other node.
                            angleStartDeg = angleNext; // The arc will end at the angle of the other node.
                            totalSweepDeg = 180;    // The magnitude of the sweep.
                        } else { // N > 2 (original logic for more than 2 nodes)
                            angleStartDeg = angleNext;
                            const angleEndTargetDeg = anglePrev; // Visually, arc starts from prev and goes to next (backwards)
                            totalSweepDeg = (angleStartDeg - angleEndTargetDeg + 360) % 360;
                            // Fallback for N > 2 if sweep is 0 unexpectedly
                            if (totalSweepDeg === 0 && angleStartDeg === angleEndTargetDeg && progress > 0) {
                                totalSweepDeg = 360;
                            }
                        }
                        const currentSweepAmount = progress * totalSweepDeg;
                        theta1Deg = angleStartDeg - currentSweepAmount; // Arc data starts here
                        theta2Deg = angleStartDeg;                   // Arc data ends here
                    }
                } else { // N === 1 (since N=0 is guarded at the function start)
                    // For a single node, sweep a full circle starting from the top.
                    theta1Deg = -90;
                    totalSweepDeg = 360;
                    theta2Deg = theta1Deg + progress * totalSweepDeg;
                }

                // Draw the arc if the sweep is significant
                if (Math.abs(progress * totalSweepDeg) > 1e-3) {
                    const startPoint = pointOnCircle(outerPos.x, outerPos.y, arcOuterRadius, theta1Deg);
                    const endPoint = pointOnCircle(outerPos.x, outerPos.y, arcOuterRadius, theta2Deg);

                    const actualSweepAngle = progress * totalSweepDeg;
                    // largeArcFlag: 1 if arc sweep is > 180 degrees.
                    // For exactly 360 degrees, SVG needs largeArcFlag=1 if start/end points are identical.
                    // The original logic: Math.abs(actualSweepAngle) >= 360 ? 1 : (Math.abs(actualSweepAngle) > 180 ? 1 : 0) handles this well.
                    const largeArcFlag = Math.abs(actualSweepAngle) >= 360 ? 1 : (Math.abs(actualSweepAngle) > 180 ? 1 : 0);
                    
                    // sweepFlagArc: 1 for positive angle direction (typically counter-clockwise).
                    // All our totalSweepDeg are positive magnitudes, and theta calculations respect direction.
                    let sweepFlagArc = 1;
                    if (N === 1 && type === "Bwd") {
                        sweepFlagArc = 0; // For N=1 and "Bwd" type, sweep Clockwise
                    }

                    const pathData = `M ${startPoint.x} ${startPoint.y} A ${arcOuterRadius} ${arcOuterRadius} 0 ${largeArcFlag} ${sweepFlagArc} ${endPoint.x} ${endPoint.y}`;
                    computeArc.setAttribute('d', pathData);
                    computeArc.setAttribute('stroke', device.compute.color || 'gray');
                    computeArc.setAttribute('visibility', 'visible');
                } else {
                    computeArc.setAttribute('visibility', 'hidden');
                }
            } else if (computeArc) { computeArc.setAttribute('visibility', 'hidden'); }
        }
        // ... end of existing updateSVG logic

        syncAnimationHeaderVisibility(); // Ensure header reflects current state after draw
    }

    // --- UI Update Functions ---
    function updateStatusDisplay(state) {
        let statusText = `Cycle: ${state.current_frame || 0}`; // Default frame to 0 if undefined
        if (state.is_complete) {
            statusText += " | Status: Complete";
        } else if (state.is_paused) {
            statusText += " | Status: Paused";
        } else {
            statusText += " | Status: Running";
        }
        if (state.target_cycle !== null && state.target_cycle > (state.current_frame || 0) && !state.is_complete) {
            statusText += ` (Target: ${state.target_cycle})`;
        }
        statusArea.textContent = statusText;
        runToCycleInput.value = state.current_frame || 0;
        syncAnimationHeaderVisibility(); // Also update header as status text changes
    }

    function updateControls(state) {
        const canPotentiallyRun = simulationInitialized && simulationConfig && simulationConfig.N > 0;
        const isEffectivelyRunning = !state.is_paused && !state.is_complete;
    
        if (playBtn) playBtn.disabled = !canPotentiallyRun || isEffectivelyRunning || state.is_complete;
        if (pauseBtn) pauseBtn.disabled = !canPotentiallyRun || !isEffectivelyRunning;
        
        if (speedSlider) {
            speedSlider.disabled = !canPotentiallyRun;
            
            let speedForUI = initialSpeedDisplay; // Default to initial
            if (state && typeof state.speed_level !== 'undefined') {
                speedForUI = state.speed_level;
            } else if (simulationState && typeof simulationState.speed_level !== 'undefined') { 
                // Fallback to main thread's global simulationState if argument doesn't have it
                speedForUI = simulationState.speed_level;
            }
            // else, speedForUI remains initialSpeedDisplay

            if (document.activeElement !== speedSlider) {
                if (String(speedSlider.value) !== String(speedForUI)) {
                    speedSlider.value = speedForUI;
                }
            }
            if (speedValue) {
                if (document.activeElement !== speedSlider) {
                    speedValue.textContent = speedForUI;
                } else {
                    // Input listener handles textContent when slider is active
                    // speedValue.textContent = speedSlider.value; // This would be correct here too
                }
            }
        }
        
        if (runToBtn) runToBtn.disabled = !canPotentiallyRun || state.is_complete;
        if (runToCycleInput) runToCycleInput.disabled = !canPotentiallyRun || state.is_complete;
    
        syncAnimationHeaderVisibility();
    }

    function displayLegends(config) {
        if (memoryLegendArea) {
            memoryLegendArea.style.display = config.memory_legend ? 'block' : 'none';
            if (config.memory_legend && memoryLegendArea.querySelector('pre')) memoryLegendArea.querySelector('pre').textContent = config.memory_legend;
        }
        if (computeLegendArea) {
            computeLegendArea.style.display = config.compute_legend ? 'block' : 'none';
            if (config.compute_legend && computeLegendArea.querySelector('pre')) computeLegendArea.querySelector('pre').textContent = config.compute_legend;
        }
    }

    function displayCompletionStats(stats) {
        if (completionPopup && completionArea && stats && stats.text) {
            completionArea.textContent = stats.text;
            completionPopup.style.display = 'block';
        } else {
            hideCompletionPopup();
        }
    }

    function hideCompletionPopup() {
        if (completionPopup) completionPopup.style.display = 'none';
    }

    function displayError(message) {
        if (errorMessageDiv) {
            errorMessageDiv.textContent = message;
            errorMessageDiv.classList.toggle('visible', !!message);
        }
        syncAnimationHeaderVisibility(); // Error message affects header visibility
    }

    function handleApiError(errorMsg, context) {
        const fullMsg = `Error ${context}: ${errorMsg}`;
        console.error(fullMsg);
        displayError(fullMsg);
    }

    function setFormEnabled(isEnabled) {
        Array.from(paramsForm.elements).forEach(el => {
            if (el.type === 'submit') {
                el.disabled = false;
                el.textContent = isEnabled ? 'Prepare Simulation' : 'Reset Simulation';
            } else {
                // Keep specific form inputs like 'attn_type' and 'chunk_type' disabled if in prepare mode (isEnabled)
                if (isEnabled && (el.id === 'attn_type' || el.id === 'chunk_type')) {
                    el.disabled = true;
                } else {
                    el.disabled = !isEnabled;
                }
            }
        });

        if (isEnabled) { // "Prepare" mode
            console.log("Main: Setting UI to PREPARE mode.");
            simulationInitialized = false;
            if (simulationConfig) simulationConfig.N = 0; // Affects updateControls logic

            // Explicitly disable all simulation-specific controls
            if (playBtn) playBtn.disabled = true;
            if (pauseBtn) pauseBtn.disabled = true;
            if (speedSlider) speedSlider.disabled = true;
            if (runToCycleInput) runToCycleInput.disabled = true;
            if (runToBtn) runToBtn.disabled = true;
            if (speedValue) speedValue.style.opacity = '0.7';

            hideTorusPlot();
            torusPlotInitialized = false;
            // Update controls to reflect this "prepare" state definitively
            updateControls({ is_paused: true, is_complete: false, N: 0 });
        } else { // "Simulation Active/Starting" mode
            console.log("Main: Setting UI to SIMULATION ACTIVE/STARTING mode.");
            if (speedValue) speedValue.style.opacity = '1';
            // Controls (play, pause, etc.) will be enabled/disabled by `updateControls`
            // when the first `stateUpdate` message is received from the worker,
            // based on the actual initial state of the simulation (e.g., if it starts paused).
        }
        syncAnimationHeaderVisibility();
    }

    // --- SVG Helper Functions (Unchanged) ---
    function addSvgElement(parent, tagName, attributes = {}) { const el = document.createElementNS(svgNS, tagName); for (const attr in attributes) { el.setAttribute(attr, attributes[attr]); } parent.appendChild(el); return el; }
    function addSvgCircle(parent, cx, cy, r, attributes = {}) { return addSvgElement(parent, 'circle', { cx, cy, r, ...attributes }); }
    function addSvgRect(parent, x, y, width, height, attributes = {}) { return addSvgElement(parent, 'rect', { x, y, width, height, ...attributes }); }
    function addSvgPath(parent, d, attributes = {}) { return addSvgElement(parent, 'path', { d, ...attributes }); }
    function addSvgText(parent, x, y, textContent, attributes = {}) { attributes['dominant-baseline'] = attributes['dominant-baseline'] || 'central'; attributes['text-anchor'] = attributes['text-anchor'] || 'middle'; const el = addSvgElement(parent, 'text', { x, y, ...attributes }); updateMultiLineText(el, textContent, x, y, parseFloat(attributes['font-size'] || 1)); return el; }
    function createMarker(id, color, size, refX, markerWidth, markerHeight) { const marker = document.createElementNS(svgNS, 'marker'); marker.setAttribute('id', id); marker.setAttribute('viewBox', `0 0 ${size*1.2} ${markerHeight*1.2}`); marker.setAttribute('markerWidth', markerWidth); marker.setAttribute('markerHeight', markerHeight); marker.setAttribute('refX', refX); marker.setAttribute('refY', markerHeight / 2); marker.setAttribute('orient', 'auto-start-reverse'); addSvgElement(marker, 'path', { d: `M 0 0 L ${size} ${markerHeight / 2} L 0 ${markerHeight} z`, fill: color }); return marker; }
    function updateMultiLineText(textElement, text, x, y, fontSize) { textElement.textContent = ''; textElement.setAttribute('x', x); textElement.setAttribute('y', y); if (fontSize) { textElement.setAttribute('font-size', fontSize); } textElement.setAttribute('dominant-baseline', 'central'); textElement.setAttribute('text-anchor', 'middle'); const lines = String(text).split('\n'); const numLines = lines.length; const lineSpacingEm = 1.2; const initialDyEm = numLines > 1 ? - (numLines - 1) * 0.5 * lineSpacingEm : 0; lines.forEach((line, index) => { const tspan = document.createElementNS(svgNS, "tspan"); tspan.setAttribute("x", x); tspan.setAttribute("dy", index === 0 ? `${initialDyEm}em` : `${lineSpacingEm}em`); tspan.textContent = line; textElement.appendChild(tspan); }); }
    function pointOnCircle(cx, cy, radius, angleDegrees) { const angleRadians = angleDegrees * Math.PI / 180; return { x: cx + radius * Math.cos(angleRadians), y: cy + radius * Math.sin(angleRadians) }; }
    function getColorForDevice(index, total, baseHue) { const startLightness = 90; const endLightness = 10; const saturation = 95; if (total <= 0) { return `hsl(${baseHue}, ${saturation}%, 50%)`; } const validatedHue = ((baseHue % 360) + 360) % 360; const factor = (total > 1) ? index / (total - 1) : 0; const currentLightness = startLightness + factor * (endLightness - startLightness); return `hsl(${validatedHue}, ${saturation}%, ${currentLightness}%)`; }

    // --- Torus Plot Functions (Unchanged) ---
    function linspace(start, end, num) { if (num <= 1) return [start]; const step = (end - start) / (num - 1); const arr = []; for (let i = 0; i < num; i++) { arr.push(start + i * step); } return arr; }
    function hslToRgbString(h, s, l) { s = 0.75; l = 0.65; let r, g, b; if (s === 0) { r = g = b = l; } else { const hue2rgb = (p, q, t) => { if (t < 0) t += 1; if (t > 1) t -= 1; if (t < 1 / 6) return p + (q - p) * 6 * t; if (t < 1 / 2) return q; if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6; return p; }; const q = l < 0.5 ? l * (1 + s) : l + s - l * s; const p = 2 * l - q; h /= 360; r = hue2rgb(p, q, h + 1 / 3); g = hue2rgb(p, q, h); b = hue2rgb(p, q, h - 1 / 3); } const to255 = x => Math.round(x * 255); return `rgb(${to255(r)},${to255(g)},${to255(b)})`; }
    
    function drawTorusPlot(N_slices, M_nodes_per_slice) { 
        if (!torusPlotContainer) {
            return;
        }

        // Handle special case of N_slices = 2, should draw a cyclinder instead of a torus
        if (N_slices === 2) {
            // Call the dedicated function for the N_slices = 2 cylinder case
            drawCylinderForTwoSlices(M_nodes_per_slice, torusPlotContainer);
            return;
        }
    

        const R_major = 4.0; 
        const r_minor = 1.0; 
        const twist_factor = 1; 
        const phi_slices = linspace(0, 2 * Math.PI, N_slices + 1).slice(0, N_slices); 
        const theta_nodes_base = linspace(0, 2 * Math.PI, M_nodes_per_slice + 1).slice(0, M_nodes_per_slice); 
        const nodes = []; 
        const all_x = []; 
        const all_y = []; 
        const all_z = []; 
        const node_colors = []; 
        const slice_labels = []; 
        const face_i = []; 
        const face_j = []; 
        const face_k = []; 
        const node_wire_colors_rgb = []; 
        for (let j = 0; j < M_nodes_per_slice; j++) { 
            const hue = (j * 360 / M_nodes_per_slice) % 360; 
            node_wire_colors_rgb.push(hslToRgbString(hue, 0.75, 0.65)); 
        } 
        for (let k = 0; k < N_slices; k++) { 
            const phi = phi_slices[k]; 
            const slice_nodes_coords = []; 
            let sum_x = 0, sum_y = 0, sum_z = 0; 
            const baseVertexIndex = k * M_nodes_per_slice; 
            for (let j = 0; j < M_nodes_per_slice; j++) { 
                const theta_base = theta_nodes_base[j]; 
                const theta_twisted = theta_base + twist_factor * phi; 
                const radius_factor = R_major + r_minor * Math.cos(theta_twisted); 
                const x = radius_factor * Math.cos(phi); 
                const y = radius_factor * Math.sin(phi); 
                const z = r_minor * Math.sin(theta_twisted); 
                slice_nodes_coords.push([x, y, z]); 
                all_x.push(x); 
                all_y.push(y); 
                all_z.push(z); 
                node_colors.push(node_wire_colors_rgb[j]); 
                sum_x += x; 
                sum_y += y; 
                sum_z += z; 
            } 
            nodes.push(slice_nodes_coords); 
            const center_x = sum_x / M_nodes_per_slice; 
            const center_y = sum_y / M_nodes_per_slice; 
            const center_z = sum_z / M_nodes_per_slice; 
            slice_labels.push({ x: center_x, y: center_y, z: center_z, 
                                text: k.toString(), font: { color: 'darkgreen', size: 14 }, showarrow: false, }); 
            if (M_nodes_per_slice >= 3) {  
                const v0_idx = baseVertexIndex + 0; 
                for (let m = 0; m < M_nodes_per_slice - 2; m++) { 
                    const v1_idx = baseVertexIndex + (m + 1); 
                    const v2_idx = baseVertexIndex + (m + 2); 
                    face_i.push(v0_idx); 
                    face_j.push(v1_idx); 
                    face_k.push(v2_idx); 
                } 
            } 
        } 
        const traces = []; 
        if (face_i.length > 0) { 
            traces.push({ type: 'mesh3d', x: all_x, y: all_y, z: all_z, 
                            i: face_i, j: face_j, k: face_k, 
                            color: 'black', opacity: 0.3, flatshading: true, hoverinfo: 'none', showlegend: false }); 
        } 
        traces.push({ x: all_x, y: all_y, z: all_z, mode: 'markers', type: 'scatter3d', 
                        marker: { color: node_colors, size: 6, symbol: 'circle', line: { color: 'black', width: 1.5 }, opacity: 1.0 }, 
                        showlegend: false, hoverinfo: 'none' }); 
        
        let legendTitle = '';
        for (let j = 0; j < M_nodes_per_slice; j++) { 
            const wire_x = []; 
            const wire_y = []; 
            const wire_z = []; 
            for (let k = 0; k < N_slices; k++) { 
                if (nodes && nodes[k] && nodes[k][j] && nodes[k][j].length === 3) { 
                    wire_x.push(nodes[k][j][0]); 
                    wire_y.push(nodes[k][j][1]); 
                    wire_z.push(nodes[k][j][2]); 
                } 
            } 
            if (nodes && nodes[0] && nodes[0][j] && nodes[0][j].length === 3) { 
                wire_x.push(nodes[0][j][0]); 
                wire_y.push(nodes[0][j][1]); 
                wire_z.push(nodes[0][j][2]); 
            } 
            traces.push({ x: wire_x, y: wire_y, z: wire_z, 
                            mode: 'lines', type: 'scatter3d', name: `Seq. ${j}`, 
                             line: { color: node_wire_colors_rgb[j], width: 3 }, hoverinfo: 'none', showlegend: true }); 
        } 
         
        const layout = {
            title: {
                text: `<b>Model Stages: ${N_slices}<br>(Hypothetical) Concurrent Rings: ${M_nodes_per_slice}</b>`,
                x: 0.5, xanchor: 'center', font: { size: 16, family: 'sans-serif' }
            },
            showlegend: true,
            legend: { title: { text: legendTitle }, x: 0, xanchor: 'left', y: 1, yanchor: 'top', bgcolor: 'rgba(255,255,255,0.7)' },
            margin: { l: 0, r: 0, b: 0, t: 40 },
            scene: {
                xaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
                yaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
                zaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
                aspectmode: 'data',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } },
                bgcolor: 'rgba(0,0,0,0)',
                annotations: slice_labels
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
    
        Plotly.newPlot(torusPlotContainer, traces, layout, { responsive: true });
    }

    function drawCylinderForTwoSlices(M_nodes_per_slice, plotContainerElement) {
        if (!plotContainerElement) {
            console.error("Plot container element not provided for drawCylinderForTwoSlices.");
            return;
        }
    
        const cylinder_radius = 1.0;
        const cylinder_height = 4.0;
        
        const newSliceLabelFontSize = 14; // Increased font size for slice labels
    
        const theta_nodes = linspace(0, 2 * Math.PI, M_nodes_per_slice + 1).slice(0, M_nodes_per_slice);
    
        const all_x = [];
        const all_y = [];
        const all_z = [];
        const node_colors_per_sequence = [];
    
        // Arrays for face indices: one for sides, one for caps
        const face_i_sides = [], face_j_sides = [], face_k_sides = [];
        const face_i_caps = [], face_j_caps = [], face_k_caps = [];
    
    
        for (let j = 0; j < M_nodes_per_slice; j++) {
            const hue = (j * 360 / M_nodes_per_slice) % 360;
            node_colors_per_sequence.push(hslToRgbString(hue, 0.75, 0.65));
        }
    
        const nodes_slice_coords = [[], []];
    
        const z0 = -cylinder_height / 2;
        for (let j = 0; j < M_nodes_per_slice; j++) {
            const theta = theta_nodes[j];
            const x = cylinder_radius * Math.cos(theta);
            const y = cylinder_radius * Math.sin(theta);
            nodes_slice_coords[0].push([x, y, z0]);
            all_x.push(x); all_y.push(y); all_z.push(z0);
        }
    
        const z1 = cylinder_height / 2;
        for (let j = 0; j < M_nodes_per_slice; j++) {
            const theta = theta_nodes[j];
            const x = cylinder_radius * Math.cos(theta);
            const y = cylinder_radius * Math.sin(theta);
            nodes_slice_coords[1].push([x, y, z1]);
            all_x.push(x); all_y.push(y); all_z.push(z1);
        }
    
        const marker_node_colors = [];
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < M_nodes_per_slice; j++) {
                marker_node_colors.push(node_colors_per_sequence[j]);
            }
        }
    
        // Create faces for the cylinder SIDES
        for (let j = 0; j < M_nodes_per_slice; j++) {
            const next_j = (j + 1) % M_nodes_per_slice;
            const idx_s0_j = j;
            const idx_s0_next_j = next_j;
            const idx_s1_j = j + M_nodes_per_slice;
            const idx_s1_next_j = next_j + M_nodes_per_slice;
    
            // These faces are for the sides. We will not use them for the shaded mesh.
            // If you needed a transparent mesh for sides, you'd use these.
            // face_i_sides.push(idx_s0_j, idx_s0_next_j);
            // face_j_sides.push(idx_s0_next_j, idx_s1_next_j);
            // face_k_sides.push(idx_s1_j, idx_s1_j);
        }
    
        // Create faces for the cylinder CAPS (bottom and top)
        // Bottom cap (slice 0)
        if (M_nodes_per_slice >= 3) {
            for (let j = 1; j < M_nodes_per_slice - 1; j++) {
                face_i_caps.push(0);
                face_j_caps.push(j);
                face_k_caps.push(j + 1);
            }
        }
        // Top cap (slice 1)
        if (M_nodes_per_slice >= 3) {
            const base_idx_s1 = M_nodes_per_slice;
            for (let j = 1; j < M_nodes_per_slice - 1; j++) {
                face_i_caps.push(base_idx_s1 + 0);
                face_j_caps.push(base_idx_s1 + j);
                face_k_caps.push(base_idx_s1 + j + 1);
            }
        }
    
        const traces = [];
    
        // Cylinder CAP mesh (shaded)
        if (face_i_caps.length > 0) {
            traces.push({
                type: 'mesh3d',
                x: all_x, y: all_y, z: all_z,
                i: face_i_caps, j: face_j_caps, k: face_k_caps, // Use only cap faces
                color: 'black', opacity: 0.3, // Adjust opacity as needed
                flatshading: true, hoverinfo: 'none', showlegend: false
            });
        }
    
        // Node markers (same as before)
        traces.push({
            x: all_x, y: all_y, z: all_z,
            mode: 'markers', type: 'scatter3d',
            marker: {
                color: marker_node_colors, size: 6, symbol: 'circle',
                line: { color: 'black', width: 1.5 }, opacity: 1.0
            },
            showlegend: false, hoverinfo: 'none'
        });
    
        // Lines connecting corresponding nodes (cylinder height lines - "shared links")
        for (let j = 0; j < M_nodes_per_slice; j++) {
            traces.push({
                x: [nodes_slice_coords[0][j][0], nodes_slice_coords[1][j][0]],
                y: [nodes_slice_coords[0][j][1], nodes_slice_coords[1][j][1]],
                z: [nodes_slice_coords[0][j][2], nodes_slice_coords[1][j][2]],
                mode: 'lines', type: 'scatter3d',
                name: `Seq. ${j}`,
                line: { color: node_colors_per_sequence[j], width: 3 },
                hoverinfo: 'none', showlegend: true
            });
        }
    
        // Lines for the circular edges of the cylinder caps
        [[nodes_slice_coords[0], "Bottom Cap Edge"], [nodes_slice_coords[1], "Top Cap Edge"]].forEach(cap_data => {
            const cap_nodes = cap_data[0];
            const cap_x = cap_nodes.map(n => n[0]);
            const cap_y = cap_nodes.map(n => n[1]);
            const cap_z = cap_nodes.map(n => n[2]);
            if (M_nodes_per_slice > 1) {
                cap_x.push(cap_nodes[0][0]);
                cap_y.push(cap_nodes[0][1]);
                cap_z.push(cap_nodes[0][2]);
            }
            traces.push({
                x: cap_x, y: cap_y, z: cap_z,
                mode: 'lines', type: 'scatter3d', name: cap_data[1],
                line: { color: 'dimgray', width: 1.5 },
                hoverinfo: 'none', showlegend: false
            });
        });
    
        // Slice labels with increased font size
        const slice_labels = [
            { x: 0, y: 0, z: z0, text: '0', font: { color: 'darkgreen', size: newSliceLabelFontSize }, showarrow: false},
            { x: 0, y: 0, z: z1, text: '1', font: { color: 'darkgreen', size: newSliceLabelFontSize }, showarrow: false}
        ];
    
        const layout = {
            title: {
                text: `<b>Model Stages: 2<br>(Hypothetical) Concurrent Rings: ${M_nodes_per_slice}</b>`,
                x: 0.5, xanchor: 'center', font: { size: 16, family: 'sans-serif' }
            },
            showlegend: true,
            legend: { title: { text: '' }, x: 0, xanchor: 'left', y: 1, yanchor: 'top', bgcolor: 'rgba(255,255,255,0.7)' },
            margin: { l: 0, r: 0, b: 0, t: 40 },
            scene: {
                xaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
                yaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
                zaxis: { visible: false, showgrid: false, zeroline: false, automargin: true },
                aspectmode: 'data',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } },
                bgcolor: 'rgba(0,0,0,0)',
                annotations: slice_labels
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
    
        Plotly.newPlot(plotContainerElement, traces, layout, { responsive: true });
    }

    function showTorusPlot() { if (torusPlotContainer && horizontalResizer) { const parentHeight = torusPlotContainer.parentElement.clientHeight; const resizerHeight = horizontalResizer.offsetHeight || 8; const availableHeight = parentHeight - resizerHeight; const initialTopPercent = 0.60; let initialTopPx = Math.max(minPaneHeight, Math.floor(availableHeight * initialTopPercent)); let initialBottomPx = Math.max(minPaneHeight, availableHeight - initialTopPx); if (initialTopPx + initialBottomPx > availableHeight) { initialBottomPx = availableHeight - initialTopPx; if (initialBottomPx < minPaneHeight) { initialBottomPx = minPaneHeight; initialTopPx = availableHeight - initialBottomPx; } } if (svgContainer) { svgContainer.style.height = `${initialTopPx}px`; } torusPlotContainer.style.height = `${initialBottomPx}px`; torusPlotContainer.style.display = 'block'; horizontalResizer.style.display = 'block'; if (torusPlotInitialized) { requestAnimationFrame(() => { try { Plotly.Plots.resize(torusPlotContainer); } catch (e) { console.warn("Plotly resize failed in showTorusPlot (rAF):", e); } }); } } }
    function hideTorusPlot() { if (torusPlotContainer && horizontalResizer) { torusPlotContainer.style.display = 'none'; horizontalResizer.style.display = 'none'; if(svgContainer) svgContainer.style.height = ''; torusPlotContainer.style.height = ''; try { Plotly.purge(torusPlotContainer); } catch (e) { console.warn("Could not purge torus plot.", e); } } }

    

    // --- Initial Setup on Page Load ---
    if (memoryLegendArea) memoryLegendArea.style.display = 'none';
    if (computeLegendArea) computeLegendArea.style.display = 'none';

    setFormEnabled(true); // Start in "prepare" mode (disables controls, sets button to "Prepare")
    // updateControls and updateStatusDisplay are implicitly called by setFormEnabled
    // to reflect the initial non-initialized, paused state.
    hideCompletionPopup();
    syncAnimationHeaderVisibility(); // Initial sync for the header
    updateControls(simulationState);

}); // End DOMContentLoaded