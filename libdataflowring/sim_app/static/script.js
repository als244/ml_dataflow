// static/script.js

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
    // **NEW** References for completion popup
    const completionPopup = document.getElementById('completion-popup');
    const closeCompletionPopupBtn = document.getElementById('closeCompletionPopupBtn');


    // --- Simulation State ---
    let simulationState = { current_frame: 0, is_paused: true, is_complete: false, speed_level: 50, target_cycle: null, max_frames: 30000, completion_stats: {}, devices: [] };
    let simulationConfig = { N: 0, total_layers: 0, total_layers_non_head: 0, memory_legend: "", compute_legend: "" };
    let currentIntervalSec = 1.0;
    let animationTimer = null;
    let simulationInitialized = false;
    let isFetching = false;

    // --- SVG Rendering Constants ---
    const svgNS = "http://www.w3.org/2000/svg";
    const viewBoxWidth = 50;
    const centerX = viewBoxWidth / 2;
    const effectiveCenterY = viewBoxWidth / 2;
    const totalDistance           = effectiveCenterY * 0.98;
    const innerRadius             = effectiveCenterY * 0.35;
    const innerNodeRadius         = effectiveCenterY * 0.12;
    const outerNodeRadius         = effectiveCenterY * 0.12;
    const stallNodeRadius         = outerNodeRadius * 1.2;
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
    const deviceOpacity           = 0.4;
    const innerNodeOpacity        = 0.8;
    const stallNodeOpacity        = 0.9;
    let svgElements = {};
    let nodePositions = {};
    let drawingBounds = { minY: Infinity, maxY: -Infinity, minX: Infinity, maxX: -Infinity };

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


    // --- API Communication Functions ---

    async function startSimulation(event) {
        event.preventDefault();
        displayError('');
        hideCompletionPopup();

        // **** MODIFIED: Check the initialization flag ****
        if (simulationInitialized) {
        // **** END MODIFIED ****

            // --- Perform UI RESET ---
            console.log("Resetting UI for new simulation...");
            stopAnimationLoop();

            // Clear SVG and Hide/Clear Legends
            if(svg) {
                svg.innerHTML = '';
            }
            if (memoryLegendArea) {
                memoryLegendArea.style.display = 'none';
                if(memoryLegendArea.querySelector('pre')) memoryLegendArea.querySelector('pre').textContent = '';
            }
            if (computeLegendArea) {
                computeLegendArea.style.display = 'none';
                 if(computeLegendArea.querySelector('pre')) computeLegendArea.querySelector('pre').textContent = '';
            }

            setFormEnabled(true); // Re-enable form, disable controls, set button text back
            simulationState = { ...simulationState, is_paused: true, is_complete: false, current_frame: 0 }; // Reset local state parts
            updateStatusDisplay(simulationState);
            updateControls(simulationState); // Ensure controls reflect reset state

            // Reset simulation config variables
            simulationConfig.memory_legend = "";
            simulationConfig.compute_legend = "";
            simulationConfig.N = 0;

            // **** ADDED: Reset the initialization flag ****
            simulationInitialized = false;
            // **** END ADDED ****

            console.log("UI Reset complete. Ready to start new simulation.");
            return; // Exit function after UI reset
        }

        // --- START New Simulation ---
        // (Code is the same as before, but now only runs if simulationInitialized is false)
        console.log("Attempting to start simulation...");
        if (memoryLegendArea) memoryLegendArea.style.display = 'none';
        if (computeLegendArea) computeLegendArea.style.display = 'none';
        stopAnimationLoop();

        const formData = new FormData(paramsForm);
        console.log("Starting simulation with params:", Object.fromEntries(formData));

        setFormEnabled(false); // Disable form after reading data
        statusArea.textContent = "Status: Initializing...";

        try {
            const response = await fetch('/start_simulation', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok && data.success) {
                console.log("Simulation started successfully via API.");
                simulationConfig = data.config;
                simulationState = data.state;
                currentIntervalSec = data.interval_sec;

                // **** ADDED: Set initialization flag on success ****
                simulationInitialized = true;
                // **** END ADDED ****

                setupSVG(simulationConfig);
                window.requestAnimationFrame(() => { updateSVG(simulationState); });
                displayLegends(simulationConfig);
                updateStatusDisplay(simulationState);
                updateControls(simulationState);
                runToCycleInput.max = simulationState.max_frames;
                if (!simulationState.is_paused) {
                    fetchUpdate();
                }
                console.log("Initialization complete.");

            } else {
                handleApiError(data.error || response.statusText, "starting simulation");
                setFormEnabled(true);
                simulationState = { ...simulationState, is_paused: true, is_complete: false };
                simulationInitialized = false; // Ensure flag is false on error
                updateStatusDisplay(simulationState);
                updateControls(simulationState);
            }
        } catch (error) {
            handleApiError(error.message, "starting simulation (network/fetch error)");
            setFormEnabled(true);
            simulationState = { ...simulationState, is_paused: true, is_complete: false };
            simulationInitialized = false; // Ensure flag is false on error
            updateStatusDisplay(simulationState);
            updateControls(simulationState);
        }
    }

    async function fetchUpdate() {
        clearTimeout(animationTimer);
        if (simulationState.is_paused || simulationState.is_complete || isFetching) { return; }
        isFetching = true;
        try {
            const response = await fetch('/get_state_update');
            const data = await response.json();
            if (response.ok && data.success) {
                simulationState = data.state; currentIntervalSec = data.interval_sec;
                 window.requestAnimationFrame(() => { updateSVG(simulationState); });
                updateStatusDisplay(simulationState); updateControls(simulationState);
                if (simulationState.is_complete) {
                     // **MODIFIED** Call displayCompletionStats to show popup
                     displayCompletionStats(simulationState.completion_stats);
                     console.log("Simulation complete.");
                 } else if (!simulationState.is_paused) {
                     const delay = Math.max(1, currentIntervalSec * 1000);
                     animationTimer = setTimeout(fetchUpdate, delay);
                 }
            } else {
                handleApiError(data.error || response.statusText, "fetching update");
                simulationState.is_paused = true; updateControls(simulationState);
            }
        } catch (error) {
            handleApiError(error.message, "fetching update (network error)");
            simulationState.is_paused = true; updateControls(simulationState);
        } finally { isFetching = false; }
    }

    async function sendControlCommand(command, value = null) {
        displayError(''); console.log(`Sending command: ${command}`, value !== null ? `Value: ${value}` : '');
        const wasPaused = simulationState.is_paused; stopAnimationLoop(); hideCompletionPopup(); // Hide popup on control action
        try {
            const body = { command }; if (value !== null) { body.value = value; }
            const response = await fetch('/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const data = await response.json();
            if (response.ok && data.success) {
                 console.log(`Command '${command}' successful.`);
                simulationState = { ...simulationState, ...data.state_summary }; currentIntervalSec = data.interval_sec;
                if (command === 'restart') {
                    setupSVG(simulationConfig); await fetchInitialStateAfterReset();
                } else {
                    updateStatusDisplay(simulationState); updateControls(simulationState); runToCycleInput.value = simulationState.current_frame;
                     if (!simulationState.is_paused && !simulationState.is_complete) { fetchUpdate(); }
                     else if (simulationState.is_complete) { await fetchFullStateAfterCommand(); displayCompletionStats(simulationState.completion_stats); }
                     else if (!wasPaused && simulationState.is_paused) { await fetchFullStateAfterCommand(); }
                }
            } else {
                handleApiError(data.error || response.statusText, `sending command '${command}'`);
                 if (!wasPaused && !simulationState.is_complete) { fetchUpdate(); }
            }
        } catch (error) {
            handleApiError(error.message, `sending command '${command}' (network error)`);
             if (!wasPaused && !simulationState.is_complete) { fetchUpdate(); }
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

     async function fetchInitialStateAfterReset() {
         try {
             const response = await fetch('/get_state_update'); const data = await response.json();
             if (response.ok && data.success) {
                 simulationState = data.state; currentIntervalSec = data.interval_sec;
                 window.requestAnimationFrame(() => { updateSVG(simulationState); });
                 updateStatusDisplay(simulationState); updateControls(simulationState);
             } else { handleApiError(data.error, "fetching state after reset"); }
         } catch (error) { handleApiError(error.message, "fetching state after reset (network error)"); }
     }


    // --- SVG Setup Function ---
    // setupSVG remains unchanged
    function setupSVG(config) {
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
        for (let i = 0; i < N; i++) { const deviceId = `dev_${i}`; const outerPos = nodePositions.outer[i]; const innerPos = nodePositions.inner[i]; const stallPos = nodePositions.stall[i]; const unitDir = nodePositions.unit[i]; const baseColor = getColorForDevice(i, N); const devGroup = addSvgElement(g, 'g', { id: deviceId }); svgElements[`${deviceId}_outer_circle`] = addSvgCircle(devGroup, outerPos.x, outerPos.y, outerNodeRadius, { fill: baseColor, stroke: 'black', 'stroke-width': baseStrokeWidth, opacity: deviceOpacity, id: `${deviceId}_outer_circle` }); const innerSquareSide = innerNodeRadius * 1.414; svgElements[`${deviceId}_inner_square`] = addSvgRect(devGroup, innerPos.x - innerSquareSide / 2, innerPos.y - innerSquareSide / 2, innerSquareSide, innerSquareSide, { fill: baseColor, stroke: 'black', 'stroke-width': baseStrokeWidth, opacity: innerNodeOpacity, id: `${deviceId}_inner_square` }); svgElements[`${deviceId}_stall_node`] = addSvgCircle(devGroup, stallPos.x, stallPos.y, stallNodeRadius, { fill: 'red', stroke: baseColor, 'stroke-width': baseStrokeWidth * 2, opacity: stallNodeOpacity, visibility: 'hidden', id: `${deviceId}_stall_node` }); svgElements[`${deviceId}_finish_indicator`] = addSvgCircle(devGroup, stallPos.x, stallPos.y, stallNodeRadius, { fill: 'lime', stroke: baseColor, 'stroke-width': baseStrokeWidth * 2, opacity: stallNodeOpacity, visibility: 'hidden', id: `${deviceId}_finish_indicator` }); svgElements[`${deviceId}_outer_label`] = addSvgText(devGroup, outerPos.x, outerPos.y, `D${i}`, { 'font-size': outerLabelFontSize, fill: 'black', id: `${deviceId}_outer_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_inner_label`] = addSvgText(devGroup, innerPos.x, innerPos.y, `Home`, { 'font-size': innerLabelFontSize, fill: 'black', id: `${deviceId}_inner_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_stall_label`] = addSvgText(devGroup, stallPos.x, stallPos.y, "", { 'font-size': stallLabelFontSize, fill: 'white', 'font-weight': 'bold', visibility: 'hidden', id: `${deviceId}_stall_label`, 'pointer-events': 'none' }); const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const inStartX = innerPos.x + unitDir.x * innerNodeRadius + offset.x; const inStartY = innerPos.y + unitDir.y * innerNodeRadius + offset.y; const outStartX = outerPos.x - unitDir.x * outerNodeRadius - offset.x; const outStartY = outerPos.y - unitDir.y * outerNodeRadius - offset.y; svgElements[`${deviceId}_in_arrow`] = addSvgPath(devGroup, `M ${inStartX} ${inStartY}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_in_arrow` }); svgElements[`${deviceId}_in_label`] = addSvgText(devGroup, inStartX, inStartY, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_in_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_out_arrow`] = addSvgPath(devGroup, `M ${outStartX} ${outStartY}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_out_arrow` }); svgElements[`${deviceId}_out_label`] = addSvgText(devGroup, outStartX, outStartY, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_out_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_ring_arrow`] = addSvgPath(devGroup, `M ${outerPos.x} ${outerPos.y}`, { stroke: 'gray', 'stroke-width': baseStrokeWidth * 1.5, fill: 'none', 'marker-end': 'url(#arrowhead)', visibility: 'hidden', id: `${deviceId}_ring_arrow` }); svgElements[`${deviceId}_ring_label`] = addSvgText(devGroup, outerPos.x, outerPos.y, "", { 'font-size': transferLabelFontSize, visibility: 'hidden', id: `${deviceId}_ring_label`, 'pointer-events': 'none' }); svgElements[`${deviceId}_compute_arc`] = addSvgPath(devGroup, "", { stroke: 'gray', 'stroke-width': baseStrokeWidth * 3, fill: 'none', visibility: 'hidden', id: `${deviceId}_compute_arc` }); }
    }

    // --- SVG Update Function ---
    // updateSVG remains unchanged
    function updateSVG(state) {
        const N = simulationConfig.N; if (N <= 0 || !state.devices) return;
        for (let i = 0; i < N; i++) { const device = state.devices.find(d => d.id === i); if (!device) continue; const deviceId = `dev_${i}`; const outerPos = nodePositions.outer[i]; const innerPos = nodePositions.inner[i]; const stallPos = nodePositions.stall[i]; const unitDir = nodePositions.unit[i]; const transferDist = Math.max(0, Math.hypot(outerPos.x - innerPos.x, outerPos.y - innerPos.y) - outerNodeRadius - innerNodeRadius); updateMultiLineText(svgElements[`${deviceId}_outer_label`], device.status_text || `D${i}`, outerPos.x, outerPos.y, outerLabelFontSize); svgElements[`${deviceId}_stall_node`].setAttribute('visibility', device.status === "Stalled" ? 'visible' : 'hidden'); svgElements[`${deviceId}_finish_indicator`].setAttribute('visibility', device.status === "Finished" ? 'visible' : 'hidden'); if (device.status === "Stalled") { updateMultiLineText(svgElements[`${deviceId}_stall_label`], device.stall_reason || "Stalled", stallPos.x, stallPos.y, stallLabelFontSize); svgElements[`${deviceId}_stall_label`].setAttribute('visibility', 'visible'); } else { svgElements[`${deviceId}_stall_label`].setAttribute('visibility', 'hidden'); } updateMultiLineText(svgElements[`${deviceId}_inner_label`], `Home`, innerPos.x, innerPos.y, innerLabelFontSize); const inArrow = svgElements[`${deviceId}_in_arrow`]; const inLabel = svgElements[`${deviceId}_in_label`]; if (device.inbound && device.inbound.progress > 1e-6 && transferDist > 1e-6) { const progress = Math.min(1.0, device.inbound.progress); const lenProg = progress * transferDist; const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const startX = innerPos.x + unitDir.x * innerNodeRadius + offset.x; const startY = innerPos.y + unitDir.y * innerNodeRadius + offset.y; const endX = startX + unitDir.x * lenProg; const endY = startY + unitDir.y * lenProg; const midX = (startX + endX) / 2; const midY = (startY + endY) / 2; const labelOffsetX = perpVec.x * labelOffsetDistance; const labelOffsetY = perpVec.y * labelOffsetDistance; inArrow.setAttribute('d', `M ${startX} ${startY} L ${endX} ${endY}`); inArrow.setAttribute('stroke', device.inbound.color || 'gray'); inArrow.setAttribute('visibility', 'visible'); updateMultiLineText(inLabel, device.inbound.label || '', midX + labelOffsetX, midY + labelOffsetY, transferLabelFontSize); inLabel.setAttribute('fill', device.inbound.color || 'gray'); inLabel.setAttribute('visibility', 'visible'); } else { inArrow.setAttribute('visibility', 'hidden'); inLabel.setAttribute('visibility', 'hidden'); } const outArrow = svgElements[`${deviceId}_out_arrow`]; const outLabel = svgElements[`${deviceId}_out_label`]; if (device.outbound && device.outbound.progress > 1e-6 && transferDist > 1e-6) { const progress = Math.min(1.0, device.outbound.progress); const lenProg = progress * transferDist; const perpVec = { x: -unitDir.y, y: unitDir.x }; const offset = { x: perpVec.x * arrowOffsetDist, y: perpVec.y * arrowOffsetDist }; const startX = outerPos.x - unitDir.x * outerNodeRadius - offset.x; const startY = outerPos.y - unitDir.y * outerNodeRadius - offset.y; const endX = startX - unitDir.x * lenProg; const endY = startY - unitDir.y * lenProg; const midX = (startX + endX) / 2; const midY = (startY + endY) / 2; const labelOffsetX = -perpVec.x * labelOffsetDistance; const labelOffsetY = -perpVec.y * labelOffsetDistance; outArrow.setAttribute('d', `M ${startX} ${startY} L ${endX} ${endY}`); outArrow.setAttribute('stroke', device.outbound.color || 'gray'); outArrow.setAttribute('visibility', 'visible'); updateMultiLineText(outLabel, device.outbound.label || '', midX + labelOffsetX, midY + labelOffsetY, transferLabelFontSize); outLabel.setAttribute('fill', device.outbound.color || 'gray'); outLabel.setAttribute('visibility', 'visible'); } else { outArrow.setAttribute('visibility', 'hidden'); outLabel.setAttribute('visibility', 'hidden'); } const ringArrow = svgElements[`${deviceId}_ring_arrow`]; const ringLabel = svgElements[`${deviceId}_ring_label`]; if (device.peer && device.peer.progress > 1e-6 && N > 1) { const progress = Math.min(1.0, device.peer.progress); const peerId = device.peer.target_peer; if (peerId >= 0 && peerId < N && peerId !== i) { const targetPos = nodePositions.outer[peerId]; const startPos = outerPos; const dx = targetPos.x - startPos.x; const dy = targetPos.y - startPos.y; const dist = Math.hypot(dx, dy); const startAngleRad = Math.atan2(dy, dx); const startEdge = pointOnCircle(startPos.x, startPos.y, outerNodeRadius, startAngleRad * 180 / Math.PI); const endAngleRad = Math.atan2(-dy, -dx); const endEdge = pointOnCircle(targetPos.x, targetPos.y, outerNodeRadius, endAngleRad * 180 / Math.PI); const currentEdgeX = startEdge.x + (endEdge.x - startEdge.x) * progress; const currentEdgeY = startEdge.y + (endEdge.y - startEdge.y) * progress; const sweepFlag = device.peer.direction > 0 ? 1 : 0; let arcRadius = dist * 0.6; if (Math.hypot(currentEdgeX - startEdge.x, currentEdgeY - startEdge.y) > 1e-3) { const pathData = `M ${startEdge.x} ${startEdge.y} A ${arcRadius} ${arcRadius} 0 0 ${sweepFlag} ${currentEdgeX} ${currentEdgeY}`; ringArrow.setAttribute('d', pathData); ringArrow.setAttribute('stroke', device.peer.color || 'indigo'); ringArrow.setAttribute('visibility', 'visible'); const midX = (startEdge.x + currentEdgeX) / 2; const midY = (startEdge.y + currentEdgeY) / 2; const vecCenterX = midX - centerX; const vecCenterY = midY - effectiveCenterY; const vecCenterMag = Math.hypot(vecCenterX, vecCenterY); const outX = vecCenterMag > 1e-6 ? vecCenterX / vecCenterMag : 0; const outY = vecCenterMag > 1e-6 ? vecCenterY / vecCenterMag : 0; const labelOffsetMag = labelOffsetDistance * 2.0; const labelX = midX + outX * labelOffsetMag; const labelY = midY + outY * labelOffsetMag; updateMultiLineText(ringLabel, device.peer.label || '', labelX, labelY, transferLabelFontSize); ringLabel.setAttribute('fill', device.peer.color || 'indigo'); ringLabel.setAttribute('visibility', 'visible'); } else { ringArrow.setAttribute('visibility', 'hidden'); ringLabel.setAttribute('visibility', 'hidden'); } } else { ringArrow.setAttribute('visibility', 'hidden'); ringLabel.setAttribute('visibility', 'hidden'); } } else { ringArrow.setAttribute('visibility', 'hidden'); ringLabel.setAttribute('visibility', 'hidden'); } const computeArc = svgElements[`${deviceId}_compute_arc`]; if (device.compute && device.compute.progress > 1e-6) { const progress = Math.min(1.0, device.compute.progress); const type = device.compute.type; const arcOuterRadius = outerNodeRadius * computeArcRadiusScale; const anglePrev = nodePositions.angleToPrev[i]; const angleNext = nodePositions.angleToNext[i]; let theta1Deg = 0, theta2Deg = 0, totalSweepDeg = 0; if (N > 1) { if (type === "Fwd") { const angleStartDeg = (device.compute.layer > 0 || N <= 1) ? anglePrev : (angleNext - 180); const angleEndTargetDeg = angleNext; totalSweepDeg = (angleEndTargetDeg - angleStartDeg + 360) % 360; if (N === 1) totalSweepDeg = 360; theta1Deg = angleStartDeg; theta2Deg = angleStartDeg + progress * totalSweepDeg; } else if (type === "Head") { theta1Deg = anglePrev; totalSweepDeg = 360; theta2Deg = theta1Deg + progress * totalSweepDeg; } else { const angleStartDeg = angleNext; const angleEndTargetDeg = anglePrev; totalSweepDeg = (angleStartDeg - angleEndTargetDeg + 360) % 360; if (N === 1) totalSweepDeg = 360; const currentSweepDeg = progress * totalSweepDeg; theta1Deg = angleStartDeg - currentSweepDeg; theta2Deg = angleStartDeg; } } else { totalSweepDeg = 360; theta1Deg = -90; theta2Deg = theta1Deg + progress * totalSweepDeg; } if (Math.abs(progress * totalSweepDeg) > 1e-3) { const startPoint = pointOnCircle(outerPos.x, outerPos.y, arcOuterRadius, theta1Deg); const endPoint = pointOnCircle(outerPos.x, outerPos.y, arcOuterRadius, theta2Deg); const largeArcFlag = Math.abs(progress * totalSweepDeg) >= 360 ? 1 : (Math.abs(progress * totalSweepDeg) > 180 ? 1 : 0); const sweepFlagArc = 1; const pathData = `M ${startPoint.x} ${startPoint.y} A ${arcOuterRadius} ${arcOuterRadius} 0 ${largeArcFlag} ${sweepFlagArc} ${endPoint.x} ${endPoint.y}`; computeArc.setAttribute('d', pathData); computeArc.setAttribute('stroke', device.compute.color || 'gray'); computeArc.setAttribute('visibility', 'visible'); } else { computeArc.setAttribute('visibility', 'hidden'); } } else { computeArc.setAttribute('visibility', 'hidden'); }
        }
    }

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
        Array.from(paramsForm.elements).forEach(el => { if (isEnabled && (el.id === 'attn_type' || el.id == 'chunk_type' || el.id === 'min_chunk_size')) { el.disabled = true; } else if (el !== submitButton) { el.disabled = !isEnabled; } });
        controlElements.forEach(control => { control.disabled = isEnabled; });
        if(speedValue) speedValue.style.opacity = isEnabled ? '0.7' : '1'; // Adjust opacity if needed
        if (submitButton) { submitButton.textContent = isEnabled ? 'Prepare Simulation' : 'Reset Simulation'; submitButton.disabled = false; }
        if (isEnabled) { playBtn.disabled = true; pauseBtn.disabled = true; speedSlider.disabled = true; runToCycleInput.disabled = true; runToBtn.disabled = true; }
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
    function getColorForDevice(index, total) { if (total <= 0) return 'gray'; const hue = (index * 360 / total) % 360; return `hsl(${hue}, 75%, 65%)`; }

    // --- Initial Setup ---
    // Hide legends initially - displayLegends will show them when simulation starts
    if (memoryLegendArea) memoryLegendArea.style.display = 'none';
    if (computeLegendArea) computeLegendArea.style.display = 'none';

    setFormEnabled(true);
    updateControls(simulationState);
    updateStatusDisplay(simulationState);
    hideCompletionPopup(); // Ensure popup is hidden initially

}); // End DOMContentLoaded