/**
 * Genosense - Palm Oil Monitoring System
 * Cellular Automata Implementation for Infection Prediction
 * 
 * This file contains client-side code for visualizing and interacting with
 * the Cellular Automata prediction model results.
 */

/**
 * Initialize prediction controls
 */
function initPredictionControls() {
    // Set up time slider
    const timeSlider = document.getElementById('prediction-timeframe');
    if (timeSlider) {
        timeSlider.addEventListener('input', function() {
            // Update timeframe
            app.predictionTimeframe = parseInt(this.value);
            
            // Update prediction layer
            updatePredictionLayer();
        });
    }
    
    // Set up days input
    const daysInput = document.getElementById('prediction-days');
    if (daysInput) {
        daysInput.addEventListener('change', function() {
            // Ensure value is in range
            if (this.value < 1) this.value = 1;
            if (this.value > 365) this.value = 365;
        });
    }
    
    // Set up predict button
    const predictButton = document.getElementById('generate-prediction');
    if (predictButton) {
        predictButton.addEventListener('click', function() {
            generatePrediction();
        });
    }
    
    // Set up play button
    const playButton = document.getElementById('play-prediction');
    if (playButton) {
        playButton.addEventListener('click', function() {
            if (app.animationTimer) {
                // Pause if already playing
                pausePredictionAnimation();
                this.textContent = '▶ Play';
            } else {
                // Play animation
                playPredictionAnimation();
                this.textContent = '⏸ Pause';
            }
        });
    }
    
    // Set up reset button
    const resetButton = document.getElementById('reset-prediction');
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            resetPredictionAnimation();
            
            // Reset play button text
            const playButton = document.getElementById('play-prediction');
            if (playButton) {
                playButton.textContent = '▶ Play';
            }
        });
    }
}

/**
 * Update the prediction time slider based on available timeframes
 */
function updatePredictionSlider() {
    if (!app.data.predictions) {
        return;
    }
    
    const slider = document.getElementById('prediction-timeframe');
    if (slider) {
        // Set slider range
        slider.min = 0;
        slider.max = app.data.predictions.timeframes.length - 1;
        slider.value = app.predictionTimeframe;
    }
}

/**
 * Update prediction summary with statistics
 */
function updatePredictionSummary() {
    if (!app.data.predictions || !app.data.predictions.timeframes) {
        return;
    }
    
    // Get initial and final states
    const initialState = app.data.predictions.initial_state;
    const finalState = app.data.predictions.final_state;
    
    // Calculate growth statistics
    const initialCount = initialState.length;
    const finalCount = finalState.length;
    const growthCount = finalCount - initialCount;
    const growthPercent = (growthCount / initialCount * 100).toFixed(1);
    
    // Calculate severity changes
    let initialSeveritySum = 0;
    let finalSeveritySum = 0;
    
    initialState.forEach(point => {
        initialSeveritySum += point.level;
    });
    
    finalState.forEach(point => {
        finalSeveritySum += point.level;
    });
    
    const initialAvgSeverity = initialSeveritySum / initialCount;
    const finalAvgSeverity = finalSeveritySum / finalCount;
    const severityChange = ((finalAvgSeverity - initialAvgSeverity) * 100).toFixed(1);
    
    // Calculate approximate affected area
    // Find the bounding box of points
    let minLat = Number.MAX_VALUE;
    let maxLat = Number.MIN_VALUE;
    let minLng = Number.MAX_VALUE;
    let maxLng = Number.MIN_VALUE;
    
    finalState.forEach(point => {
        minLat = Math.min(minLat, point.lat);
        maxLat = Math.max(maxLat, point.lat);
        minLng = Math.min(minLng, point.lng);
        maxLng = Math.max(maxLng, point.lng);
    });
    
    // Convert to approximate square kilometers (rough estimate)
    // 1 degree latitude is approximately 111 km
    // 1 degree longitude varies with latitude, roughly 111 * cos(latitude) km
    const avgLat = (minLat + maxLat) / 2;
    const latDistance = (maxLat - minLat) * 111;
    const lngDistance = (maxLng - minLng) * 111 * Math.cos(avgLat * Math.PI / 180);
    const areaKm = latDistance * lngDistance;
    
    // Get final timeframe
    const finalTimeframe = app.data.predictions.timeframes[app.data.predictions.timeframes.length - 1];
    
    // Update prediction summary HTML
    const predictionSummary = document.getElementById('prediction-summary');
    if (predictionSummary) {
        predictionSummary.innerHTML = `
            <h3>Prediction Summary</h3>
            <div class="summary-item">
                <span class="summary-label">Prediction Period:</span>
                <span class="summary-value">${finalTimeframe.day} days</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Growth Rate:</span>
                <span class="summary-value ${growthCount > 0 ? 'text-danger' : 'text-success'}">
                    ${growthCount > 0 ? '+' : ''}${growthPercent}%
                </span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Severity Change:</span>
                <span class="summary-value ${severityChange > 0 ? 'text-danger' : 'text-success'}">
                    ${severityChange > 0 ? '+' : ''}${severityChange}%
                </span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Affected Area:</span>
                <span class="summary-value">${areaKm.toFixed(2)} km²</span>
            </div>
        `;
    }
}
