/**
 * Genosense - Palm Oil Monitoring System
 * Map Controls and Interaction
 */

/**
 * Initialize map controls and UI interactions
 */
function initMapControls() {
    // Initialize prediction controls
    initPredictionControls();
    
    // Set up model info button
    const modelInfoButton = document.getElementById('model-info-button');
    if (modelInfoButton) {
        modelInfoButton.addEventListener('click', function() {
            fetchModelInfo();
        });
    }
    
    // Set up base map toggler
    const toggleBaseMapButton = document.getElementById('toggle-basemap');
    if (toggleBaseMapButton) {
        toggleBaseMapButton.addEventListener('click', function() {
            toggleBaseMap();
        });
    }
    
    // Set up export data button
    const exportDataButton = document.getElementById('export-data');
    if (exportDataButton) {
        exportDataButton.addEventListener('click', function() {
            exportMapData();
        });
    }
}

/**
 * Fetch and display model information
 */
function fetchModelInfo() {
    // Show loading indicator
    showLoader(true);
    
    // Fetch model info from API
    fetch('/api/model_info')
        .then(response => response.json())
        .then(data => {
            // Display model info in a modal
            showModelInfoModal(data);
            
            // Hide loading indicator
            showLoader(false);
        })
        .catch(error => {
            console.error('Error fetching model info:', error);
            showFlashMessage('Error fetching model information', 'error');
            showLoader(false);
        });
}

/**
 * Show modal with model information
 * @param {Array} models - Array of model information objects
 */
function showModelInfoModal(models) {
    // Create modal container
    const modalContainer = document.createElement('div');
    modalContainer.className = 'modal-container';
    
    // Create modal content
    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content';
    
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.className = 'modal-close';
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', function() {
        document.body.removeChild(modalContainer);
    });
    
    // Add header
    const header = document.createElement('h2');
    header.textContent = 'Model Information';
    
    // Create model cards
    const modelList = document.createElement('div');
    modelList.className = 'model-list';
    
    models.forEach(model => {
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        
        // Add model status indicator
        const statusIndicator = document.createElement('div');
        statusIndicator.className = `model-status ${model.active ? 'active' : 'inactive'}`;
        statusIndicator.textContent = model.active ? 'Active' : 'Inactive';
        
        // Create model content
        modelCard.innerHTML = `
            <h3>${model.name}</h3>
            <p>${model.description}</p>
            <div class="model-details">
                <div class="detail-item">
                    <span class="detail-label">Type:</span>
                    <span class="detail-value">${model.type}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Accuracy:</span>
                    <span class="detail-value">${(model.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Last Updated:</span>
                    <span class="detail-value">${model.last_updated}</span>
                </div>
            </div>
        `;
        
        // Add status indicator to card
        modelCard.appendChild(statusIndicator);
        
        // Add card to list
        modelList.appendChild(modelCard);
    });
    
    // Assemble modal
    modalContent.appendChild(closeButton);
    modalContent.appendChild(header);
    modalContent.appendChild(modelList);
    modalContainer.appendChild(modalContent);
    
    // Add event listener to close when clicking outside
    modalContainer.addEventListener('click', function(e) {
        if (e.target === modalContainer) {
            document.body.removeChild(modalContainer);
        }
    });
    
    // Add to document
    document.body.appendChild(modalContainer);
    
    // Add styles if not already present
    if (!document.getElementById('modal-styles')) {
        const modalStyles = document.createElement('style');
        modalStyles.id = 'modal-styles';
        modalStyles.textContent = `
            .modal-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 2000;
                animation: fadeIn 0.3s ease-out;
            }
            
            .modal-content {
                background-color: var(--secondary);
                border-radius: var(--border-radius);
                padding: 20px;
                max-width: 800px;
                width: 80%;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
            }
            
            .modal-close {
                position: absolute;
                top: 10px;
                right: 10px;
                background: none;
                border: none;
                font-size: 24px;
                color: var(--light);
                cursor: pointer;
            }
            
            .model-list {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .model-card {
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: var(--border-radius);
                padding: 15px;
                position: relative;
            }
            
            .model-status {
                position: absolute;
                top: 15px;
                right: 15px;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 12px;
            }
            
            .model-status.active {
                background-color: var(--success);
            }
            
            .model-status.inactive {
                background-color: var(--warning);
            }
            
            .model-details {
                margin-top: 15px;
            }
            
            .detail-item {
                margin-bottom: 8px;
            }
            
            .detail-label {
                font-weight: bold;
                margin-right: 5px;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        `;
        document.head.appendChild(modalStyles);
    }
}

/**
 * Toggle between different base maps
 */
function toggleBaseMap() {
    // Get current visible layer
    const layers = document.querySelectorAll('.leaflet-layer');
    const currentVisible = layers[0].style.display !== 'none' ? 0 : 1;
    
    // Toggle visibility
    layers[0].style.display = currentVisible === 0 ? 'none' : 'block';
    layers[1].style.display = currentVisible === 1 ? 'none' : 'block';
    
    // Update button text
    const toggleButton = document.getElementById('toggle-basemap');
    if (toggleButton) {
        toggleButton.textContent = currentVisible === 0 ? 'Show Dark Map' : 'Show Satellite';
    }
}

/**
 * Export current map data as GeoJSON
 */
function exportMapData() {
    // Create GeoJSON structure
    const geojson = {
        type: 'FeatureCollection',
        features: []
    };
    
    // Add infection data
    app.data.infections.forEach(infection => {
        geojson.features.push({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [infection.lng, infection.lat]
            },
            properties: {
                level: infection.level,
                date: infection.date
            }
        });
    });
    
    // Add prediction data if available
    if (app.data.predictions && app.currentView === 'prediksi') {
        const currentTimeframe = app.data.predictions.timeframes[app.predictionTimeframe];
        
        currentTimeframe.points.forEach(point => {
            geojson.features.push({
                type: 'Feature',
                geometry: {
                    type: 'Point',
                    coordinates: [point.lng, point.lat]
                },
                properties: {
                    level: point.level,
                    predicted: true,
                    day: currentTimeframe.day
                }
            });
        });
    }
    
    // Convert to JSON string
    const dataStr = JSON.stringify(geojson, null, 2);
    
    // Create download link
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    // Create filename with date
    const date = new Date().toISOString().split('T')[0];
    const filename = `genosense_export_${date}.geojson`;
    
    // Create and trigger download link
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', filename);
    linkElement.style.display = 'none';
    
    document.body.appendChild(linkElement);
    linkElement.click();
    document.body.removeChild(linkElement);
    
    // Show success message
    showFlashMessage('Data exported successfully', 'success');
}
