/**
 * Genosense - Palm Oil Monitoring System
 * Leaflet Map Initialization and Controls
 */

/**
 * Initialize the leaflet map
 */
function initMap() {
    // Create the map in the 'map' container
    app.map = L.map('map', {
        center: [-2.5489, 118.0149], // Center of Indonesia
        zoom: 5,
        minZoom: 4,
        maxZoom: 18,
        zoomControl: true,
        attributionControl: true
    });
    
    // Add the dark basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(app.map);
    
    // Add satellite layer
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    });
    
    // Add layer control
    const baseMaps = {
        "Dark": L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }),
        "Satellite": satellite
    };
    
    // Add the layer control to the map
    L.control.layers(baseMaps).addTo(app.map);
    
    // Create infection layer group
    app.layers.infections = L.layerGroup().addTo(app.map);
    
    // Create prediction layer group
    app.layers.predictions = L.layerGroup().addTo(app.map);
    
    // Add map legend
    addMapLegend();
    
    // Add map event listeners
    app.map.on('click', onMapClick);
}

/**
 * Update the infection layer with current data
 */
function updateInfectionLayer() {
    // Clear existing markers
    app.layers.infections.clearLayers();
    
    // Add markers for each infection point
    app.data.infections.forEach(infection => {
        // Determine marker class based on infection level
        let markerClass = 'infection-marker ';
        if (infection.level < 0.3) {
            markerClass += 'infection-marker-low';
        } else if (infection.level < 0.7) {
            markerClass += 'infection-marker-medium';
        } else {
            markerClass += 'infection-marker-high';
        }
        
        // Calculate marker size based on infection level (10-30px)
        const size = 10 + (infection.level * 20);
        
        // Create custom marker
        const marker = L.divIcon({
            className: markerClass,
            html: '',
            iconSize: [size, size]
        });
        
        // Create marker and add to layer
        const infectionMarker = L.marker([infection.lat, infection.lng], { icon: marker })
            .bindPopup(`
                <div class="infection-popup">
                    <h3>Infection Point</h3>
                    <p>Level: ${(infection.level * 100).toFixed(1)}%</p>
                    <p>Date: ${infection.date || 'Unknown'}</p>
                    <div class="infection-level">
                        <div class="infection-level-fill" style="width: ${infection.level * 100}%"></div>
                    </div>
                </div>
            `);
        
        app.layers.infections.addLayer(infectionMarker);
    });
    
    // If there are infection points, fit the map to bounds
    if (app.data.infections.length > 0) {
        const markers = Object.values(app.layers.infections._layers);
        const group = L.featureGroup(markers);
        app.map.fitBounds(group.getBounds(), { padding: [50, 50] });
    }
}

/**
 * Update the prediction layer with current prediction data
 */
function updatePredictionLayer() {
    // Clear existing predictions
    app.layers.predictions.clearLayers();
    
    // If no prediction data or not in prediction view, do nothing
    if (!app.data.predictions || app.currentView !== 'prediksi') {
        return;
    }
    
    // Get the current timeframe data
    const timeframe = app.data.predictions.timeframes[app.predictionTimeframe];
    
    // Update day display
    document.getElementById('prediction-day-display').textContent = `Day ${timeframe.day}`;
    
    // Add markers for each prediction point
    timeframe.points.forEach(point => {
        // Determine marker class based on infection level
        let markerClass = 'infection-marker prediction-overlay ';
        if (point.level < 0.3) {
            markerClass += 'infection-marker-low';
        } else if (point.level < 0.7) {
            markerClass += 'infection-marker-medium';
        } else {
            markerClass += 'infection-marker-high';
        }
        
        // Calculate marker size based on infection level (8-25px)
        const size = 8 + (point.level * 17);
        
        // Create custom marker
        const marker = L.divIcon({
            className: markerClass,
            html: '',
            iconSize: [size, size]
        });
        
        // Create marker and add to layer
        const predictionMarker = L.marker([point.lat, point.lng], { icon: marker })
            .bindPopup(`
                <div class="infection-popup">
                    <h3>Predicted Infection</h3>
                    <p>Level: ${(point.level * 100).toFixed(1)}%</p>
                    <p>Day: ${timeframe.day}</p>
                    <div class="infection-level">
                        <div class="infection-level-fill" style="width: ${point.level * 100}%"></div>
                    </div>
                </div>
            `);
        
        app.layers.predictions.addLayer(predictionMarker);
    });
}

/**
 * Handle map click events
 * @param {Event} e - The click event
 */
function onMapClick(e) {
    // If in development mode and prediksi view, allow adding sample data
    if (window.location.hostname === 'localhost' && app.currentView === 'prediksi') {
        if (confirm('Add sample infection data at this location?')) {
            addSampleData(e.latlng.lat, e.latlng.lng);
        }
    }
}

/**
 * Add sample infection data for development and testing
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 */
function addSampleData(lat, lng) {
    // Show loading indicator
    showLoader(true);
    
    // Send request to add sample data
    fetch('/api/sample_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ lat, lng })
    })
        .then(response => response.json())
        .then(data => {
            showFlashMessage('Sample data added', 'success');
            
            // Reload infection data
            loadInfectionData();
            
            // Hide loading indicator
            showLoader(false);
        })
        .catch(error => {
            console.error('Error adding sample data:', error);
            showFlashMessage('Error adding sample data', 'error');
            showLoader(false);
        });
}

/**
 * Add a legend to the map
 */
function addMapLegend() {
    // Create a custom legend control
    const legend = L.control({ position: 'bottomright' });
    
    legend.onAdd = function(map) {
        const div = L.DomUtil.create('div', 'map-legend');
        div.innerHTML = `
            <h4>Infection Level</h4>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(39, 174, 96, 0.8);"></div>
                <div class="legend-label">Low (0-30%)</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(243, 156, 18, 0.8);"></div>
                <div class="legend-label">Medium (30-70%)</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(192, 57, 43, 0.8);"></div>
                <div class="legend-label">High (70-100%)</div>
            </div>
        `;
        return div;
    };
    
    legend.addTo(app.map);
}
