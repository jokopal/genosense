/**
 * Genosense - Palm Oil Monitoring System
 * Main JavaScript file
 */

// Global state
const app = {
    currentView: 'dashboard',
    map: null,
    layers: {
        infections: null,
        predictions: null
    },
    data: {
        infections: [],
        predictions: null,
        trends: []
    },
    charts: {
        trend: null,
        pattern: null
    },
    predictionTimeframe: 0,
    animationTimer: null
};

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

/**
 * Initialize the application
 */
function initApp() {
    // Initialize the map
    initMap();
    
    // Set up navigation event listeners
    setupNavigation();
    
    // Load initial data
    loadInfectionData();
    
    // Initialize charts
    initCharts();
    
    // Set up flash message handling
    setupFlashMessages();
}

/**
 * Set up navigation between different views
 */
function setupNavigation() {
    // Get all navigation items
    const navItems = document.querySelectorAll('.nav-item');
    
    // Add click event listener to each navigation item
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            const view = item.getAttribute('data-view');
            changeView(view);
            
            // Update active class
            navItems.forEach(navItem => navItem.classList.remove('active'));
            item.classList.add('active');
        });
    });
    
    // Set the default active view
    document.querySelector('.nav-item[data-view="dashboard"]').classList.add('active');
}

/**
 * Change the current view
 * @param {string} view - The view to display
 */
function changeView(view) {
    // Hide all panels
    const panels = document.querySelectorAll('.panel');
    panels.forEach(panel => panel.classList.add('panel-hidden'));
    
    // Show the selected panel
    const selectedPanel = document.querySelector(`.${view}-panel`);
    if (selectedPanel) {
        selectedPanel.classList.remove('panel-hidden');
    }
    
    // Update the current view
    app.currentView = view;
    
    // Perform view-specific actions
    switch (view) {
        case 'dashboard':
            // Refresh map
            app.map.invalidateSize();
            break;
        case 'pola':
            // Load pattern data if needed
            loadPatternData();
            break;
        case 'trend':
            // Load trend data
            loadTrendData();
            break;
        case 'prediksi':
            // Show prediction controls
            document.querySelector('.prediction-controls').classList.remove('panel-hidden');
            // Generate prediction if not already done
            if (!app.data.predictions) {
                generatePrediction();
            }
            break;
        default:
            // Hide prediction controls for other views
            document.querySelector('.prediction-controls').classList.add('panel-hidden');
            break;
    }
}

/**
 * Load infection data from the API
 */
function loadInfectionData() {
    // Show loading indicator
    showLoader(true);
    
    // Fetch infection data from API
    fetch('/api/infection_data')
        .then(response => response.json())
        .then(data => {
            // Store the data
            app.data.infections = data;
            
            // Update the map with infection data
            updateInfectionLayer();
            
            // Hide loading indicator
            showLoader(false);
        })
        .catch(error => {
            console.error('Error loading infection data:', error);
            showFlashMessage('Error loading infection data', 'error');
            showLoader(false);
        });
}

/**
 * Load trend data from the API
 */
function loadTrendData() {
    // Show loading indicator
    showLoader(true);
    
    // Fetch trend data from API
    fetch('/api/trend_data')
        .then(response => response.json())
        .then(data => {
            // Store the data
            app.data.trends = data;
            
            // Update trend chart
            updateTrendChart();
            
            // Hide loading indicator
            showLoader(false);
        })
        .catch(error => {
            console.error('Error loading trend data:', error);
            showFlashMessage('Error loading trend data', 'error');
            showLoader(false);
        });
}

/**
 * Load pattern data for analysis
 */
function loadPatternData() {
    // In a real implementation, this would fetch pattern data from the API
    // For now, we'll use the infection data to show patterns
    
    // Use infection data to show patterns
    if (app.data.infections.length > 0) {
        updatePatternChart();
    }
}

/**
 * Generate prediction using the Cellular Automata model
 */
function generatePrediction() {
    // Show loading indicator
    showLoader(true);
    
    // Get prediction days from input
    const days = parseInt(document.getElementById('prediction-days').value) || 30;
    
    // Request prediction from API
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ days: days })
    })
        .then(response => response.json())
        .then(data => {
            // Store the prediction data
            app.data.predictions = data;
            
            // Reset timeframe
            app.predictionTimeframe = 0;
            
            // Update the prediction layer
            updatePredictionLayer();
            
            // Hide loading indicator
            showLoader(false);
        })
        .catch(error => {
            console.error('Error generating prediction:', error);
            showFlashMessage('Error generating prediction', 'error');
            showLoader(false);
        });
}

/**
 * Show or hide the loading indicator
 * @param {boolean} show - Whether to show or hide the loader
 */
function showLoader(show) {
    const loader = document.querySelector('.loader');
    if (show) {
        loader.style.display = 'block';
    } else {
        loader.style.display = 'none';
    }
}

/**
 * Display a flash message to the user
 * @param {string} message - The message to display
 * @param {string} type - The type of message (success, error, warning)
 */
function showFlashMessage(message, type = 'success') {
    const flashContainer = document.querySelector('.flash-container');
    
    // Create a new flash message element
    const flashElement = document.createElement('div');
    flashElement.className = `flash-message flash-${type}`;
    flashElement.textContent = message;
    
    // Add the message to the container
    flashContainer.appendChild(flashElement);
    
    // Remove the message after 5 seconds
    setTimeout(() => {
        flashElement.style.opacity = '0';
        setTimeout(() => {
            flashContainer.removeChild(flashElement);
        }, 300);
    }, 5000);
}

/**
 * Set up flash message handling for server-side messages
 */
function setupFlashMessages() {
    // This would handle flash messages coming from the server
    // For now, we'll just set up the container
    const flashContainer = document.createElement('div');
    flashContainer.className = 'flash-container';
    document.body.appendChild(flashContainer);
}

/**
 * Play prediction animation
 */
function playPredictionAnimation() {
    // Clear any existing animation timer
    if (app.animationTimer) {
        clearInterval(app.animationTimer);
    }
    
    // Set up animation interval
    app.animationTimer = setInterval(() => {
        // Advance to next timeframe
        app.predictionTimeframe++;
        
        // Check if we've reached the end
        if (app.predictionTimeframe >= app.data.predictions.timeframes.length) {
            app.predictionTimeframe = 0;
        }
        
        // Update prediction slider
        const slider = document.getElementById('prediction-timeframe');
        slider.value = app.predictionTimeframe;
        
        // Update the prediction layer
        updatePredictionLayer();
    }, 1000); // 1 second per timeframe
}

/**
 * Pause prediction animation
 */
function pausePredictionAnimation() {
    if (app.animationTimer) {
        clearInterval(app.animationTimer);
        app.animationTimer = null;
    }
}

/**
 * Reset prediction animation to initial state
 */
function resetPredictionAnimation() {
    pausePredictionAnimation();
    app.predictionTimeframe = 0;
    
    // Update prediction slider
    const slider = document.getElementById('prediction-timeframe');
    slider.value = app.predictionTimeframe;
    
    // Update the prediction layer
    updatePredictionLayer();
}
