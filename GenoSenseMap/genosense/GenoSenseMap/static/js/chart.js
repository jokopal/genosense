/**
 * Genosense - Palm Oil Monitoring System
 * Chart and Data Visualization Functions
 */

/**
 * Initialize charts used in the application
 */
function initCharts() {
    // Set up the trend chart (empty initially)
    const trendCtx = document.getElementById('trend-chart').getContext('2d');
    app.charts.trend = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Infection Count',
                data: [],
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                label: 'Average Severity (%)',
                data: [],
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Infection Count',
                        color: '#e74c3c'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                y1: {
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Severity (%)',
                        color: '#f39c12'
                    },
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Infection Trend Over Time',
                    color: 'rgba(255, 255, 255, 0.9)',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.datasetIndex === 0) {
                                label += context.parsed.y;
                            } else {
                                label += context.parsed.y.toFixed(1) + '%';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
    
    // Set up the pattern chart (empty initially)
    const patternCtx = document.getElementById('pattern-chart').getContext('2d');
    app.charts.pattern = new Chart(patternCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Infection Distribution',
                data: [],
                backgroundColor: function(context) {
                    const value = context.raw ? context.raw.z : 0;
                    if (value < 0.3) return 'rgba(39, 174, 96, 0.7)';
                    if (value < 0.7) return 'rgba(243, 156, 18, 0.7)';
                    return 'rgba(192, 57, 43, 0.7)';
                },
                borderColor: 'rgba(255, 255, 255, 0.3)',
                borderWidth: 1,
                pointRadius: function(context) {
                    const value = context.raw ? context.raw.z : 0;
                    return 5 + (value * 10); // 5-15px based on severity
                },
                pointHoverRadius: function(context) {
                    const value = context.raw ? context.raw.z : 0;
                    return 7 + (value * 12); // 7-19px based on severity
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Longitude',
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Latitude',
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Spatial Distribution of G. boninense Infections',
                    color: 'rgba(255, 255, 255, 0.9)',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return [
                                `Latitude: ${point.y.toFixed(4)}`,
                                `Longitude: ${point.x.toFixed(4)}`,
                                `Severity: ${(point.z * 100).toFixed(1)}%`
                            ];
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update the trend chart with current data
 */
function updateTrendChart() {
    if (!app.data.trends || !app.charts.trend) {
        return;
    }
    
    // Extract data for chart
    const labels = app.data.trends.map(item => item.date);
    const countData = app.data.trends.map(item => item.count);
    const severityData = app.data.trends.map(item => item.avg_level * 100); // Convert to percentage
    
    // Update chart data
    app.charts.trend.data.labels = labels;
    app.charts.trend.data.datasets[0].data = countData;
    app.charts.trend.data.datasets[1].data = severityData;
    
    // Update chart
    app.charts.trend.update();
    
    // Update trend summary
    updateTrendSummary();
}

/**
 * Update the pattern chart with current data
 */
function updatePatternChart() {
    if (!app.data.infections || !app.charts.pattern) {
        return;
    }
    
    // Prepare data for scatter plot
    const scatterData = app.data.infections.map(infection => ({
        x: infection.lng,
        y: infection.lat,
        z: infection.level // Used for coloring and sizing
    }));
    
    // Update chart data
    app.charts.pattern.data.datasets[0].data = scatterData;
    
    // Update chart
    app.charts.pattern.update();
    
    // Update pattern analysis
    updatePatternAnalysis();
}

/**
 * Update the trend summary with statistics
 */
function updateTrendSummary() {
    if (!app.data.trends || app.data.trends.length === 0) {
        return;
    }
    
    // Calculate trend statistics
    const latestData = app.data.trends[app.data.trends.length - 1];
    const earliestData = app.data.trends[0];
    
    // Calculate growth rate
    const countGrowth = latestData.count - earliestData.count;
    const countGrowthPercent = (countGrowth / earliestData.count * 100).toFixed(1);
    
    // Calculate severity change
    const severityChange = (latestData.avg_level - earliestData.avg_level) * 100;
    const severityChangePercent = severityChange.toFixed(1);
    
    // Calculate total infections
    const totalInfections = app.data.trends.reduce((sum, item) => sum + item.count, 0);
    
    // Update summary HTML
    const trendSummary = document.getElementById('trend-summary');
    trendSummary.innerHTML = `
        <h3>Trend Analysis</h3>
        <div class="summary-item">
            <span class="summary-label">Total Detected Infections:</span>
            <span class="summary-value">${totalInfections}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Growth Rate:</span>
            <span class="summary-value ${countGrowth > 0 ? 'text-danger' : 'text-success'}">
                ${countGrowth > 0 ? '+' : ''}${countGrowthPercent}%
            </span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Severity Change:</span>
            <span class="summary-value ${severityChange > 0 ? 'text-danger' : 'text-success'}">
                ${severityChange > 0 ? '+' : ''}${severityChangePercent}%
            </span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Time Period:</span>
            <span class="summary-value">${earliestData.date} to ${latestData.date}</span>
        </div>
    `;
}

/**
 * Update the pattern analysis with spatial statistics
 */
function updatePatternAnalysis() {
    if (!app.data.infections || app.data.infections.length === 0) {
        return;
    }
    
    // Calculate spatial statistics
    
    // Calculate center of mass
    let totalLat = 0;
    let totalLng = 0;
    let totalWeight = 0;
    
    app.data.infections.forEach(infection => {
        const weight = infection.level;
        totalLat += infection.lat * weight;
        totalLng += infection.lng * weight;
        totalWeight += weight;
    });
    
    const centerLat = (totalWeight > 0) ? totalLat / totalWeight : 0;
    const centerLng = (totalWeight > 0) ? totalLng / totalWeight : 0;
    
    // Calculate average distance from center (spread)
    let totalDistance = 0;
    
    app.data.infections.forEach(infection => {
        const dLat = infection.lat - centerLat;
        const dLng = infection.lng - centerLng;
        const distance = Math.sqrt(dLat * dLat + dLng * dLng);
        totalDistance += distance;
    });
    
    const avgDistance = app.data.infections.length > 0 ? 
        totalDistance / app.data.infections.length : 0;
    
    // Calculate approximate area (convert to km²)
    // 1 degree is approximately 111 km at the equator
    const areaKm = Math.PI * Math.pow(avgDistance * 111, 2);
    
    // Calculate clustering index (ratio of points within close proximity)
    const proximityThreshold = 0.05; // ~5.5 km
    let closeProximityCount = 0;
    
    for (let i = 0; i < app.data.infections.length; i++) {
        for (let j = i + 1; j < app.data.infections.length; j++) {
            const dLat = app.data.infections[i].lat - app.data.infections[j].lat;
            const dLng = app.data.infections[i].lng - app.data.infections[j].lng;
            const distance = Math.sqrt(dLat * dLat + dLng * dLng);
            
            if (distance < proximityThreshold) {
                closeProximityCount++;
            }
        }
    }
    
    const maxPossiblePairs = (app.data.infections.length * (app.data.infections.length - 1)) / 2;
    const clusteringIndex = maxPossiblePairs > 0 ? 
        closeProximityCount / maxPossiblePairs : 0;
    
    // Update pattern analysis HTML
    const patternAnalysis = document.getElementById('pattern-analysis');
    patternAnalysis.innerHTML = `
        <h3>Spatial Pattern Analysis</h3>
        <div class="summary-item">
            <span class="summary-label">Infection Center:</span>
            <span class="summary-value">${centerLat.toFixed(4)}, ${centerLng.toFixed(4)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Affected Area:</span>
            <span class="summary-value">${areaKm.toFixed(2)} km²</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Clustering Index:</span>
            <span class="summary-value">${(clusteringIndex * 100).toFixed(1)}%</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Spread Pattern:</span>
            <span class="summary-value">
                ${clusteringIndex > 0.5 ? 'Clustered' : clusteringIndex > 0.2 ? 'Mixed' : 'Dispersed'}
            </span>
        </div>
    `;
}
