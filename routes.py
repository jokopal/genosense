import os
import json
from datetime import datetime
from flask import render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

from app import app, db
from models import ImageData, InfectionData, PredictionModel
from ml_models import process_image, get_model_info
from cellular_automata import predict_spread

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with the interactive map and sidebar"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with information on Ganoderma and classification methods"""
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Image upload page and processing"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Create database entry
            new_image = ImageData(filename=filename)
            db.session.add(new_image)
            db.session.commit()
            
            # Process image (this would trigger the ML model)
            result_path = process_image(filepath, new_image.id)
            
            # Update database entry with result
            new_image.processed = True
            new_image.result_path = result_path
            db.session.commit()
            
            flash('File successfully uploaded and processed')
            return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/api/infection_data')
def get_infection_data():
    """API endpoint to get infection data for the map"""
    infections = InfectionData.query.all()
    data = [{
        'id': infection.id,
        'lat': infection.latitude,
        'lng': infection.longitude,
        'level': infection.infection_level,
        'date': infection.date_recorded.strftime('%Y-%m-%d')
    } for infection in infections]
    
    return jsonify(data)

@app.route('/api/trend_data')
def get_trend_data():
    """API endpoint to get trend data for visualization"""
    # Group infections by date and calculate average severity
    infections = InfectionData.query.all()
    
    # Process data for trend visualization
    dates = {}
    for infection in infections:
        date_str = infection.date_recorded.strftime('%Y-%m-%d')
        if date_str not in dates:
            dates[date_str] = {'count': 0, 'total_level': 0}
        
        dates[date_str]['count'] += 1
        dates[date_str]['total_level'] += infection.infection_level
    
    trend_data = [{
        'date': date,
        'count': data['count'],
        'avg_level': data['total_level'] / data['count'] if data['count'] > 0 else 0
    } for date, data in dates.items()]
    
    # Sort by date
    trend_data.sort(key=lambda x: x['date'])
    
    return jsonify(trend_data)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to get infection spread prediction using Cellular Automata"""
    data = request.json
    
    # Get current infection data from database
    infections = InfectionData.query.all()
    current_state = [{
        'lat': infection.latitude,
        'lng': infection.longitude,
        'level': infection.infection_level
    } for infection in infections]
    
    # Get prediction parameters from request
    days = data.get('days', 30)
    
    # Generate prediction using cellular automata
    prediction = predict_spread(current_state, days)
    
    return jsonify(prediction)

@app.route('/api/model_info')
def model_info():
    """API endpoint to get information about the ML models"""
    return jsonify(get_model_info())

# Sample data route for development purposes
@app.route('/api/sample_data', methods=['POST'])
def add_sample_data():
    """Add sample infection data for development and testing"""
    if not app.debug:
        return jsonify({"error": "This endpoint is only available in debug mode"}), 403
    
    # Add 20 sample infection points around a center point
    center_lat = request.json.get('lat', 0)
    center_lng = request.json.get('lng', 0)
    
    for i in range(20):
        # Create random points within ~5km of center
        lat_offset = (np.random.random() - 0.5) * 0.1
        lng_offset = (np.random.random() - 0.5) * 0.1
        
        new_infection = InfectionData(
            latitude=center_lat + lat_offset,
            longitude=center_lng + lng_offset,
            infection_level=np.random.random(),
            date_recorded=datetime.utcnow()
        )
        db.session.add(new_infection)
    
    db.session.commit()
    return jsonify({"status": "Sample data added"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
