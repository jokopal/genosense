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

# Import new modules for model handling
from model_utils.model_loader import get_available_models, load_model, clear_model_cache
from model_utils.image_processor import ImageProcessor
from model_utils.gcp_config import get_gcp_config, is_gcp_configured, setup_gcp_credentials, get_available_gcp_models

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
            
            try:
                # Process image (this would trigger the ML model)
                result_path = process_image(filepath, new_image.id)
                
                # Update database entry with result
                new_image.processed = True
                new_image.result_path = result_path
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                flash(f'Error processing image: {str(e)}')
                new_image.processed = False
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
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        app.logger.error(f"Error serving uploaded file {filename}: {str(e)}")
        return "File not found", 404

@app.route('/api/models')
def get_models_list():
    """API endpoint to get all available ML models (local and GCP)"""
    # Get local models
    local_models = get_available_models()
    
    # Get GCP models if configured
    gcp_models = get_available_gcp_models() if is_gcp_configured() else {}
    
    # Combine all models
    all_models = {**local_models, **gcp_models}
    
    # Convert to list format for API
    models_list = [
        {
            'id': model_id,
            'name': model_info['name'],
            'type': model_info['type'],
            'description': model_info['description'],
            'accuracy': model_info['accuracy'],
            'last_updated': model_info['last_updated'],
            'active': model_info['active'],
            'location': 'gcp' if model_info.get('from_gcs', False) else 'local'
        } 
        for model_id, model_info in all_models.items()
    ]
    
    return jsonify({
        'models': models_list,
        'gcp_configured': is_gcp_configured()
    })

@app.route('/api/gcp/configure', methods=['POST'])
def configure_gcp():
    """API endpoint to configure GCP credentials"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    
    # Check for required fields
    if 'credentials' not in data:
        return jsonify({"error": "GCP credentials required"}), 400
    
    # Set up GCP credentials
    success = setup_gcp_credentials(data['credentials'])
    
    if success:
        return jsonify({"status": "GCP credentials configured successfully"})
    else:
        return jsonify({"error": "Failed to configure GCP credentials"}), 500

@app.route('/api/process_image_with_model', methods=['POST'])
def process_image_with_model():
    """API endpoint to process an image with a specific model"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    
    # Check for required fields
    if 'image_id' not in data or 'model_id' not in data:
        return jsonify({"error": "Image ID and model ID required"}), 400
    
    # Get the image record
    image = ImageData.query.get(data['image_id'])
    if not image:
        return jsonify({"error": f"Image with ID {data['image_id']} not found"}), 404
    
    # Get the image file path
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file {image.filename} not found"}), 404
    
    # Create image processor with the specified model
    try:
        processor = ImageProcessor(model_id=data['model_id'], use_gpu=data.get('use_gpu', False))
        
        # Process the image
        success, infection_data = processor.detect_infections(image_path, image.id)
        
        if success:
            return jsonify({
                "status": "success",
                "infections_found": len(infection_data) > 0,
                "infection_data": infection_data
            })
        else:
            return jsonify({"error": "Failed to process image"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    """API endpoint to upload a new model file"""
    # Check if the post request has the file part
    if 'model_file' not in request.files:
        return jsonify({"error": "No model file part"}), 400
    
    model_file = request.files['model_file']
    if model_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check model type
    model_type = request.form.get('model_type')
    if model_type not in ['keras', 'pytorch']:
        return jsonify({"error": "Invalid model type. Must be 'keras' or 'pytorch'"}), 400
    
    # Save the model file
    filename = secure_filename(model_file.filename)
    models_dir = os.path.join(os.path.dirname(__file__), 'models', 'saved')
    os.makedirs(models_dir, exist_ok=True)
    
    filepath = os.path.join(models_dir, filename)
    model_file.save(filepath)
    
    # Add model to database
    model_name = request.form.get('name', filename)
    model_description = request.form.get('description', 'Uploaded model')
    accuracy = float(request.form.get('accuracy', 0.8))
    
    new_model = PredictionModel(
        name=model_name,
        model_type=model_type,
        accuracy=accuracy,
        active=True
    )
    
    db.session.add(new_model)
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "message": "Model uploaded successfully",
        "model_id": new_model.id
    })
