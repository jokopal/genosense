from datetime import datetime
from app import db

class ImageData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    result_path = db.Column(db.String(255), nullable=True)
    
    def __repr__(self):
        return f'<ImageData {self.filename}>'

class InfectionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    infection_level = db.Column(db.Float, nullable=False)  # 0-1 value indicating severity
    date_recorded = db.Column(db.DateTime, default=datetime.utcnow)
    source_image_id = db.Column(db.Integer, db.ForeignKey('image_data.id'), nullable=True)
    
    def __repr__(self):
        return f'<InfectionData lat:{self.latitude}, lng:{self.longitude}, level:{self.infection_level}>'

class PredictionModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # ANN, UNet, etc.
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    accuracy = db.Column(db.Float, nullable=True)
    active = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<PredictionModel {self.name} ({self.model_type})>'
