"""
Script untuk menyiapkan database dan menambahkan data sampel untuk development
"""

import os
import sys
from datetime import datetime, timedelta
import random

# Pastikan variable lingkungan sudah diatur
if 'DATABASE_URL' not in os.environ:
    print("ERROR: Environment variable DATABASE_URL belum diatur.")
    print("Contoh: export DATABASE_URL=postgresql://username:password@localhost:5432/genosense")
    sys.exit(1)

from app import app, db
from models import ImageData, InfectionData, PredictionModel

def setup_database():
    """Buat tabel database"""
    with app.app_context():
        db.create_all()
        print("Database tabel telah dibuat.")

def add_sample_data():
    """Tambahkan data sampel untuk development/testing"""
    with app.app_context():
        # Buat sampel Model Prediksi
        if PredictionModel.query.count() == 0:
            models = [
                PredictionModel(name="UNet Segmentation", model_type="UNet", 
                               accuracy=0.89, active=True),
                PredictionModel(name="ANN Classifier", model_type="ANN", 
                               accuracy=0.92, active=True),
                PredictionModel(name="Cellular Automata", model_type="CA", 
                               accuracy=0.85, active=True),
            ]
            db.session.add_all(models)
            db.session.commit()
            print("Sample prediction models added.")
        
        # Buat sampel data infeksi
        if InfectionData.query.count() == 0:
            # Lokasi sekitar Malaysia/Indonesia (area perkebunan kelapa sawit)
            base_lat, base_lng = 3.140853, 101.693207  # Kuala Lumpur
            
            # Buat 25 titik sampel
            infection_points = []
            for i in range(25):
                # Lokasi acak dalam radius 50km
                lat_offset = (random.random() - 0.5) * 0.5
                lng_offset = (random.random() - 0.5) * 0.5
                
                # Tanggal acak dalam 30 hari terakhir
                days_ago = random.randint(0, 30)
                date = datetime.utcnow() - timedelta(days=days_ago)
                
                # Level infeksi acak (0.1 - 0.9)
                infection_level = round(random.uniform(0.1, 0.9), 2)
                
                infection_points.append(
                    InfectionData(
                        latitude=base_lat + lat_offset,
                        longitude=base_lng + lng_offset,
                        infection_level=infection_level,
                        date_recorded=date
                    )
                )
            
            db.session.add_all(infection_points)
            db.session.commit()
            print("Sample infection data added.")
        
        print("Setup complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--with-sample-data":
        setup_database()
        add_sample_data()
    else:
        setup_database()
        print("Untuk menambahkan data sampel, jalankan: python setup_db.py --with-sample-data")