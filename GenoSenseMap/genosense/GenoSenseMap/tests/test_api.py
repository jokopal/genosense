import unittest
import json
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from app import db
from models import ImageData, InfectionData, PredictionModel

class APITestCase(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        
        with app.app_context():
            db.create_all()
            
            # Add test data
            test_model = PredictionModel(
                name="Test Model",
                model_type="keras",
                accuracy=0.95,
                active=True
            )
            db.session.add(test_model)
            db.session.commit()
            
            test_image = ImageData(
                filename="test_image.jpg",
                processed=True,
                result_path="uploads/test_image_result.jpg"
            )
            db.session.add(test_image)
            db.session.commit()
            
            test_infection = InfectionData(
                latitude=3.1234,
                longitude=101.5678,
                infection_level=0.8,
                source_image_id=test_image.id
            )
            db.session.add(test_infection)
            db.session.commit()
    
    def tearDown(self):
        """Tear down test environment"""
        with app.app_context():
            db.session.remove()
            db.drop_all()
    
    def test_get_infection_data(self):
        """Test get_infection_data API endpoint"""
        response = self.client.get('/api/infection_data')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('infections', data)
        self.assertEqual(len(data['infections']), 1)
        
    def test_get_models_list(self):
        """Test get_models_list API endpoint"""
        response = self.client.get('/api/models')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('models', data)
        self.assertIn('gcp_configured', data)
        
    def test_predict(self):
        """Test predict API endpoint"""
        data = {
            'infections': [
                {'lat': 3.1234, 'lng': 101.5678, 'level': 0.8}
            ],
            'days': 7
        }
        response = self.client.post('/api/predict', 
                                   data=json.dumps(data),
                                   content_type='application/json')
        result = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', result)
        self.assertIn('days', result)
        
    def test_model_info(self):
        """Test model_info API endpoint"""
        response = self.client.get('/api/model_info')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('models', data)
        
if __name__ == '__main__':
    unittest.main()