import unittest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_models import process_image, get_model_info
from main import app
from app import db
from models import ImageData

class MLModelsTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['UPLOAD_FOLDER'] = 'uploads'
        self.client = app.test_client()
        
        # Ensure the uploads folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        with app.app_context():
            db.create_all()
            
            # Add test data
            test_image = ImageData(
                filename="test_image.jpg",
                processed=False
            )
            db.session.add(test_image)
            db.session.commit()
            self.test_image_id = test_image.id
    
    def tearDown(self):
        """Tear down test environment"""
        with app.app_context():
            db.session.remove()
            db.drop_all()
    
    @patch('ml_models.ImageProcessor')
    def test_process_image(self, mock_processor):
        """Test process_image function"""
        # Mock the ImageProcessor
        mock_instance = MagicMock()
        mock_processor.return_value = mock_instance
        mock_instance.detect_infections.return_value = (True, [
            {'lat': 3.1234, 'lng': 101.5678, 'level': 0.8}
        ])
        
        # Create a test image file
        test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_image.jpg')
        with open(test_image_path, 'w') as f:
            f.write('test image data')
        
        # Call the function
        with app.app_context():
            result_path = process_image(test_image_path, self.test_image_id)
            
            # Check that the ImageProcessor was called correctly
            mock_processor.assert_called_once()
            mock_instance.detect_infections.assert_called_once_with(test_image_path, self.test_image_id)
            
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
    
    def test_get_model_info(self):
        """Test get_model_info function"""
        # Call the function
        result = get_model_info()
        
        # Check the result
        self.assertIsInstance(result, dict)
        self.assertIn('models', result)
        self.assertIn('total_models', result)
        
if __name__ == '__main__':
    unittest.main()