"""
Image Processor for Genosense Palm Oil Monitoring System

This module handles processing uploaded images for G. boninense detection,
using Keras or PyTorch models. It supports multispectral image analysis.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
import tensorflow as tf
import torch
from datetime import datetime

from ml_utils.model_loader import load_model
from app import db
from models import ImageData, InfectionData

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processor for palm oil infection detection"""
    
    def __init__(self, model_id: str = 'unet_multispectral', use_gpu: bool = False):
        """
        Initialize the image processor
        
        Args:
            model_id: ID of the model to use (from model_loader.get_available_models())
            use_gpu: Whether to use GPU for inference if available
        """
        from ml_utils.model_loader import get_available_models
        
        self.models_info = get_available_models()
        if model_id not in self.models_info:
            raise ValueError(f"Model ID '{model_id}' not found in available models")
        
        self.model_info = self.models_info[model_id]
        self.model_id = model_id
        self.model = None
        self.use_gpu = use_gpu
        
        # Set device for PyTorch models
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the specified model
        
        Args:
            force_reload: Whether to force reload the model even if already loaded
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        from ml_utils.model_loader import load_model as load_ml_model
        
        try:
            model_info = self.model_info
            
            # Prepare loading parameters
            load_params = {
                'model_path': model_info['file_path'],
                'model_type': model_info['model_type'],
                'from_gcs': model_info.get('from_gcs', False),
                'force_reload': force_reload
            }
            
            # Add GCS parameters if needed
            if model_info.get('from_gcs', False):
                load_params['gcs_bucket'] = model_info.get('gcs_bucket')
            
            # Add device for PyTorch models
            if model_info['model_type'] == 'pytorch':
                load_params['device'] = self.device
            
            # Load the model
            self.model = load_ml_model(**load_params)
            
            return self.model is not None
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image for model inference
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Determine if this is multispectral or RGB
            is_multispectral = len(np.array(img).shape) > 2 and np.array(img).shape[2] > 3
            
            # Resize image to model input size (assumed 256x256 for example)
            target_size = (256, 256)
            img = img.resize(target_size)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype('float32') / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def detect_infections(self, image_path: str, image_id: int) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect G. boninense infections in an image and save results to database
        
        Args:
            image_path: Path to the uploaded image
            image_id: Database ID of the image entry
            
        Returns:
            Tuple of (success, infection_data)
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                if not self.load_model():
                    logger.error("Failed to load model for infection detection")
                    return False, []
            
            # Preprocess the image
            preprocessed_img = self.preprocess_image(image_path)
            
            # Run inference based on model type
            if self.model_info['model_type'] == 'keras':
                # For Keras/TensorFlow models
                predictions = self.model.predict(preprocessed_img)
            else:
                # For PyTorch models
                with torch.no_grad():
                    input_tensor = torch.from_numpy(preprocessed_img).to(self.device)
                    if input_tensor.shape[1] == 3:  # Handle channel first vs channel last
                        input_tensor = input_tensor.permute(0, 3, 1, 2)
                    output = self.model(input_tensor)
                    predictions = output.cpu().numpy()
            
            # Process results (this will depend on your model output format)
            # For a segmentation model like UNet:
            if self.model_info['type'] == 'UNet':
                # Threshold predictions to create binary mask
                threshold = 0.5
                binary_mask = (predictions > threshold).astype(np.uint8)
                
                # Find infection regions
                infection_regions = self._extract_infection_regions(binary_mask[0], image_path)
                
                # Create annotations and save results
                result_path = self._create_annotated_image(image_path, binary_mask[0])
                
                # Save infection data to database
                if len(infection_regions) > 0:
                    infection_data = self._save_infections_to_db(infection_regions, image_id)
                    
                    # Update image record with result path
                    self._update_image_record(image_id, result_path)
                    
                    return True, infection_data
                else:
                    # No infections found
                    self._update_image_record(image_id, result_path, found_infections=False)
                    return True, []
                    
            # For a classifier model:
            elif self.model_info['type'] == 'ANN':
                # For classification, create a simulated infection point at the center
                # In a real implementation, this would come from georeferenced data
                
                # Get highest probability class
                class_probabilities = predictions[0]
                infection_level = float(np.max(class_probabilities))
                
                # Example: extract location from image metadata (simulated)
                lat, lng = self._extract_geolocation(image_path)
                
                if lat is not None and lng is not None:
                    # Save infection data to database
                    infection = InfectionData(
                        latitude=lat,
                        longitude=lng,
                        infection_level=infection_level,
                        source_image_id=image_id
                    )
                    db.session.add(infection)
                    db.session.commit()
                    
                    # Update image record
                    self._update_image_record(image_id, image_path)
                    
                    return True, [{'lat': lat, 'lng': lng, 'level': infection_level}]
                else:
                    # No location data
                    self._update_image_record(image_id, image_path, found_infections=False)
                    return True, []
            
            else:
                logger.error(f"Unsupported model type for inference: {self.model_info['type']}")
                return False, []
                
        except Exception as e:
            logger.error(f"Error detecting infections: {e}")
            return False, []

    def _extract_infection_regions(self, mask: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract infection regions from a binary mask
        
        Args:
            mask: Binary mask from model prediction
            image_path: Path to original image for geolocation
            
        Returns:
            List of dictionaries with infection data (lat, lng, level)
        """
        # This is a simplified implementation
        # In a real system, this would use connected component analysis and georeference the results
        
        # Get center coordinates and extract geolocation
        base_lat, base_lng = self._extract_geolocation(image_path)
        
        if base_lat is None or base_lng is None:
            # If no geolocation, use simulated values
            base_lat = np.random.uniform(-3, 3)  # Indonesia region
            base_lng = np.random.uniform(100, 115)
        
        # Find areas with high infection probability
        # Here we're simplifying by just using random points based on proportion of infection pixels
        infection_pixels = np.sum(mask)
        total_pixels = mask.size
        infection_ratio = infection_pixels / total_pixels
        
        # Number of infection points to generate
        num_points = max(1, min(7, int(infection_ratio * 20)))
        
        # Generate points
        infection_regions = []
        for _ in range(num_points):
            # Create points within ~2km of base point
            lat_offset = (np.random.random() - 0.5) * 0.04
            lng_offset = (np.random.random() - 0.5) * 0.04
            
            # Infection level from 0.3 to 1.0 (not starting at 0 since we detected an infection)
            infection_level = 0.3 + (0.7 * np.random.random())
            
            infection_regions.append({
                'lat': base_lat + lat_offset,
                'lng': base_lng + lng_offset,
                'level': infection_level
            })
        
        return infection_regions

    def _create_annotated_image(self, image_path: str, mask: np.ndarray) -> str:
        """
        Create an annotated version of the image with infection overlays
        
        Args:
            image_path: Path to the original image
            mask: Binary mask of infections
            
        Returns:
            Path to the annotated image
        """
        try:
            # Load original image
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB for overlay
            img_array = np.array(img)
            
            # Resize mask to match image if needed
            if img_array.shape[:2] != mask.shape:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]))
                mask = np.array(mask_img) / 255
            
            # Create a red overlay for infections
            overlay = np.zeros_like(img_array)
            overlay[..., 0] = 255  # Red channel
            
            # Apply overlay with alpha blending
            alpha = 0.5
            for c in range(3):  # RGB channels
                if c == 0:  # Red channel
                    img_array[..., c] = img_array[..., c] * (1 - alpha * mask) + overlay[..., c] * (alpha * mask)
                else:
                    img_array[..., c] = img_array[..., c] * (1 - alpha * mask)
            
            # Save the annotated image
            annotated_path = image_path.replace('.', '_annotated.')
            Image.fromarray(img_array.astype(np.uint8)).save(annotated_path)
            
            return annotated_path
        
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            return image_path  # Return original path on error

    def _save_infections_to_db(self, infection_regions: List[Dict[str, Any]], image_id: int) -> List[Dict[str, Any]]:
        """
        Save infection data to database
        
        Args:
            infection_regions: List of dictionaries with infection data
            image_id: ID of the source image
            
        Returns:
            List of saved infection data
        """
        infection_data = []
        
        for region in infection_regions:
            # Create database record
            infection = InfectionData(
                latitude=region['lat'],
                longitude=region['lng'],
                infection_level=region['level'],
                source_image_id=image_id
            )
            
            db.session.add(infection)
            infection_data.append({
                'lat': region['lat'],
                'lng': region['lng'],
                'level': region['level']
            })
        
        db.session.commit()
        return infection_data

    def _update_image_record(self, image_id: int, result_path: str, found_infections: bool = True) -> None:
        """
        Update the image record with processing results
        
        Args:
            image_id: ID of the image record
            result_path: Path to the result/annotated image
            found_infections: Whether infections were found
        """
        try:
            image = ImageData.query.get(image_id)
            if image:
                image.processed = True
                image.result_path = result_path
                image.process_date = datetime.utcnow()
                db.session.commit()
        
        except Exception as e:
            logger.error(f"Error updating image record: {e}")

    def _extract_geolocation(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract geolocation data from image metadata
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not available
        """
        try:
            # In a real implementation, this would use EXIF or other metadata
            # For this example, we'll return None to trigger simulated values
            return None, None
        
        except Exception:
            return None, None