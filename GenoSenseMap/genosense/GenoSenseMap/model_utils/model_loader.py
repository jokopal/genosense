"""
Model Loader for Genosense Palm Oil Monitoring System

This module handles loading machine learning models from local storage or Google Cloud Storage.
It supports both Keras (.h5) and PyTorch (.pt) models.
"""

import os
import logging
from typing import Union, Optional, Dict, Any

import tensorflow as tf
from keras.models import load_model as load_keras_model
import torch
from google.cloud import storage

logger = logging.getLogger(__name__)

# Model cache to avoid reloading models
_MODEL_CACHE = {}

def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str,
                      credentials_path: Optional[str] = None) -> bool:
    """
    Downloads a model file from Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        source_blob_name: Path to the model file in the bucket
        destination_file_name: Local path to save the downloaded file
        credentials_path: Path to the GCP credentials JSON file (optional)
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        # Set credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        
        # Download the file
        blob.download_to_filename(destination_file_name)
        logger.info(f"Downloaded model from GCS: {source_blob_name} to {destination_file_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading model from GCS: {e}")
        return False

def load_model(model_path: str, model_type: str = 'keras', from_gcs: bool = False,
               gcs_bucket: Optional[str] = None, force_reload: bool = False,
               **kwargs) -> Union[tf.keras.Model, torch.nn.Module, None]:
    """
    Loads a machine learning model from file or downloads it from Google Cloud Storage.
    
    Args:
        model_path: Path to the model file or GCS blob path
        model_type: 'keras' or 'pytorch'
        from_gcs: Whether to download the model from GCS
        gcs_bucket: GCS bucket name (required if from_gcs is True)
        force_reload: Whether to force reload even if model is in cache
        **kwargs: Additional arguments to pass to the model loading function
        
    Returns:
        Loaded model or None if loading failed
    """
    # Check if model is already loaded and not forcing reload
    cache_key = f"{model_path}_{model_type}"
    if cache_key in _MODEL_CACHE and not force_reload:
        logger.info(f"Using cached model: {model_path}")
        return _MODEL_CACHE[cache_key]
    
    # Set up local path
    local_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')
    os.makedirs(local_models_dir, exist_ok=True)
    
    local_model_path = model_path
    
    # Download from GCS if needed
    if from_gcs:
        if not gcs_bucket:
            logger.error("GCS bucket name required when from_gcs is True")
            return None
        
        # Generate local path from GCS path
        filename = os.path.basename(model_path)
        local_model_path = os.path.join(local_models_dir, filename)
        
        # Download the model
        success = download_from_gcs(
            bucket_name=gcs_bucket,
            source_blob_name=model_path,
            destination_file_name=local_model_path,
            credentials_path=kwargs.get('credentials_path')
        )
        
        if not success:
            logger.error(f"Failed to download model from GCS: {model_path}")
            return None
    
    # Load the model based on type
    try:
        if model_type.lower() == 'keras':
            # Load Keras model
            model = load_keras_model(local_model_path, compile=kwargs.get('compile', True))
            logger.info(f"Successfully loaded Keras model: {local_model_path}")
        
        elif model_type.lower() == 'pytorch':
            # Load PyTorch model
            model = torch.load(local_model_path, map_location=kwargs.get('device', 'cpu'))
            if isinstance(model, Dict):  # Handle state_dict case
                if 'model_class' in kwargs and 'model_args' in kwargs:
                    model_class = kwargs['model_class']
                    model_instance = model_class(**kwargs['model_args'])
                    model_instance.load_state_dict(model['state_dict'] if 'state_dict' in model else model)
                    model = model_instance
            
            # Set to eval mode for inference
            if hasattr(model, 'eval'):
                model.eval()
            
            logger.info(f"Successfully loaded PyTorch model: {local_model_path}")
        
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
        
        # Cache the model
        _MODEL_CACHE[cache_key] = model
        return model
    
    except Exception as e:
        logger.error(f"Error loading model {local_model_path}: {e}")
        return None

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Returns information about all available models, both local and in GCS.
    
    Returns:
        Dictionary of model information
    """
    models_info = {
        'unet_multispectral': {
            'name': 'UNet Multispectral',
            'type': 'UNet',
            'description': 'UNet architecture for segmentation of multispectral imagery',
            'accuracy': 0.89,
            'last_updated': '2023-05-15',
            'active': True,
            'model_type': 'keras',
            'file_path': 'unet_multispectral.h5',
            'from_gcs': False
        },
        'ann_classifier': {
            'name': 'ANN Classifier',
            'type': 'ANN',
            'description': 'Artificial Neural Network for classification of palm oil infections',
            'accuracy': 0.92,
            'last_updated': '2023-06-10',
            'active': True,
            'model_type': 'pytorch',
            'file_path': 'ann_classifier.pt',
            'from_gcs': False
        }
    }
    
    return models_info

def clear_model_cache():
    """Clears the model cache to free memory"""
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    logger.info("Model cache cleared")