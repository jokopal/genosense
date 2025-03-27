"""
Google Cloud Platform configuration for Genosense

This module handles GCP configuration and authentication.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# GCP configuration
GCP_CONFIG = {
    'project_id': os.environ.get('GCP_PROJECT_ID'),
    'model_bucket': os.environ.get('GCP_MODEL_BUCKET', 'genosense-models'),
    'credentials_path': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
}

def get_gcp_config() -> Dict[str, Any]:
    """
    Get the GCP configuration
    
    Returns:
        Dictionary with GCP configuration
    """
    return GCP_CONFIG

def is_gcp_configured() -> bool:
    """
    Check if GCP is properly configured
    
    Returns:
        True if GCP is configured, False otherwise
    """
    return bool(GCP_CONFIG['project_id'] and GCP_CONFIG['credentials_path'] and
                os.path.exists(GCP_CONFIG['credentials_path']))

def get_available_gcp_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about models available in GCP storage
    
    Returns:
        Dictionary of model information for models in GCP
    """
    if not is_gcp_configured():
        logger.warning("GCP not properly configured, cannot retrieve model list")
        return {}
    
    # In a real implementation, this would query the GCP bucket
    # For now, return static model information
    return {
        'unet_multispectral_gcp': {
            'name': 'UNet Multispectral (GCP)',
            'type': 'UNet',
            'description': 'UNet architecture for segmentation of multispectral imagery (GCP)',
            'accuracy': 0.91,
            'last_updated': '2023-08-20',
            'active': True,
            'model_type': 'keras',
            'file_path': 'models/unet_multispectral_v2.h5',
            'from_gcs': True,
            'gcs_bucket': GCP_CONFIG['model_bucket']
        },
        'ann_classifier_gcp': {
            'name': 'ANN Classifier (GCP)',
            'type': 'ANN',
            'description': 'Improved ANN for classification of palm oil infections (GCP)',
            'accuracy': 0.94,
            'last_updated': '2023-09-15',
            'active': True,
            'model_type': 'pytorch',
            'file_path': 'models/ann_classifier_v2.pt',
            'from_gcs': True,
            'gcs_bucket': GCP_CONFIG['model_bucket']
        }
    }

def setup_gcp_credentials(credentials_json: Dict[str, Any]) -> bool:
    """
    Set up GCP credentials from JSON
    
    Args:
        credentials_json: GCP credentials JSON
        
    Returns:
        True if credentials were set up successfully, False otherwise
    """
    try:
        # Create credentials directory if it doesn't exist
        credentials_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials')
        os.makedirs(credentials_dir, exist_ok=True)
        
        # Write credentials to file
        credentials_path = os.path.join(credentials_dir, 'gcp_credentials.json')
        with open(credentials_path, 'w') as f:
            json.dump(credentials_json, f)
        
        # Update environment variable and config
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        GCP_CONFIG['credentials_path'] = credentials_path
        
        logger.info("GCP credentials set up successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up GCP credentials: {e}")
        return False