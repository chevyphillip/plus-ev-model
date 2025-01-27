"""Model management system for loading and saving trained models."""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from typing import Dict, Any, Optional, Union
from src.models.enhanced_props_model import EnhancedPropsModel
from src.models.h2o_props_model import H2OPropsModel

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model persistence, loading, and versioning."""
    
    def __init__(
        self,
        models_dir: str = 'models',
        max_age_days: int = 7,
        use_h2o: bool = True
    ) -> None:
        """Initialize model manager.
        
        Args:
            models_dir: Directory to store models
            max_age_days: Maximum age of models before retraining
            use_h2o: Whether to use H2O AutoML models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self.use_h2o = use_h2o
        
    def _get_model_path(self, prop_type: str) -> Path:
        """Get path for model files.
        
        Args:
            prop_type: Type of prop model
            
        Returns:
            Path to model directory
        """
        return self.models_dir / prop_type
        
    def _get_metadata_path(self, prop_type: str) -> Path:
        """Get path for model metadata.
        
        Args:
            prop_type: Type of prop model
            
        Returns:
            Path to metadata file
        """
        return self._get_model_path(prop_type) / 'metadata.json'
        
    def _save_metadata(
        self,
        prop_type: str,
        metrics: Dict[str, float],
        feature_names: list,
        model_type: str
    ) -> None:
        """Save model metadata.
        
        Args:
            prop_type: Type of prop model
            metrics: Model performance metrics
            feature_names: List of feature names
            model_type: Type of model (enhanced or h2o)
        """
        metadata = {
            'prop_type': prop_type,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'feature_names': feature_names,
            'model_type': model_type
        }
        
        metadata_path = self._get_metadata_path(prop_type)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _load_metadata(self, prop_type: str) -> Optional[Dict[str, Any]]:
        """Load model metadata if it exists.
        
        Args:
            prop_type: Type of prop model
            
        Returns:
            Metadata dictionary if found, None otherwise
        """
        metadata_path = self._get_metadata_path(prop_type)
        if not metadata_path.exists():
            return None
            
        with open(metadata_path) as f:
            return json.load(f)
            
    def _is_model_valid(self, metadata: Dict[str, Any]) -> bool:
        """Check if model is still valid based on age.
        
        Args:
            metadata: Model metadata
            
        Returns:
            True if model is still valid
        """
        trained_at = datetime.fromisoformat(metadata['trained_at'])
        age = datetime.now() - trained_at
        return age.days <= self.max_age_days
        
    def save_model(
        self,
        model: Union[EnhancedPropsModel, H2OPropsModel],
        metrics: Dict[str, float]
    ) -> None:
        """Save model and metadata.
        
        Args:
            model: Trained model to save
            metrics: Model performance metrics
        """
        model_dir = self._get_model_path(model.prop_type)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'model.joblib'
        joblib.dump(model, model_path)
        
        # Save metadata
        model_type = 'h2o' if isinstance(model, H2OPropsModel) else 'enhanced'
        self._save_metadata(
            model.prop_type,
            metrics,
            model.feature_names,
            model_type
        )
        
        logger.info(f"Saved {model.prop_type} model to {model_dir}")
        
    def load_model(self, prop_type: str) -> Optional[Union[EnhancedPropsModel, H2OPropsModel]]:
        """Load model if valid version exists.
        
        Args:
            prop_type: Type of prop model
            
        Returns:
            Loaded model if valid version exists, None otherwise
        """
        # Check metadata first
        metadata = self._load_metadata(prop_type)
        if not metadata or not self._is_model_valid(metadata):
            return None
            
        # Load model
        model_path = self._get_model_path(prop_type) / 'model.joblib'
        if not model_path.exists():
            return None
            
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded {prop_type} model trained at {metadata['trained_at']}")
            return model
        except Exception as e:
            logger.error(f"Error loading {prop_type} model: {str(e)}")
            return None
            
    def get_model(
        self,
        prop_type: str,
        force_retrain: bool = False
    ) -> Union[EnhancedPropsModel, H2OPropsModel]:
        """Get model, training new one if needed.
        
        Args:
            prop_type: Type of prop model
            force_retrain: Force model retraining
            
        Returns:
            Trained model
        """
        if not force_retrain:
            model = self.load_model(prop_type)
            if model is not None:
                return model
                
        # Train new model
        logger.info(f"Training new {prop_type} model...")
        if self.use_h2o:
            model = H2OPropsModel(prop_type=prop_type)
        else:
            model = EnhancedPropsModel(prop_type=prop_type)
            
        metrics = model.train()
        
        # Save model
        self.save_model(model, metrics)
        
        return model
