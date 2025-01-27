"""Tests for extended ensemble model."""

import pytest
import numpy as np
from src.models.extended_ensemble_model import ExtendedEnsembleModel

def test_model_initialization():
    """Test model initialization."""
    model = ExtendedEnsembleModel(prop_type='points')
    assert model.prop_type == 'points'
    assert len(model.models) == 4
    assert 'rf' in model.models
    assert 'gb' in model.models
    assert 'xgb' in model.models
    assert 'lasso' in model.models

def test_invalid_prop_type():
    """Test initialization with invalid prop type."""
    with pytest.raises(ValueError):
        ExtendedEnsembleModel(prop_type='invalid_type')

def test_range_thresholds():
    """Test range threshold calculation."""
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model = ExtendedEnsembleModel(prop_type='points')
    thresholds = model._determine_range_thresholds(y)
    assert len(thresholds) == 4
    assert all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1))

def test_range_label():
    """Test range label assignment."""
    model = ExtendedEnsembleModel(prop_type='points')
    model.range_thresholds = [10, 20, 30, 40]
    
    assert model._get_range_label(5) == 'range_0'
    assert model._get_range_label(15) == 'range_1'
    assert model._get_range_label(25) == 'range_2'
    assert model._get_range_label(35) == 'range_3'
    assert model._get_range_label(45) == 'range_4'

def test_weight_optimization():
    """Test ensemble weight optimization."""
    model = ExtendedEnsembleModel(prop_type='points')
    
    # Mock predictions
    predictions = {
        'rf': np.array([1, 2, 3]),
        'gb': np.array([1.1, 2.1, 3.1]),
        'xgb': np.array([0.9, 1.9, 2.9]),
        'lasso': np.array([1.2, 2.2, 3.2])
    }
    y_true = np.array([1, 2, 3])
    
    weights = model._optimize_weights(predictions, y_true)
    
    assert len(weights) == 4
    assert all(0 <= w <= 1 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6

def test_prediction_with_confidence():
    """Test prediction with confidence score."""
    model = ExtendedEnsembleModel(prop_type='points')
    
    # Mock data for testing
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    y = np.random.rand(10)
    
    # Train on mock data
    model.scaler.fit(X)
    for m in model.models.values():
        m.fit(X, y)
    
    # Make prediction
    features = np.random.rand(1, 5)
    result = model.predict(features, return_confidence=True)
    
    assert 'prediction' in result
    assert 'confidence' in result
    assert isinstance(result['prediction'], float)
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1

def test_prediction_without_confidence():
    """Test prediction without confidence score."""
    model = ExtendedEnsembleModel(prop_type='points')
    
    # Mock data for testing
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    y = np.random.rand(10)
    
    # Train on mock data
    model.scaler.fit(X)
    for m in model.models.values():
        m.fit(X, y)
    
    # Make prediction
    features = np.random.rand(1, 5)
    result = model.predict(features, return_confidence=False)
    
    assert 'prediction' in result
    assert result['confidence'] is None
    assert isinstance(result['prediction'], float)

if __name__ == '__main__':
    pytest.main([__file__])
