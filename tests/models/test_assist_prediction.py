import pytest
import pandas as pd
import numpy as np
from src.models.assist_prediction import AssistPredictionModel
from unittest.mock import patch, MagicMock

@pytest.fixture
def sample_player_data():
    return pd.DataFrame({
        'player_id': [1] * 10,
        'player_name': ['Test Player'] * 10,
        'season': ['2023-24'] * 10,
        'ast': [6, 4, 7, 3, 8, 5, 4, 6, 7, 5],
        'pts': [20, 18, 22, 15, 25, 19, 17, 21, 23, 20],
        'min': [32, 30, 35, 28, 36, 31, 29, 33, 34, 32],
        'fg_pct': [0.45, 0.42, 0.48, 0.40, 0.50, 0.44, 0.41, 0.46, 0.49, 0.45],
        'plus_minus': [5, 2, 8, -2, 10, 4, 1, 6, 9, 5]
    })

@pytest.fixture
def model():
    return AssistPredictionModel(db_path=":memory:")

def test_prepare_features(model, sample_player_data):
    X, y = model.prepare_features(sample_player_data)
    
    # Check if features are correct
    assert list(X.columns) == [
        'ast_rolling_5',
        'pts_rolling_5',
        'min_rolling_5',
        'fg_pct_rolling_5',
        'plus_minus_rolling_5'
    ]
    
    # Check if target is binary
    assert set(y.unique()) == {True, False}
    
    # Check if rolling averages are calculated correctly
    assert len(X) == len(sample_player_data) - 1  # -1 because we can't predict the last game

def test_train_model(model):
    with patch('src.models.assist_prediction.AssistPredictionModel._get_player_recent_games') as mock_get_data:
        # Create mock data
        mock_data = pd.DataFrame({
            'player_id': [1] * 100,
            'player_name': ['Test Player'] * 100,
            'season': ['2023-24'] * 100,
            'ast': np.random.randint(0, 10, 100),
            'pts': np.random.randint(10, 30, 100),
            'min': np.random.randint(20, 40, 100),
            'fg_pct': np.random.uniform(0.3, 0.6, 100),
            'plus_minus': np.random.randint(-10, 11, 100)
        })
        mock_get_data.return_value = mock_data
        
        # Train model
        metrics = model.train()
        
        # Check if metrics are calculated
        assert set(metrics.keys()) == {
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc'
        }
        
        # Check if metrics are in valid range [0, 1]
        assert all(0 <= v <= 1 for v in metrics.values())

def test_predict_player(model, sample_player_data):
    with patch('src.models.assist_prediction.AssistPredictionModel._get_player_recent_games') as mock_get_data:
        mock_get_data.return_value = sample_player_data
        
        # Train model first
        model.train()
        
        # Make prediction
        prediction = model.predict_player(1)
        
        # Check prediction structure
        assert set(prediction.keys()) == {
            'player_name',
            'probability_5plus_assists',
            'prediction',
            'recent_assist_avg'
        }
        
        # Check value types
        assert isinstance(prediction['player_name'], str)
        assert isinstance(prediction['probability_5plus_assists'], float)
        assert isinstance(prediction['prediction'], bool)
        assert isinstance(prediction['recent_assist_avg'], float)
        
        # Check probability is in valid range [0, 1]
        assert 0 <= prediction['probability_5plus_assists'] <= 1

def test_invalid_player_id(model):
    with patch('src.models.assist_prediction.AssistPredictionModel._get_player_recent_games') as mock_get_data:
        mock_get_data.return_value = pd.DataFrame()  # Empty DataFrame
        
        with pytest.raises(ValueError, match="No data found for player_id"):
            model.predict_player(999999)  # Non-existent player ID
