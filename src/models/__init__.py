"""Machine learning models for NBA player predictions."""

from .assist_prediction import AssistPredictionModel
from .predict_assists import predict_player_assists

__all__ = ["AssistPredictionModel", "predict_player_assists"]
