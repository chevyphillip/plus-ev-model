"""Test script for player props predictions."""

import logging
from typing import Dict, Any, List
from src.models.player_props_model import PlayerPropsModel
from src.core.edge_calculator import EdgeCalculator, BettingLine

logger = logging.getLogger(__name__)

def test_lebron_predictions() -> None:
    """Test predictions for LeBron James props."""
    try:
        # Initialize models
        points_model = PlayerPropsModel(prop_type='points')
        assists_model = PlayerPropsModel(prop_type='assists')
        rebounds_model = PlayerPropsModel(prop_type='rebounds')
        
        calculator = EdgeCalculator(
            min_edge=0.05,
            kelly_fraction=0.5,
            confidence_threshold=0.6
        )
        
        # LeBron's player ID
        lebron_id = 2544
        
        # Test points prediction
        points_pred = points_model.predict_player(lebron_id)
        points_line = BettingLine(
            over=25.5,
            over_odds=-110,
            under=25.5,
            under_odds=-110
        )
        points_analysis = calculator.analyze_prop_bet(points_pred, points_line)
        
        # Test assists prediction
        assists_pred = assists_model.predict_player(lebron_id)
        assists_line = BettingLine(
            over=7.5,
            over_odds=-110,
            under=7.5,
            under_odds=-110
        )
        assists_analysis = calculator.analyze_prop_bet(assists_pred, assists_line)
        
        # Test rebounds prediction
        rebounds_pred = rebounds_model.predict_player(lebron_id)
        rebounds_line = BettingLine(
            over=7.5,
            over_odds=-110,
            under=7.5,
            under_odds=-110
        )
        rebounds_analysis = calculator.analyze_prop_bet(rebounds_pred, rebounds_line)
        
        # Print results
        print("\nLeBron James Predictions:")
        
        print("\nPoints:")
        print(f"Predicted: {points_pred['predicted_value']:.1f}")
        print(f"Recent Average: {points_pred['recent_average']:.1f}")
        print(f"Last 5 Games: {points_pred['last_5_games']}")
        if points_analysis['confidence'] >= calculator.confidence_threshold:
            print(f"Confidence: {points_analysis['confidence']:.2f}")
            if points_analysis['over']['edge']:
