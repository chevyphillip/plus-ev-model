import logging
from typing import Dict, List
from src.models.assist_prediction import AssistPredictionModel

logger = logging.getLogger(__name__)

def predict_player_assists(player_ids: List[int], db_path: str = 'data/nba_stats.duckdb') -> List[Dict]:
    """
    Predict 5+ assists probability for a list of players
    
    Args:
        player_ids: List of player IDs to predict for
        db_path: Path to the DuckDB database
        
    Returns:
        List of predictions with player details and probabilities
    """
    # Initialize and train model
    model = AssistPredictionModel(db_path)
    model.train()
    
    # Make predictions for each player
    predictions = []
    for player_id in player_ids:
        try:
            prediction = model.predict_player(player_id)
            predictions.append(prediction)
            logger.info(f"Prediction for {prediction['player_name']}: "
                       f"{prediction['probability_5plus_assists']:.1%} chance of 5+ assists "
                       f"(Recent avg: {prediction['recent_assist_avg']:.1f})")
        except ValueError as e:
            logger.warning(f"Could not predict for player_id {player_id}: {str(e)}")
            continue
    
    return predictions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Predict for top NBA playmakers
    player_ids = [
        1629027,  # Trae Young
        1630169,  # Tyrese Haliburton
        201935,   # James Harden
        203999,   # Nikola Jokić
        101108    # Chris Paul
    ]
    
    predictions = predict_player_assists(player_ids)
    
    print("\nPredictions Summary:")
    print("-------------------")
    for pred in predictions:
        print(f"{pred['player_name']}:")
        print(f"  Probability of 5+ assists: {pred['probability_5plus_assists']:.1%}")
        print(f"  Recent assist average: {pred['recent_assist_avg']:.1f}")
        print(f"  Prediction: {'OVER' if pred['prediction'] else 'UNDER'} 5 assists")
        print()
