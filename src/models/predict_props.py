"""Example script demonstrating player props prediction and edge calculation."""

import logging
from typing import Dict, Any, List
from src.models.player_props_model import PlayerPropsModel
from src.core.edge_calculator import EdgeCalculator, BettingLine

logger = logging.getLogger(__name__)

def analyze_player_prop(
    player_id: int,
    prop_type: str,
    line: float,
    over_odds: int = -110,
    under_odds: int = -110,
    db_path: str = 'data/nba_stats.duckdb'
) -> Dict[str, Any]:
    """Analyze a player prop bet.
    
    Args:
        player_id: NBA player ID
        prop_type: Type of prop (points, assists, rebounds)
        line: Prop line (e.g. 22.5)
        over_odds: American odds for over
        under_odds: American odds for under
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary containing analysis and recommendation
    """
    try:
        # Initialize model and calculator
        model = PlayerPropsModel(db_path=db_path, prop_type=prop_type)
        calculator = EdgeCalculator(
            min_edge=0.05,
            kelly_fraction=0.5,
            confidence_threshold=0.6
        )
        
        # Get model prediction
        prediction = model.predict_player(player_id)
        
        # Create betting line object
        betting_line = BettingLine(
            over=line,
            over_odds=over_odds,
            under=line,
            under_odds=under_odds
        )
        
        # Analyze the bet
        analysis = calculator.analyze_prop_bet(prediction, betting_line)
        
        # Get recommendation
        recommendation = calculator.get_bet_recommendation(analysis)
        
        return {
            'analysis': analysis,
            'recommendation': recommendation
        }
        
    except Exception as e:
        logger.error(f"Error analyzing prop bet: {str(e)}")
        raise

def analyze_multiple_props(
    props: List[Dict[str, Any]],
    db_path: str = 'data/nba_stats.duckdb'
) -> List[Dict[str, Any]]:
    """Analyze multiple player props.
    
    Args:
        props: List of dictionaries containing:
            - player_id: NBA player ID
            - prop_type: Type of prop
            - line: Prop line
            - over_odds: Odds for over (optional)
            - under_odds: Odds for under (optional)
        db_path: Path to DuckDB database
        
    Returns:
        List of analysis results
    """
    results = []
    
    for prop in props:
        try:
            result = analyze_player_prop(
                player_id=prop['player_id'],
                prop_type=prop['prop_type'],
                line=prop['line'],
                over_odds=prop.get('over_odds', -110),
                under_odds=prop.get('under_odds', -110),
                db_path=db_path
            )
            results.append(result)
            
        except Exception as e:
            logger.error(
                f"Error analyzing prop for player {prop['player_id']}: {str(e)}"
            )
            continue
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    example_props = [
        {
            'player_id': 2544,  # LeBron James
            'prop_type': 'points',
            'line': 25.5,
            'over_odds': -110,
            'under_odds': -110
        },
        {
            'player_id': 2544,  # LeBron James
            'prop_type': 'assists',
            'line': 7.5,
            'over_odds': -115,
            'under_odds': -105
        }
    ]
    
    results = analyze_multiple_props(example_props)
    
    # Print results
    for result in results:
        analysis = result['analysis']
        recommendation = result['recommendation']
        
        print(f"\nAnalysis for {analysis['player_name']} {analysis['prop_type']}:")
        print(f"Line: {analysis['line']}")
        print(f"Predicted Value: {analysis['predicted_value']:.1f}")
        print(f"Recent Average: {analysis['recent_average']:.1f}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        
        if recommendation:
            print("\nRecommended Bet:")
            print(f"Type: {recommendation['bet_type'].upper()}")
            print(f"Edge: {recommendation['edge']:.1%}")
            print(f"Kelly Bet: {recommendation['kelly_bet']:.1%}")
        else:
            print("\nNo bet recommended")
