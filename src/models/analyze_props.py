"""Script to analyze real-time player props using the Odds API."""

import logging
from typing import Dict, Any, Set
from datetime import datetime, timezone
from src.models.prop_bet_analyzer import PropBetAnalyzer
from src.data.odds_api import OddsAPIClient
from src.data.sync_players import get_player_mapping, sync_players

logger = logging.getLogger(__name__)

def analyze_props() -> None:
    """Analyze current player props for betting edges."""
    try:
        # Sync player data
        sync_players()
        
        # Get player mapping from database
        player_ids = get_player_mapping()
        logger.info(f"Found {len(player_ids)} active players")
        
        # Initialize analyzer
        analyzer = PropBetAnalyzer(
            min_edge=0.05,
            kelly_fraction=0.25,
            confidence_threshold=0.6
        )
        
        # Find edges across all props
        edges = analyzer.find_edges(
            prop_types={'points', 'assists', 'rebounds'},
            player_ids=player_ids
        )
        
        # Print results
        print(f"\nFound {len(edges)} props with edges:")
        for edge in edges:
            print(f"\n{edge['player_name']} - {edge['prop_type']}")
            print(f"Line: {edge['line']}")
            print(f"Model prediction: {edge['model_prediction']:.1f}")
            print(f"Recent average: {edge['recent_average']:.1f}")
            print(f"Last 5 games: {edge['last_5_games']}")
            
            # Print over/under analysis
            for side in ['over', 'under']:
                if edge[side]['edge'] >= analyzer.min_edge:
                    print(f"\n{side.upper()}:")
                    print(f"Edge: {edge[side]['edge']:.1%}")
                    print(f"EV: ${edge[side]['ev_dollars']:.2f} ({edge[side]['ev_percent']:.1%})")
                    print(f"Kelly bet: ${edge[side]['kelly_bet']:.2f}")
                    print(f"Best odds: {edge[side]['best_odds']}")
        
    except Exception as e:
        logger.error(f"Error analyzing props: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    analyze_props()
