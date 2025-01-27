                                                                                x3 cf cxbvnd ÎÏvarsx ≈≈¥∂"""Analyzer for finding edges in player props using model predictions and real-time odds."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from src.models.player_props_model import PlayerPropsModel
from src.data.odds_api import OddsAPIClient, PropMarket
from src.core.devig import DevigCalculator, get_kelly_bet
from src.core.edge_calculator import EdgeCalculator, BettingLine

logger = logging.getLogger(__name__)

class PropBetAnalyzer:
    """Analyzes player props for betting edges."""
    
    def __init__(
        self,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25,
        confidence_threshold: float = 0.6,
        bankroll: float = 10000,
        db_path: str = 'data/nba_stats.duckdb'
    ) -> None:
        """Initialize the analyzer.
        
        Args:
            min_edge: Minimum edge required for bet recommendation
            kelly_fraction: Fraction of Kelly criterion to use
            confidence_threshold: Minimum model confidence required
            bankroll: Bankroll for Kelly bet sizing
            db_path: Path to DuckDB database
        """
        # Initialize components
        self.odds_client = OddsAPIClient()
        self.devig_calc = DevigCalculator()
        self.edge_calc = EdgeCalculator(
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize models
        self.points_model = PlayerPropsModel(db_path=db_path, prop_type='points')
        self.assists_model = PlayerPropsModel(db_path=db_path, prop_type='assists')
        self.rebounds_model = PlayerPropsModel(db_path=db_path, prop_type='rebounds')
        
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
    
    def _get_model_for_prop(self, prop_type: str) -> PlayerPropsModel:
        """Get the appropriate model for a prop type."""
        if prop_type == 'points':
            return self.points_model
        elif prop_type == 'assists':
            return self.assists_model
        elif prop_type == 'rebounds':
            return self.rebounds_model
        else:
            raise ValueError(f"Unsupported prop type: {prop_type}")
    
    def _calculate_model_probabilities(
        self,
        prediction: Dict[str, Any],
        line: float
    ) -> Dict[str, float]:
        """Calculate over/under probabilities from model prediction.
        
        Args:
            prediction: Model prediction dictionary
            line: Prop line (e.g. 22.5 points)
            
        Returns:
            Dictionary with over/under probabilities
        """
        pred_value = prediction['predicted_value']
        recent_std = float(np.std(prediction['last_5_games']))
        
        # Use normal distribution to estimate probabilities
        over_prob = 1 - float(
            np.exp(-0.5 * ((line - pred_value) / recent_std) ** 2)
            / (recent_std * np.sqrt(2 * np.pi))
        )
        under_prob = 1 - over_prob
        
        return {
            'over_probability': over_prob,
            'under_probability': under_prob
        }
    
    def analyze_prop(
        self,
        market: PropMarket,
        player_id: int
    ) -> Dict[str, Any]:
        """Analyze a single prop market for potential edges.
        
        Args:
            market: Prop market from OddsAPI
            player_id: NBA player ID
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Get model prediction
            model = self._get_model_for_prop(market.prop_type)
            prediction = model.predict_player(player_id)
            
            # Calculate true probabilities from sharp books
            sharp_probs = self.devig_calc.calculate_true_probability(market)
            
            # Calculate model probabilities
            model_probs = self._calculate_model_probabilities(
                prediction,
                market.line
            )
            
            # Get best available odds
            best_over_odds = min(
                (book.over_odds for book in market.sharp_odds),
                key=lambda x: self.devig_calc.get_raw_probabilities(x)
            )
            best_under_odds = min(
                (book.under_odds for book in market.sharp_odds),
                key=lambda x: self.devig_calc.get_raw_probabilities(x)
            )
            
            # Calculate edges
            over_edge = model_probs['over_probability'] - sharp_probs['over_probability']
            under_edge = model_probs['under_probability'] - sharp_probs['under_probability']
            
            # Calculate EV
            over_ev = self.devig_calc.calculate_ev(
                model_probs['over_probability'],
                best_over_odds
            )
            under_ev = self.devig_calc.calculate_ev(
                model_probs['under_probability'],
                best_under_odds
            )
            
            # Calculate Kelly bets
            over_kelly = get_kelly_bet(
                self.bankroll,
                model_probs['over_probability'],
                best_over_odds,
                self.kelly_fraction
            ) if over_edge >= self.min_edge else 0
            
            under_kelly = get_kelly_bet(
                self.bankroll,
                model_probs['under_probability'],
                best_under_odds,
                self.kelly_fraction
            ) if under_edge >= self.min_edge else 0
            
            return {
                'player_name': market.player_name,
                'team': market.team,
                'prop_type': market.prop_type,
                'line': market.line,
                'timestamp': datetime.utcnow().isoformat(),
                'model_prediction': prediction['predicted_value'],
                'recent_average': prediction['recent_average'],
                'last_5_games': prediction['last_5_games'],
                'over': {
                    'model_probability': model_probs['over_probability'],
                    'sharp_probability': sharp_probs['over_probability'],
                    'best_odds': best_over_odds,
                    'edge': over_edge,
                    'ev_dollars': over_ev['ev_dollars'],
                    'ev_percent': over_ev['ev_percent'],
                    'kelly_bet': over_kelly
                },
                'under': {
                    'model_probability': model_probs['under_probability'],
                    'sharp_probability': sharp_probs['under_probability'],
                    'best_odds': best_under_odds,
                    'edge': under_edge,
                    'ev_dollars': under_ev['ev_dollars'],
                    'ev_percent': under_ev['ev_percent'],
                    'kelly_bet': under_kelly
                }
            }
            
        except Exception as e:
            logger.error(
                f"Error analyzing {market.prop_type} prop for {market.player_name}: {str(e)}"
            )
            return {}  # Return empty dict instead of None for type safety
    
    def find_edges(
        self,
        prop_types: Optional[Set[str]] = None,
        player_ids: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """Find edges across all available player props.
        
        Args:
            prop_types: Set of prop types to analyze
            player_ids: Dictionary mapping player names to IDs
            
        Returns:
            List of analysis results for props with edges
        """
        if not prop_types:
            prop_types = {'points', 'assists', 'rebounds'}
            
        edges = []
        
        try:
            # Fetch available props
            markets = self.odds_client.get_player_props(prop_types=prop_types)
            logger.info(f"Found {len(markets)} available props")
            
            # Analyze each market
            for market in markets:
                # Skip if we don't have player ID
                if not player_ids or market.player_name not in player_ids:
                    logger.warning(f"No player ID found for {market.player_name}")
                    continue
                
                player_id = player_ids[market.player_name]
                analysis = self.analyze_prop(market, player_id)
                
                if analysis:
                    # Check if there's an edge
                    if (analysis['over']['edge'] >= self.min_edge or 
                        analysis['under']['edge'] >= self.min_edge):
                        edges.append(analysis)
            
            logger.info(f"Found {len(edges)} props with edges")
            return edges
            
        except Exception as e:
            logger.error(f"Error finding edges: {str(e)}")
            return []

def main() -> None:
    """Example usage of PropBetAnalyzer"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize analyzer
    analyzer = PropBetAnalyzer(
        min_edge=0.05,
        kelly_fraction=0.25,
        confidence_threshold=0.6
    )
    
    # Example player IDs (you would need a complete mapping)
    player_ids = {
        'LeBron James': 2544,
        'Stephen Curry': 201939,
        'Kevin Durant': 201142
    }
    
    # Find edges
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
        
        # Print over/under analysis
        for side in ['over', 'under']:
            if edge[side]['edge'] >= analyzer.min_edge:
                print(f"\n{side.upper()}:")
                print(f"Edge: {edge[side]['edge']:.1%}")
                print(f"EV: ${edge[side]['ev_dollars']:.2f} ({edge[side]['ev_percent']:.1%})")
                print(f"Kelly bet: ${edge[side]['kelly_bet']:.2f}")
                print(f"Best odds: {edge[side]['best_odds']}")

if __name__ == "__main__":
    main()
