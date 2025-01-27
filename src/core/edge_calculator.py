"""Calculate betting edges by comparing model predictions to sportsbook lines."""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BettingLine:
    """Container for betting line information."""
    over: float  # Over line (e.g. 22.5 points)
    over_odds: int  # American odds for over (e.g. -110)
    under: float  # Under line (should match over)
    under_odds: int  # American odds for under (e.g. -110)

def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability.
    
    Args:
        american_odds: Odds in American format (e.g. -110, +150)
        
    Returns:
        Implied probability as decimal (0-1)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def decimal_odds(american_odds: int) -> float:
    """Convert American odds to decimal odds.
    
    Args:
        american_odds: Odds in American format
        
    Returns:
        Decimal odds (e.g. 1.91)
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def calculate_edge(
    model_prob: float,
    market_prob: float,
    min_edge: float = 0.05
) -> Optional[float]:
    """Calculate edge between model probability and market probability.
    
    Args:
        model_prob: Model's predicted probability
        market_prob: Market implied probability
        min_edge: Minimum edge required to consider bet
        
    Returns:
        Edge as percentage if above min_edge, otherwise None
    """
    edge = model_prob - market_prob
    return float(edge) if abs(edge) >= min_edge else None

def kelly_criterion(
    prob: float,
    decimal_odds: float,
    fraction: float = 1.0
) -> float:
    """Calculate Kelly Criterion bet size.
    
    Args:
        prob: Probability of winning
        decimal_odds: Decimal odds offered
        fraction: Fraction of Kelly to use (default full Kelly)
        
    Returns:
        Recommended bet size as percentage of bankroll
    """
    q = 1 - prob  # Probability of losing
    bet = (prob * (decimal_odds - 1) - q) / (decimal_odds - 1)
    return max(0, bet * fraction)

class EdgeCalculator:
    """Calculate betting edges and recommended bet sizes."""
    
    def __init__(
        self,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.5,
        confidence_threshold: float = 0.6
    ) -> None:
        """Initialize the edge calculator.
        
        Args:
            min_edge: Minimum edge required to consider bet
            kelly_fraction: Fraction of Kelly criterion to use
            confidence_threshold: Minimum model confidence required
        """
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.confidence_threshold = confidence_threshold
    
    def analyze_prop_bet(
        self,
        prediction: Dict[str, Any],
        line: BettingLine
    ) -> Dict[str, Any]:
        """Analyze a player prop bet for potential edges.
        
        Args:
            prediction: Model prediction dictionary
            line: Betting line information
            
        Returns:
            Dictionary containing analysis results
        """
        pred_value = prediction['predicted_value']
        recent_avg = prediction['recent_average']
        last_5 = prediction['last_5_games']
        
        # Calculate over/under probabilities
        over_market_prob = implied_probability(line.over_odds)
        under_market_prob = implied_probability(line.under_odds)
        
        # Calculate standard deviation of recent games
        recent_std = float(np.std(last_5))
        
        # Use normal distribution to estimate probabilities
        over_model_prob = 1 - float(
            np.exp(-0.5 * ((line.over - pred_value) / recent_std) ** 2)
            / (recent_std * np.sqrt(2 * np.pi))
        )
        under_model_prob = 1 - over_model_prob
        
        # Calculate edges
        over_edge = calculate_edge(
            over_model_prob,
            over_market_prob,
            self.min_edge
        )
        under_edge = calculate_edge(
            under_model_prob,
            under_market_prob,
            self.min_edge
        )
        
        # Calculate Kelly sizes if edges exist
        over_kelly = kelly_criterion(
            over_model_prob,
            decimal_odds(line.over_odds),
            self.kelly_fraction
        ) if over_edge else 0.0
        
        under_kelly = kelly_criterion(
            under_model_prob,
            decimal_odds(line.under_odds),
            self.kelly_fraction
        ) if under_edge else 0.0
        
        # Calculate confidence score (0-1)
        confidence = 1 - (recent_std / pred_value)
        confidence = max(0, min(1, confidence))
        
        return {
            'player_name': prediction['player_name'],
            'prop_type': prediction['prop_type'],
            'line': line.over,  # Same as line.under
            'predicted_value': pred_value,
            'recent_average': recent_avg,
            'last_5_games': last_5,
            'recent_std': recent_std,
            'confidence': confidence,
            'over': {
                'odds': line.over_odds,
                'market_prob': over_market_prob,
                'model_prob': over_model_prob,
                'edge': over_edge,
                'kelly': over_kelly if confidence >= self.confidence_threshold else 0.0
            },
            'under': {
                'odds': line.under_odds,
                'market_prob': under_market_prob,
                'model_prob': under_model_prob,
                'edge': under_edge,
                'kelly': under_kelly if confidence >= self.confidence_threshold else 0.0
            }
        }
    
    def get_bet_recommendation(
        self,
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get betting recommendation if an edge exists.
        
        Args:
            analysis: Analysis results from analyze_prop_bet
            
        Returns:
            Dictionary with bet recommendation or None if no edge
        """
        if analysis['confidence'] < self.confidence_threshold:
            return None
        
        # Find side with larger edge
        over_edge = analysis['over']['edge'] or 0
        under_edge = analysis['under']['edge'] or 0
        
        if max(over_edge, under_edge) < self.min_edge:
            return None
        
        side = 'over' if over_edge > under_edge else 'under'
        edge = over_edge if side == 'over' else under_edge
        kelly = analysis[side]['kelly']
        
        if kelly <= 0:
            return None
        
        return {
            'player_name': analysis['player_name'],
            'prop_type': analysis['prop_type'],
            'bet_type': side,
            'line': analysis['line'],
            'odds': analysis[side]['odds'],
            'edge': edge,
            'kelly_bet': kelly,
            'confidence': analysis['confidence'],
            'predicted_value': analysis['predicted_value'],
            'recent_average': analysis['recent_average'],
            'last_5_games': analysis['last_5_games']
        }
