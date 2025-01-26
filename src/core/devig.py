"""
Devig module for calculating true probabilities from sharp book odds.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, NoReturn, Tuple
import math
import numpy as np
from decimal import Decimal, ROUND_HALF_UP


@dataclass
class BookOdds:
    """Represents odds from a single bookmaker"""
    book_name: str
    over_odds: int  # American odds format
    under_odds: int
    weight: float = 1.0  # Book weight for probability calculation


@dataclass
class PropMarket:
    """Represents a player prop market with multiple book odds"""
    player_name: str
    team: str  # Player's team
    prop_type: str  # points, rebounds, assists, etc.
    line: float
    sharp_odds: List[BookOdds]
    timestamp: str
    
    def __post_init__(self) -> None:
        """Validate odds data"""
        if not self.sharp_odds:
            raise ValueError("Must provide at least one book's odds")
        if not all(0 < book.weight <= 1 for book in self.sharp_odds):
            raise ValueError("Book weights must be between 0 and 1")
        total_weight = sum(book.weight for book in self.sharp_odds)
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point differences
            raise ValueError(f"Book weights must sum to 1, got {total_weight}")


class DevigCalculator:
    """Handles devig calculations for prop markets"""
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal format"""
        if american_odds == 0:
            return 1.0
        if american_odds > 0:
            return round(american_odds / 100 + 1, 4)
        return round(-100 / abs(american_odds) + 1, 4)

    @staticmethod
    def decimal_to_probability(decimal_odds: float) -> float:
        """Convert decimal odds to probability"""
        if decimal_odds == 0:
            raise ZeroDivisionError("Cannot convert zero odds to probability")
        if decimal_odds < 0:
            raise ValueError("Decimal odds must be positive")
        return round(1 / decimal_odds, 4)

    def get_raw_probabilities(self, odds: int) -> float:
        """Get raw probability from American odds"""
        decimal = self.american_to_decimal(odds)
        return self.decimal_to_probability(decimal)

    def balanced_devig(self, prob1: float, prob2: float) -> Tuple[float, float]:
        """
        Apply balanced devigging method.
        
        This method:
        1. Converts raw probabilities to odds
        2. Averages the odds
        3. Converts back to probabilities
        4. Normalizes to sum to 1
        """
        if prob1 <= 0 or prob2 <= 0:
            return 0.5, 0.5
            
        # Convert probabilities to odds
        odds1 = 1 / prob1
        odds2 = 1 / prob2
        
        # Average the odds
        avg_odds1 = (odds1 + odds2) / 2
        avg_odds2 = (odds1 + odds2) / 2
        
        # Convert back to probabilities
        fair_prob1 = 1 / avg_odds1
        fair_prob2 = 1 / avg_odds2
        
        # Normalize
        total = fair_prob1 + fair_prob2
        if total > 0:
            fair_prob1 = fair_prob1 / total
            fair_prob2 = fair_prob2 / total
        else:
            fair_prob1 = fair_prob2 = 0.5
        
        # Round to 4 decimal places
        return (
            float(Decimal(str(fair_prob1)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            float(Decimal(str(fair_prob2)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
        )

    def weighted_average_probability(self, probabilities: List[float], weights: List[float]) -> float:
        """Calculate weighted arithmetic mean of probabilities"""
        if len(probabilities) != len(weights):
            raise ValueError("Must have same number of probabilities and weights")
        if not all(0 <= p <= 1 for p in probabilities):
            raise ValueError("All probabilities must be between 0 and 1")
        if not all(0 <= w <= 1 for w in weights):
            raise ValueError("All weights must be between 0 and 1")
        if not 0.99 <= sum(weights) <= 1.01:  # Allow small floating point differences
            raise ValueError("Weights must sum to 1")
        
        # Calculate weighted average
        result = sum(p * w for p, w in zip(probabilities, weights))
        
        # Round to 4 decimal places
        return float(Decimal(str(result)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

    def calculate_true_probability(self, market: PropMarket) -> Dict[str, float]:
        """Calculate true probabilities for over/under from sharp books"""
        
        # Convert odds to probabilities for each book
        over_probs: List[float] = []
        under_probs: List[float] = []
        weights: List[float] = []
        
        for book in market.sharp_odds:
            # Convert odds to probabilities
            over_prob = self.get_raw_probabilities(book.over_odds)
            under_prob = self.get_raw_probabilities(book.under_odds)
            
            # Apply balanced devig
            devigged_over, devigged_under = self.balanced_devig(over_prob, under_prob)
            
            over_probs.append(devigged_over)
            under_probs.append(devigged_under)
            weights.append(book.weight)
        
        # Calculate weighted average probabilities
        true_over_prob = self.weighted_average_probability(over_probs, weights)
        true_under_prob = self.weighted_average_probability(under_probs, weights)
        
        return {
            "over_probability": true_over_prob,
            "under_probability": true_under_prob
        }

    def calculate_ev(self, 
                    true_prob: float, 
                    odds: int, 
                    bet_size: float = 100) -> Dict[str, float]:
        """
        Calculate expected value using decimal odds.
        
        EV = (Probability * (Decimal Odds - 1)) - ((1 - Probability) * 1)
        
        Args:
            true_prob: True probability of the outcome
            odds: American odds
            bet_size: Bet size in dollars (default: $100)
            
        Returns:
            Dictionary with EV in dollars and percentage
        """
        if not 0 <= true_prob <= 1:
            raise ValueError("True probability must be between 0 and 1")
            
        # Convert to decimal odds
        decimal_odds = self.american_to_decimal(odds)
        
        # Calculate EV using decimal odds formula
        ev_percent = (true_prob * (decimal_odds - 1)) - ((1 - true_prob) * 1)
        ev_percent *= 100  # Convert to percentage
        
        # Calculate dollar EV
        ev_dollars = ev_percent * bet_size / 100
        
        # Round to 2 decimal places
        ev_dec = Decimal(str(ev_dollars)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        ev_percent_dec = Decimal(str(ev_percent)).quantize(Decimal('0.01'), 
                                                         rounding=ROUND_HALF_UP)
        
        return {
            "ev_dollars": float(ev_dec),
            "ev_percent": float(ev_percent_dec)
        }


def get_kelly_bet(bankroll: float, 
                  true_prob: float, 
                  odds: int, 
                  fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion bet size with fractional Kelly for risk management.
    
    The Kelly Criterion finds the optimal bet size that maximizes long-term growth
    rate of the bankroll. We use fractional Kelly (typically 25%) to reduce variance.
    """
    if not 0 <= true_prob <= 1:
        raise ValueError("True probability must be between 0 and 1")
    if not 0 < fraction <= 1:
        raise ValueError("Kelly fraction must be between 0 and 1")
        
    # Convert to decimal odds
    decimal_odds = DevigCalculator.american_to_decimal(odds)
    
    # Calculate full Kelly fraction: (bp - q)/b where:
    # b = decimal odds - 1
    # p = probability of winning
    # q = probability of losing
    b = decimal_odds - 1
    p = true_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly
    fractional_kelly = kelly * fraction
    
    # Calculate bet size
    bet_size = max(0, fractional_kelly * bankroll)
    
    # Round to nearest dollar
    return round(bet_size)