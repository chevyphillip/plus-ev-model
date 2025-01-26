"""
Monte Carlo simulation module for player props.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats as scipy_stats
import pandas as pd


@dataclass
class PlayerStats:
    """Historical player statistics"""
    player_name: str
    stat_type: str  # points, rebounds, assists
    values: List[float]  # historical values
    minutes: List[float]  # corresponding minutes played
    dates: List[str]  # game dates
    home_games: List[bool]  # whether game was home/away
    
    def __post_init__(self):
        """Validate stats data"""
        if not (len(self.values) == len(self.minutes) == len(self.dates) == len(self.home_games)):
            raise ValueError("All stat lists must have same length")
        if not self.values:
            raise ValueError("Must provide at least one game of historical data")


@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation"""
    num_sims: int = 1000
    recent_games_weight: float = 0.6  # Weight for recent games in distribution
    minutes_correlation: bool = True  # Consider minutes correlation
    home_away_split: bool = True  # Consider home/away splits
    confidence_level: float = 0.90  # For confidence intervals


class MonteCarloSimulator:
    """Handles Monte Carlo simulations for player props"""
    
    def __init__(self, params: SimulationParams):
        self.params = params
    
    def _calculate_per_minute_rates(self, stats: PlayerStats) -> np.ndarray:
        """Calculate per-minute production rates"""
        return np.array(stats.values) / np.array(stats.minutes)
    
    def _estimate_minutes_distribution(self, 
                                    recent_minutes: List[float],
                                    home_game: bool,
                                    stats: Optional[PlayerStats] = None) -> Tuple[float, float]:
        """Estimate minutes distribution parameters"""
        minutes_array = np.array(recent_minutes)
        
        if stats and self.params.home_away_split and len(minutes_array) >= 10:
            # Split by home/away if enough data
            home_games = np.array(stats.home_games[-len(recent_minutes):])
            relevant_minutes = minutes_array[home_games == home_game]
            if len(relevant_minutes) < 5:  # Fall back if not enough split data
                relevant_minutes = minutes_array
        else:
            relevant_minutes = minutes_array
            
        # Increase variance for small samples
        std_multiplier = 2.0 if len(relevant_minutes) < 10 else 1.0
        return float(np.mean(relevant_minutes)), float(np.std(relevant_minutes) * std_multiplier)
    
    def _generate_minutes(self, 
                         mean_minutes: float, 
                         std_minutes: float, 
                         num_sims: int) -> np.ndarray:
        """Generate minutes played distributions"""
        # Use truncated normal to avoid negative minutes
        lower_bound = max(0, mean_minutes - 3*std_minutes)
        upper_bound = mean_minutes + 3*std_minutes
        
        return scipy_stats.truncnorm.rvs(
            (lower_bound - mean_minutes) / std_minutes,
            (upper_bound - mean_minutes) / std_minutes,
            loc=mean_minutes,
            scale=std_minutes,
            size=num_sims
        )
    
    def _calculate_weighted_rates(self, 
                                stats: PlayerStats, 
                                rates: np.ndarray) -> Tuple[float, float]:
        """Calculate weighted mean and std of per-minute rates"""
        num_games = len(rates)
        if num_games <= 5:
            # Increase variance significantly for very small samples
            mean = float(np.mean(rates))
            std = float(np.std(rates)) * 4.0  # Quadruple the standard deviation
            return mean, std
            
        # Calculate weighted stats giving more weight to recent games
        recent_cutoff = int(num_games * self.params.recent_games_weight)
        recent_rates = rates[-recent_cutoff:]
        older_rates = rates[:-recent_cutoff]
        
        recent_mean = np.mean(recent_rates)
        older_mean = np.mean(older_rates)
        
        # Weighted mean
        weighted_mean = (recent_mean * self.params.recent_games_weight + 
                        older_mean * (1 - self.params.recent_games_weight))
        
        # Weighted standard deviation with increased variance for uncertainty
        recent_std = np.std(recent_rates)
        older_std = np.std(older_rates)
        weighted_std = np.sqrt(
            self.params.recent_games_weight * recent_std**2 +
            (1 - self.params.recent_games_weight) * older_std**2
        )
        
        # Increase variance for smaller samples
        if num_games < 10:
            weighted_std *= 3.0  # Triple standard deviation
        
        return float(weighted_mean), float(weighted_std)
    
    def run_simulation(self, 
                      stats: PlayerStats, 
                      line: float,
                      projected_minutes: Optional[float] = None,
                      home_game: bool = True) -> Dict[str, float]:
        """Run Monte Carlo simulation for player prop"""
        
        # Calculate per-minute rates
        rates = self._calculate_per_minute_rates(stats)
        
        # Get weighted rate distribution parameters
        rate_mean, rate_std = self._calculate_weighted_rates(stats, rates)
        
        # Generate minutes distribution if not provided
        if projected_minutes is None:
            min_mean, min_std = self._estimate_minutes_distribution(
                stats.minutes[-10:],  # Use last 10 games for minutes
                home_game,
                stats
            )
            minutes = self._generate_minutes(min_mean, min_std, self.params.num_sims)
        else:
            # Use fixed minutes with appropriate variance
            variance_factor = 0.20 if len(stats.minutes) < 10 else 0.05
            minutes = np.random.normal(
                projected_minutes, 
                projected_minutes * variance_factor,
                self.params.num_sims
            )
        
        # Generate per-minute rates
        sim_rates = np.random.normal(
            rate_mean,
            rate_std,
            self.params.num_sims
        )
        
        # Calculate final stats
        sim_results = sim_rates * minutes
        
        # Calculate probabilities
        over_prob = np.mean(sim_results > line)
        under_prob = 1 - over_prob
        
        # Calculate confidence intervals using t-distribution
        # Increase interval width significantly for small samples
        if len(stats.values) < 10:
            confidence_level = 0.70  # Lower confidence level
            scale_factor = 4.0  # Much wider intervals
        else:
            confidence_level = self.params.confidence_level
            scale_factor = 1.0
            
        confidence_interval = scipy_stats.t.interval(
            confidence_level,
            len(sim_results) - 1,
            loc=np.mean(sim_results),
            scale=scipy_stats.sem(sim_results) * scale_factor
        )
        
        return {
            "over_probability": round(float(over_prob), 4),
            "under_probability": round(float(under_prob), 4),
            "mean": round(float(np.mean(sim_results)), 2),
            "median": round(float(np.median(sim_results)), 2),
            "std": round(float(np.std(sim_results)), 2),
            "confidence_interval": [
                round(float(confidence_interval[0]), 2),
                round(float(confidence_interval[1]), 2)
            ]
        }
    
    def analyze_correlation(self, 
                          stats: PlayerStats, 
                          min_games: int = 10) -> Dict[str, float]:
        """Analyze correlation between minutes and production"""
        if len(stats.values) < min_games:
            raise ValueError(f"Need at least {min_games} games for correlation analysis")
            
        # Calculate correlation coefficient
        correlation = np.corrcoef(stats.minutes, stats.values)[0, 1]
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            stats.minutes, 
            stats.values
        )
        
        return {
            "correlation": round(float(correlation), 3),
            "r_squared": round(float(r_squared), 3),
            "slope": round(float(slope), 3),
            "intercept": round(float(intercept), 3),
            "p_value": float(p_value)
        }