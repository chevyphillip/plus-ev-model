"""
Tests for Monte Carlo simulation
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from src.core.monte_carlo import MonteCarloSimulator, PlayerStats, SimulationParams


def create_test_stats(num_games: int = 20) -> PlayerStats:
    """Create test player statistics"""
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic test data
    base_minutes = 32
    base_points = 20
    minutes = np.random.normal(base_minutes, 4, num_games)
    points = minutes * (base_points/base_minutes) + np.random.normal(0, 3, num_games)
    
    # Generate dates and home/away
    base_date = datetime(2024, 1, 1)
    dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") 
            for i in range(num_games)]
    home_games = [bool(i % 2) for i in range(num_games)]
    
    return PlayerStats(
        player_name="Test Player",
        stat_type="points",
        values=points.tolist(),
        minutes=minutes.tolist(),
        dates=dates,
        home_games=home_games
    )


def test_player_stats_validation():
    """Test PlayerStats validation"""
    # Valid stats
    stats = create_test_stats()
    assert stats  # Should not raise exception
    
    # Invalid stats - mismatched lengths
    with pytest.raises(ValueError):
        PlayerStats(
            player_name="Test Player",
            stat_type="points",
            values=[20.0],
            minutes=[32.0, 30.0],  # Different length
            dates=["2024-01-01"],
            home_games=[True]
        )
    
    # Invalid stats - empty data
    with pytest.raises(ValueError):
        PlayerStats(
            player_name="Test Player",
            stat_type="points",
            values=[],
            minutes=[],
            dates=[],
            home_games=[]
        )


def test_simulation_params():
    """Test SimulationParams defaults and validation"""
    # Default params
    params = SimulationParams()
    assert params.num_sims == 1000
    assert 0 < params.recent_games_weight < 1
    assert params.minutes_correlation is True
    assert params.home_away_split is True
    assert 0 < params.confidence_level < 1


def test_per_minute_rates():
    """Test per-minute rate calculations"""
    simulator = MonteCarloSimulator(SimulationParams())
    stats = create_test_stats()
    
    rates = simulator._calculate_per_minute_rates(stats)
    assert len(rates) == len(stats.values)
    assert all(rate > 0 for rate in rates)  # Rates should be positive
    assert np.mean(rates) * 32 >= 15  # Reasonable points per game at 32 minutes


def test_minutes_distribution():
    """Test minutes distribution estimation"""
    simulator = MonteCarloSimulator(SimulationParams())
    stats = create_test_stats()
    
    # Test home game minutes
    mean, std = simulator._estimate_minutes_distribution(stats.minutes, True)
    assert 20 <= mean <= 40  # Reasonable minutes range
    assert 0 < std <= 10  # Reasonable standard deviation
    
    # Test away game minutes
    mean, std = simulator._estimate_minutes_distribution(stats.minutes, False)
    assert 20 <= mean <= 40
    assert 0 < std <= 10


def test_generate_minutes():
    """Test minutes generation"""
    simulator = MonteCarloSimulator(SimulationParams())
    
    mean_minutes = 32
    std_minutes = 4
    num_sims = 1000
    
    minutes = simulator._generate_minutes(mean_minutes, std_minutes, num_sims)
    
    assert len(minutes) == num_sims
    assert all(m >= 0 for m in minutes)  # No negative minutes
    assert abs(np.mean(minutes) - mean_minutes) < 2  # Close to target mean
    assert abs(np.std(minutes) - std_minutes) < 1  # Close to target std


def test_weighted_rates():
    """Test weighted rate calculations"""
    simulator = MonteCarloSimulator(SimulationParams())
    stats = create_test_stats()
    
    rates = simulator._calculate_per_minute_rates(stats)
    mean, std = simulator._calculate_weighted_rates(stats, rates)
    
    assert 0.4 <= mean <= 1.0  # Reasonable points per minute
    assert 0 < std <= 0.3  # Reasonable standard deviation


def test_run_simulation():
    """Test full simulation run"""
    simulator = MonteCarloSimulator(SimulationParams())
    stats = create_test_stats()
    
    # Test with line near average production
    result = simulator.run_simulation(stats, 20.5)
    
    # Check result structure and values
    assert "over_probability" in result
    assert "under_probability" in result
    assert "mean" in result
    assert "median" in result
    assert "std" in result
    assert "confidence_interval" in result
    
    # Verify probabilities sum to 1
    assert abs(result["over_probability"] + result["under_probability"] - 1) < 0.0001
    
    # Verify reasonable ranges
    assert 15 <= result["mean"] <= 30
    assert 15 <= result["median"] <= 30
    assert 0 < result["std"] <= 10
    assert len(result["confidence_interval"]) == 2
    assert result["confidence_interval"][0] < result["confidence_interval"][1]


def test_correlation_analysis():
    """Test minutes-production correlation analysis"""
    simulator = MonteCarloSimulator(SimulationParams())
    stats = create_test_stats(30)  # Use 30 games for better correlation analysis
    
    result = simulator.analyze_correlation(stats)
    
    # Check result structure
    assert "correlation" in result
    assert "r_squared" in result
    assert "slope" in result
    assert "intercept" in result
    assert "p_value" in result
    
    # Verify reasonable ranges
    assert -1 <= result["correlation"] <= 1
    assert 0 <= result["r_squared"] <= 1
    assert result["slope"] > 0  # Should be positive correlation
    assert isinstance(result["p_value"], float)


def test_simulation_with_fixed_minutes():
    """Test simulation with fixed projected minutes"""
    simulator = MonteCarloSimulator(SimulationParams())
    stats = create_test_stats()
    
    projected_minutes = 35
    result = simulator.run_simulation(stats, 20.5, projected_minutes=projected_minutes)
    
    # Results should reflect higher minutes
    assert result["mean"] > 20  # Higher production with more minutes
    assert result["confidence_interval"][0] < result["mean"] < result["confidence_interval"][1]


def test_insufficient_data():
    """Test handling of insufficient data"""
    simulator = MonteCarloSimulator(SimulationParams())
    
    # Create stats with minimal data
    stats = create_test_stats(5)
    
    # Should still run but with wider confidence intervals
    result = simulator.run_simulation(stats, 20.5)
    assert result["confidence_interval"][1] - result["confidence_interval"][0] > 5
    
    # Correlation analysis should fail
    with pytest.raises(ValueError):
        simulator.analyze_correlation(stats, min_games=10)