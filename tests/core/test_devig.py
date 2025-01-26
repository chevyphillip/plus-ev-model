"""
Tests for devig calculations
"""
import pytest
from decimal import Decimal
from src.core.devig import DevigCalculator, BookOdds, PropMarket


def test_american_to_decimal():
    calculator = DevigCalculator()
    
    # Test positive American odds
    assert calculator.american_to_decimal(100) == 2.0
    assert calculator.american_to_decimal(150) == 2.5
    
    # Test negative American odds
    assert calculator.american_to_decimal(-110) == 1.9091
    assert calculator.american_to_decimal(-150) == 1.6667
    
    # Test edge cases
    assert calculator.american_to_decimal(0) == 1.0
    assert calculator.american_to_decimal(-100) == 2.0


def test_decimal_to_probability():
    calculator = DevigCalculator()
    
    # Test common decimal odds
    assert calculator.decimal_to_probability(2.0) == 0.5
    assert calculator.decimal_to_probability(1.5) == 0.6667
    
    # Test edge cases
    assert calculator.decimal_to_probability(1.0) == 1.0
    with pytest.raises(ZeroDivisionError):
        calculator.decimal_to_probability(0)


def test_geometric_mean_probability():
    calculator = DevigCalculator()
    
    # Test equal weights
    probs = [0.5, 0.52, 0.48]
    weights = [1/3, 1/3, 1/3]
    assert abs(calculator.geometric_mean_probability(probs, weights) - 0.4994) < 0.0001
    
    # Test different weights
    weights = [0.5, 0.3, 0.2]
    assert abs(calculator.geometric_mean_probability(probs, weights) - 0.5008) < 0.0001
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        calculator.geometric_mean_probability([0.5], [0.5, 0.5])


def test_calculate_true_probability():
    calculator = DevigCalculator()
    
    # Create test market
    market = PropMarket(
        player_name="Test Player",
        prop_type="points",
        line=20.5,
        sharp_odds=[
            BookOdds("Pinnacle", -110, -110, 0.5),
            BookOdds("BetOnline", -108, -112, 0.5)
        ],
        timestamp="2024-01-23T12:00:00"
    )
    
    result = calculator.calculate_true_probability(market)
    
    # Verify probabilities sum to 1
    assert abs(result["over_probability"] + result["under_probability"] - 1.0) < 0.0001
    
    # Verify reasonable probability range
    assert 0.45 <= result["over_probability"] <= 0.55
    assert 0.45 <= result["under_probability"] <= 0.55


def test_calculate_ev():
    calculator = DevigCalculator()
    
    # Test positive EV
    result = calculator.calculate_ev(0.55, -110, 100)
    assert result["ev_dollars"] > 0
    assert result["ev_percent"] > 0
    
    # Test negative EV
    result = calculator.calculate_ev(0.45, -110, 100)
    assert result["ev_dollars"] < 0
    assert result["ev_percent"] < 0
    
    # Test invalid probability
    with pytest.raises(ValueError):
        calculator.calculate_ev(1.1, -110, 100)
    with pytest.raises(ValueError):
        calculator.calculate_ev(-0.1, -110, 100)


def test_get_kelly_bet():
    from src.core.devig import get_kelly_bet
    
    # Test positive edge
    bet_size = get_kelly_bet(10000, 0.55, -110, fraction=0.25)
    assert bet_size > 0
    assert bet_size < 10000  # Bet should be less than bankroll
    
    # Test negative edge
    bet_size = get_kelly_bet(10000, 0.45, -110, fraction=0.25)
    assert bet_size == 0  # Should not bet with negative edge
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        get_kelly_bet(10000, 1.1, -110, 0.25)  # Invalid probability
    with pytest.raises(ValueError):
        get_kelly_bet(10000, 0.55, -110, 1.5)  # Invalid fraction


def test_prop_market_validation():
    # Test valid market
    valid_market = PropMarket(
        player_name="Test Player",
        prop_type="points",
        line=20.5,
        sharp_odds=[
            BookOdds("Pinnacle", -110, -110, 0.6),
            BookOdds("BetOnline", -108, -112, 0.4)
        ],
        timestamp="2024-01-23T12:00:00"
    )
    assert valid_market  # Should not raise exception
    
    # Test invalid weights
    with pytest.raises(ValueError):
        PropMarket(
            player_name="Test Player",
            prop_type="points",
            line=20.5,
            sharp_odds=[
                BookOdds("Pinnacle", -110, -110, 0.6),
                BookOdds("BetOnline", -108, -112, 0.6)  # Weights sum > 1
            ],
            timestamp="2024-01-23T12:00:00"
        )
    
    # Test empty odds
    with pytest.raises(ValueError):
        PropMarket(
            player_name="Test Player",
            prop_type="points",
            line=20.5,
            sharp_odds=[],
            timestamp="2024-01-23T12:00:00"
        )