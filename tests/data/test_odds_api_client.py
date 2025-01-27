"""Tests for OddsAPIClient using real data."""

import pytest
from pytest import MonkeyPatch
import os
from typing import cast, Generator
from src.data.odds_api_client import (
    OddsAPIClient,
    GameDict,
    PropsDict,
    BookmakerDict,
    MarketDict,
    OutcomeDict
)

@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.getenv('ODDS_API_KEY')
    if not key:
        pytest.skip("ODDS_API_KEY environment variable not set")
    return key

@pytest.fixture
def client(api_key: str) -> OddsAPIClient:
    """Create OddsAPIClient instance."""
    return OddsAPIClient(api_key=api_key)

def test_init_with_api_key(api_key: str) -> None:
    """Test initialization with API key."""
    client = OddsAPIClient(api_key=api_key)
    assert client.api_key == api_key

def test_init_without_api_key() -> None:
    """Test initialization fails without API key."""
    with pytest.raises(ValueError):
        OddsAPIClient()

def test_get_nba_games(client: OddsAPIClient) -> None:
    """Test fetching NBA games with real API."""
    games = client.get_nba_games()
    
    assert isinstance(games, list)
    if games:  # Only test if games are available
        game = cast(GameDict, games[0])
        assert isinstance(game['id'], str)
        assert isinstance(game['sport_key'], str)
        assert game['sport_key'] == 'basketball_nba'
        assert isinstance(game['home_team'], str)
        assert isinstance(game['away_team'], str)
        assert isinstance(game['commence_time'], str)

def test_get_player_props(client: OddsAPIClient) -> None:
    """Test fetching player props with real API."""
    games = client.get_nba_games()
    
    if not games:
        pytest.skip("No NBA games available")
        
    # Get props for first game
    game = cast(GameDict, games[0])
    props = client.get_player_props(game['id'], ['player_points'])
    
    assert isinstance(props, dict)
    if 'bookmakers' in props:  # Only test if props are available
        assert len(props['bookmakers']) > 0
        bookmaker = cast(BookmakerDict, props['bookmakers'][0])
        assert isinstance(bookmaker['key'], str)
        assert isinstance(bookmaker['markets'], list)
        
        market = cast(MarketDict, bookmaker['markets'][0])
        assert market['key'] == 'player_points'
        assert len(market['outcomes']) > 0
        
        outcome = cast(OutcomeDict, market['outcomes'][0])
        assert isinstance(outcome['name'], str)  # Over/Under
        assert isinstance(outcome['description'], str)  # Player name
        assert isinstance(outcome['price'], str)  # Odds
        assert isinstance(outcome['point'], float)  # Line

def test_get_best_odds(client: OddsAPIClient) -> None:
    """Test getting best odds with real API."""
    games = client.get_nba_games()
    
    if not games:
        pytest.skip("No NBA games available")
        
    # Get first game
    game = cast(GameDict, games[0])
    
    # Get props to find a valid player
    props = client.get_player_props(game['id'], ['player_points'])
    if not props or 'bookmakers' not in props or not props['bookmakers']:
        pytest.skip("No props available")
        
    # Get first available player
    bookmaker = cast(BookmakerDict, props['bookmakers'][0])
    market = cast(MarketDict, bookmaker['markets'][0])
    outcome = cast(OutcomeDict, market['outcomes'][0])
    player_name = outcome['description']
    
    # Test getting best odds
    best_odds = client.get_best_odds(game['id'], player_name, 'points')
    
    assert isinstance(best_odds, dict)
    assert 'over' in best_odds
    assert 'under' in best_odds
    
    for side in ['over', 'under']:
        assert 'odds' in best_odds[side]
        assert 'book' in best_odds[side]
        assert 'line' in best_odds[side]
        assert isinstance(best_odds[side]['odds'], int)
        assert isinstance(best_odds[side]['line'], float)
        assert isinstance(best_odds[side]['book'], str)
        assert best_odds[side]['book'] != ''

def test_rate_limit_handling(client: OddsAPIClient) -> None:
    """Test rate limit error handling."""
    # Make multiple requests to potentially hit rate limit
    for _ in range(10):
        try:
            games = client.get_nba_games()
            assert isinstance(games, list)
        except RuntimeError as e:
            assert str(e) == "API rate limit exceeded"
            return
            
    # If we didn't hit rate limit, test passes
    assert True

if __name__ == '__main__':
    pytest.main([__file__])
