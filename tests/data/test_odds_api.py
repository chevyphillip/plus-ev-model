"""
Tests for ODDS API integration
"""
import pytest
from unittest.mock import Mock, patch, PropertyMock
from datetime import datetime
import json
import requests

from src.data.odds_api import OddsAPIClient, Team, Player, Event, PlayerPropOdds


@pytest.fixture
def mock_session():
    """Create mock session with responses"""
    session = Mock()
    
    def mock_get(url, params=None):
        response = Mock()
        response.raise_for_status.return_value = None
        response.headers = {
            "x-requests-remaining": "100",
            "x-requests-used": "50"
        }
        
        if "events" in url and "odds" not in url:
            response.json.return_value = [
                {
                    "id": "75504a3843124cb9fd021ef3ccbec2f1",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2025-01-24T00:00:00Z",
                    "home_team": "Boston Celtics",
                    "away_team": "Los Angeles Lakers"
                }
            ]
        elif "odds" in url:
            response.json.return_value = {
                "id": "75504a3843124cb9fd021ef3ccbec2f1",
                "sport_key": "basketball_nba",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "player_points",
                                "outcomes": [
                                    {
                                        "name": "Over 25.5",
                                        "description": "Jayson Tatum",
                                        "price": 1.91
                                    },
                                    {
                                        "name": "Under 25.5",
                                        "description": "Jayson Tatum",
                                        "price": 1.95
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "key": "betonlineag",
                        "title": "BetOnline",
                        "markets": [
                            {
                                "key": "player_points",
                                "outcomes": [
                                    {
                                        "name": "Over 25.5",
                                        "description": "Jayson Tatum",
                                        "price": 1.90
                                    },
                                    {
                                        "name": "Under 25.5",
                                        "description": "Jayson Tatum",
                                        "price": 1.96
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        elif "teams" in url:
            response.json.return_value = [
                {
                    "id": "team1",
                    "full_name": "Boston Celtics",
                    "name": "Celtics",
                    "city": "Boston"
                }
            ]
        elif "players" in url:
            response.json.return_value = [
                {
                    "id": "player1",
                    "full_name": "Jayson Tatum",
                    "first_name": "Jayson",
                    "last_name": "Tatum"
                }
            ]
        else:
            response.json.return_value = [{"key": "basketball_nba"}]
        
        return response
    
    session.get = mock_get
    return session


@pytest.fixture
def mock_error_session():
    """Create mock session that raises errors"""
    session = Mock()
    
    def mock_get(url, params=None):
        response = Mock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Client Error",
            response=Mock(status_code=401)
        )
        return response
    
    session.get = mock_get
    return session


@pytest.fixture
def client(mock_session):
    """Create test client with mock session"""
    with patch('requests.Session') as mock_session_class:
        mock_session_class.return_value = mock_session
        client = OddsAPIClient("test_key")
        return client


def test_initialization():
    """Test client initialization"""
    # Test with valid key
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.json.return_value = [{"key": "basketball_nba"}]
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        client = OddsAPIClient("test_key")
        assert client.api_key == "test_key"
    
    # Test with invalid key
    with patch('requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Client Error",
            response=Mock(status_code=401)
        )
        mock_session.return_value.get.return_value = mock_response
        
        with pytest.raises(ValueError, match="ODDS API key not found"):
            OddsAPIClient("invalid_key")
    
    # Test without key
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="ODDS API key not found"):
            OddsAPIClient()


def test_normalize_player_name(client):
    """Test player name normalization"""
    assert client._normalize_player_name("JAYSON TATUM") == "Jayson Tatum"
    assert client._normalize_player_name("LeBron James Jr.") == "Lebron James"
    assert client._normalize_player_name("Jaren Jackson III") == "Jaren Jackson"


def test_parse_prop_type(client):
    """Test market to prop type conversion"""
    assert client._parse_prop_type("player_points") == "points"
    assert client._parse_prop_type("player_rebounds") == "rebounds"
    assert client._parse_prop_type("player_assists") == "assists"
    assert client._parse_prop_type("unknown_market") == "unknown_market"


def test_extract_line_and_odds(client):
    """Test line and odds extraction"""
    # Test with handicap
    outcome = {"handicap": "25.5", "price": 1.91}
    line, odds = client._extract_line_and_odds(outcome)
    assert line == 25.5
    assert -115 <= odds <= -105  # Sharp book range
    
    # Test with name
    outcome = {"name": "Over 25.5", "price": 1.91}
    line, odds = client._extract_line_and_odds(outcome)
    assert line == 25.5
    assert -115 <= odds <= -105
    
    # Test with American odds
    outcome = {"name": "Over 25.5", "odds": -110}
    line, odds = client._extract_line_and_odds(outcome)
    assert line == 25.5
    assert -115 <= odds <= -105
    
    # Test invalid outcome
    with pytest.raises(ValueError):
        client._extract_line_and_odds({"name": "Invalid", "price": 1.91})


def test_get_teams(client):
    """Test teams fetching"""
    teams = client.get_teams()
    assert len(teams) > 0
    assert isinstance(teams[0], Team)
    assert teams[0].full_name == "Boston Celtics"


def test_get_players(client):
    """Test players fetching"""
    players = client.get_players("team1")
    assert len(players) > 0
    assert isinstance(players[0], Player)
    assert players[0].full_name == "Jayson Tatum"


def test_get_events(client):
    """Test events fetching"""
    events = client.get_events()
    assert len(events) > 0
    assert isinstance(events[0], Event)
    assert events[0].home_team == "Boston Celtics"
    assert events[0].away_team == "Los Angeles Lakers"


def test_get_player_props(client):
    """Test player props fetching"""
    props = client.get_player_props(
        prop_types={"points"},
        player_names={"Jayson Tatum"}
    )
    assert len(props) > 0
    assert props[0].player_name == "Jayson Tatum"
    assert props[0].prop_type == "points"
    assert len(props[0].sharp_odds) > 0


def test_api_error_handling(mock_error_session):
    """Test API error handling"""
    with patch('requests.Session') as mock_session_class:
        mock_session_class.return_value = mock_error_session
        
        with pytest.raises(ValueError, match="ODDS API key not found"):
            OddsAPIClient("test_key")


def test_sharp_odds_extraction(client):
    """Test sharp book odds extraction"""
    bookmakers = [
        {
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "Over 25.5",
                            "price": 1.91
                        },
                        {
                            "name": "Under 25.5",
                            "price": 1.95
                        }
                    ]
                }
            ]
        },
        {
            "key": "betonlineag",
            "title": "BetOnline",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "Over 25.5",
                            "price": 1.90
                        },
                        {
                            "name": "Under 25.5",
                            "price": 1.96
                        }
                    ]
                }
            ]
        }
    ]
    
    sharp_odds = client._get_sharp_odds(bookmakers, "player_points")
    assert len(sharp_odds) == 2
    
    # Verify Pinnacle odds
    pinnacle = next(o for o in sharp_odds if o.book_name == "Pinnacle")
    assert abs(pinnacle.weight - 0.6) < 0.01
    assert -115 <= pinnacle.over_odds <= -105
    assert -115 <= pinnacle.under_odds <= -105
    
    # Verify BetOnline odds
    betonline = next(o for o in sharp_odds if o.book_name == "BetOnline")
    assert abs(betonline.weight - 0.4) < 0.01
    assert -115 <= betonline.over_odds <= -105
    assert -115 <= betonline.under_odds <= -105


def test_weight_normalization(client):
    """Test odds weight normalization"""
    # Create mock bookmakers with only one sharp book
    bookmakers = [
        {
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "Over 25.5",
                            "price": 1.91
                        },
                        {
                            "name": "Under 25.5",
                            "price": 1.95
                        }
                    ]
                }
            ]
        }
    ]
    
    sharp_odds = client._get_sharp_odds(bookmakers, "player_points")
    assert len(sharp_odds) == 1
    assert sharp_odds[0].weight == 1.0  # Should normalize to 1