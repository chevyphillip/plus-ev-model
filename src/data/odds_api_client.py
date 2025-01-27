"""Client for interacting with The Odds API."""

import os
import logging
from typing import Dict, List, Optional, Any, TypedDict, cast
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class GameDict(TypedDict):
    """Type definition for game data."""
    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str

class OutcomeDict(TypedDict):
    """Type definition for prop outcome."""
    name: str
    description: str
    price: str
    point: float

class MarketDict(TypedDict):
    """Type definition for prop market."""
    key: str
    outcomes: List[OutcomeDict]

class BookmakerDict(TypedDict):
    """Type definition for bookmaker data."""
    key: str
    title: str
    markets: List[MarketDict]

class PropsDict(TypedDict):
    """Type definition for props response."""
    id: str
    bookmakers: List[BookmakerDict]

class OddsAPIClient:
    """Client for fetching odds data from The Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the client.
        
        Args:
            api_key: API key for The Odds API. If not provided, will look for
                    ODDS_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ODDS_API_KEY environment variable.")
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> Any:
        """Make API request with rate limit handling.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Log remaining requests
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                remaining_int = int(remaining)
                if remaining_int < 10:
                    logger.warning(f"Low on API requests: {remaining} remaining")
                else:
                    logger.info(f"API requests remaining: {remaining}")
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error("API rate limit exceeded")
                raise RuntimeError("API rate limit exceeded") from e
            raise
    
    def get_nba_games(self) -> List[GameDict]:
        """Get current NBA games.
        
        Returns:
            List of game dictionaries
            
        Raises:
            RuntimeError: If API rate limit is exceeded
            requests.exceptions.RequestException: If request fails
        """
        url = f"{self.BASE_URL}/sports/basketball_nba/events"
        params = {
            'apiKey': self.api_key
        }
        
        try:
            response = self._make_request(url, params)
            if not isinstance(response, list):
                logger.error("Unexpected response format")
                return []
            return cast(List[GameDict], response)
        except Exception as e:
            logger.error(f"Error fetching NBA games: {str(e)}")
            return []
    
    def get_player_props(
        self,
        game_id: str,
        markets: Optional[List[str]] = None
    ) -> PropsDict:
        """Get player props for a specific game.
        
        Args:
            game_id: Game ID from get_nba_games()
            markets: List of markets to fetch. Defaults to all available markets.
                    Options: player_points, player_rebounds, player_assists,
                            player_threes, player_blocks, player_steals
        
        Returns:
            Dictionary containing odds data
            
        Raises:
            RuntimeError: If API rate limit is exceeded
            requests.exceptions.RequestException: If request fails
        """
        if not markets:
            markets = [
                'player_points',
                'player_rebounds', 
                'player_assists',
                'player_threes',
                'player_blocks',
                'player_steals'
            ]
        
        url = f"{self.BASE_URL}/sports/basketball_nba/events/{game_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }
        
        try:
            response = self._make_request(url, params)
            if not isinstance(response, dict):
                logger.error("Unexpected response format")
                return cast(PropsDict, {})
            return cast(PropsDict, response)
        except Exception as e:
            logger.error(f"Error fetching props for game {game_id}: {str(e)}")
            return cast(PropsDict, {})
    
    def get_best_odds(
        self,
        game_id: str,
        player_name: str,
        prop_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get best available odds for a player prop.
        
        Args:
            game_id: Game ID from get_nba_games()
            player_name: Player name to find odds for
            prop_type: Type of prop (points, rebounds, assists, threes)
            
        Returns:
            Dictionary containing best odds for over/under
        """
        props = self.get_player_props(game_id, [f'player_{prop_type}'])
        
        if not props or 'bookmakers' not in props:
            return {}
        
        best_odds: Dict[str, Dict[str, Any]] = {
            'over': {'odds': -1000000, 'book': '', 'line': 0},
            'under': {'odds': -1000000, 'book': '', 'line': 0}
        }
        
        for book in props['bookmakers']:
            for market in book['markets']:
                if market['key'] == f'player_{prop_type}':
                    for outcome in market['outcomes']:
                        if outcome['description'] == player_name:
                            odds = int(outcome['price'])
                            side = outcome['name'].lower()
                            if side in best_odds:
                                current_odds = cast(int, best_odds[side]['odds'])
                                if odds > current_odds:
                                    best_odds[side] = {
                                        'odds': odds,
                                        'book': book['title'],
                                        'line': float(outcome['point'])
                                    }
        
        return best_odds

def main() -> None:
    """Example usage of OddsAPIClient."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        client = OddsAPIClient()
        
        # Get today's games
        games = client.get_nba_games()
        logger.info(f"Found {len(games)} NBA games")
        
        # Get props for first game
        if games:
            game = games[0]
            logger.info(f"\nGetting props for {game['away_team']} @ {game['home_team']}")
            
            props = client.get_player_props(game['id'])
            if props and 'bookmakers' in props:
                for book in props['bookmakers'][:1]:  # Just show first book
                    logger.info(f"\n{book['title']} odds:")
                    for market in book['markets']:
                        logger.info(f"\n{market['key']}:")
                        for outcome in market['outcomes'][:3]:  # Show first 3 lines
                            logger.info(
                                f"{outcome['description']}: "
                                f"{outcome['name']} {outcome['point']} ({outcome['price']})"
                            )
                            
    except RuntimeError as e:
        logger.error(f"API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    main()
