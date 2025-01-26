"""
Module for fetching and processing NBA player props odds from ODDS API.
"""
from dataclasses import dataclass
from datetime import datetime, UTC
import os
from typing import Dict, List, Optional, Set, Tuple, Any, Dict, List
import requests
from dotenv import load_dotenv
import re

from src.core.devig import BookOdds, PropMarket


@dataclass
class Event:
    """NBA game event"""
    id: str
    home_team: str
    away_team: str
    commence_time: str


class OddsAPIClient:
    """Client for interacting with ODDS API"""
    
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT_KEY = "basketball_nba"
    REGIONS = ["eu", "us", "us2", "us_dfs", "us_ex"]
    MARKETS = [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_blocks",
        "player_steals",
        "player_first_basket",
        "player_double_double",
        "player_triple_double",
        "player_method_of_first_basket"
    ]
    SHARP_BOOKS = {"pinnacle", "betonlineag"}  # Case-insensitive
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize client with API key"""
        load_dotenv()
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("ODDS API key not found")
        
        self.session = requests.Session()
        
        # Test API key validity
        try:
            self._make_request(f"{self.SPORT_KEY}/events", {})
        except Exception as e:
            if "401" in str(e):
                raise ValueError("Invalid API key")
            raise
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to ODDS API"""
        url = f"{self.BASE_URL}/{endpoint}"
        params["apiKey"] = self.api_key
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                if e.response.status_code == 401:
                    raise ValueError("Invalid API key")
            raise Exception(f"API Error: {str(e)}")
    
    def get_events(self) -> List[Event]:
        """Get today's NBA games"""
        events: List[Event] = []
        response = self._make_request(
            f"{self.SPORT_KEY}/events",
            {}
        )
        
        if isinstance(response, list):
            for event_data in response:
                if isinstance(event_data, dict):
                    event_id = str(event_data.get("id", ""))
                    home_team = str(event_data.get("home_team", ""))
                    away_team = str(event_data.get("away_team", ""))
                    commence_time = str(event_data.get("commence_time", ""))
                    
                    if event_id and home_team and away_team and commence_time:
                        events.append(Event(
                            id=event_id,
                            home_team=home_team,
                            away_team=away_team,
                            commence_time=commence_time
                        ))
                    else:
                        print(f"Skipping invalid event data: {event_data}")
        
        return events
    
    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name for consistent matching"""
        # Remove Jr., III, etc.
        suffixes = [r"\bJr\.?", r"\bSr\.?", r"\bII+", r"\bIII+", r"\bIV"]
        normalized = name
        for suffix in suffixes:
            normalized = re.sub(suffix, "", normalized, flags=re.IGNORECASE)
        
        # Convert to title case and remove extra whitespace
        return " ".join(normalized.title().split())
    
    def _parse_prop_type(self, market: str) -> str:
        """Convert API market to internal prop type"""
        market_map = {
            "player_points": "points",
            "player_rebounds": "rebounds",
            "player_assists": "assists",
            "player_threes": "threes",
            "player_blocks": "blocks",
            "player_steals": "steals",
            "player_first_basket": "first_basket",
            "player_double_double": "double_double",
            "player_triple_double": "triple_double",
            "player_method_of_first_basket": "first_basket_method"
        }
        return market_map.get(market, market)
    
    def _extract_line_and_odds(self, outcome: Dict[str, Any]) -> Tuple[float, float]:
        """Extract line value and decimal odds from outcome"""
        # Get line value
        if "point" in outcome:
            line = float(outcome["point"])
        elif "handicap" in outcome:
            line = float(outcome["handicap"])
        else:
            # Extract from name (e.g., "Over 22.5")
            try:
                line = float(outcome["name"].split()[-1])
            except (IndexError, ValueError):
                raise ValueError(f"Could not extract line from outcome: {outcome}")
        
        # Get decimal odds
        if "price" in outcome:
            odds = float(outcome["price"])
        else:
            raise ValueError(f"No odds found in outcome: {outcome}")
        
        return line, odds
    
    def _get_sharp_odds(self, bookmakers: List[Dict[str, Any]], market_key: str, player: str) -> List[BookOdds]:
        """Extract odds from sharp books with proper weighting"""
        sharp_odds = []
        total_weight = 0
        
        for book in bookmakers:
            if not isinstance(book, dict):
                continue
                
            book_key = book.get("key", "").lower()
            if book_key not in self.SHARP_BOOKS:
                continue
            
            # Find the relevant market
            for market in book.get("markets", []):
                if not isinstance(market, dict) or market.get("key") != market_key:
                    continue
                
                # Find over/under outcomes for this player
                outcomes = []
                for outcome in market.get("outcomes", []):
                    if not isinstance(outcome, dict):
                        continue
                    if str(outcome.get("description", "")) == player:
                        outcomes.append(outcome)
                
                if len(outcomes) != 2:
                    continue
                
                try:
                    # Get over/under outcomes
                    over_outcome = next(o for o in outcomes if o.get("name", "").startswith("Over"))
                    under_outcome = next(o for o in outcomes if o.get("name", "").startswith("Under"))
                    
                    # Extract line and odds
                    line, over_odds = self._extract_line_and_odds(over_outcome)
                    _, under_odds = self._extract_line_and_odds(under_outcome)
                    
                    # Convert decimal odds to American
                    over_american = round((over_odds - 1) * 100) if over_odds >= 2 else round(-100 / (over_odds - 1))
                    under_american = round((under_odds - 1) * 100) if under_odds >= 2 else round(-100 / (under_odds - 1))
                    
                    # Assign weights (Pinnacle higher weight)
                    weight = 0.6 if book_key == "pinnacle" else 0.4
                    total_weight += weight
                    
                    sharp_odds.append(BookOdds(
                        book_name=str(book.get("title", "")),
                        over_odds=over_american,
                        under_odds=under_american,
                        weight=weight
                    ))
                except (StopIteration, KeyError, ValueError) as e:
                    print(f"Error processing odds for {player} from {book_key}: {str(e)}")
                    continue
        
        # Normalize weights if we don't have all sharp books
        if sharp_odds and total_weight != 1:
            for odds in sharp_odds:
                odds.weight = odds.weight / total_weight
        
        return sharp_odds
    
    def _get_player_team(self, event: Event, player_name: str, market_data: Dict[str, Any]) -> str:
        """Determine player's team based on market data"""
        # Look for team indicator in market data
        for book in market_data.get("bookmakers", []):
            if not isinstance(book, dict):
                continue
            for market in book.get("markets", []):
                if not isinstance(market, dict):
                    continue
                for outcome in market.get("outcomes", []):
                    if not isinstance(outcome, dict):
                        continue
                    if str(outcome.get("description", "")) == player_name:
                        team = str(outcome.get("team", ""))
                        if team:
                            return team
        
        # Default to home team if no team info found
        return event.home_team
    
    def get_player_props(self, 
                        prop_types: Optional[Set[str]] = None,
                        player_names: Optional[Set[str]] = None) -> List[PropMarket]:
        """
        Fetch player props odds from ODDS API.
        
        Args:
            prop_types: Set of prop types to fetch (points, rebounds, assists, etc.)
            player_names: Optional set of player names to filter for
        
        Returns:
            List of PropMarket objects with sharp book odds
        """
        if not prop_types:
            prop_types = set(self._parse_prop_type(m) for m in self.MARKETS)
        
        markets = []
        
        try:
            # Get today's games
            events = self.get_events()
            print(f"Found {len(events)} NBA games")
            
            # Fetch odds for each game
            for event in events:
                print(f"\nFetching odds for {event.away_team} @ {event.home_team}")
                
                params = {
                    "regions": ",".join(self.REGIONS),
                    "markets": ",".join(
                        m for m in self.MARKETS 
                        if self._parse_prop_type(m) in prop_types
                    ),
                    "oddsFormat": "decimal"  # Request decimal odds directly
                }
                
                try:
                    response = self._make_request(
                        f"{self.SPORT_KEY}/events/{event.id}/odds",
                        params
                    )
                    
                    # Track processed players to avoid duplicates
                    processed = set()
                    
                    # Process each market type
                    for market_key in self.MARKETS:
                        if self._parse_prop_type(market_key) not in prop_types:
                            continue
                            
                        # Find all players with odds for this market
                        players = set()
                        for book in response.get("bookmakers", []):
                            if not isinstance(book, dict):
                                continue
                            for market in book.get("markets", []):
                                if not isinstance(market, dict) or market.get("key") != market_key:
                                    continue
                                for outcome in market.get("outcomes", []):
                                    if isinstance(outcome, dict):
                                        player = self._normalize_player_name(
                                            str(outcome.get("description", ""))
                                        )
                                        if player and (not player_names or player in player_names):
                                            players.add(player)
                        
                        # Get odds for each player
                        for player in players:
                            if player in processed:
                                continue
                                
                            # Get sharp book odds
                            sharp_odds = self._get_sharp_odds(
                                response.get("bookmakers", []),
                                market_key,
                                player
                            )
                            
                            if sharp_odds:
                                # Get line from first sharp book
                                line = None
                                for book in response.get("bookmakers", []):
                                    if not isinstance(book, dict):
                                        continue
                                    if book.get("key", "").lower() in self.SHARP_BOOKS:
                                        for market in book.get("markets", []):
                                            if not isinstance(market, dict) or market.get("key") != market_key:
                                                continue
                                            for outcome in market.get("outcomes", []):
                                                if isinstance(outcome, dict) and str(outcome.get("description", "")) == player:
                                                    try:
                                                        line, _ = self._extract_line_and_odds(outcome)
                                                        break
                                                    except ValueError:
                                                        continue
                                            if line is not None:
                                                break
                                    if line is not None:
                                        break
                                
                                if line is not None:
                                    # Get player's team
                                    team = self._get_player_team(event, player, response)
                                    
                                    markets.append(PropMarket(
                                        player_name=player,
                                        team=team,
                                        prop_type=self._parse_prop_type(market_key),
                                        line=line,
                                        sharp_odds=sharp_odds,
                                        timestamp=datetime.now(UTC).isoformat()
                                    ))
                                    processed.add(player)
                
                except Exception as e:
                    print(f"Error fetching odds for event {event.id}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error fetching events: {str(e)}")
        
        return markets
    
    def get_remaining_requests(self) -> Dict[str, int]:
        """Get API usage information"""
        try:
            response = self.session.get(
                "https://api.the-odds-api.com/v4/sports",
                params={"apiKey": self.api_key}
            )
            return {
                "requests_remaining": int(
                    response.headers.get("x-requests-remaining", 0)
                ),
                "requests_used": int(response.headers.get("x-requests-used", 0))
            }
        except Exception as e:
            print(f"Error checking API usage: {str(e)}")
            return {"requests_remaining": 0, "requests_used": 0}


def main() -> None:
    """Example usage of OddsAPIClient"""
    client = OddsAPIClient()
    
    # Check API usage
    usage = client.get_remaining_requests()
    print(f"\nAPI Usage:")
    print(f"Requests remaining: {usage['requests_remaining']}")
    print(f"Requests used: {usage['requests_used']}")
    
    # Get today's games
    events = client.get_events()
    print(f"\nFound {len(events)} NBA games today:")
    for event in events:
        print(f"{event.away_team} @ {event.home_team}")
        print(f"Start time: {event.commence_time}\n")
    
    # Fetch props for specific players and types
    props = client.get_player_props(
        prop_types={"points", "rebounds", "assists"},
        player_names=None  # Get all players
    )
    
    print(f"\nFetched {len(props)} player props:")
    for prop in props:
        print(f"\n{prop.player_name} ({prop.team}) - {prop.prop_type}")
        print(f"Line: {prop.line}")
        for odds in prop.sharp_odds:
            print(
                f"{odds.book_name}: Over {odds.over_odds} / "
                f"Under {odds.under_odds} (weight: {odds.weight})"
            )


if __name__ == "__main__":
    main()