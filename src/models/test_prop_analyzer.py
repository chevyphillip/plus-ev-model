"""Test script for prop bet analyzer."""

import logging
from typing import Dict, Any, List, Set, Optional, Callable
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from src.models.prop_bet_analyzer import PropBetAnalyzer
from src.data.odds_api import PropMarket, BookOdds, OddsAPIClient

logger = logging.getLogger(__name__)

def create_mock_markets() -> List[PropMarket]:
    """Create mock prop markets for testing."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Create mock prop market for LeBron
    lebron_points = PropMarket(
        player_name="LeBron James",
        team="Los Angeles Lakers",
        prop_type="points",
        line=25.5,
        sharp_odds=[
            BookOdds(
                book_name="Pinnacle",
                over_odds=-110,
                under_odds=-110,
                weight=0.6
            ),
            BookOdds(
                book_name="BetOnline",
                over_odds=-108,
                under_odds=-112,
                weight=0.4
            )
        ],
        timestamp=timestamp
    )
    
    lebron_assists = PropMarket(
        player_name="LeBron James",
        team="Los Angeles Lakers",
        prop_type="assists",
        line=7.5,
        sharp_odds=[
            BookOdds(
                book_name="Pinnacle",
                over_odds=-115,
                under_odds=-105,
                weight=0.6
            ),
            BookOdds(
                book_name="BetOnline",
                over_odds=-112,
                under_odds=-108,
                weight=0.4
            )
        ],
        timestamp=timestamp
    )
    
    return [lebron_points, lebron_assists]

class MockPlayerPropsModel:
    """Mock player props model for testing."""
    
    def predict_player(self, player_id: int) -> Dict[str, Any]:
        """Return mock prediction."""
        if player_id == 2544:  # LeBron
            return {
                'player_name': 'LeBron James',
                'predicted_value': 28.5,
                'recent_average': 27.2,
                'last_5_games': [25, 28, 30, 26, 27],
                'prop_type': 'points'
            }
        raise ValueError(f"No mock data for player_id {player_id}")

class MockOddsAPIClient(OddsAPIClient):
    """Mock OddsAPI client for testing."""
    
    def __init__(self) -> None:
        """Initialize mock client."""
        with patch('requests.Session') as mock_session:
            mock_session.return_value.get.return_value = MagicMock(
                headers={"x-requests-remaining": "1000", "x-requests-used": "0"}
            )
            super().__init__(api_key="dummy_key")
    
    def get_player_props(
        self,
        prop_types: Optional[Set[str]] = None,
        player_names: Optional[Set[str]] = None
    ) -> List[PropMarket]:
        """Return mock prop markets."""
        return create_mock_markets()
    
    def get_remaining_requests(self) -> Dict[str, int]:
        """Mock API usage info."""
        return {
            "requests_remaining": 1000,
            "requests_used": 0
        }

def test_with_mock_data() -> None:
    """Test analyzer with mock prop market data."""
    try:
        # Initialize analyzer with mock clients
        analyzer = PropBetAnalyzer(
            min_edge=0.05,
            kelly_fraction=0.25,
            confidence_threshold=0.6
        )
        analyzer.odds_client = MockOddsAPIClient()
        analyzer.points_model = MockPlayerPropsModel()
        analyzer.assists_model = MockPlayerPropsModel()
        analyzer.rebounds_model = MockPlayerPropsModel()
        
        # Get mock markets
        markets = create_mock_markets()
        lebron_points, lebron_assists = markets
        
        # Test individual prop analysis
        lebron_id = 2544
        points_analysis = analyzer.analyze_prop(lebron_points, lebron_id)
        assists_analysis = analyzer.analyze_prop(lebron_assists, lebron_id)
        
        # Print results
        print("\nLeBron James Props Analysis:")
        
        print("\nPoints:")
        print(f"Line: {points_analysis['line']}")
        print(f"Model prediction: {points_analysis['predicted_value']:.1f}")
        print(f"Recent average: {points_analysis['recent_average']:.1f}")
        print(f"Last 5 games: {points_analysis['last_5_games']}")
        
        for side in ['over', 'under']:
            if points_analysis[side]['edge'] >= analyzer.min_edge:
                print(f"\n{side.upper()}:")
                print(f"Edge: {points_analysis[side]['edge']:.1%}")
                print(f"EV: ${points_analysis[side]['ev_dollars']:.2f} ({points_analysis[side]['ev_percent']:.1%})")
                print(f"Kelly bet: ${points_analysis[side]['kelly_bet']:.2f}")
        
        print("\nAssists:")
        print(f"Line: {assists_analysis['line']}")
        print(f"Model prediction: {assists_analysis['predicted_value']:.1f}")
        print(f"Recent average: {assists_analysis['recent_average']:.1f}")
        print(f"Last 5 games: {assists_analysis['last_5_games']}")
        
        for side in ['over', 'under']:
            if assists_analysis[side]['edge'] >= analyzer.min_edge:
                print(f"\n{side.upper()}:")
                print(f"Edge: {assists_analysis[side]['edge']:.1%}")
                print(f"EV: ${assists_analysis[side]['ev_dollars']:.2f} ({assists_analysis[side]['ev_percent']:.1%})")
                print(f"Kelly bet: ${assists_analysis[side]['kelly_bet']:.2f}")
        
        # Test edge finding across multiple props
        player_ids = {"LeBron James": 2544}
        edges = analyzer.find_edges(
            prop_types={'points', 'assists'},
            player_ids=player_ids
        )
        
        print(f"\nFound {len(edges)} props with edges")
        
    except Exception as e:
        logger.error(f"Error testing prop analyzer: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_with_mock_data()
