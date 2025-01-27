"""Find edges between model predictions and sportsbook odds."""

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, TypedDict, cast
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
from pathlib import Path
from scipy import stats
from src.data.db_config import get_db_connection
from src.models.enhanced_props_model import EnhancedPropsModel
from src.models.ensemble_scorer_model import HighScorerEnsemble
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class EventDict(TypedDict):
    id: str
    away_team: str
    home_team: str

class OutcomeDict(TypedDict):
    name: str
    description: str
    point: float
    price: int

class MarketDict(TypedDict):
    key: str
    outcomes: List[OutcomeDict]

class BookmakerDict(TypedDict):
    key: str
    markets: List[MarketDict]

class OddsDict(TypedDict):
    bookmakers: List[BookmakerDict]

class PropEdgeFinder:
    """Find edges in player props using model predictions."""
    
    def __init__(
        self,
        api_key: str,
        bookmakers: List[str] = ['pinnacle', 'betonlineag', 'betmgm', 'betrivers'],
        min_edge: float = 5.0,  # Minimum edge percentage to report
        force_retrain: bool = False  # Force model retraining
    ) -> None:
        """Initialize edge finder.
        
        Args:
            api_key: Odds API key
            bookmakers: List of bookmakers to check
            min_edge: Minimum edge percentage to report
        """
        self.api_key = api_key
        self.bookmakers = bookmakers
        self.min_edge = min_edge
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # Initialize model manager with force_retrain option
        self.model_manager = ModelManager()
        
        # Initialize models
        logger.info("Initializing high scorer model...")
        self.points_model = HighScorerEnsemble()
        self.points_model.train()  # Always train high scorer model since it's fast
        
        logger.info("Initializing prop models...")
        self.props_models = {}
        for prop_type in ['points', 'rebounds', 'assists', 'threes']:
            logger.info(f"Loading/training {prop_type} model...")
            self.props_models[prop_type] = self.model_manager.get_model(
                prop_type,
                force_retrain=force_retrain
            )
        
    def _get_events(self) -> List[EventDict]:
        """Get today's NBA events.
        
        Returns:
            List of event dictionaries
        """
        url = f"{self.base_url}/sports/basketball_nba/events"
        params = {'apiKey': self.api_key}
        
        logger.info(f"Getting events from {url}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        events = response.json()
        if not isinstance(events, list):
            raise ValueError("Expected list of events")
        
        logger.info(f"Found {len(events)} events")
        return [
            EventDict(
                id=event['id'],
                away_team=event['away_team'],
                home_team=event['home_team']
            )
            for event in events
        ]
    
    def _get_event_odds(
        self,
        event_id: str,
        markets: List[str]
    ) -> OddsDict:
        """Get odds for an event.
        
        Args:
            event_id: Event ID
            markets: List of markets to fetch
            
        Returns:
            Dictionary with odds data
        """
        url = f"{self.base_url}/sports/basketball_nba/events/{event_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }
        
        logger.info(f"Getting odds from {url}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        odds = response.json()
        logger.info(f"Found {len(odds.get('bookmakers', []))} bookmakers")
        return cast(OddsDict, odds)
    
    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal.
        
        Args:
            american_odds: American odds format
            
        Returns:
            Decimal odds
        """
        if american_odds > 0:
            return (american_odds / 100.0) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def _calculate_implied_probability(self, decimal_odds: float) -> float:
        """Calculate implied probability from decimal odds.
        
        Args:
            decimal_odds: Decimal odds format
            
        Returns:
            Implied probability (0-1)
        """
        return 1 / decimal_odds
    
    def _get_player_id(self, player_name: str) -> Optional[int]:
        """Get NBA API player ID from name.
        
        Args:
            player_name: Player's full name
            
        Returns:
            Player ID if found, None otherwise
        """
        conn = get_db_connection(use_motherduck=True)
        try:
            query = f"""
                SELECT DISTINCT player_id
                FROM player_stats
                WHERE player_name = '{player_name}'
                ORDER BY game_date DESC
                LIMIT 1
            """
            
            result = conn.execute(query).fetchone()
            logger.info(f"Found player ID {result[0] if result else None} for {player_name}")
            return int(result[0]) if result else None
            
        finally:
            conn.close()
    
    def find_edges(self) -> pd.DataFrame:
        """Find edges between model predictions and odds.
        
        Returns:
            DataFrame with edge opportunities
        """
        edges = []
        
        # Get today's events
        try:
            events = self._get_events()
        except Exception as e:
            logger.error(f"Error getting events: {str(e)}")
            return pd.DataFrame()
        
        for event in events:
            logger.info(f"Processing {event['away_team']} @ {event['home_team']}")
            
            # Get odds for all prop markets
            markets = [
                'player_points',
                'player_rebounds',
                'player_assists',
                'player_threes'
            ]
            
            try:
                odds = self._get_event_odds(event['id'], markets)
            except Exception as e:
                logger.error(f"Error getting odds for event {event['id']}: {str(e)}")
                continue
            
            # Process each bookmaker
            for bookmaker in odds['bookmakers']:
                if bookmaker['key'] not in self.bookmakers:
                    continue
                
                # Process each market
                for market in bookmaker['markets']:
                    prop_type = market['key'].replace('player_', '')
                    
                    # Process each player line
                    for outcome in market['outcomes']:
                        if outcome['name'] not in ['Over', 'Under']:
                            continue
                            
                        player_name = outcome['description']
                        line = float(outcome['point'])
                        price = int(outcome['price'])
                        
                        # Get player ID
                        player_id = self._get_player_id(player_name)
                        if not player_id:
                            logger.warning(f"Could not find player ID for {player_name}")
                            continue
                        
                        # Get model prediction
                        try:
                            if prop_type == 'points' and line >= 17.8:
                                prediction = self.points_model.predict(player_id)
                                if not isinstance(prediction, dict) or 'predicted_points' not in prediction:
                                    raise ValueError("Invalid prediction format")
                                pred_value = prediction['predicted_points']
                            else:
                                prediction = self.props_models[prop_type].predict_player(
                                    player_id
                                )
                                pred_value = prediction['predicted_value']
                            
                            # Calculate decimal odds and implied probability
                            decimal_odds = self._american_to_decimal(price)
                            implied_prob = self._calculate_implied_probability(decimal_odds)
                            
                            # Sanity check for predictions too far from line
                            if abs(pred_value - line) > line:
                                logger.warning(f"Prediction {pred_value} too far from line {line} for {player_name}")
                                continue  # Skip unrealistic predictions
                            
                            # Calculate fair probability using normal distribution
                            if prop_type == 'points':
                                std_dev = max(4.5, pred_value * 0.25)  # 25% of prediction
                            elif prop_type == 'rebounds':
                                std_dev = max(2.0, pred_value * 0.3)  # 30% of prediction
                            elif prop_type == 'assists':
                                std_dev = max(1.5, pred_value * 0.3)  # 30% of prediction
                            else:  # threes
                                std_dev = max(1.0, pred_value * 0.35)  # 35% of prediction
                            
                            # Calculate model probability
                            if outcome['name'] == 'Over':
                                model_prob = float(1 - stats.norm.cdf(line, pred_value, std_dev))
                            else:  # Under
                                model_prob = float(stats.norm.cdf(line, pred_value, std_dev))
                            
                            # Calculate edge as difference between prediction and line
                            edge = pred_value - line if outcome['name'] == 'Over' else line - pred_value
                            
                            # Convert to percentage
                            edge_pct = (edge / line) * 100
                            
                            # Apply confidence penalty for predictions far from line
                            confidence_penalty = min(abs(edge) / line, 0.5)
                            edge_pct = edge_pct * (1 - confidence_penalty)
                            
                            if abs(edge_pct) >= self.min_edge:
                                logger.info(f"Found edge: {player_name} {prop_type} {outcome['name']} {line} ({edge_pct:.1f}%)")
                                edges.append({
                                    'game': f"{event['away_team']} @ {event['home_team']}",
                                    'player': player_name,
                                    'prop_type': prop_type,
                                    'line': line,
                                    'prediction': pred_value,
                                    'odds': price,
                                    'book': bookmaker['key'],
                                    'bet_type': outcome['name'],
                                    'edge': edge_pct,
                                    'model_prob': model_prob,
                                    'implied_prob': implied_prob
                                })
                            
                        except Exception as e:
                            logger.error(
                                f"Error processing {player_name} {prop_type}: {str(e)}"
                            )
                            continue
        
        # Convert to DataFrame and sort
        if not edges:
            logger.info("No edges found")
            return pd.DataFrame()
            
        df = pd.DataFrame(edges)
        df = df.sort_values('edge', ascending=False)
        
        return df
    
    def generate_report(self) -> None:
        """Generate edge report and save to CSV."""
        logger.info("Finding prop betting edges...")
        
        # Get edges
        edges_df = self.find_edges()
        
        if edges_df.empty:
            logger.info("No edges found meeting criteria")
            return
        
        # Split into over/under
        overs = edges_df[edges_df['bet_type'] == 'Over']
        unders = edges_df[edges_df['bet_type'] == 'Under']
        
        # Get top 5 of each
        top_overs = overs.head(5)
        top_unders = unders.head(5)
        
        # Combine and save
        report = pd.concat([top_overs, top_unders])
        
        # Create reports directory if needed
        reports_dir = Path('data/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Format datetime for filename
        now = datetime.now().strftime('%Y%m%d_%H%M')
        filename = reports_dir / f'prop_edges_{now}.csv'
        
        # Save report
        report.to_csv(filename, index=False)
        
        logger.info(f"Edge report saved to {filename}")
        
        # Log summary
        logger.info("\nTop Overs:")
        for _, row in top_overs.iterrows():
            logger.info(
                f"{row['player']} {row['prop_type']} Over {row['line']} "
                f"({row['odds']}) - {row['edge']:.1f}% edge"
            )
        
        logger.info("\nTop Unders:")
        for _, row in top_unders.iterrows():
            logger.info(
                f"{row['player']} {row['prop_type']} Under {row['line']} "
                f"({row['odds']}) - {row['edge']:.1f}% edge"
            )

def main() -> int:
    """Find prop betting edges and generate report.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        import argparse
        
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Find prop betting edges')
        parser.add_argument(
            '--force-retrain',
            action='store_true',
            help='Force model retraining'
        )
        args = parser.parse_args()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Get API key from environment
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            raise ValueError("ODDS_API_KEY environment variable not set")
        
        # Create edge finder with force_retrain option
        finder = PropEdgeFinder(
            api_key=api_key,
            force_retrain=args.force_retrain
        )
        
        # Generate report
        finder.generate_report()
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
