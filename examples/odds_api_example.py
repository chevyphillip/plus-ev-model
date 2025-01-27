"""Example usage of OddsAPIClient."""

import os
import logging
from typing import cast, Dict, Any, List
from tabulate import tabulate
from datetime import datetime
from src.data.odds_api_client import (
    OddsAPIClient,
    GameDict,
    PropsDict,
    BookmakerDict,
    MarketDict,
    OutcomeDict
)

logger = logging.getLogger(__name__)

def format_odds(odds: int) -> str:
    """Format American odds for display."""
    if odds > 0:
        return f"+{odds}"
    return str(odds)

def display_props(
    props: PropsDict,
    prop_type: str,
    min_line: float = 0.0
) -> None:
    """Display props in a table format.
    
    Args:
        props: Props data from API
        prop_type: Type of prop to display
        min_line: Minimum line to display
    """
    if not props or 'bookmakers' not in props:
        logger.info("No props available")
        return
        
    # Collect all props
    rows: List[List[Any]] = []
    for book in props['bookmakers']:
        for market in book['markets']:
            if market['key'] == f'player_{prop_type}':
                for outcome in market['outcomes']:
                    if float(outcome['point']) >= min_line:
                        rows.append([
                            outcome['description'],
                            outcome['point'],
                            format_odds(int(outcome['price'])),
                            outcome['name'],
                            book['title']
                        ])
    
    if not rows:
        logger.info(f"No {prop_type} props found")
        return
        
    # Sort by player name and line
    rows.sort(key=lambda x: (x[0], x[1]))
    
    # Display table
    headers = ['Player', 'Line', 'Odds', 'Side', 'Book']
    print(tabulate(rows, headers=headers, tablefmt='grid'))

def main() -> None:
    """Example usage of OddsAPIClient."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize client
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            raise ValueError("ODDS_API_KEY environment variable not set")
            
        client = OddsAPIClient(api_key=api_key)
        
        # Get today's games
        games = client.get_nba_games()
        logger.info(f"Found {len(games)} NBA games")
        
        # Display each game's props
        for game in games:
            game = cast(GameDict, game)
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print("=" * 50)
            
            # Get all prop types
            props = client.get_player_props(game['id'])
            
            # Display points props
            print("\nPoints Props (15+ only):")
            display_props(props, 'points', min_line=15.0)
            
            # Display assists props
            print("\nAssists Props (5+ only):")
            display_props(props, 'assists', min_line=5.0)
            
            # Display rebounds props
            print("\nRebounds Props (5+ only):")
            display_props(props, 'rebounds', min_line=5.0)
            
            # Display threes props
            print("\nThrees Props (2+ only):")
            display_props(props, 'threes', min_line=2.0)
            
            # Get best odds example
            if props and 'bookmakers' in props:
                book = cast(BookmakerDict, props['bookmakers'][0])
                market = cast(MarketDict, book['markets'][0])
                outcome = cast(OutcomeDict, market['outcomes'][0])
                player = outcome['description']
                
                print(f"\nBest Odds for {player}:")
                for prop in ['points', 'assists', 'rebounds', 'threes']:
                    best = client.get_best_odds(game['id'], player, prop)
                    if best:
                        print(f"\n{prop.title()}:")
                        for side, odds in best.items():
                            print(
                                f"{side.title()}: {odds['line']} "
                                f"({format_odds(odds['odds'])} @ {odds['book']})"
                            )
            
            input("\nPress Enter to continue...")
            
    except RuntimeError as e:
        logger.error(f"API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    main()
