"""Fix fg3m (3-pointers made) calculation in player_stats table."""

import logging
import sys
from src.data.db_config import get_db_connection
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def convert_date(date_str: str) -> str:
    """Convert NBA API date format to database format.
    
    Args:
        date_str: Date string in format 'MMM DD YYYY'
        
    Returns:
        Date string in format 'YYYY-MM-DD'
    """
    try:
        # Parse NBA API date format
        dt = datetime.strptime(date_str, '%b %d %Y')
        # Convert to database format
        return dt.strftime('%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Error converting date {date_str}: {str(e)}")
        return None

def fix_fg3m_calculation() -> None:
    """Fix fg3m using actual NBA API data."""
    conn = get_db_connection(use_motherduck=False)
    try:
        logger.info("Fixing fg3m calculation...")
        
        # Get all player IDs
        player_ids = conn.execute("""
            SELECT DISTINCT player_id 
            FROM player_stats 
            ORDER BY player_id
        """).fetchall()
        
        for (player_id,) in player_ids:
            logger.info(f"Processing player {player_id}...")
            
            try:
                # Get player game log
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=SeasonAll.all
                )
                
                # Process each game
                for game in gamelog.get_data_frames()[0].to_dict('records'):
                    game_date = convert_date(game['GAME_DATE'])
                    if not game_date:
                        continue
                        
                    fg3m = game['FG3M']
                    
                    # Update fg3m for this game
                    conn.execute("""
                        UPDATE player_stats
                        SET fg3m = ?
                        WHERE player_id = ?
                        AND game_date = ?
                    """, [fg3m, player_id, game_date])
                
                # Sleep to avoid API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing player {player_id}: {str(e)}")
                continue
        
        logger.info("Fixed fg3m calculation")
        
    finally:
        conn.close()

def main() -> int:
    """Fix fg3m calculation.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        fix_fg3m_calculation()
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
