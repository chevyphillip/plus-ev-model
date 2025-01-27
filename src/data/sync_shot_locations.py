"""Sync shot location data from NBA API."""

import logging
import sys
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import duckdb
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.library.parameters import Season
from nba_api.stats.static import players
from src.data.db_config import get_db_connection
from requests.exceptions import Timeout, RequestException
from nba_api.stats.library.http import NBAStatsHTTP

# Configure NBA API
NBAStatsHTTP.HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

logger = logging.getLogger(__name__)

class ShotLocationSync:
    """Sync and process shot location data."""
    
    def __init__(
        self,
        db_path: str = 'data/nba_stats.duckdb'
    ) -> None:
        """Initialize shot location sync.
        
        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
        
    def _create_shot_locations_table(self) -> None:
        """Create shot locations table if it doesn't exist."""
        conn = get_db_connection(use_motherduck=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shot_locations (
                    player_id INTEGER,
                    game_date DATE,
                    shot_zone_basic VARCHAR,
                    shot_zone_area VARCHAR,
                    shot_zone_range VARCHAR,
                    shot_distance FLOAT,
                    loc_x FLOAT,
                    loc_y FLOAT,
                    shot_made INTEGER,
                    shot_type VARCHAR,
                    shot_attempted INTEGER,
                    game_id VARCHAR,
                    period INTEGER,
                    minutes_remaining INTEGER,
                    seconds_remaining INTEGER,
                    PRIMARY KEY (player_id, game_id, loc_x, loc_y)
                )
            """)
            
        finally:
            conn.close()
    
    def _get_existing_games(self) -> List[str]:
        """Get list of games already in database.
        
        Returns:
            List of game IDs
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            result = conn.execute("""
                SELECT DISTINCT game_id 
                FROM shot_locations
            """).fetchdf()
            
            return result['game_id'].tolist() if not result.empty else []
            
        finally:
            conn.close()
    
    def _get_active_players(self) -> List[int]:
        """Get list of active players.
        
        Returns:
            List of player IDs
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            result = conn.execute("""
                SELECT DISTINCT player_id
                FROM player_stats
                WHERE game_date >= CURRENT_DATE - INTERVAL '1 year'
            """).fetchdf()
            
            return result['player_id'].tolist()
            
        finally:
            conn.close()
    
    def _fetch_shot_data_with_retry(
        self,
        player_id: int,
        season: str,
        max_retries: int = 3,
        base_delay: float = 2.0
    ) -> Optional[pd.DataFrame]:
        """Fetch shot data for a player.
        
        Args:
            player_id: NBA player ID
            season: Season string (e.g., '2023-24')
            
        Returns:
            DataFrame with shot data or None if no data
        """
        for attempt in range(max_retries):
            try:
                # Add delay between requests
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(delay)
                
                shot_data = shotchartdetail.ShotChartDetail(
                    player_id=player_id,
                    team_id=0,
                    season_nullable=season,
                    context_measure_simple='FGA',
                    timeout=60  # Increase timeout
                ).get_data_frames()[0]
            
                if shot_data.empty:
                    return None
                
                # Clean and transform data
                shot_data['GAME_DATE'] = pd.to_datetime(
                    shot_data['GAME_DATE']
                ).dt.date
                
                # Rename columns
                columns = {
                    'PLAYER_ID': 'player_id',
                    'GAME_DATE': 'game_date',
                    'SHOT_ZONE_BASIC': 'shot_zone_basic',
                    'SHOT_ZONE_AREA': 'shot_zone_area',
                    'SHOT_ZONE_RANGE': 'shot_zone_range',
                    'SHOT_DISTANCE': 'shot_distance',
                    'LOC_X': 'loc_x',
                    'LOC_Y': 'loc_y',
                    'SHOT_MADE_FLAG': 'shot_made',
                    'SHOT_TYPE': 'shot_type',
                    'SHOT_ATTEMPTED_FLAG': 'shot_attempted',
                    'GAME_ID': 'game_id',
                    'PERIOD': 'period',
                    'MINUTES_REMAINING': 'minutes_remaining',
                    'SECONDS_REMAINING': 'seconds_remaining'
                }
                
                shot_data = shot_data.rename(columns=columns)
                
                # Select relevant columns
                return shot_data[list(columns.values())]
                
            except (Timeout, RequestException) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Error fetching shot data for player {player_id} "
                        f"after {max_retries} attempts: {str(e)}"
                    )
                    return None
                logger.warning(
                    f"Attempt {attempt + 1} failed for player {player_id}: {str(e)}"
                )
                continue
                
            except Exception as e:
                logger.error(f"Error fetching shot data for player {player_id}: {str(e)}")
                return None
    
    def sync_shot_data(
        self,
        days_back: int = 7
    ) -> Dict[str, int]:
        """Sync shot location data.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary with sync statistics
        """
        # Create table if needed
        self._create_shot_locations_table()
        
        # Get existing games
        existing_games = set(self._get_existing_games())
        
        # Get active players
        players = self._get_active_players()
        
        # Track statistics
        stats = {
            'players_processed': 0,
            'shots_added': 0,
            'errors': 0
        }
        
        # Current season
        current_season = f"{datetime.now().year-1}-{str(datetime.now().year)[2:]}"
        
        conn = get_db_connection(use_motherduck=False)
        try:
            # Add initial delay
            time.sleep(1.0)
            
            for i, player_id in enumerate(players):
                # Add delay between players
                if i > 0:
                    time.sleep(1.5)  # 1.5 seconds between players
                try:
                    # Get shot data with retry
                    shot_data = self._fetch_shot_data_with_retry(
                        player_id,
                        current_season
                    )
                    
                    if shot_data is None:
                        continue
                    
                    # Filter to recent games
                    cutoff_date = datetime.now().date() - timedelta(days=days_back)
                    shot_data = shot_data[
                        (shot_data['game_date'] >= cutoff_date) &
                        (~shot_data['game_id'].isin(existing_games))
                    ]
                    
                    if shot_data.empty:
                        continue
                    
                    # Convert DataFrame columns to match table schema
                    shot_data = shot_data.astype({
                        'player_id': 'int32',
                        'shot_made': 'int32',
                        'shot_attempted': 'int32',
                        'period': 'int32',
                        'minutes_remaining': 'int32',
                        'seconds_remaining': 'int32',
                        'shot_distance': 'float64',
                        'loc_x': 'float64',
                        'loc_y': 'float64'
                    })
                    
                    # Insert data row by row
                    for _, row in shot_data.iterrows():
                        conn.execute("""
                            INSERT INTO shot_locations (
                                player_id, game_date, shot_zone_basic,
                                shot_zone_area, shot_zone_range, shot_distance,
                                loc_x, loc_y, shot_made, shot_type,
                                shot_attempted, game_id, period,
                                minutes_remaining, seconds_remaining
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT DO NOTHING
                        """, [
                            row['player_id'], row['game_date'], row['shot_zone_basic'],
                            row['shot_zone_area'], row['shot_zone_range'], row['shot_distance'],
                            row['loc_x'], row['loc_y'], row['shot_made'], row['shot_type'],
                            row['shot_attempted'], row['game_id'], row['period'],
                            row['minutes_remaining'], row['seconds_remaining']
                        ])
                    
                    stats['shots_added'] += len(shot_data)
                    stats['players_processed'] += 1
                    
                except Exception as e:
                    logger.error(
                        f"Error processing player {player_id}: {str(e)}"
                    )
                    stats['errors'] += 1
            
            return stats
            
        finally:
            conn.close()
    
    def calculate_shot_zone_stats(self) -> None:
        """Calculate shot zone statistics for each player."""
        conn = get_db_connection(use_motherduck=False)
        try:
            # Create stats table if needed
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_shot_stats (
                    player_id INTEGER,
                    game_date DATE,
                    
                    -- Three point zones
                    corner_3_attempts INTEGER,
                    corner_3_made INTEGER,
                    above_break_3_attempts INTEGER,
                    above_break_3_made INTEGER,
                    
                    -- Shot clock
                    early_shot_clock_3_attempts INTEGER,
                    early_shot_clock_3_made INTEGER,
                    late_shot_clock_3_attempts INTEGER,
                    late_shot_clock_3_made INTEGER,
                    
                    -- Shot types
                    catch_shoot_3_attempts INTEGER,
                    catch_shoot_3_made INTEGER,
                    pullup_3_attempts INTEGER,
                    pullup_3_made INTEGER,
                    
                    -- Rolling averages
                    corner_3_pct_5game FLOAT,
                    above_break_3_pct_5game FLOAT,
                    catch_shoot_3_pct_5game FLOAT,
                    pullup_3_pct_5game FLOAT,
                    
                    PRIMARY KEY (player_id, game_date)
                )
            """)
            
            # Calculate daily stats
            conn.execute("""
                INSERT INTO player_shot_stats
                WITH daily_stats AS (
                    SELECT
                        player_id,
                        game_date,
                        
                        -- Corner threes
                        COUNT(*) FILTER (
                            WHERE shot_zone_basic = 'Corner 3'
                        ) as corner_3_attempts,
                        SUM(shot_made) FILTER (
                            WHERE shot_zone_basic = 'Corner 3'
                        ) as corner_3_made,
                        
                        -- Above break threes
                        COUNT(*) FILTER (
                            WHERE shot_zone_basic = 'Above Break 3'
                        ) as above_break_3_attempts,
                        SUM(shot_made) FILTER (
                            WHERE shot_zone_basic = 'Above Break 3'
                        ) as above_break_3_made,
                        
                        -- Shot clock
                        COUNT(*) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining >= 18
                        ) as early_shot_clock_3_attempts,
                        SUM(shot_made) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining >= 18
                        ) as early_shot_clock_3_made,
                        COUNT(*) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining <= 4
                        ) as late_shot_clock_3_attempts,
                        SUM(shot_made) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining <= 4
                        ) as late_shot_clock_3_made,
                        
                        -- Shot types (approximated by location and timing)
                        COUNT(*) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining >= 16
                        ) as catch_shoot_3_attempts,
                        SUM(shot_made) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining >= 16
                        ) as catch_shoot_3_made,
                        COUNT(*) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining < 16
                        ) as pullup_3_attempts,
                        SUM(shot_made) FILTER (
                            WHERE shot_zone_basic LIKE '%3'
                            AND minutes_remaining < 16
                        ) as pullup_3_made
                        
                    FROM shot_locations
                    GROUP BY player_id, game_date
                ),
                rolling_stats AS (
                    SELECT
                        *,
                        AVG(CAST(corner_3_made AS FLOAT) / 
                            NULLIF(corner_3_attempts, 0)) OVER (
                            PARTITION BY player_id
                            ORDER BY game_date
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) as corner_3_pct_5game,
                        AVG(CAST(above_break_3_made AS FLOAT) / 
                            NULLIF(above_break_3_attempts, 0)) OVER (
                            PARTITION BY player_id
                            ORDER BY game_date
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) as above_break_3_pct_5game,
                        AVG(CAST(catch_shoot_3_made AS FLOAT) / 
                            NULLIF(catch_shoot_3_attempts, 0)) OVER (
                            PARTITION BY player_id
                            ORDER BY game_date
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) as catch_shoot_3_pct_5game,
                        AVG(CAST(pullup_3_made AS FLOAT) / 
                            NULLIF(pullup_3_attempts, 0)) OVER (
                            PARTITION BY player_id
                            ORDER BY game_date
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) as pullup_3_pct_5game
                    FROM daily_stats
                )
                SELECT * FROM rolling_stats
                ON CONFLICT (player_id, game_date) DO UPDATE
                SET
                    corner_3_attempts = EXCLUDED.corner_3_attempts,
                    corner_3_made = EXCLUDED.corner_3_made,
                    above_break_3_attempts = EXCLUDED.above_break_3_attempts,
                    above_break_3_made = EXCLUDED.above_break_3_made,
                    early_shot_clock_3_attempts = EXCLUDED.early_shot_clock_3_attempts,
                    early_shot_clock_3_made = EXCLUDED.early_shot_clock_3_made,
                    late_shot_clock_3_attempts = EXCLUDED.late_shot_clock_3_attempts,
                    late_shot_clock_3_made = EXCLUDED.late_shot_clock_3_made,
                    catch_shoot_3_attempts = EXCLUDED.catch_shoot_3_attempts,
                    catch_shoot_3_made = EXCLUDED.catch_shoot_3_made,
                    pullup_3_attempts = EXCLUDED.pullup_3_attempts,
                    pullup_3_made = EXCLUDED.pullup_3_made,
                    corner_3_pct_5game = EXCLUDED.corner_3_pct_5game,
                    above_break_3_pct_5game = EXCLUDED.above_break_3_pct_5game,
                    catch_shoot_3_pct_5game = EXCLUDED.catch_shoot_3_pct_5game,
                    pullup_3_pct_5game = EXCLUDED.pullup_3_pct_5game
            """)
            
            # Verify stats
            result = conn.execute("""
                SELECT COUNT(*) FROM player_shot_stats
            """).fetchone()
            
            if result is None:
                raise ValueError("Failed to verify shot stats")
                
            logger.info(f"Calculated shot stats for {result[0]} player-games")
            
        finally:
            conn.close()

def main() -> int:
    """Sync shot location data and calculate stats.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        syncer = ShotLocationSync()
        
        # Sync recent shot data
        logger.info("Syncing shot location data...")
        stats = syncer.sync_shot_data()
        
        logger.info("Sync complete:")
        logger.info(f"Players processed: {stats['players_processed']}")
        logger.info(f"Shots added: {stats['shots_added']}")
        logger.info(f"Errors: {stats['errors']}")
        
        # Calculate shot zone stats
        logger.info("\nCalculating shot zone statistics...")
        syncer.calculate_shot_zone_stats()
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
