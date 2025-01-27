"""NBA career statistics data pipeline with async processing and database integration."""

import logging
import os
import asyncio
import time
from datetime import datetime
import pandas as pd
from nba_api.stats.static import players  # type: ignore
from nba_api.stats.endpoints import (  # type: ignore
    playercareerstats,
    commonplayerinfo,
    leaguedashplayerstats
)
from typing import Dict, Any, List, Optional, Union, Iterator, NoReturn
from tqdm import tqdm  # type: ignore
import requests
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from db_config import DatabaseConfig, get_db_connection  # type: ignore
import duckdb
from asyncio import Semaphore
import aiohttp

# Configure logging
def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'nba_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# API settings
CONCURRENT_REQUESTS = 20  # maximum concurrent requests
REQUEST_TIMEOUT = 30  # timeout in seconds
MAX_RETRIES = 3  # number of retries
BATCH_SIZE = 20  # very small batch size for rapid syncing

# NBA API headers
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

def setup_database(conn: duckdb.DuckDBPyConnection) -> bool:
    """Set up database schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            -- Player Info
            player_id INTEGER,
            player_name VARCHAR,
            position VARCHAR,
            team VARCHAR,
            height VARCHAR,
            weight VARCHAR,
            age INTEGER,
            
            -- Season Info
            season VARCHAR,
            games_played INTEGER,
            minutes DOUBLE,
            
            -- Basic Stats
            pts_avg DOUBLE,
            ast_avg DOUBLE,
            reb_avg DOUBLE,
            stl_avg DOUBLE,
            blk_avg DOUBLE,
            fg_pct DOUBLE,
            fg3_pct DOUBLE,
            ft_pct DOUBLE,
            tov_avg DOUBLE,
            
            -- Advanced Metrics
            usg_pct DOUBLE,
            net_rating DOUBLE,
            fantasy_pts DOUBLE,
            
            -- Efficiency Metrics
            pts_per_min DOUBLE,
            pts_per_48 DOUBLE,
            ast_per_min DOUBLE,
            ast_per_48 DOUBLE,
            reb_per_min DOUBLE,
            reb_per_48 DOUBLE,
            
            -- Career Stats
            career_games INTEGER,
            career_pts DOUBLE,
            career_ast DOUBLE,
            career_reb DOUBLE,
            
            -- Metadata
            last_updated TIMESTAMP,
            PRIMARY KEY (player_id, season)
        )
    """)
    return True

class NBAStatsCollector:
    """Async NBA stats collector with database integration."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.semaphore = Semaphore(CONCURRENT_REQUESTS)
        self.db_config = db_config
        self.batch_stats: List[Dict[str, Any]] = []
    
    async def get_player_info(self, player_id: int, player_name: str) -> Optional[Dict[str, Any]]:
        """Get player info asynchronously."""
        try:
            async with self.semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        start_time = time.time()
                        info = commonplayerinfo.CommonPlayerInfo(
                            player_id=player_id,
                            timeout=REQUEST_TIMEOUT,
                            headers=HEADERS
                        ).get_data_frames()[0]
                        
                        elapsed = time.time() - start_time
                        logger.debug(f"Got info for {player_name} in {elapsed:.2f}s")
                        # Calculate age from birthdate
                        birthdate = pd.to_datetime(info['BIRTHDATE'].iloc[0])
                        age = (datetime.now() - birthdate).days // 365
                        
                        return {
                            'position': info['POSITION'].iloc[0],
                            'team': info['TEAM_NAME'].iloc[0],
                            'height': info['HEIGHT'].iloc[0],
                            'weight': info['WEIGHT'].iloc[0],
                            'age': age
                        }
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            logger.error(f"Failed to get info for {player_name}: {str(e)}")
                            return None
                        await asyncio.sleep(0.1 * (2 ** attempt))  # minimal delay between retries
        except Exception as e:
            logger.error(f"Unexpected error in get_player_info: {str(e)}")
            return None
        return None
    
    async def get_career_stats(self, player_id: int, player_name: str) -> Optional[Dict[str, Any]]:
        """Get career stats asynchronously."""
        try:
            async with self.semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        start_time = time.time()
                        career = playercareerstats.PlayerCareerStats(
                            player_id=player_id,
                            per_mode36="PerGame",
                            timeout=REQUEST_TIMEOUT,
                            headers=HEADERS
                        ).get_data_frames()
                        
                        if len(career) == 0 or len(career[0]) == 0:
                            logger.warning(f"No career stats found for {player_name}")
                            return None
                        
                        current_season = career[0].iloc[-1]
                        
                        elapsed = time.time() - start_time
                        logger.debug(f"Got career stats for {player_name} in {elapsed:.2f}s")
                        # Get advanced stats with retry
                        advanced_stats = pd.DataFrame()
                        for adv_attempt in range(MAX_RETRIES):
                            try:
                                advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                            player_id_nullable=player_id,
                            season=current_season['SEASON_ID'],
                            per_mode_detailed="PerGame",
                            measure_type_detailed_defense="Advanced",
                            timeout=REQUEST_TIMEOUT,
                            headers=HEADERS
                                ).get_data_frames()[0]
                                break
                            except Exception as e:
                                if adv_attempt == MAX_RETRIES - 1:
                                    logger.warning(f"Failed to get advanced stats for {player_name}: {str(e)}")
                                else:
                                    await asyncio.sleep(0.1 * (2 ** adv_attempt))
                        
                        # Calculate efficiency metrics
                        minutes = float(current_season['MIN'])
                        pts = float(current_season['PTS'])
                        ast = float(current_season['AST'])
                        reb = float(current_season['REB'])
                        
                        pts_per_min = pts / minutes if minutes > 0 else 0
                        ast_per_min = ast / minutes if minutes > 0 else 0
                        reb_per_min = reb / minutes if minutes > 0 else 0
                        
                        # Calculate fantasy points (DraftKings format)
                        fantasy_pts = (
                            pts + 
                            (reb * 1.25) + 
                            (ast * 1.5) + 
                            (float(current_season['STL']) * 2) + 
                            (float(current_season['BLK']) * 2) -
                            float(current_season.get('TOV', 0))
                        )
                        
                        return {
                            # Basic stats
                            'season': current_season['SEASON_ID'],
                            'games_played': int(current_season['GP']),
                            'minutes': float(current_season['MIN']),
                            'pts_avg': float(current_season['PTS']),
                            'ast_avg': float(current_season['AST']),
                            'reb_avg': float(current_season['REB']),
                            'stl_avg': float(current_season['STL']),
                            'blk_avg': float(current_season['BLK']),
                            'fg_pct': float(current_season['FG_PCT']),
                            'fg3_pct': float(current_season['FG3_PCT']),
                            'ft_pct': float(current_season['FT_PCT']),
                            'career_games': int(career[1]['GP'].sum()),
                            'career_pts': float(career[1]['PTS'].mean()),
                            'career_ast': float(career[1]['AST'].mean()),
                            'career_reb': float(career[1]['REB'].mean()),
                            
                            # Advanced metrics
                            'usg_pct': float(advanced_stats['USG_PCT'].iloc[0]) if not advanced_stats.empty else 0,
                            'net_rating': float(advanced_stats['NET_RATING'].iloc[0]) if not advanced_stats.empty else 0,
                            'fantasy_pts': fantasy_pts,
                            
                            # Efficiency metrics
                            'pts_per_min': pts_per_min,
                            'pts_per_48': pts_per_min * 48,
                            'ast_per_min': ast_per_min,
                            'ast_per_48': ast_per_min * 48,
                            'reb_per_min': reb_per_min,
                            'reb_per_48': reb_per_min * 48
                        }
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            logger.error(f"Failed to get career stats for {player_name}: {str(e)}")
                            return None
                        await asyncio.sleep(0.1 * (2 ** attempt))  # minimal delay between retries
        except Exception as e:
            logger.error(f"Unexpected error in get_career_stats: {str(e)}")
            return None
        return None
    
    async def process_player(self, player: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single player asynchronously."""
        try:
            player_id = player['id']
            player_name = player['full_name']
            
            # Get player info and career stats concurrently
            info_task = asyncio.create_task(self.get_player_info(player_id, player_name))
            career_task = asyncio.create_task(self.get_career_stats(player_id, player_name))
            
            info, career = await asyncio.gather(info_task, career_task)
            
            if not info or not career:
                return None
            
            # Combine all stats
            stats = {
                'player_id': player_id,
                'player_name': player_name,
                **info,
                **career,
                'last_updated': datetime.now()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing {player['full_name']}: {str(e)}")
            return None
    
    def sync_batch_to_db(self, stats: List[Dict[str, Any]]) -> bool:
        """Sync a batch of stats to the database."""
        if not stats:
            return True
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(stats)
            
            # Insert into local DuckDB
            conn = duckdb.connect(self.db_config.local_path)
            setup_database(conn)
            
            # Insert data
            conn.execute("""
                INSERT OR REPLACE INTO player_stats 
                SELECT * FROM df
            """)
            conn.close()
            
            logger.info(f"Successfully inserted {len(df)} records into local database")
            
            # Sync to MotherDuck
            if self.db_config.use_motherduck:
                logger.info("Syncing to MotherDuck...")
                self.db_config.sync_to_motherduck()
                logger.info("Successfully synced to MotherDuck")
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise
        return True
    
    async def process_all_players(self) -> None:
        """Process all active players concurrently with batching."""
        # Get active players
        active_players = [p for p in players.get_active_players() if p['is_active']]
        logger.info(f"Found {len(active_players)} active players")
        
        # Process players with progress bar
        with tqdm(total=len(active_players), desc="Processing players") as pbar:
            # Process players in smaller batches
            total_batches = (len(active_players) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, len(active_players), BATCH_SIZE):
                batch_num = i // BATCH_SIZE + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                batch = active_players[i:i + BATCH_SIZE]
                tasks = []
                for player in batch:
                    task = asyncio.create_task(self.process_player(player))
                    task.add_done_callback(lambda _: pbar.update(1))
                    tasks.append(task)
                
                # Process batch
                batch_results = await asyncio.gather(*tasks)
                valid_results = [r for r in batch_results if r is not None]
                
                # Sync batch to database
                if valid_results:
                    self.sync_batch_to_db(valid_results)
                

async def main() -> None:
    """Main async function."""
    start_time = time.time()
    logger.info("Starting NBA stats collection")
    
    try:
        # Initialize database config
        db_config = DatabaseConfig(use_motherduck=True)
        
        # Initialize collector
        collector = NBAStatsCollector(db_config)
        
        # Process all players
        await collector.process_all_players()
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Completed NBA stats collection in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
