"""NBA statistics data pipeline."""

import logging
import os
from datetime import datetime, date
from typing import Dict, Any, Optional, List, Callable, Union
import time
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,  # type: ignore
    playergamelog,          # type: ignore
    commonplayerinfo        # type: ignore
)
from duckdb import DuckDBPyConnection
from pandas import DataFrame, Index, Series
from dotenv import load_dotenv
from src.data.db_config import DatabaseConfig

load_dotenv()
logger = logging.getLogger(__name__)

from nba_api.stats.static import players  # type: ignore
from nba_api.stats.library.parameters import SeasonAll  # type: ignore

# Configure NBA API settings
NBA_API_DELAY = float(os.getenv('NBA_API_DELAY', '3.0'))  # Further increased delay
NBA_API_TIMEOUT = int(os.getenv('NBA_API_TIMEOUT', '180'))  # Further increased timeout
MAX_RETRIES = 3
BATCH_SIZE = 25  # Further reduced batch size


class NBADataPipeline:
    """Pipeline for fetching and processing NBA player statistics."""
    
    def __init__(self, use_motherduck: bool = True) -> None:
        """Initialize the pipeline with database connection.
        
        Args:
            use_motherduck: Whether to use MotherDuck cloud database
        """
        db_config = DatabaseConfig(use_motherduck)
        self.conn: DuckDBPyConnection = db_config.connect()
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize the database schema."""
        self.conn.execute("""
            -- Create base stats table if it doesn't exist
            CREATE TABLE IF NOT EXISTS player_stats (
                -- Player identification
                player_id INTEGER,
                game_date DATE,
                PRIMARY KEY (player_id, game_date),
                player_name VARCHAR,
                team_id INTEGER,
                team_abbreviation VARCHAR,
                position VARCHAR,  -- Added player position
                
                -- Basic stats
                age DOUBLE,
                gp INTEGER,
                min DOUBLE,
                fgm DOUBLE,
                fga DOUBLE,
                fg_pct DOUBLE,
                fg3m DOUBLE,
                fg3a DOUBLE,
                fg3_pct DOUBLE,
                ftm DOUBLE,
                fta DOUBLE,
                ft_pct DOUBLE,
                oreb DOUBLE,
                dreb DOUBLE,
                reb DOUBLE,
                ast DOUBLE,
                stl DOUBLE,
                blk DOUBLE,
                tov DOUBLE,
                pts DOUBLE,
                plus_minus DOUBLE,
                
                -- Rolling averages (last 5 games)
                pts_rolling_5 DOUBLE,
                ast_rolling_5 DOUBLE,
                reb_rolling_5 DOUBLE,
                fg3m_rolling_5 DOUBLE,
                
                -- Home/Away splits
                pts_home DOUBLE,
                pts_away DOUBLE,
                ast_home DOUBLE,
                ast_away DOUBLE,
                reb_home DOUBLE,
                reb_away DOUBLE,
                
                -- Opponent strength indicators
                opp_pts_allowed_avg DOUBLE,  -- Average points allowed by opponent
                opp_reb_rate DOUBLE,         -- Opponent rebound rate
                opp_ast_allowed_avg DOUBLE,  -- Average assists allowed by opponent
                
                -- Seasonal trends
                pts_last_season DOUBLE,
                ast_last_season DOUBLE,
                reb_last_season DOUBLE,
                
                -- Metadata
                last_updated TIMESTAMP,
                is_home BOOLEAN              -- Whether last game was home
            );
            
            -- Create indices for efficient querying
            CREATE INDEX IF NOT EXISTS player_stats_team_idx ON player_stats(team_id);
            CREATE INDEX IF NOT EXISTS player_stats_date_idx ON player_stats(game_date)
        """)
    
    def _make_api_request(self, request_fn: Callable[[], Any], player_name: str, request_type: str) -> Any:  # type: ignore
        """Make API request with retries and exponential backoff."""
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    sleep_time = NBA_API_DELAY * (2.0 ** float(attempt))
                    logger.info(f"Retry {attempt + 1}/{MAX_RETRIES} for {player_name} {request_type} after {sleep_time}s delay")
                    time.sleep(sleep_time)
                return request_fn()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"All retries failed for {player_name} {request_type}: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {player_name} {request_type}: {str(e)}")
                time.sleep(NBA_API_DELAY)  # Add delay before retry
        raise Exception(f"All retries failed for {player_name} {request_type}")
    
    def _calculate_rolling_averages(self, player_id: int, player_name: str) -> Dict[str, Optional[float]]:
        """Calculate rolling averages for last 5 games.
        
        Args:
            player_id: NBA player ID
            player_name: Name of player for logging
            
        Returns:
            Dictionary containing rolling averages
        """
        try:
            # Get player's game log
            def get_gamelog() -> DataFrame:
                try:
                    frames: List[Any] = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season='2024-25',  # Current season
                        timeout=NBA_API_TIMEOUT
                    ).get_data_frames()
                    
                    if not frames or len(frames[0]) == 0:
                        return DataFrame()
                    
                    df = DataFrame(frames[0].copy())
                    # Convert date format
                    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
                    return df
                except Exception as e:
                    logger.error(f"Error getting game log for player {player_id}: {str(e)}")
                    return DataFrame()
            
            gamelog: DataFrame = self._make_api_request(get_gamelog, player_name, "game log")
            
            if len(gamelog) == 0:
                return {
                    'pts_rolling_5': None,
                    'ast_rolling_5': None,
                    'reb_rolling_5': None,
                    'fg3m_rolling_5': None
                }
            
            # Calculate rolling averages
            recent_games = gamelog.head(5)
            return {
                'pts_rolling_5': float(recent_games['PTS'].mean()),
                'ast_rolling_5': float(recent_games['AST'].mean()),
                'reb_rolling_5': float(recent_games['REB'].mean()),
                'fg3m_rolling_5': float(recent_games['FG3M'].mean())
            }
        except Exception as e:
            logger.error(f"Error calculating rolling averages for player {player_id}: {str(e)}")
            return {
                'pts_rolling_5': None,
                'ast_rolling_5': None,
                'reb_rolling_5': None,
                'fg3m_rolling_5': None
            }
    
    def _get_player_position(self, player_id: int, player_name: str) -> str:
        """Get player's position.
        
        Args:
            player_id: NBA player ID
            player_name: Name of player for logging
            
        Returns:
            Player position (e.g., 'G', 'F', 'C') or 'Unknown' if not found
        """
        try:
            def get_player_info() -> DataFrame:
                frames: List[Any] = commonplayerinfo.CommonPlayerInfo(
                    player_id=player_id,
                    timeout=NBA_API_TIMEOUT
                ).get_data_frames()
                return DataFrame(frames[0].copy())
            
            info: DataFrame = self._make_api_request(get_player_info, player_name, "position info")
            return str(info['POSITION'].iloc[0]) if not info.empty else 'Unknown'
        except Exception as e:
            logger.warning(f"Could not get position for player {player_id}, using Unknown: {str(e)}")
            return 'Unknown'
    
    def _calculate_home_away_splits(self, player_id: int, player_name: str) -> Dict[str, Optional[Union[float, bool, date]]]:
        """Calculate home/away splits.
        
        Args:
            player_id: NBA player ID
            player_name: Name of player for logging
            
        Returns:
            Dictionary containing home/away averages
        """
        try:
            # Get player's game log
            def get_gamelog() -> DataFrame:
                try:
                    frames: List[Any] = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season='2024-25',  # Current season
                        timeout=NBA_API_TIMEOUT
                    ).get_data_frames()
                    
                    if not frames or len(frames[0]) == 0:
                        return DataFrame()
                    
                    df = DataFrame(frames[0].copy())
                    # Convert date format
                    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed').dt.strftime('%Y-%m-%d')
                    return df
                except Exception as e:
                    logger.error(f"Error getting game log for player {player_id}: {str(e)}")
                    return DataFrame()
            
            gamelog: DataFrame = self._make_api_request(get_gamelog, player_name, "game log for splits")
            
            if len(gamelog) == 0:
                return {
                    'pts_home': None, 'pts_away': None,
                    'ast_home': None, 'ast_away': None,
                    'reb_home': None, 'reb_away': None,
                    'is_home': None,
                    'game_date': None
                }
            
            # Split into home and away games
            home_games = gamelog[gamelog['MATCHUP'].str.contains(' vs. ')]
            away_games = gamelog[gamelog['MATCHUP'].str.contains(' @ ')]
            
            # Get latest game location
            is_home = ' vs. ' in gamelog['MATCHUP'].iloc[0] if len(gamelog) > 0 else None
            
            return {
                'pts_home': float(home_games['PTS'].mean()) if len(home_games) > 0 else None,
                'pts_away': float(away_games['PTS'].mean()) if len(away_games) > 0 else None,
                'ast_home': float(home_games['AST'].mean()) if len(home_games) > 0 else None,
                'ast_away': float(away_games['AST'].mean()) if len(away_games) > 0 else None,
                'reb_home': float(home_games['REB'].mean()) if len(home_games) > 0 else None,
                'reb_away': float(away_games['REB'].mean()) if len(away_games) > 0 else None,
                'is_home': is_home,
                'game_date': gamelog['GAME_DATE'].iloc[0].date() if len(gamelog) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error calculating home/away splits for player {player_id}: {str(e)}")
            return {
                'pts_home': None, 'pts_away': None,
                'ast_home': None, 'ast_away': None,
                'reb_home': None, 'reb_away': None,
                'is_home': None,
                'game_date': None
            }
    
    def fetch_player_stats(self) -> DataFrame:
        """Fetch current player statistics from NBA API.
        
        Returns:
            DataFrame containing player statistics with last_updated timestamp
        """
        try:
            # Add delay to avoid rate limiting
            time.sleep(NBA_API_DELAY)
            
            # Get stats from NBA API
            logger.info("Fetching basic player stats from NBA API...")
            def get_league_stats() -> List[DataFrame]:
                frames: List[Any] = leaguedashplayerstats.LeagueDashPlayerStats(
                    timeout=NBA_API_TIMEOUT,
                    per_mode_detailed='PerGame',
                    season='2024-25',
                    season_type_all_star='Regular Season'
                ).get_data_frames()
                
                if not frames or len(frames[0]) == 0:
                    return []
                
                # Filter for active players
                df = DataFrame(frames[0].copy())
                active_players = df[df['MIN'] > 0]  # Only players who have played minutes
                return [active_players]
            
            raw_stats: List[DataFrame] = self._make_api_request(
                get_league_stats, 
                "all players", 
                "league dashboard"
            )
            logger.info(f"Retrieved basic stats for {len(raw_stats[0])} players")
            
            if not raw_stats or len(raw_stats) == 0:
                raise ValueError("No data returned from NBA API")
            
            # Convert to DataFrame and process basic stats
            stats = DataFrame(raw_stats[0])
            stats.columns = Index([col.lower() for col in stats.columns])
            stats['last_updated'] = datetime.now()
            
            # Process players in batches
            enhanced_stats: List[Dict[str, Any]] = []
            total_players = len(stats)
            batch_size = BATCH_SIZE
            
            for batch_start in range(0, total_players, batch_size):
                batch_end = min(batch_start + batch_size, total_players)
                batch_players = stats.iloc[batch_start:batch_end]
                
                batch_num = str(int(batch_start/batch_size) + 1)
                total_batches = str(int((total_players + batch_size - 1)/batch_size))
                logger.info(f"Processing batch {batch_num}/{total_batches} "
                          f"(players {str(batch_start + 1)}-{str(batch_end)}/{str(total_players)})")
                
                batch_enhanced_stats: List[Dict[str, Any]] = []
                for i, (_, player) in enumerate(batch_players.iterrows()):
                    player_name = str(player['player_name'])
                    player_id = int(player['player_id'])
                    
                    try:
                        # Get additional metrics with rate limiting
                        current_player_num = batch_start + i + 1
                        logger.debug(f"Processing {player_name} ({str(current_player_num)}/{str(total_players)})")
                        
                        # Get all metrics in parallel for each player
                        rolling_avgs = self._calculate_rolling_averages(player_id, player_name)
                        position = self._get_player_position(player_id, player_name)
                        splits = self._calculate_home_away_splits(player_id, player_name)
                        
                        # Combine all stats
                        enhanced_player = {
                            **player.to_dict(),
                            'position': position,
                            **rolling_avgs,
                            **splits
                        }
                        batch_enhanced_stats.append(enhanced_player)
                        
                        # Add small delay between players
                        time.sleep(NBA_API_DELAY)
                        
                    except Exception as e:
                        logger.error(f"Error processing player {player_name}: {str(e)}")
                        # Continue with next player on error
                        continue
                
                # Add batch results to overall stats and save progress
                enhanced_stats.extend(batch_enhanced_stats)
                logger.info(f"Successfully processed {len(batch_enhanced_stats)} players in batch")
                
                # Add delay between batches
                if batch_end < total_players:
                    delay = NBA_API_DELAY * 2
                    logger.info(f"Batch complete, waiting {delay}s before next batch...")
                    time.sleep(delay)
            
            return DataFrame(enhanced_stats)
            
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            raise
    
    def store_stats_in_db(self, stats: pd.DataFrame) -> None:
        """Store player statistics in the database.
        
        Args:
            stats: DataFrame containing player statistics
        """
        try:
            logger.info("Storing enhanced player statistics in database...")
            # Convert DataFrame to DuckDB table
            self.conn.register('stats_df', stats)
            
            # Only insert new records
            logger.debug("Executing database insert...")
            self.conn.execute("""
                INSERT INTO player_stats (
                    -- Player identification
                    player_id, player_name, team_id, team_abbreviation, position,
                    
                    -- Basic stats
                    age, gp, min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                    ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk,
                    tov, pts, plus_minus,
                    
                    -- Rolling averages
                    pts_rolling_5, ast_rolling_5, reb_rolling_5, fg3m_rolling_5,
                    
                    -- Home/Away splits
                    pts_home, pts_away, ast_home, ast_away, reb_home, reb_away,
                    
                    -- Opponent strength indicators
                    opp_pts_allowed_avg, opp_reb_rate, opp_ast_allowed_avg,
                    
                    -- Seasonal trends
                    pts_last_season, ast_last_season, reb_last_season,
                    
                    -- Metadata
                    last_updated, game_date, is_home
                )
                SELECT 
                    -- Player identification
                    player_id, player_name, team_id, team_abbreviation, position,
                    
                    -- Basic stats
                    age, gp, min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                    ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk,
                    tov, pts, plus_minus,
                    
                    -- Rolling averages
                    pts_rolling_5, ast_rolling_5, reb_rolling_5, fg3m_rolling_5,
                    
                    -- Home/Away splits
                    pts_home, pts_away, ast_home, ast_away, reb_home, reb_away,
                    
                    -- Opponent strength indicators
                    opp_pts_allowed_avg, opp_reb_rate, opp_ast_allowed_avg,
                    
                    -- Seasonal trends
                    pts_last_season, ast_last_season, reb_last_season,
                    
                    -- Metadata
                    last_updated, game_date, is_home
                FROM stats_df
                ON CONFLICT (player_id, game_date) DO UPDATE SET
                    -- Player identification
                    player_name = EXCLUDED.player_name,
                    team_id = EXCLUDED.team_id,
                    team_abbreviation = EXCLUDED.team_abbreviation,
                    position = EXCLUDED.position,
                    
                    -- Basic stats
                    age = EXCLUDED.age,
                    gp = EXCLUDED.gp,
                    min = EXCLUDED.min,
                    fgm = EXCLUDED.fgm,
                    fga = EXCLUDED.fga,
                    fg_pct = EXCLUDED.fg_pct,
                    fg3m = EXCLUDED.fg3m,
                    fg3a = EXCLUDED.fg3a,
                    fg3_pct = EXCLUDED.fg3_pct,
                    ftm = EXCLUDED.ftm,
                    fta = EXCLUDED.fta,
                    ft_pct = EXCLUDED.ft_pct,
                    oreb = EXCLUDED.oreb,
                    dreb = EXCLUDED.dreb,
                    reb = EXCLUDED.reb,
                    ast = EXCLUDED.ast,
                    stl = EXCLUDED.stl,
                    blk = EXCLUDED.blk,
                    tov = EXCLUDED.tov,
                    pts = EXCLUDED.pts,
                    plus_minus = EXCLUDED.plus_minus,
                    
                    -- Rolling averages
                    pts_rolling_5 = EXCLUDED.pts_rolling_5,
                    ast_rolling_5 = EXCLUDED.ast_rolling_5,
                    reb_rolling_5 = EXCLUDED.reb_rolling_5,
                    fg3m_rolling_5 = EXCLUDED.fg3m_rolling_5,
                    
                    -- Home/Away splits
                    pts_home = EXCLUDED.pts_home,
                    pts_away = EXCLUDED.pts_away,
                    ast_home = EXCLUDED.ast_home,
                    ast_away = EXCLUDED.ast_away,
                    reb_home = EXCLUDED.reb_home,
                    reb_away = EXCLUDED.reb_away,
                    
                    -- Opponent strength indicators
                    opp_pts_allowed_avg = EXCLUDED.opp_pts_allowed_avg,
                    opp_reb_rate = EXCLUDED.opp_reb_rate,
                    opp_ast_allowed_avg = EXCLUDED.opp_ast_allowed_avg,
                    
                    -- Seasonal trends
                    pts_last_season = EXCLUDED.pts_last_season,
                    ast_last_season = EXCLUDED.ast_last_season,
                    reb_last_season = EXCLUDED.reb_last_season,
                    
                    -- Metadata
                    last_updated = EXCLUDED.last_updated,
                    game_date = EXCLUDED.game_date,
                    is_home = EXCLUDED.is_home
            """)
            
            # Cleanup
            self.conn.unregister('stats_df')
            logger.info("Successfully stored player statistics in database")
            
        except Exception as e:
            logger.error(f"Error storing player stats: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
        self.conn = None  # type: ignore


def update_player_stats(use_motherduck: bool = True) -> None:
    """Main pipeline execution function to fetch and store current player statistics.
    
    Args:
        use_motherduck: Whether to use MotherDuck cloud database
    """
    pipeline = None
    try:
        pipeline = NBADataPipeline(use_motherduck)
        stats = pipeline.fetch_player_stats()
        pipeline.store_stats_in_db(stats)
        logger.info("Successfully updated player statistics")
    except Exception as e:
        logger.error(f"Failed to update player stats: {str(e)}")
        raise
    finally:
        if pipeline:
            pipeline.close()

if __name__ == "__main__":
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    update_player_stats()
