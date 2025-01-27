"""Script to sync player game stats from NBA API to database."""

import logging
from typing import Dict, List, Optional, Set
import duckdb
from db_config import DatabaseConfig
from nba_api.stats.endpoints import playergamelog
from datetime import datetime, timezone
import pandas as pd
import time
from tqdm import tqdm
import requests
from requests.exceptions import Timeout, RequestException
import signal
import sys

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
should_exit = False

def signal_handler(signum, frame):
    """Handle interrupt signal."""
    global should_exit
    if not should_exit:
        print("\nGracefully shutting down... (Press Ctrl+C again to force quit)")
        should_exit = True
    else:
        print("\nForce quitting...")
        sys.exit(1)

def create_player_stats_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create player_stats table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            player_id INTEGER,
            player_name VARCHAR,
            team_id INTEGER,
            team_abbreviation VARCHAR,
            game_date DATE,
            is_home BOOLEAN,
            min INTEGER,
            pts INTEGER,
            ast INTEGER,
            reb INTEGER,
            fg_pct FLOAT,
            fg3_pct FLOAT,
            ft_pct FLOAT,
            plus_minus INTEGER,
            
            -- Home/Away averages
            pts_home FLOAT,
            pts_away FLOAT,
            ast_home FLOAT,
            ast_away FLOAT,
            reb_home FLOAT,
            reb_away FLOAT,
            
            -- Rolling averages
            pts_rolling_5 FLOAT,
            pts_rolling_10 FLOAT,
            pts_rolling_20 FLOAT,
            ast_rolling_5 FLOAT,
            ast_rolling_10 FLOAT,
            ast_rolling_20 FLOAT,
            reb_rolling_5 FLOAT,
            reb_rolling_10 FLOAT,
            reb_rolling_20 FLOAT,
            fg_pct_rolling_5 FLOAT,
            fg_pct_rolling_10 FLOAT,
            fg_pct_rolling_20 FLOAT,
            fg3_pct_rolling_5 FLOAT,
            fg3_pct_rolling_10 FLOAT,
            fg3_pct_rolling_20 FLOAT,
            ft_pct_rolling_5 FLOAT,
            ft_pct_rolling_10 FLOAT,
            ft_pct_rolling_20 FLOAT,
            plus_minus_rolling_5 FLOAT,
            plus_minus_rolling_10 FLOAT,
            plus_minus_rolling_20 FLOAT,
            
            -- Opponent metrics
            opp_pts_allowed_avg FLOAT,
            opp_ast_allowed_avg FLOAT,
            opp_reb_rate FLOAT,
            
            PRIMARY KEY (player_id, game_date)
        )
    """)

def get_synced_players(conn: duckdb.DuckDBPyConnection) -> Set[int]:
    """Get set of player IDs that have already been synced."""
    result = conn.execute("""
        SELECT DISTINCT player_id
        FROM player_stats
    """).fetchall()
    return {row[0] for row in result} if result else set()

def safe_get_column(df: pd.DataFrame, col: str, default: Optional[str] = None) -> pd.Series:
    """Safely get column from DataFrame, returning default if not found."""
    return df[col] if col in df.columns else pd.Series([default] * len(df))

def get_player_game_stats(player_id: int, retries: int = 3) -> pd.DataFrame:
    """Get game stats for a player from NBA API.
    
    Args:
        player_id: NBA player ID
        retries: Number of times to retry on failure
        
    Returns:
        DataFrame with player game stats
    """
    for attempt in range(retries):
        try:
            # Get game logs for current season
            game_logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2023-24',
                timeout=30
            ).get_data_frames()[0]
            
            if game_logs.empty:
                return pd.DataFrame()
            
            # Print available columns for debugging
            logger.debug(f"Available columns: {game_logs.columns.tolist()}")
            
            # Convert date string to datetime
            game_logs['game_date'] = pd.to_datetime(safe_get_column(game_logs, 'GAME_DATE'))
            
            # Create new DataFrame with required columns
            df = pd.DataFrame()
            
            # Map NBA API columns to our schema with safe column access
            df['team_id'] = pd.to_numeric(safe_get_column(game_logs, 'Team_ID'), errors='coerce')
            df['team_abbreviation'] = safe_get_column(game_logs, 'Team_Abbreviation')
            df['game_date'] = game_logs['game_date']
            df['is_home'] = safe_get_column(game_logs, 'MATCHUP').str.contains('vs', na=False)
            df['min'] = pd.to_numeric(safe_get_column(game_logs, 'MIN'), errors='coerce')
            df['pts'] = pd.to_numeric(safe_get_column(game_logs, 'PTS'), errors='coerce')
            df['ast'] = pd.to_numeric(safe_get_column(game_logs, 'AST'), errors='coerce')
            df['reb'] = pd.to_numeric(safe_get_column(game_logs, 'REB'), errors='coerce')
            df['fg_pct'] = pd.to_numeric(safe_get_column(game_logs, 'FG_PCT'), errors='coerce')
            df['fg3_pct'] = pd.to_numeric(safe_get_column(game_logs, 'FG3_PCT'), errors='coerce')
            df['ft_pct'] = pd.to_numeric(safe_get_column(game_logs, 'FT_PCT'), errors='coerce')
            df['plus_minus'] = pd.to_numeric(safe_get_column(game_logs, 'PLUS_MINUS'), errors='coerce')
            
            # Fill NaN values with 0
            numeric_cols = ['min', 'pts', 'ast', 'reb', 'fg_pct', 'fg3_pct', 'ft_pct', 'plus_minus']
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Calculate rolling averages
            for window in [5, 10, 20]:
                df[f'pts_rolling_{window}'] = df['pts'].rolling(window, min_periods=1).mean()
                df[f'ast_rolling_{window}'] = df['ast'].rolling(window, min_periods=1).mean()
                df[f'reb_rolling_{window}'] = df['reb'].rolling(window, min_periods=1).mean()
                df[f'fg_pct_rolling_{window}'] = df['fg_pct'].rolling(window, min_periods=1).mean()
                df[f'fg3_pct_rolling_{window}'] = df['fg3_pct'].rolling(window, min_periods=1).mean()
                df[f'ft_pct_rolling_{window}'] = df['ft_pct'].rolling(window, min_periods=1).mean()
                df[f'plus_minus_rolling_{window}'] = df['plus_minus'].rolling(window, min_periods=1).mean()
            
            # Calculate home/away splits
            home_games = df[df['is_home']]
            away_games = df[~df['is_home']]
            
            pts_home = home_games['pts'].mean() if not home_games.empty else 0
            pts_away = away_games['pts'].mean() if not away_games.empty else 0
            ast_home = home_games['ast'].mean() if not home_games.empty else 0
            ast_away = away_games['ast'].mean() if not away_games.empty else 0
            reb_home = home_games['reb'].mean() if not home_games.empty else 0
            reb_away = away_games['reb'].mean() if not away_games.empty else 0
            
            # Add home/away averages
            df['pts_home'] = pts_home
            df['pts_away'] = pts_away
            df['ast_home'] = ast_home
            df['ast_away'] = ast_away
            df['reb_home'] = reb_home
            df['reb_away'] = reb_away
            
            # For now, use placeholder values for opponent metrics
            df['opp_pts_allowed_avg'] = 110.0
            df['opp_ast_allowed_avg'] = 25.0
            df['opp_reb_rate'] = 0.5
            
            return df
            
        except (Timeout, RequestException) as e:
            if attempt == retries - 1:  # Last attempt
                logger.error(f"Error getting game stats for player {player_id} after {retries} attempts: {str(e)}")
                return pd.DataFrame()
            else:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
        except Exception as e:
            logger.error(f"Error getting game stats for player {player_id}: {str(e)}")
            return pd.DataFrame()

def sync_player_stats() -> None:
    """Sync player game stats from NBA API to database."""
    try:
        # Initialize database config for local storage first
        db_config = DatabaseConfig(use_motherduck=False)
        
        # Connect to local database
        with db_config.connect() as conn:
            # Create table if needed
            create_player_stats_table(conn)
            
            # Get active players
            active_players = conn.execute("""
                SELECT player_id, full_name
                FROM nba_players
                WHERE is_active = true
            """).fetchall()
            
            if not active_players:
                logger.error("No active players found in database")
                return
            
            # Get already synced players
            synced_players = get_synced_players(conn)
            remaining_players = [(pid, name) for pid, name in active_players if pid not in synced_players]
            
            logger.info(f"Found {len(remaining_players)} players to sync")
            
            # Sync stats for each player
            for player_id, player_name in tqdm(remaining_players, desc="Syncing player stats"):
                if should_exit:
                    logger.info("Received exit signal, stopping gracefully...")
                    break
                    
                try:
                    # Get player's game stats
                    game_logs = get_player_game_stats(player_id)
                    
                    if not game_logs.empty:
                        # Create temporary table from DataFrame
                        conn.register('temp_game_logs', game_logs)
                        
                        # Insert into database
                        conn.execute("""
                            INSERT INTO player_stats (
                                player_id,
                                player_name,
                                team_id,
                                team_abbreviation,
                                game_date,
                                is_home,
                                min,
                                pts,
                                ast,
                                reb,
                                fg_pct,
                                fg3_pct,
                                ft_pct,
                                plus_minus,
                                pts_home,
                                pts_away,
                                ast_home,
                                ast_away,
                                reb_home,
                                reb_away,
                                pts_rolling_5,
                                pts_rolling_10,
                                pts_rolling_20,
                                ast_rolling_5,
                                ast_rolling_10,
                                ast_rolling_20,
                                reb_rolling_5,
                                reb_rolling_10,
                                reb_rolling_20,
                                fg_pct_rolling_5,
                                fg_pct_rolling_10,
                                fg_pct_rolling_20,
                                fg3_pct_rolling_5,
                                fg3_pct_rolling_10,
                                fg3_pct_rolling_20,
                                ft_pct_rolling_5,
                                ft_pct_rolling_10,
                                ft_pct_rolling_20,
                                plus_minus_rolling_5,
                                plus_minus_rolling_10,
                                plus_minus_rolling_20,
                                opp_pts_allowed_avg,
                                opp_ast_allowed_avg,
                                opp_reb_rate
                            )
                            SELECT
                                ?,  -- player_id
                                ?,  -- player_name
                                team_id,
                                team_abbreviation,
                                game_date,
                                is_home,
                                min,
                                pts,
                                ast,
                                reb,
                                fg_pct,
                                fg3_pct,
                                ft_pct,
                                plus_minus,
                                pts_home,
                                pts_away,
                                ast_home,
                                ast_away,
                                reb_home,
                                reb_away,
                                pts_rolling_5,
                                pts_rolling_10,
                                pts_rolling_20,
                                ast_rolling_5,
                                ast_rolling_10,
                                ast_rolling_20,
                                reb_rolling_5,
                                reb_rolling_10,
                                reb_rolling_20,
                                fg_pct_rolling_5,
                                fg_pct_rolling_10,
                                fg_pct_rolling_20,
                                fg3_pct_rolling_5,
                                fg3_pct_rolling_10,
                                fg3_pct_rolling_20,
                                ft_pct_rolling_5,
                                ft_pct_rolling_10,
                                ft_pct_rolling_20,
                                plus_minus_rolling_5,
                                plus_minus_rolling_10,
                                plus_minus_rolling_20,
                                opp_pts_allowed_avg,
                                opp_ast_allowed_avg,
                                opp_reb_rate
                            FROM temp_game_logs
                        """, [player_id, player_name])
                        
                        logger.info(f"Synced {len(game_logs)} games for {player_name}")
                    
                    # Sleep to avoid API rate limiting
                    time.sleep(2)  # Increased sleep time
                    
                except Exception as e:
                    logger.error(f"Error syncing stats for {player_name}: {str(e)}")
                    continue
            
            logger.info("Successfully synced player stats")
            
    except Exception as e:
        logger.error(f"Error syncing player stats: {str(e)}")
        raise

def main() -> None:
    """Sync player stats and print summary."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Sync player stats to local DB
    sync_player_stats()
    
    # Sync to MotherDuck
    logger.info("Syncing data to MotherDuck...")
    db_config = DatabaseConfig(use_motherduck=True)
    db_config.sync_to_motherduck()
    
    # Print summary using MotherDuck connection
    with db_config.connect() as conn:
        result = conn.execute("""
            SELECT 
                COUNT(DISTINCT player_id) as num_players,
                COUNT(*) as num_games,
                MIN(game_date) as earliest_game,
                MAX(game_date) as latest_game
            FROM player_stats
        """).fetchone()
        
        if result:
            print("\nPlayer Stats Summary:")
            print(f"Number of players: {result[0]}")
            print(f"Number of games: {result[1]}")
            print(f"Date range: {result[2]} to {result[3]}")
        else:
            print("\nNo player stats found in database")

if __name__ == "__main__":
    main()
