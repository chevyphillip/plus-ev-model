"""NBA statistics data pipeline."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats  # type: ignore
import json
import time
import duckdb
from duckdb import DuckDBPyConnection

from src.data.db_config import DatabaseConfig, get_db_connection

logger = logging.getLogger(__name__)

class NBADataPipeline:
    def __init__(self, use_motherduck: bool = True) -> None:
        """Initialize the NBA data pipeline.
        
        Args:
            use_motherduck: Whether to use MotherDuck cloud database
        """
        self.db_config = DatabaseConfig(use_motherduck)
        self.conn: Optional[DuckDBPyConnection] = None
        
    def set_connection(self, conn: DuckDBPyConnection) -> None:
        """Set the database connection.
        
        Args:
            conn: DuckDB connection
        """
        self.conn = conn
        
    def _init_schema(self) -> None:
        """Initialize database schema if not exists"""
        if not self.conn:
            raise ValueError("Database connection not initialized")
            
        # Create schema if not exists
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                    player_id INTEGER,
                    season VARCHAR,
                    player_name VARCHAR,
                    team_id VARCHAR,
                    team_abbreviation VARCHAR,
                    age REAL,
                    gp INTEGER,
                    min REAL,
                    fgm REAL,
                    fga REAL,
                    fg_pct REAL,
                    fg3m REAL,
                    fg3a REAL,
                    fg3_pct REAL,
                    ftm REAL,
                    fta REAL,
                    ft_pct REAL,
                    oreb REAL,
                    dreb REAL,
                    reb REAL,
                    ast REAL,
                    stl REAL,
                    blk REAL,
                    tov REAL,
                    pts REAL,
                    plus_minus REAL,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (player_id, season)
                );

                CREATE TABLE IF NOT EXISTS player_season_averages (
                    player_id INTEGER,
                    player_name VARCHAR,
                    start_season VARCHAR,
                    end_season VARCHAR,
                    num_seasons INTEGER,
                    avg_gp REAL,
                    avg_min REAL,
                    avg_pts REAL,
                    avg_reb REAL,
                    avg_ast REAL,
                    avg_stl REAL,
                    avg_blk REAL,
                    avg_tov REAL,
                    avg_fg_pct REAL,
                    avg_fg3_pct REAL,
                    avg_ft_pct REAL,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (player_id, start_season, end_season)
                );

                CREATE TABLE IF NOT EXISTS data_metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR,
                    last_updated TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_stats(season);
                CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_stats(player_id);
                CREATE INDEX IF NOT EXISTS idx_season_averages_player ON player_season_averages(player_id);
            """)
    
    def fetch_player_stats(self, season: str = "2024-25") -> pd.DataFrame:
        """Fetch current season player stats from NBA API"""
        try:
            # Initialize API client with timeout and retry logic
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base',
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Origin': 'https://www.nba.com',
                    'Referer': 'https://www.nba.com/'
                },
                timeout=10
            )
            
            # Parse and validate JSON response
            try:
                response_data = json.loads(stats.get_normalized_json())
                logger.debug(f"API status code: {response_data.get('statusCode')}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response: {str(e)}")
                raise ValueError("Invalid JSON response from API")
            
            # Get DataFrame from API client with error handling
            try:
                df = stats.get_data_frames()[0]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to retrieve data frame: {str(e)}")
                logger.error("Failed to process API response")
                raise ValueError("Missing expected data structure in API response")
            
            if df.empty:
                logger.warning(f"Empty DataFrame received for season {season}")
                return pd.DataFrame()
                
            return self._validate_data(df)
        except Exception as e:
            logger.error(f"Failed to fetch player stats: {str(e)}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and transform API response data"""
        required_columns = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
            'AGE', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
            'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
            'AST', 'STL', 'BLK', 'TOV', 'PTS', 'PLUS_MINUS'
        ]
        
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        df = df[required_columns].copy()
        df.columns = df.columns.str.lower()
        df['last_updated'] = datetime.now()
        return df
    
    def store_stats_in_db(self, df: pd.DataFrame, season: str | None = None) -> None:
        """Store player stats for a specific season into database"""
        if not self.conn:
            raise ValueError("Database connection not initialized")
            
        try:
            if season is not None:
                df['season'] = season
            elif 'season' not in df.columns:
                df['season'] = datetime.now().year
            
            self.conn.register('temp_stats', df)
            self.conn.execute("""
                INSERT OR REPLACE INTO player_stats (
                    player_id, season, player_name, team_id, team_abbreviation,
                    age, gp, min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                    ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk,
                    tov, pts, plus_minus, last_updated
                )
                SELECT 
                    player_id, season, player_name, team_id::VARCHAR, team_abbreviation,
                    age, gp, min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                    ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk,
                    tov, pts, plus_minus, last_updated::TIMESTAMP
                FROM temp_stats
            """)
            logger.info(f"Upserted {len(df)} player records for season {season}")
        except Exception as e:
            logger.error(f"Database operation failed: {str(e)}")
            raise

    def fetch_multiple_seasons(self, start_season: int = 2021, num_seasons: int = 4) -> Dict[str, pd.DataFrame]:
        """Fetch stats for multiple seasons with rate limiting"""
        seasons_data = {}
        
        for year in range(start_season, start_season + num_seasons):
            season = f"{year}-{str(year+1)[2:]}"
            try:
                logger.info(f"Fetching data for season {season}")
                df = self.fetch_player_stats(season)
                if not df.empty:
                    seasons_data[season] = df
                # Rate limiting - wait 1 second between requests
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to fetch season {season}: {str(e)}")
                continue
        
        return seasons_data

    def calculate_season_averages(self, start_season: int = 2021, num_seasons: int = 4) -> None:
        """Calculate and store player averages across specified seasons"""
        if not self.conn:
            raise ValueError("Database connection not initialized")
            
        seasons_data = self.fetch_multiple_seasons(start_season, num_seasons)
        
        if not seasons_data:
            logger.error("No season data available for averaging")
            return

        # Store individual season data
        for season, df in seasons_data.items():
            self.store_stats_in_db(df, season)

        # Calculate averages across seasons
        try:
            assert self.conn is not None  # For type checking
            self.conn.execute(f"""
                INSERT OR REPLACE INTO player_season_averages
                SELECT 
                    player_id,
                    MAX(player_name) as player_name,
                    MIN(season) as start_season,
                    MAX(season) as end_season,
                    COUNT(DISTINCT season) as num_seasons,
                    AVG(gp) as avg_gp,
                    AVG(min) as avg_min,
                    AVG(pts) as avg_pts,
                    AVG(reb) as avg_reb,
                    AVG(ast) as avg_ast,
                    AVG(stl) as avg_stl,
                    AVG(blk) as avg_blk,
                    AVG(tov) as avg_tov,
                    AVG(fg_pct) as avg_fg_pct,
                    AVG(fg3_pct) as avg_fg3_pct,
                    AVG(ft_pct) as avg_ft_pct,
                    NOW() as last_updated
                FROM player_stats
                WHERE season >= '{start_season}-{str(start_season+1)[2:]}'
                AND season <= '{start_season+num_seasons-1}-{str(start_season+num_seasons)[2:]}'
                GROUP BY player_id
            """)
            
            # Update metadata
            self.conn.execute("""
                INSERT OR REPLACE INTO data_metadata (key, value, last_updated)
                VALUES ('last_average_calculation', NOW()::VARCHAR, NOW())
            """)
            
            logger.info("Successfully calculated and stored season averages")
        except Exception as e:
            logger.error(f"Failed to calculate season averages: {str(e)}")
            raise

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            assert self.conn is not None  # For type checking
            self.conn.close()

def update_player_stats(use_motherduck: bool = True, start_season: int = 2021, num_seasons: int = 4) -> None:
    """Main pipeline execution function to fetch and calculate multi-season averages.
    
    Args:
        use_motherduck: Whether to use MotherDuck cloud database
        start_season: First season to fetch (e.g., 2021 for 2021-22 season)
        num_seasons: Number of seasons to fetch
    """
    try:
        pipeline = NBADataPipeline(use_motherduck)
        conn = get_db_connection(use_motherduck)
        pipeline.set_connection(conn)
        pipeline._init_schema()
        pipeline.calculate_season_averages(start_season, num_seasons)
        
        # Sync to MotherDuck if enabled
        if use_motherduck:
            db_config = DatabaseConfig(use_motherduck)
            db_config.sync_to_motherduck()
            
        logger.info(f"Successfully processed {num_seasons} seasons of player stats starting from {start_season}")
    except Exception as e:
        logger.error(f"Failed to update player stats: {str(e)}")
        raise
    finally:
        if 'pipeline' in locals():
            pipeline.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_player_stats()
