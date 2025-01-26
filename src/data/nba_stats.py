"""NBA statistics data pipeline."""

import logging
import os
import pandas as pd
from datetime import datetime
import time
from nba_api.stats.endpoints import leaguedashplayerstats  # type: ignore
from duckdb import DuckDBPyConnection
from pandas import DataFrame, Index
from dotenv import load_dotenv
from src.data.db_config import DatabaseConfig

load_dotenv()
logger = logging.getLogger(__name__)

# Configure NBA API delay
NBA_API_DELAY = float(os.getenv('NBA_API_DELAY', '1.0'))


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
            DROP TABLE IF EXISTS player_stats;
            
            CREATE TABLE player_stats (
                player_id INTEGER PRIMARY KEY,
                player_name VARCHAR,
                team_id INTEGER,
                team_abbreviation VARCHAR,
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
                last_updated TIMESTAMP
            );
            
            CREATE UNIQUE INDEX IF NOT EXISTS player_stats_id_idx ON player_stats(player_id)
        """)
    
    def fetch_player_stats(self) -> DataFrame:
        """Fetch current player statistics from NBA API.
        
        Returns:
            DataFrame containing player statistics with last_updated timestamp
        """
        try:
            # Add delay to avoid rate limiting
            time.sleep(NBA_API_DELAY)
            
            # Get stats from NBA API
            raw_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                timeout=30,
                per_mode_detailed='PerGame'
            ).get_data_frames()
            
            if not raw_stats or len(raw_stats) == 0:
                raise ValueError("No data returned from NBA API")
            
            # Convert to DataFrame and process
            stats = DataFrame(raw_stats[0])
            stats.columns = Index([col.lower() for col in stats.columns])
            stats['last_updated'] = datetime.now()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            raise
    
    def store_stats_in_db(self, stats: pd.DataFrame) -> None:
        """Store player statistics in the database.
        
        Args:
            stats: DataFrame containing player statistics
        """
        try:
            # Map DataFrame columns to our schema
            stats_to_insert = stats[[
                'player_id', 'player_name', 'team_id', 'team_abbreviation',
                'age', 'gp', 'min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a',
                'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb',
                'ast', 'stl', 'blk', 'tov', 'pts', 'plus_minus', 'last_updated'
            ]]
            
            # Convert DataFrame to DuckDB table
            self.conn.register('stats_df', stats_to_insert)
            
            # Insert new data with explicit column list
            self.conn.execute("""
                INSERT INTO player_stats (
                    player_id, player_name, team_id, team_abbreviation,
                    age, gp, min, fgm, fga, fg_pct, fg3m, fg3a,
                    fg3_pct, ftm, fta, ft_pct, oreb, dreb, reb,
                    ast, stl, blk, tov, pts, plus_minus, last_updated
                )
                SELECT * FROM stats_df
                ON CONFLICT (player_id) DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    team_id = EXCLUDED.team_id,
                    team_abbreviation = EXCLUDED.team_abbreviation,
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
                    last_updated = EXCLUDED.last_updated
            """)
            
            # Cleanup
            self.conn.unregister('stats_df')
            
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
    logging.basicConfig(level=logging.INFO)
    update_player_stats()
