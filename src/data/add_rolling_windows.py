"""Add rolling window statistics to player_stats table."""

import logging
import sys
from typing import List
import duckdb
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def add_rolling_windows() -> None:
    """Add rolling window statistics for all prop types."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Add rolling windows for each stat type
        stats = ['pts', 'reb', 'ast', 'fg3m', 'fg_pct', 'fg3_pct', 'ft_pct', 'plus_minus']
        windows = [5, 10, 20]
        
        for stat in stats:
            for window in windows:
                # Check if column exists
                result = conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = 'player_stats' 
                    AND column_name = '{stat}_rolling_{window}'
                """).fetchone()
                
                if result[0] == 0:
                    logger.info(f"Adding {stat}_rolling_{window} column...")
                    
                    # Add rolling average column
                    conn.execute(f"""
                        ALTER TABLE player_stats 
                        ADD COLUMN {stat}_rolling_{window} FLOAT
                    """)
                    
                    # Calculate rolling averages
                    conn.execute(f"""
                        WITH rolling_stats AS (
                            SELECT 
                                *,
                                AVG({stat}) OVER (
                                    PARTITION BY player_id 
                                    ORDER BY game_date 
                                    ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW
                                ) as rolling_avg
                            FROM player_stats
                        )
                        UPDATE player_stats
                        SET {stat}_rolling_{window} = rolling_stats.rolling_avg
                        FROM rolling_stats
                        WHERE player_stats.player_id = rolling_stats.player_id
                        AND player_stats.game_date = rolling_stats.game_date
                    """)
                    
                    logger.info(f"Added {stat}_rolling_{window} column")
        
        # Add home/away splits if they don't exist
        splits = ['pts', 'ast', 'reb', 'fg3m']
        for stat in splits:
            # Check home column
            result = conn.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'player_stats' 
                AND column_name = '{stat}_home'
            """).fetchone()
            
            if result[0] == 0:
                logger.info(f"Adding {stat}_home and {stat}_away columns...")
                
                # Add home/away columns
                conn.execute(f"""
                    ALTER TABLE player_stats 
                    ADD COLUMN {stat}_home FLOAT;
                    
                    ALTER TABLE player_stats 
                    ADD COLUMN {stat}_away FLOAT;
                """)
                
                # Calculate home/away averages
                conn.execute(f"""
                    WITH home_stats AS (
                        SELECT 
                            player_id,
                            AVG({stat}) as home_avg
                        FROM player_stats
                        WHERE is_home = true
                        GROUP BY player_id
                    ),
                    away_stats AS (
                        SELECT 
                            player_id,
                            AVG({stat}) as away_avg
                        FROM player_stats
                        WHERE is_home = false
                        GROUP BY player_id
                    )
                    UPDATE player_stats
                    SET 
                        {stat}_home = home_stats.home_avg,
                        {stat}_away = away_stats.away_avg
                    FROM home_stats, away_stats
                    WHERE player_stats.player_id = home_stats.player_id
                    AND player_stats.player_id = away_stats.player_id
                """)
                
                logger.info(f"Added {stat}_home and {stat}_away columns")
        
        logger.info("All rolling windows and splits added successfully")
        
    finally:
        conn.close()

def main() -> int:
    """Add rolling window statistics.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        add_rolling_windows()
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
