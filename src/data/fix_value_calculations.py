"""Fix value calculations and opponent correlations in player statistics."""

import logging
import sys
from typing import Dict, Any, Tuple, Optional, cast
import duckdb
import pandas as pd
import numpy as np
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def fix_rolling_averages() -> None:
    """Fix rolling average calculations in player_stats table."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Update rolling averages using window functions
        conn.execute("""
            WITH rolling_stats AS (
                SELECT 
                    player_id,
                    game_date,
                    pts,
                    ast,
                    reb,
                    AVG(pts) OVER (
                        PARTITION BY player_id 
                        ORDER BY game_date 
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) as new_pts_rolling_5,
                    AVG(ast) OVER (
                        PARTITION BY player_id 
                        ORDER BY game_date 
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) as new_ast_rolling_5,
                    AVG(reb) OVER (
                        PARTITION BY player_id 
                        ORDER BY game_date 
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) as new_reb_rolling_5
                FROM player_stats
            )
            UPDATE player_stats ps
            SET 
                pts_rolling_5 = rs.new_pts_rolling_5,
                ast_rolling_5 = rs.new_ast_rolling_5,
                reb_rolling_5 = rs.new_reb_rolling_5
            FROM rolling_stats rs
            WHERE ps.player_id = rs.player_id 
            AND ps.game_date = rs.game_date
        """)
        
        logger.info("Rolling averages updated successfully")
        
        # Verify the update
        verification_result: Optional[Tuple[int, int, int, int]] = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(pts_rolling_5) as pts_rolling_count,
                COUNT(ast_rolling_5) as ast_rolling_count,
                COUNT(reb_rolling_5) as reb_rolling_count
            FROM player_stats
        """).fetchone()
        
        if verification_result is None:
            raise ValueError("Failed to get verification counts")
            
        total_rows, pts_count, ast_count, reb_count = verification_result
        
        logger.info(f"Verification - Total rows: {total_rows}")
        logger.info(f"Rows with rolling pts: {pts_count}")
        logger.info(f"Rows with rolling ast: {ast_count}")
        logger.info(f"Rows with rolling reb: {reb_count}")
        
    except Exception as e:
        logger.error(f"Failed to update rolling averages: {str(e)}")
        raise
    finally:
        conn.close()

def update_opponent_stats() -> None:
    """Update opponent statistics calculations."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Calculate opponent averages using proper window functions
        conn.execute("""
            WITH opp_stats AS (
                SELECT 
                    player_id,
                    game_date,
                    team_id as opp_team_id,
                    AVG(pts) OVER (
                        PARTITION BY team_id
                        ORDER BY game_date 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as new_opp_pts_allowed_avg,
                    AVG(ast) OVER (
                        PARTITION BY team_id
                        ORDER BY game_date 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as new_opp_ast_allowed_avg,
                    AVG(reb) OVER (
                        PARTITION BY team_id
                        ORDER BY game_date 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as new_opp_reb_avg
                FROM player_stats
            )
            UPDATE player_stats ps
            SET 
                opp_pts_allowed_avg = os.new_opp_pts_allowed_avg,
                opp_ast_allowed_avg = os.new_opp_ast_allowed_avg,
                opp_reb_rate = os.new_opp_reb_avg / 40.0  -- Normalize to a rate between 0-1
            FROM opp_stats os
            WHERE ps.player_id = os.player_id 
            AND ps.game_date = os.game_date
        """)
        
        logger.info("Opponent statistics updated successfully")
        
        # Verify the update
        result: Optional[Tuple[int, int, int, int]] = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(opp_pts_allowed_avg) as pts_count,
                COUNT(opp_ast_allowed_avg) as ast_count,
                COUNT(opp_reb_rate) as reb_rate_count
            FROM player_stats
        """).fetchone()
        
        if result is None:
            raise ValueError("Failed to get verification counts")
            
        total_rows, pts_count, ast_count, reb_count = result
        
        logger.info(f"Verification - Total rows: {total_rows}")
        logger.info(f"Rows with opp pts avg: {pts_count}")
        logger.info(f"Rows with opp ast avg: {ast_count}")
        logger.info(f"Rows with opp reb rate: {reb_count}")
        
    except Exception as e:
        logger.error(f"Failed to update opponent statistics: {str(e)}")
        raise
    finally:
        conn.close()

def verify_calculations() -> None:
    """Verify all value calculations are correct."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Check for any remaining NULL values
        null_check_result: Optional[Tuple[int, int, int, int, int, int, int]] = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                SUM(CASE WHEN pts_rolling_5 IS NULL THEN 1 ELSE 0 END) as null_pts_rolling,
                SUM(CASE WHEN ast_rolling_5 IS NULL THEN 1 ELSE 0 END) as null_ast_rolling,
                SUM(CASE WHEN reb_rolling_5 IS NULL THEN 1 ELSE 0 END) as null_reb_rolling,
                SUM(CASE WHEN opp_pts_allowed_avg IS NULL THEN 1 ELSE 0 END) as null_opp_pts,
                SUM(CASE WHEN opp_ast_allowed_avg IS NULL THEN 1 ELSE 0 END) as null_opp_ast,
                SUM(CASE WHEN opp_reb_rate IS NULL THEN 1 ELSE 0 END) as null_opp_reb
            FROM player_stats
        """).fetchone()
        
        if null_check_result is None:
            raise ValueError("Failed to get null counts")
            
        (total_rows, null_pts, null_ast, null_reb, 
         null_opp_pts, null_opp_ast, null_opp_reb) = null_check_result
        
        logger.info("Null value check:")
        logger.info(f"Total rows: {total_rows}")
        logger.info(f"Null rolling pts: {null_pts}")
        logger.info(f"Null rolling ast: {null_ast}")
        logger.info(f"Null rolling reb: {null_reb}")
        logger.info(f"Null opp pts: {null_opp_pts}")
        logger.info(f"Null opp ast: {null_opp_ast}")
        logger.info(f"Null opp reb: {null_opp_reb}")
        
        # Verify calculations are within expected ranges
        range_result: Optional[Tuple[float, float, float, float, float, float]] = conn.execute("""
            SELECT 
                MIN(pts_rolling_5) as min_pts_rolling,
                MAX(pts_rolling_5) as max_pts_rolling,
                MIN(opp_pts_allowed_avg) as min_opp_pts,
                MAX(opp_pts_allowed_avg) as max_opp_pts,
                MIN(opp_reb_rate) as min_reb_rate,
                MAX(opp_reb_rate) as max_reb_rate
            FROM player_stats
        """).fetchone()
        
        if range_result is None:
            raise ValueError("Failed to get value ranges")
            
        min_pts, max_pts, min_opp_pts, max_opp_pts, min_reb_rate, max_reb_rate = range_result
        
        logger.info("\nValue ranges:")
        logger.info(f"Rolling pts range: {min_pts:.1f} to {max_pts:.1f}")
        logger.info(f"Opp pts range: {min_opp_pts:.1f} to {max_opp_pts:.1f}")
        logger.info(f"Reb rate range: {min_reb_rate:.3f} to {max_reb_rate:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to verify calculations: {str(e)}")
        raise
    finally:
        conn.close()

def main() -> int:
    """Fix value calculations in player statistics.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting value calculation fixes...")
        
        logger.info("Fixing rolling averages...")
        fix_rolling_averages()
        
        logger.info("Updating opponent statistics...")
        update_opponent_stats()
        
        logger.info("Verifying calculations...")
        verify_calculations()
        
        logger.info("Value calculation fixes complete")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to fix value calculations: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
