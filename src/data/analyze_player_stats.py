"""Analyze player statistics and value calculations."""

import logging
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def get_player_stats_summary() -> pd.DataFrame:
    """Get summary statistics for player performance metrics."""
    conn = get_db_connection(use_motherduck=False)
    try:
        df = conn.execute("""
            SELECT 
                player_name,
                COUNT(*) as games_played,
                AVG(min) as avg_minutes,
                AVG(pts) as avg_points,
                AVG(ast) as avg_assists,
                AVG(reb) as avg_rebounds,
                AVG(fg_pct) as avg_fg_pct,
                AVG(fg3_pct) as avg_fg3_pct,
                AVG(ft_pct) as avg_ft_pct,
                AVG(plus_minus) as avg_plus_minus,
                team_abbreviation
            FROM player_stats
            GROUP BY player_name, team_abbreviation
            HAVING COUNT(*) >= 5  -- Filter for players with sufficient games
            ORDER BY avg_points DESC
            LIMIT 20
        """).fetchdf()
        return df
    except Exception as e:
        logger.error(f"Failed to get player stats summary: {str(e)}")
        raise
    finally:
        conn.close()

def analyze_home_away_splits() -> pd.DataFrame:
    """Analyze performance differences between home and away games."""
    conn = get_db_connection(use_motherduck=False)
    try:
        df = conn.execute("""
            WITH player_averages AS (
                SELECT 
                    player_name,
                    team_abbreviation,
                    AVG(CASE WHEN is_home THEN pts END) as home_pts,
                    AVG(CASE WHEN NOT is_home THEN pts END) as away_pts,
                    AVG(CASE WHEN is_home THEN ast END) as home_ast,
                    AVG(CASE WHEN NOT is_home THEN ast END) as away_ast,
                    AVG(CASE WHEN is_home THEN reb END) as home_reb,
                    AVG(CASE WHEN NOT is_home THEN reb END) as away_reb,
                    COUNT(*) as total_games
                FROM player_stats
                GROUP BY player_name, team_abbreviation
                HAVING COUNT(*) >= 10
            )
            SELECT 
                player_name,
                team_abbreviation,
                home_pts - away_pts as pts_home_diff,
                home_ast - away_ast as ast_home_diff,
                home_reb - away_reb as reb_home_diff,
                total_games
            FROM player_averages
            ORDER BY ABS(pts_home_diff) DESC
            LIMIT 20
        """).fetchdf()
        return df
    except Exception as e:
        logger.error(f"Failed to analyze home/away splits: {str(e)}")
        raise
    finally:
        conn.close()

def verify_rolling_averages() -> pd.DataFrame:
    """Verify rolling average calculations."""
    conn = get_db_connection(use_motherduck=False)
    try:
        df = conn.execute("""
            WITH recent_games AS (
                SELECT 
                    player_name,
                    team_abbreviation,
                    game_date,
                    pts,
                    pts_rolling_5,
                    ast,
                    ast_rolling_5,
                    reb,
                    reb_rolling_5
                FROM player_stats
                WHERE game_date >= CURRENT_DATE - INTERVAL 30 DAY
                ORDER BY player_name, game_date DESC
            )
            SELECT 
                player_name,
                team_abbreviation,
                game_date,
                pts,
                pts_rolling_5,
                ABS(pts - pts_rolling_5) as pts_diff,
                ast,
                ast_rolling_5,
                ABS(ast - ast_rolling_5) as ast_diff,
                reb,
                reb_rolling_5,
                ABS(reb - reb_rolling_5) as reb_diff
            FROM recent_games
            WHERE pts_rolling_5 IS NOT NULL
            ORDER BY pts_diff DESC
            LIMIT 20
        """).fetchdf()
        return df
    except Exception as e:
        logger.error(f"Failed to verify rolling averages: {str(e)}")
        raise
    finally:
        conn.close()

def check_statistical_anomalies() -> pd.DataFrame:
    """Identify potential statistical anomalies."""
    conn = get_db_connection(use_motherduck=False)
    try:
        df = conn.execute("""
            WITH player_stats_z AS (
                SELECT 
                    player_name,
                    team_abbreviation,
                    game_date,
                    pts,
                    ast,
                    reb,
                    (pts - AVG(pts) OVER (PARTITION BY player_name)) / 
                        NULLIF(STDDEV(pts) OVER (PARTITION BY player_name), 0) as pts_z,
                    (ast - AVG(ast) OVER (PARTITION BY player_name)) / 
                        NULLIF(STDDEV(ast) OVER (PARTITION BY player_name), 0) as ast_z,
                    (reb - AVG(reb) OVER (PARTITION BY player_name)) / 
                        NULLIF(STDDEV(reb) OVER (PARTITION BY player_name), 0) as reb_z
                FROM player_stats
            )
            SELECT *
            FROM player_stats_z
            WHERE ABS(pts_z) > 2 OR ABS(ast_z) > 2 OR ABS(reb_z) > 2
            ORDER BY ABS(pts_z) DESC
            LIMIT 20
        """).fetchdf()
        return df
    except Exception as e:
        logger.error(f"Failed to check statistical anomalies: {str(e)}")
        raise
    finally:
        conn.close()

def analyze_value_calculations() -> Dict[str, Any]:
    """Analyze value calculations and betting implications."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Get overall statistics
        stats = {}
        
        # Points analysis
        pts_analysis = conn.execute("""
            WITH point_stats AS (
                SELECT 
                    player_name,
                    team_abbreviation,
                    AVG(pts) as avg_pts,
                    STDDEV(pts) as std_pts,
                    COUNT(*) as games,
                    AVG(pts_rolling_5) as avg_rolling_pts,
                    CORR(pts, pts_rolling_5) as pts_rolling_correlation
                FROM player_stats
                GROUP BY player_name, team_abbreviation
                HAVING COUNT(*) >= 10
            )
            SELECT 
                AVG(avg_pts) as league_avg_pts,
                AVG(std_pts) as league_avg_std,
                AVG(pts_rolling_correlation) as avg_rolling_correlation,
                COUNT(*) as num_players
            FROM point_stats
        """).fetchdf()
        stats['points'] = pts_analysis.to_dict('records')[0]
        
        # Home/Away impact
        venue_impact = conn.execute("""
            SELECT 
                AVG(CASE WHEN is_home THEN pts END) as avg_home_pts,
                AVG(CASE WHEN NOT is_home THEN pts END) as avg_away_pts,
                AVG(CASE WHEN is_home THEN ast END) as avg_home_ast,
                AVG(CASE WHEN NOT is_home THEN ast END) as avg_away_ast
            FROM player_stats
        """).fetchdf()
        stats['venue_impact'] = venue_impact.to_dict('records')[0]
        
        # Opponent impact
        opp_impact = conn.execute("""
            SELECT 
                CORR(pts, opp_pts_allowed_avg) as pts_opp_correlation,
                CORR(ast, opp_ast_allowed_avg) as ast_opp_correlation,
                CORR(reb, opp_reb_rate) as reb_opp_correlation
            FROM player_stats
        """).fetchdf()
        stats['opponent_impact'] = opp_impact.to_dict('records')[0]
        
        return stats
    except Exception as e:
        logger.error(f"Failed to analyze value calculations: {str(e)}")
        raise
    finally:
        conn.close()

def main() -> int:
    """Run comprehensive analysis of player statistics.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Analyzing player statistics...")
        
        # Get top performers summary
        logger.info("\nTop Performers Summary:")
        summary = get_player_stats_summary()
        print("\nPlayer Performance Summary:")
        print(summary)
        
        # Analyze home/away splits
        logger.info("\nAnalyzing home/away splits...")
        splits = analyze_home_away_splits()
        print("\nSignificant Home/Away Splits:")
        print(splits)
        
        # Verify rolling averages
        logger.info("\nVerifying rolling averages...")
        rolling = verify_rolling_averages()
        print("\nLargest Rolling Average Discrepancies:")
        print(rolling)
        
        # Check for anomalies
        logger.info("\nChecking for statistical anomalies...")
        anomalies = check_statistical_anomalies()
        print("\nStatistical Anomalies:")
        print(anomalies)
        
        # Analyze value calculations
        logger.info("\nAnalyzing value calculations...")
        value_analysis = analyze_value_calculations()
        print("\nValue Analysis Results:")
        for category, stats in value_analysis.items():
            print(f"\n{category.upper()}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
        
        logger.info("Analysis complete")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
