"""Fix missing team information in player_stats table."""

import logging
import sys
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def update_team_info() -> None:
    """Update team_id and team_abbreviation in player_stats from nba_players table."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # First verify the number of records needing updates
        null_count = conn.execute("""
            SELECT COUNT(*) 
            FROM player_stats 
            WHERE team_id IS NULL OR team_abbreviation IS NULL
        """).fetchone()[0]
        
        logger.info(f"Found {null_count} records with missing team information")
        
        # Update the records using nba_players table
        updated = conn.execute("""
            WITH latest_teams AS (
                SELECT player_id, team_id, team_abbreviation
                FROM nba_players
                WHERE last_updated = (
                    SELECT MAX(last_updated)
                    FROM nba_players np2
                    WHERE np2.player_id = nba_players.player_id
                )
            )
            UPDATE player_stats ps
            SET 
                team_id = lt.team_id,
                team_abbreviation = lt.team_abbreviation
            FROM latest_teams lt
            WHERE ps.player_id = lt.player_id
            AND (ps.team_id IS NULL OR ps.team_abbreviation IS NULL)
        """)
        
        # Verify the update
        remaining_nulls = conn.execute("""
            SELECT COUNT(*) 
            FROM player_stats 
            WHERE team_id IS NULL OR team_abbreviation IS NULL
        """).fetchone()[0]
        
        logger.info(f"Update complete. {null_count - remaining_nulls} records updated")
        if remaining_nulls > 0:
            logger.warning(f"{remaining_nulls} records still have missing team information")
            
            # Get sample of players still missing team info
            sample_missing = conn.execute("""
                SELECT DISTINCT player_id, player_name
                FROM player_stats
                WHERE team_id IS NULL OR team_abbreviation IS NULL
                LIMIT 5
            """).fetchdf()
            
            logger.warning("Sample players still missing team info:")
            for _, row in sample_missing.iterrows():
                logger.warning(f"Player ID: {row['player_id']}, Name: {row['player_name']}")
        
        # Sync to MotherDuck if successful
        if remaining_nulls < null_count:
            logger.info("Syncing updates to MotherDuck...")
            md_conn = get_db_connection(use_motherduck=True)
            try:
                md_conn.execute("""
                    UPDATE player_stats ps
                    SET 
                        team_id = lt.team_id,
                        team_abbreviation = lt.team_abbreviation
                    FROM (
                        SELECT player_id, team_id, team_abbreviation
                        FROM nba_players
                        WHERE last_updated = (
                            SELECT MAX(last_updated)
                            FROM nba_players np2
                            WHERE np2.player_id = nba_players.player_id
                        )
                    ) lt
                    WHERE ps.player_id = lt.player_id
                    AND (ps.team_id IS NULL OR ps.team_abbreviation IS NULL)
                """)
                logger.info("MotherDuck sync complete")
            finally:
                md_conn.close()
                
    finally:
        conn.close()

def verify_data_consistency() -> None:
    """Verify data consistency between player_stats and nba_players."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Check for mismatches between player_stats and nba_players
        mismatches = conn.execute("""
            WITH latest_teams AS (
                SELECT player_id, team_id, team_abbreviation
                FROM nba_players
                WHERE last_updated = (
                    SELECT MAX(last_updated)
                    FROM nba_players np2
                    WHERE np2.player_id = nba_players.player_id
                )
            )
            SELECT 
                ps.player_id,
                ps.player_name,
                ps.team_id as ps_team_id,
                ps.team_abbreviation as ps_team_abbr,
                lt.team_id as np_team_id,
                lt.team_abbreviation as np_team_abbr
            FROM player_stats ps
            JOIN latest_teams lt ON ps.player_id = lt.player_id
            WHERE ps.team_id != lt.team_id 
               OR ps.team_abbreviation != lt.team_abbreviation
            LIMIT 5
        """).fetchdf()
        
        if not mismatches.empty:
            logger.warning("Found team information mismatches:")
            logger.warning(mismatches)
    finally:
        conn.close()

def main() -> int:
    """Update team information in player_stats table.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting team information update...")
        update_team_info()
        
        logger.info("Verifying data consistency...")
        verify_data_consistency()
        
        logger.info("Process complete")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to update team information: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
