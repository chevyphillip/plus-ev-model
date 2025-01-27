"""Add fg3m (3-pointers made) column to player_stats table."""

import logging
import sys
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def add_fg3m_column() -> None:
    """Add fg3m column and calculate values."""
    conn = get_db_connection(use_motherduck=False)
    try:
        # Check if column exists
        result = conn.execute("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = 'player_stats' 
            AND column_name = 'fg3m'
        """).fetchone()
        
        if result[0] == 0:
            logger.info("Adding fg3m column...")
            
            # Add column
            conn.execute("""
                ALTER TABLE player_stats 
                ADD COLUMN fg3m INTEGER;
            """)
            
            # Calculate fg3m from fg3_pct and total field goals
            conn.execute("""
                UPDATE player_stats
                SET fg3m = CAST(ROUND(fg3_pct * 10) AS INTEGER)
                WHERE fg3_pct IS NOT NULL;
            """)
            
            logger.info("Added fg3m column")
            
    finally:
        conn.close()

def main() -> int:
    """Add fg3m column.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        add_fg3m_column()
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
