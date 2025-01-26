"""Check data in MotherDuck database."""

import logging
import sys
from pandas import DataFrame
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def fetch_sample_data() -> DataFrame:
    """Query sample data from MotherDuck.
    
    Returns:
        DataFrame containing sample player statistics
        
    Raises:
        Exception: If database query fails
    """
    conn = get_db_connection(use_motherduck=True)
    try:
        return conn.execute("""
            SELECT player_name, team_abbreviation, pts, ast 
            FROM player_stats 
            LIMIT 5
        """).fetchdf()
    finally:
        conn.close()

def main() -> int:
    """Display sample data from MotherDuck database.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Fetching sample data from MotherDuck...")
        result = fetch_sample_data()
        
        print("\nData in MotherDuck:")
        print(result)
        print()
        
        logger.info("Successfully retrieved sample data")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
