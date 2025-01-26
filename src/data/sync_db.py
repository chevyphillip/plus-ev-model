"""Sync local database to MotherDuck."""

import logging
import sys
from src.data.db_config import DatabaseConfig

logger = logging.getLogger(__name__)

def main() -> int:
    """Sync local database to MotherDuck."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting database sync to MotherDuck...")
        config = DatabaseConfig()
        config.sync_to_motherduck()
        logger.info("Database sync completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to sync database: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
