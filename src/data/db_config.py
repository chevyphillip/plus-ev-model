"""Database configuration and connection management."""

import os
import logging
from typing import Dict, Any, Tuple, Optional, cast
from pathlib import Path
from duckdb import DuckDBPyConnection
import duckdb
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    # Configure logging
    logger = logging.getLogger(__name__)
    """Database configuration and connection management."""
    
    def __init__(self, use_motherduck: bool = True) -> None:
        """Initialize database configuration.
        
        Args:
            use_motherduck: Whether to use MotherDuck cloud database
        """
        self.use_motherduck = use_motherduck
        local_path = os.getenv('LOCAL_DB_PATH', 'data/nba_stats.duckdb')
        self.local_path = str(Path(local_path).resolve())
        self.motherduck_token = os.getenv('MOTHERDUCK_TOKEN')
        
        if self.use_motherduck and not self.motherduck_token:
            raise ValueError("MOTHERDUCK_TOKEN environment variable is required when using MotherDuck")
    
    def get_connection_string(self) -> str:
        """Get the appropriate database connection string."""
        if self.use_motherduck:
            return "md:nba-ml-model-db"
        return self.local_path
    
    def connect(self) -> DuckDBPyConnection:
        """Create a database connection."""
        conn_str = self.get_connection_string()
        
        # Validate local path exists or can be created
        if not self.use_motherduck:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        
        config: Dict[str, Any] = {
            "motherduck_token": self.motherduck_token
        } if self.use_motherduck else {}
        
        return duckdb.connect(conn_str, config=config)
    
    def sync_to_motherduck(self) -> None:
        """Sync local database to MotherDuck."""
        if not self.use_motherduck:
            return
            
        # Connect to local database
        local_conn = duckdb.connect(self.local_path)
        
        try:
            # Connect to MotherDuck
            md_conn = self.connect()
            
            # Get list of tables from local database
            tables = local_conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()
            
            # Sync each table
            for (table_name,) in tables:
                print(f"Syncing table {table_name}...")
                
                # Create table in MotherDuck if it doesn't exist
                # Get table schema
                result: Optional[Tuple[str]] = cast(
                    Optional[Tuple[str]], 
                    local_conn.execute(f"""
                        SELECT sql FROM sqlite_master 
                        WHERE type='table' AND name='{table_name}'
                    """).fetchone()
                )
                
                if result is None or not result[0]:
                    self.logger.error(f"Could not find schema for table {table_name}")
                    continue
                
                # Create table in MotherDuck
                try:
                    md_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    md_conn.execute(result[0])
                except Exception as e:
                    self.logger.error(f"Failed to create table {table_name}: {str(e)}")
                    continue
                
                # Copy data
                temp_file = f"temp_{table_name}.parquet"
                try:
                    local_conn.execute(f"""
                        COPY (SELECT * FROM {table_name}) 
                        TO '{temp_file}' (FORMAT PARQUET)
                    """)
                    
                    md_conn.execute(f"""
                        COPY {table_name} FROM '{temp_file}'
                    """)
                except Exception as e:
                    self.logger.error(f"Failed to copy data for table {table_name}: {str(e)}")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
            print("Database sync complete!")
            
        finally:
            local_conn.close()
            if 'md_conn' in locals():
                md_conn.close()

def get_db_connection(use_motherduck: bool = True) -> DuckDBPyConnection:
    """Get a database connection.
    
    Args:
        use_motherduck: Whether to use MotherDuck cloud database
        
    Returns:
        DuckDB connection
    """
    config = DatabaseConfig(use_motherduck)
    return config.connect()
