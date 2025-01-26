"""Database configuration and connection management."""

import os
from typing import Optional
from duckdb import DuckDBPyConnection
import duckdb
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    """Database configuration and connection management."""
    
    def __init__(self, use_motherduck: bool = True) -> None:
        """Initialize database configuration.
        
        Args:
            use_motherduck: Whether to use MotherDuck cloud database
        """
        self.use_motherduck = use_motherduck
        self.local_path = os.getenv('LOCAL_DB_PATH', 'data/nba_stats.duckdb')
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
        
        config = {
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
                schema = local_conn.execute(f"""
                    SELECT sql FROM sqlite_master 
                    WHERE type='table' AND name='{table_name}'
                """).fetchone()[0]
                
                md_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                md_conn.execute(schema)
                
                # Copy data
                local_conn.execute(f"""
                    COPY (SELECT * FROM {table_name}) 
                    TO 'temp_{table_name}.parquet' (FORMAT PARQUET)
                """)
                
                md_conn.execute(f"""
                    COPY {table_name} FROM 'temp_{table_name}.parquet'
                """)
                
                # Cleanup
                os.remove(f"temp_{table_name}.parquet")
                
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
