"""Comprehensive database inspection script for DuckDB and MotherDuck."""

import logging
import sys
from typing import List, Dict
import pandas as pd
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def get_table_names(conn) -> List[str]:
    """Get list of all tables in database."""
    return [
        row[0] for row in conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
    ]

def get_table_schema(conn, table_name: str) -> str:
    """Get CREATE TABLE statement for given table."""
    result = conn.execute(f"""
        SELECT sql FROM sqlite_master 
        WHERE type='table' AND name='{table_name}'
    """).fetchone()
    return result[0] if result else "Schema not found"

def get_table_stats(conn, table_name: str) -> Dict:
    """Get basic statistics for table."""
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    column_info = conn.execute(f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
    """).fetchall()
    
    return {
        "row_count": row_count,
        "columns": [
            {
                "name": col[0],
                "type": col[1],
                "nullable": col[2]
            }
            for col in column_info
        ]
    }

def get_sample_data(conn, table_name: str, limit: int = 5) -> pd.DataFrame:
    """Get sample rows from table."""
    return conn.execute(f"""
        SELECT * FROM {table_name} LIMIT {limit}
    """).fetchdf()

def check_null_counts(conn, table_name: str) -> Dict:
    """Get null value counts for each column."""
    columns = [
        col[0] for col in conn.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """).fetchall()
    ]
    
    null_counts = {}
    for col in columns:
        count = conn.execute(f"""
            SELECT COUNT(*) 
            FROM {table_name} 
            WHERE {col} IS NULL
        """).fetchone()[0]
        if count > 0:
            null_counts[col] = count
            
    return null_counts

def inspect_database(use_motherduck: bool = True) -> None:
    """Perform comprehensive database inspection."""
    db_type = "MotherDuck" if use_motherduck else "Local DuckDB"
    print(f"\n{'='*20} Inspecting {db_type} Database {'='*20}\n")
    
    conn = get_db_connection(use_motherduck=use_motherduck)
    try:
        # Get all tables
        tables = get_table_names(conn)
        print(f"Found {len(tables)} tables: {', '.join(tables)}\n")
        
        # Inspect each table
        for table in tables:
            print(f"\n{'-'*20} Table: {table} {'-'*20}")
            
            # Schema
            print("\nSchema:")
            print(get_table_schema(conn, table))
            
            # Statistics
            stats = get_table_stats(conn, table)
            print(f"\nRow count: {stats['row_count']}")
            print("\nColumns:")
            for col in stats['columns']:
                print(f"  - {col['name']}: {col['type']} (Nullable: {col['nullable']})")
            
            # Null counts
            null_counts = check_null_counts(conn, table)
            if null_counts:
                print("\nColumns with NULL values:")
                for col, count in null_counts.items():
                    print(f"  - {col}: {count} nulls")
            
            # Sample data
            print("\nSample data:")
            sample = get_sample_data(conn, table)
            print(sample)
            print("\n")
            
    finally:
        conn.close()

def main() -> int:
    """Run database inspection for both local and MotherDuck databases."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Check local database
        inspect_database(use_motherduck=False)
        
        # Check MotherDuck database
        inspect_database(use_motherduck=True)
        
        return 0
        
    except Exception as e:
        logger.error(f"Database inspection failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
