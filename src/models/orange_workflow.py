"""Export data to Orange3 format and create model analysis workflow."""

import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import duckdb
from datetime import datetime
from src.models.prop_features_config import PROP_FEATURES_CONFIG
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

def get_feature_data(
    prop_type: str,
    min_games: int = 10,
    start_date: Optional[str] = None
) -> pd.DataFrame:
    """Get feature data for Orange3 analysis.
    
    Args:
        prop_type: Type of prop to analyze
        min_games: Minimum games required for player inclusion
        start_date: Optional start date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with features and target
    """
    if prop_type not in PROP_FEATURES_CONFIG:
        raise ValueError(f"Unsupported prop type: {prop_type}")
    
    config = PROP_FEATURES_CONFIG[prop_type]
    primary_stat = config['primary_stat']
    
    # Build feature columns
    feature_cols = [
        'player_id',
        'player_name',
        'team_abbreviation',
        'game_date',
        'is_home',
        f'{primary_stat} as target_value',
        'min'
    ]
    
    # Add configured features
    feature_cols.extend(config['features'])
    
    # Build query
    query = f"""
        SELECT 
            {', '.join(feature_cols)}
        FROM player_stats
        WHERE game_date IS NOT NULL
    """
    
    if start_date:
        query += f" AND game_date >= '{start_date}'"
    
    query += " ORDER BY player_id, game_date"
    
    # Get data
    conn = get_db_connection(use_motherduck=False)
    try:
        df = conn.execute(query).fetchdf()
        
        # Filter players with minimum games
        player_games = df.groupby('player_id').size()
        valid_players = player_games[player_games >= min_games].index
        df = df[df['player_id'].isin(valid_players)]
        
        # Convert boolean to int for Orange
        df['is_home'] = df['is_home'].astype(int)
        
        # Add metadata for Orange
        feature_types = {
            'player_id': 'string',
            'player_name': 'string',
            'team_abbreviation': 'string',
            'game_date': 'time',
            'is_home': 'discrete',
            'target_value': 'continuous',
            'min': 'continuous'
        }
        
        # Add types for configured features
        for feature in config['features']:
            if feature not in feature_types:
                feature_types[feature] = 'continuous'
        
        # Save feature types
        df.attrs['feature_types'] = feature_types
        
        return df
        
    finally:
        conn.close()

def export_to_orange(
    df: pd.DataFrame,
    output_path: str,
    include_metadata: bool = True
) -> None:
    """Export DataFrame to Orange3 format.
    
    Args:
        df: DataFrame to export
        output_path: Path to save .tab file
        include_metadata: Whether to include Orange metadata
    """
    # Get feature types
    feature_types = df.attrs.get('feature_types', {})
    
    # Create Orange header
    if include_metadata:
        header = []
        
        # Add feature names and types
        for col in df.columns:
            ftype = feature_types.get(col, 'continuous')
            if ftype == 'string':
                header.append(f'{col}\tstring')
            elif ftype == 'discrete':
                header.append(f'{col}\tdiscrete')
            elif ftype == 'time':
                header.append(f'{col}\ttime')
            else:
                header.append(f'{col}\tcontinuous')
        
        # Add class variable (target)
        header.append('target_value\tcontinuous')
        
        # Write header
        with open(output_path, 'w') as f:
            f.write('\n'.join(header) + '\n')
            
        # Write data
        df.to_csv(output_path, sep='\t', index=False, mode='a')
        
    else:
        # Just write data without metadata
        df.to_csv(output_path, sep='\t', index=False)
    
    logger.info(f"Data exported to {output_path}")

def export_all_props(
    output_dir: str = 'data/orange',
    min_games: int = 10,
    days_history: int = 365
) -> None:
    """Export all prop types to Orange format.
    
    Args:
        output_dir: Directory to save .tab files
        min_games: Minimum games required for player inclusion
        days_history: Number of days of history to include
    """
    import os
    from datetime import datetime, timedelta
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate start date
    start_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')
    
    # Export each prop type
    for prop_type in PROP_FEATURES_CONFIG.keys():
        logger.info(f"\nExporting {prop_type} data...")
        
        # Get data
        df = get_feature_data(
            prop_type,
            min_games=min_games,
            start_date=start_date
        )
        
        # Export
        output_path = os.path.join(output_dir, f"{prop_type}.tab")
        export_to_orange(df, output_path)
        
        # Log stats
        logger.info(f"Exported {len(df)} rows")
        logger.info(f"Features: {', '.join(df.columns)}")

def main() -> int:
    """Export data for Orange3 analysis.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Export all prop types
        export_all_props()
        
        logger.info("\nOrange3 Export Complete!")
        logger.info("\nSuggested Orange3 Workflow:")
        logger.info("1. Load data using 'File' widget")
        logger.info("2. Connect to 'Data Table' to inspect")
        logger.info("3. Use 'Feature Statistics' for distributions")
        logger.info("4. Add 'Correlations' widget")
        logger.info("5. Try different learners:")
        logger.info("   - Gradient Boosting")
        logger.info("   - Random Forest")
        logger.info("   - Neural Network")
        logger.info("6. Use 'Test & Score' for evaluation")
        logger.info("7. Add 'ROC Analysis' and 'Lift Curve'")
        logger.info("8. Use 'Feature Importance' for selection")
        
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
