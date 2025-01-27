"""Validate data quality and implement model improvements."""

import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import duckdb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data.db_config import get_db_connection
from src.models.prop_features_config import PROP_FEATURES_CONFIG

logger = logging.getLogger(__name__)

class ModelValidator:
    """Validate and improve model performance."""
    
    def __init__(
        self,
        db_path: str = 'data/nba_stats.duckdb'
    ) -> None:
        """Initialize validator.
        
        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
    
    def check_data_quality(self) -> Dict[str, Dict[str, int]]:
        """Check data quality and completeness.
        
        Returns:
            Dictionary with quality metrics
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            quality_metrics = {}
            
            # Check each prop type
            for prop_type, config in PROP_FEATURES_CONFIG.items():
                metrics = {}
                
                # Count total rows
                result = conn.execute("""
                    SELECT COUNT(*) 
                    FROM player_stats
                    WHERE game_date IS NOT NULL
                """).fetchone()
                if result is None:
                    raise ValueError("Failed to get total row count")
                metrics['total_rows'] = int(result[0])
                
                # Check NULL values in key columns
                for feature in config['features']:
                    result = conn.execute(f"""
                        SELECT COUNT(*)
                        FROM player_stats
                        WHERE {feature} IS NULL
                        AND game_date IS NOT NULL
                    """).fetchone()
                    if result is None:
                        raise ValueError(f"Failed to get NULL count for {feature}")
                    metrics[f'{feature}_nulls'] = int(result[0])
                
                # Check rolling window completeness
                for window in config['rolling_windows']:
                    col = f"{config['primary_stat']}_rolling_{window}"
                    result = conn.execute(f"""
                        SELECT COUNT(*)
                        FROM player_stats
                        WHERE {col} IS NULL
                        AND game_date IS NOT NULL
                    """).fetchone()
                    if result is None:
                        raise ValueError(f"Failed to get NULL count for {col}")
                    metrics[f'{col}_nulls'] = int(result[0])
                
                quality_metrics[prop_type] = metrics
            
            return quality_metrics
            
        finally:
            conn.close()
    
    def validate_rolling_windows(
        self,
        prop_type: str,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """Validate rolling window calculations.
        
        Args:
            prop_type: Type of prop to validate
            sample_size: Number of samples to check
            
        Returns:
            Dictionary with validation metrics
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            config = PROP_FEATURES_CONFIG[prop_type]
            primary_stat = config['primary_stat']
            
            validation_metrics = {}
            
            # Get sample of data
            df = conn.execute(f"""
                SELECT 
                    player_id,
                    game_date,
                    {primary_stat},
                    {primary_stat}_rolling_5,
                    {primary_stat}_rolling_10,
                    {primary_stat}_rolling_20
                FROM player_stats
                WHERE game_date IS NOT NULL
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """).fetchdf()
            
            # Calculate rolling averages manually
            for window in [5, 10, 20]:
                manual_rolling = df.groupby('player_id')[primary_stat].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Compare with stored values
                stored_col = f"{primary_stat}_rolling_{window}"
                diff = np.abs(manual_rolling - df[stored_col])
                
                validation_metrics[f'{window}_day_mae'] = float(diff.mean())
                validation_metrics[f'{window}_day_max_diff'] = float(diff.max())
                
            return validation_metrics
            
        finally:
            conn.close()
    
    def run_backtesting(
        self,
        prop_type: str,
        lookback_days: int = 30,
        min_games: int = 5
    ) -> Dict[str, float]:
        """Run backtesting analysis.
        
        Args:
            prop_type: Type of prop to backtest
            lookback_days: Days to look back for testing
            min_games: Minimum games required for prediction
            
        Returns:
            Dictionary with backtesting metrics
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            config = PROP_FEATURES_CONFIG[prop_type]
            primary_stat = config['primary_stat']
            
            # Get test period data
            result = conn.execute("""
                SELECT MAX(game_date) FROM player_stats
            """).fetchone()
            if result is None or result[0] is None:
                raise ValueError("Failed to get max game date")
            end_date = result[0]
            
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get data for backtesting
            query = f"""
                WITH player_games AS (
                    SELECT player_id, COUNT(*) as games
                    FROM player_stats
                    WHERE game_date < '{start_date}'
                    GROUP BY player_id
                    HAVING COUNT(*) >= {min_games}
                )
                SELECT 
                    ps.*,
                    ps.{primary_stat} as actual_value
                FROM player_stats ps
                JOIN player_games pg ON ps.player_id = pg.player_id
                WHERE ps.game_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY ps.player_id, ps.game_date
            """
            
            df = conn.execute(query).fetchdf()
            
            # Calculate metrics
            metrics: Dict[str, float] = {
                'total_predictions': float(len(df)),
                'rmse': float(np.sqrt(mean_squared_error(
                    df[primary_stat],
                    df[f"{primary_stat}_rolling_5"]
                ))),
                'mae': float(mean_absolute_error(
                    df[primary_stat],
                    df[f"{primary_stat}_rolling_5"]
                )),
                'r2': float(r2_score(
                    df[primary_stat],
                    df[f"{primary_stat}_rolling_5"]
                ))
            }
            
            # Calculate error distributions
            errors = df[primary_stat] - df[f"{primary_stat}_rolling_5"]
            metrics.update({
                'error_std': float(errors.std()),
                'error_95th': float(np.percentile(np.abs(errors), 95)),
                'error_max': float(np.abs(errors).max())
            })
            
            return metrics
            
        finally:
            conn.close()
    
    def calculate_confidence_intervals(
        self,
        prop_type: str,
        confidence: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for predictions.
        
        Args:
            prop_type: Type of prop
            confidence: Confidence level (0-1)
            
        Returns:
            Dictionary with interval metrics
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            config = PROP_FEATURES_CONFIG[prop_type]
            primary_stat = config['primary_stat']
            
            # Calculate prediction errors
            query = f"""
                SELECT 
                    {primary_stat} - {primary_stat}_rolling_5 as error,
                    {primary_stat}_rolling_5 as predicted
                FROM player_stats
                WHERE game_date IS NOT NULL
                AND {primary_stat}_rolling_5 IS NOT NULL
            """
            
            df = conn.execute(query).fetchdf()
            
            # Calculate overall confidence interval
            error_std = df['error'].std()
            z_score = np.abs(np.percentile(
                np.random.standard_normal(10000),
                (1 - confidence) * 100
            ))
            
            interval_width = z_score * error_std
            
            # Calculate intervals by prediction range
            ranges = pd.qcut(df['predicted'], q=5)
            range_metrics = {}
            
            for name, group in df.groupby(ranges, observed=True):
                range_metrics[f"{float(name.left):.1f}-{float(name.right):.1f}"] = {
                    'mean_error': float(group['error'].mean()),
                    'error_std': float(group['error'].std()),
                    'interval_width': float(z_score * group['error'].std())
                }
            
            return {
                'overall': {
                    'mean_error': float(df['error'].mean()),
                    'error_std': float(error_std),
                    'interval_width': float(interval_width)
                },
                'by_range': range_metrics
            }
            
        finally:
            conn.close()
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("Model Validation Report")
        report.append("=" * 30)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Check data quality
        report.append("\nData Quality Check")
        report.append("-" * 20)
        quality_metrics = self.check_data_quality()
        for prop_type, metrics in quality_metrics.items():
            report.append(f"\n{prop_type.title()}:")
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value}")
        
        # Validate rolling windows
        report.append("\nRolling Window Validation")
        report.append("-" * 20)
        for prop_type in PROP_FEATURES_CONFIG.keys():
            validation_metrics = self.validate_rolling_windows(prop_type)
            report.append(f"\n{prop_type.title()}:")
            for metric, value in validation_metrics.items():
                report.append(f"  {metric}: {value:.3f}")
        
        # Run backtesting
        report.append("\nBacktesting Results")
        report.append("-" * 20)
        for prop_type in PROP_FEATURES_CONFIG.keys():
            backtest_metrics = self.run_backtesting(prop_type)
            report.append(f"\n{prop_type.title()}:")
            for metric, value in backtest_metrics.items():
                report.append(f"  {metric}: {value:.3f}")
        
        # Calculate confidence intervals
        report.append("\nConfidence Intervals (95%)")
        report.append("-" * 20)
        for prop_type in PROP_FEATURES_CONFIG.keys():
            intervals = self.calculate_confidence_intervals(prop_type)
            report.append(f"\n{prop_type.title()}:")
            
            overall = intervals['overall']
            report.append("  Overall:")
            for metric, value in overall.items():
                report.append(f"    {metric}: {value:.3f}")
            
            report.append("  By Prediction Range:")
            for range_name, metrics in intervals['by_range'].items():
                report.append(f"    {range_name}:")
                for metric, value in metrics.items():
                    report.append(f"      {metric}: {value:.3f}")
        
        return "\n".join(report)

def main() -> int:
    """Run validation and improvements.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        validator = ModelValidator()
        
        # Generate validation report
        report = validator.generate_validation_report()
        
        # Save report
        report_file = 'data/validation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
