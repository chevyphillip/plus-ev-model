"""Ensemble model for high-scoring range predictions."""

import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, cast
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import duckdb
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

class HighScorerEnsemble:
    """Ensemble model specifically for high-scoring players."""
    
    def __init__(
        self,
        scoring_threshold: float = 17.8,  # Based on validation analysis
        db_path: str = 'data/nba_stats.duckdb',
        random_state: int = 42
    ) -> None:
        """Initialize ensemble model.
        
        Args:
            scoring_threshold: Points threshold for high scorers
            db_path: Path to DuckDB database
            random_state: Random seed for reproducibility
        """
        self.scoring_threshold = scoring_threshold
        self.db_path = db_path
        self.random_state = random_state
        
        # Initialize base models
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            ),
            'lasso': LassoCV(
                cv=5,
                random_state=random_state
            )
        }
        
        self.scaler = StandardScaler()
        self.model_weights: Optional[Dict[str, float]] = None
        
    def _get_high_scorer_data(self) -> pd.DataFrame:
        """Get data for high-scoring players.
        
        Returns:
            DataFrame with high scorer statistics
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            # Get players with high scoring average
            query = f"""
                WITH player_avgs AS (
                    SELECT 
                        player_id,
                        AVG(pts) as avg_points
                    FROM player_stats
                    GROUP BY player_id
                    HAVING AVG(pts) >= {self.scoring_threshold}
                )
                SELECT 
                    ps.*,
                    -- Basic stats
                    pts as target_value,
                    min,
                    fg_pct,
                    fg3_pct,
                    ft_pct,
                    plus_minus,
                    
                    -- Rolling averages
                    pts_rolling_5,
                    pts_rolling_10,
                    pts_rolling_20,
                    
                    -- Home/Away
                    pts_home,
                    pts_away,
                    is_home,
                    
                    -- Opponent
                    opp_pts_allowed_avg,
                    opp_ast_allowed_avg,
                    opp_reb_rate
                FROM player_stats ps
                JOIN player_avgs pa ON ps.player_id = pa.player_id
                WHERE game_date IS NOT NULL
                ORDER BY ps.player_id, game_date
            """
            
            return conn.execute(query).fetchdf()
            
        finally:
            conn.close()
    
    def _prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling.
        
        Args:
            df: DataFrame with player statistics
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        # Select features
        features = [
            'min',
            'fg_pct',
            'fg3_pct', 
            'ft_pct',
            'plus_minus',
            'pts_rolling_5',
            'pts_rolling_10',
            'pts_rolling_20',
            'pts_home',
            'pts_away',
            'is_home',
            'opp_pts_allowed_avg',
            'opp_ast_allowed_avg',
            'opp_reb_rate'
        ]
        
        # Create feature matrix
        X = df[features].copy()
        y = df['target_value']
        
        return X, y
    
    def _optimize_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Optimize ensemble weights using validation performance.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            y_true: True target values
            
        Returns:
            Dictionary mapping model names to weights
        """
        # Initialize weights
        n_models = len(predictions)
        weights = np.ones(n_models) / n_models
        
        # Create prediction matrix
        pred_matrix = np.column_stack([
            predictions[model] for model in predictions.keys()
        ])
        
        # Simple optimization using validation error
        best_weights = weights.copy()
        best_rmse = float('inf')
        
        # Grid search over weight combinations
        for _ in range(100):
            # Random weights that sum to 1
            w = np.random.dirichlet(np.ones(n_models))
            ensemble_pred = pred_matrix @ w
            rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = w
        
        return dict(zip(predictions.keys(), best_weights))
    
    def train(self) -> Dict[str, float]:
        """Train ensemble model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Get high scorer data
        df = self._get_high_scorer_data()
        
        # Prepare features
        X, y = self._prepare_features(df)
        
        # Create time-based splits
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Track metrics and predictions
        metrics: Dict[str, List[float]] = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        all_predictions: Dict[str, List[float]] = {
            model_name: [] for model_name in self.models.keys()
        }
        all_true = []
        
        # Train and evaluate on each fold
        for train_idx, test_idx in tscv.split(X):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each base model
            fold_predictions = {}
            
            for name, model in self.models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Get predictions
                pred = model.predict(X_test_scaled)
                fold_predictions[name] = pred
                all_predictions[name].extend(pred)
            
            all_true.extend(y_test)
            
            # Optimize weights if not already set
            if self.model_weights is None:
                weights = self._optimize_weights(
                    fold_predictions,
                    y_test
                )
                self.model_weights = cast(Dict[str, float], weights)
            
            # Calculate ensemble predictions
            ensemble_pred = np.zeros_like(y_test, dtype=np.float64)
            if self.model_weights:
                for name, pred in fold_predictions.items():
                    ensemble_pred += pred.astype(np.float64) * self.model_weights[name]
            
            # Calculate metrics
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, ensemble_pred)))
            metrics['mae'].append(mean_absolute_error(y_test, ensemble_pred))
            metrics['r2'].append(r2_score(y_test, ensemble_pred))
        
        # Calculate final metrics
        final_metrics = {
            'rmse': float(np.mean(metrics['rmse'])),
            'mae': float(np.mean(metrics['mae'])),
            'r2': float(np.mean(metrics['r2']))
        }
        
        # Log results
        logger.info("\nEnsemble Model Weights:")
        for name, weight in self.model_weights.items():
            logger.info(f"{name}: {weight:.3f}")
        
        logger.info("\nModel Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.3f}")
        
        return final_metrics
    
    def predict(
        self,
        player_id: int
    ) -> Dict[str, Any]:
        """Predict points for a high-scoring player.
        
        Args:
            player_id: NBA player ID
            
        Returns:
            Dictionary with prediction and metadata
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            # Get player's recent data
            query = f"""
                WITH player_avg AS (
                    SELECT AVG(pts) as avg_points
                    FROM player_stats
                    WHERE player_id = {player_id}
                )
                SELECT 
                    ps.*,
                    pts as target_value,
                    pa.avg_points
                FROM player_stats ps
                CROSS JOIN player_avg pa
                WHERE ps.player_id = {player_id}
                ORDER BY game_date DESC
                LIMIT 1
            """
            
            df = conn.execute(query).fetchdf()
            
            if df.empty:
                raise ValueError(f"No data found for player {player_id}")
            
            # Check if player is a high scorer
            if df['avg_points'].iloc[0] < self.scoring_threshold:
                raise ValueError(
                    f"Player {player_id} is not a high scorer "
                    f"(avg: {df['avg_points'].iloc[0]:.1f})"
                )
            
            # Prepare features
            X, _ = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                pred = float(model.predict(X_scaled)[0])
                predictions[name] = pred
            
            # Calculate ensemble prediction
            if self.model_weights is not None:
                ensemble_pred = 0.0
                for name, pred in predictions.items():
                    ensemble_pred += float(pred) * self.model_weights[name]
            else:
                raise ValueError("Model weights not set")
            
            return {
                'player_name': df['player_name'].iloc[0],
                'predicted_points': ensemble_pred,
                'recent_average': float(df['pts_rolling_5'].iloc[0]),
                'model_predictions': predictions,
                'model_weights': self.model_weights
            }
            
        finally:
            conn.close()

def main() -> int:
    """Train and evaluate ensemble model.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Train ensemble model
        model = HighScorerEnsemble()
        metrics = model.train()
        
        logger.info("\nEnsemble Model Training Complete!")
        logger.info("Final Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
