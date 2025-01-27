"""Enhanced player props prediction model."""

import logging
import sys
from typing import Dict, List, Optional, Any, Tuple, cast, TypeVar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)
Series = TypeVar('Series', bound=pd.Series)

class EnhancedPropsModel:
    """Enhanced model for predicting player props."""
    
    def __init__(
        self,
        prop_type: str,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3
    ) -> None:
        """Initialize model.
        
        Args:
            prop_type: Type of prop to predict (points, rebounds, assists, threes)
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks contribution of each tree
            max_depth: Maximum depth of individual regression estimators
        """
        self.prop_type = prop_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        # Initialize pipeline
        self.model: Pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ))
        ])
        
        # Map prop type to column names
        prop_map = {
            'points': 'pts',
            'rebounds': 'reb',
            'assists': 'ast',
            'threes': 'fg3m'
        }
        self.db_column = prop_map[prop_type]
        
        # Define features based on prop type
        self.feature_names: List[str] = [
            f"{self.db_column}_rolling_5",
            f"{self.db_column}_rolling_10",
            f"{self.db_column}_rolling_20",
            'min',
            'is_home',
            'opp_pts_allowed_avg'
        ]
        
        if prop_type == 'points':
            self.feature_names.extend([
                'fg_pct_rolling_5',
                'fg_pct_rolling_10',
                'fg_pct_rolling_20',
                'ft_pct_rolling_5',
                'ft_pct_rolling_10',
                'ft_pct_rolling_20',
                'fg3_pct_rolling_5',
                'fg3_pct_rolling_10',
                'fg3_pct_rolling_20',
                'plus_minus_rolling_5',
                'plus_minus_rolling_10',
                'plus_minus_rolling_20',
                'pts_home',
                'pts_away'
            ])
        elif prop_type == 'rebounds':
            self.feature_names.extend([
                'fg_pct_rolling_5',
                'fg_pct_rolling_10',
                'fg_pct_rolling_20',
                'plus_minus_rolling_5',
                'plus_minus_rolling_10',
                'plus_minus_rolling_20',
                'reb_home',
                'reb_away',
                'opp_reb_rate'
            ])
        elif prop_type == 'assists':
            self.feature_names.extend([
                'fg_pct_rolling_5',
                'fg_pct_rolling_10',
                'fg_pct_rolling_20',
                'plus_minus_rolling_5',
                'plus_minus_rolling_10',
                'plus_minus_rolling_20',
                'ast_home',
                'ast_away',
                'opp_ast_allowed_avg'
            ])
        elif prop_type == 'threes':
            self.feature_names.extend([
                'fg3_pct_rolling_5',
                'fg3_pct_rolling_10',
                'fg3_pct_rolling_20',
                'plus_minus_rolling_5',
                'plus_minus_rolling_10',
                'plus_minus_rolling_20',
                'fg3m_home',
                'fg3m_away'
            ])
    
    def _get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data from database.
        
        Returns:
            Features DataFrame and target Series
        """
        conn = get_db_connection(use_motherduck=True)
        
        # Get last 2 seasons of data
        cutoff_date = datetime.now() - timedelta(days=730)
        
        # Build feature columns string
        feature_cols = ', '.join(self.feature_names)
        
        query = f"""
            SELECT 
                {self.db_column} as target,
                {feature_cols}
            FROM player_stats
            WHERE game_date >= '{cutoff_date.strftime('%Y-%m-%d')}'
            AND {self.db_column} IS NOT NULL
        """
        
        # Execute query and convert to pandas
        df = conn.execute(query).df()
        conn.close()
        
        # Split features and target
        X = df[self.feature_names]
        y = df['target']
        
        return X, y
    
    def _get_player_features(self, player_id: int) -> pd.DataFrame:
        """Get latest features for a player.
        
        Args:
            player_id: NBA API player ID
            
        Returns:
            DataFrame with player features
        """
        conn = get_db_connection(use_motherduck=True)
        
        # Build feature columns string
        feature_cols = ', '.join(self.feature_names)
        
        # Get most recent game
        query = f"""
            SELECT 
                {feature_cols}
            FROM player_stats
            WHERE player_id = {player_id}
            ORDER BY game_date DESC
            LIMIT 1
        """
        
        # Execute query and convert to pandas
        df = conn.execute(query).df()
        conn.close()
        
        return df
    
    def train(self) -> Dict[str, float]:
        """Train the model.
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.prop_type} model...")
        
        # Get training data
        X, y = self._get_training_data()
        
        if len(X) == 0:
            raise ValueError("No training data available")
            
        # Fit model
        self.model.fit(X, y)
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Get feature importances
        regressor = cast(GradientBoostingRegressor, self.model.named_steps['regressor'])
        importances = []
        for name, importance in zip(self.feature_names, regressor.feature_importances_):
            importances.append({
                'feature': name,
                'importance': importance
            })
        importances_df = pd.DataFrame(importances)
        importances_df = importances_df.sort_values('importance', ascending=False)
        
        logger.info(f"\nFeature importances for {self.prop_type}:")
        for _, row in importances_df.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.3f}")
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def predict_player(self, player_id: int) -> Dict[str, Any]:
        """Predict props for a player.
        
        Args:
            player_id: NBA API player ID
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Get player features
        X = self._get_player_features(player_id)
        
        if len(X) == 0:
            raise ValueError(f"No data found for player {player_id}")
            
        # Make prediction
        pred = self.model.predict(X)[0]
        
        # Get prediction interval
        regressor = cast(GradientBoostingRegressor, self.model.named_steps['regressor'])
        if hasattr(regressor, 'estimators_'):
            # Get predictions from all trees
            tree_preds = []
            for tree in regressor.estimators_:
                scaled_X = self.model.named_steps['scaler'].transform(
                    self.model.named_steps['imputer'].transform(X)
                )
                tree_preds.append(tree[0].predict(scaled_X)[0])
            
            # Calculate confidence interval
            std = np.std(tree_preds)
            lower = pred - (1.96 * std)
            upper = pred + (1.96 * std)
            
            return {
                'predicted_value': float(pred),
                'lower_bound': float(max(0, lower)),
                'upper_bound': float(upper),
                'confidence': float(1 - (std / pred)) if pred > 0 else 0.0
            }
        
        return {
            'predicted_value': float(pred),
            'confidence': 0.8  # Default confidence
        }

def main() -> int:
    """Train and evaluate model.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Train models for each prop type
        for prop_type in ['points', 'rebounds', 'assists', 'threes']:
            model = EnhancedPropsModel(prop_type)
            metrics = model.train()
            
            logger.info(f"\nMetrics for {prop_type}:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
