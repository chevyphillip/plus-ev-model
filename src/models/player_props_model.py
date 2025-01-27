"""Base model for predicting NBA player props."""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple, Optional
import duckdb

logger = logging.getLogger(__name__)

class PlayerPropsModel:
    """Base class for player props prediction models."""
    
    def __init__(
        self,
        db_path: str = 'data/nba_stats.duckdb',
        prop_type: str = 'points',
        rolling_windows: List[int] = [5, 10, 20],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """Initialize the player props model.
        
        Args:
            db_path: Path to DuckDB database
            prop_type: Type of prop to predict (points, assists, rebounds)
            rolling_windows: List of window sizes for rolling averages
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.db_path = db_path
        self.prop_type = prop_type
        self.rolling_windows = rolling_windows
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize models and scalers
        self.model = Ridge(random_state=random_state)
        self.scaler = StandardScaler()
        
        # Track feature names
        self.feature_names: List[str] = []
        
    def _get_base_stats_query(self, player_id: Optional[int] = None) -> str:
        """Get SQL query for base player statistics.
        
        Args:
            player_id: Optional player ID to filter for
            
        Returns:
            SQL query string
        """
        stat_col = {
            'points': 'pts',
            'assists': 'ast',
            'rebounds': 'reb'
        }.get(self.prop_type)
        
        if not stat_col:
            raise ValueError(f"Unsupported prop type: {self.prop_type}")
        
        query = f"""
            SELECT 
                player_id,
                player_name,
                team_abbreviation,
                position,
                game_date,
                is_home,
                {stat_col} as prop_value,
                min as minutes,
                pts,
                ast,
                reb,
                fg_pct,
                fg3_pct,
                ft_pct,
                plus_minus,
                
                -- Home/Away averages
                {stat_col}_home,
                {stat_col}_away,
                
                -- Rolling averages
                {stat_col}_rolling_5,
                
                -- Opponent metrics
                opp_pts_allowed_avg,
                opp_ast_allowed_avg,
                opp_reb_rate
                
            FROM player_stats
            WHERE game_date IS NOT NULL
        """
        
        if player_id is not None:
            query += f" AND player_id = {player_id}"
        
        query += " ORDER BY player_id, game_date"
        
        return query
    
    def _calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages and other time-based features.
        
        Args:
            df: DataFrame with base statistics
            
        Returns:
            DataFrame with additional rolling features
        """
        # Group by player
        grouped = df.groupby('player_id')
        
        # Calculate rolling averages for each window
        for window in self.rolling_windows:
            # Main prop value
            df[f'prop_value_rolling_{window}'] = grouped['prop_value'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Minutes played
            df[f'minutes_rolling_{window}'] = grouped['minutes'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Shooting percentages
            for pct in ['fg_pct', 'fg3_pct', 'ft_pct']:
                df[f'{pct}_rolling_{window}'] = grouped[pct].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            
            # Plus/minus
            df[f'plus_minus_rolling_{window}'] = grouped['plus_minus'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Calculate trend features
        df['prop_value_trend'] = grouped['prop_value'].transform(
            lambda x: x.rolling(5, min_periods=5).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0]
            )
        )
        
        # Calculate consistency features
        df['prop_value_std'] = grouped['prop_value'].transform(
            lambda x: x.rolling(10, min_periods=1).std()
        )
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling.
        
        Args:
            df: DataFrame with player statistics
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        # Calculate rolling features
        df = self._calculate_rolling_features(df)
        
        # Select features
        features = []
        
        # Rolling averages
        for window in self.rolling_windows:
            features.extend([
                f'prop_value_rolling_{window}',
                f'minutes_rolling_{window}',
                f'fg_pct_rolling_{window}',
                f'fg3_pct_rolling_{window}',
                f'ft_pct_rolling_{window}',
                f'plus_minus_rolling_{window}'
            ])
        
        # Trend and consistency
        features.extend([
            'prop_value_trend',
            'prop_value_std'
        ])
        
        # Home/Away and opponent features
        if self.prop_type == 'points':
            features.append('pts_home')
            features.append('pts_away')
            features.append('opp_pts_allowed_avg')
        elif self.prop_type == 'assists':
            features.append('ast_home')
            features.append('ast_away')
            features.append('opp_ast_allowed_avg')
        elif self.prop_type == 'rebounds':
            features.append('reb_home')
            features.append('reb_away')
            features.append('opp_reb_rate')
        
        # Store feature names
        self.feature_names = features
        
        # Create feature matrix and target
        X = df[features].copy()
        y = df['prop_value']
        
        return X, y
    
    def train(self) -> Dict[str, float]:
        """Train the model and return evaluation metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Get data
        with duckdb.connect(self.db_path) as conn:
            df = conn.execute(self._get_base_stats_query()).fetchdf()
        
        # Prepare features
        X, y = self._prepare_features(df)
        
        # Create time-based splits
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Track metrics across folds
        metrics = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        # Train and evaluate on each fold
        for train_idx, test_idx in tscv.split(X):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['mae'].append(mean_absolute_error(y_test, y_pred))
            metrics['r2'].append(r2_score(y_test, y_pred))
        
        # Calculate average metrics
        avg_metrics = {
            'rmse': float(np.mean(metrics['rmse'])),
            'mae': float(np.mean(metrics['mae'])),
            'r2': float(np.mean(metrics['r2']))
        }
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.model.coef_)
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature Importances:\n%s", feature_importance)
        logger.info("Model Metrics:\n%s", pd.Series(avg_metrics))
        
        return avg_metrics
    
    def predict_player(self, player_id: int) -> Dict[str, Any]:
        """Predict prop value for a player's next game.
        
        Args:
            player_id: NBA player ID
            
        Returns:
            Dictionary containing prediction and supporting data
        """
        # Get player's recent data
        with duckdb.connect(self.db_path) as conn:
            df = conn.execute(self._get_base_stats_query(player_id)).fetchdf()
        
        if df.empty:
            raise ValueError(f"No data found for player_id {player_id}")
        
        # Prepare features
        X, _ = self._prepare_features(df)
        
        # Get latest feature values
        latest_features = X.iloc[-1:].copy()
        
        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        pred_value = float(self.model.predict(latest_features_scaled)[0])
        
        # Get recent averages for context
        recent_games = df.tail(5)
        
        return {
            'player_name': df['player_name'].iloc[0],
            'predicted_value': pred_value,
            'recent_average': float(recent_games['prop_value'].mean()),
            'last_5_games': recent_games['prop_value'].tolist(),
            'prop_type': self.prop_type
        }

def train_props_model(
    prop_type: str = 'points',
    db_path: str = 'data/nba_stats.duckdb'
) -> Dict[str, float]:
    """Train a player props model and return metrics.
    
    Args:
        prop_type: Type of prop to predict
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary of evaluation metrics
    """
    model = PlayerPropsModel(db_path=db_path, prop_type=prop_type)
    metrics = model.train()
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train models for different prop types
    prop_types = ['points', 'assists', 'rebounds']
    
    for prop_type in prop_types:
        logger.info(f"\nTraining {prop_type} model...")
        metrics = train_props_model(prop_type)
        
        print(f"\n{prop_type.title()} Model Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
