"""H2O AutoML-based player props prediction model."""

import logging
import h2o
from h2o.automl import H2OAutoML
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import mlflow
from src.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

class H2OPropsModel:
    """H2O AutoML model for predicting player props."""
    
    def __init__(
        self,
        prop_type: str,
        max_models: int = 20,
        max_runtime_secs: int = 300,
        experiment_name: str = "props_prediction"
    ) -> None:
        """Initialize model.
        
        Args:
            prop_type: Type of prop to predict (points, rebounds, assists, threes)
            max_models: Maximum number of models to train
            max_runtime_secs: Maximum training time in seconds
            experiment_name: MLflow experiment name
        """
        self.prop_type = prop_type
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.experiment_name = experiment_name
        
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
        
        # Initialize H2O
        h2o.init()
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        
    def _get_training_data(self) -> pd.DataFrame:
        """Get training data from database.
        
        Returns:
            DataFrame with features and target
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
        
        return df
    
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
        """Train H2O AutoML models.
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.prop_type} model with H2O AutoML...")
        
        # Get training data
        df = self._get_training_data()
        
        if len(df) == 0:
            raise ValueError("No training data available")
        
        # Convert to H2O frame
        train = h2o.H2OFrame(df)
        
        # Split features and target
        x = self.feature_names
        y = "target"
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"h2o_automl_{self.prop_type}"):
            # Log parameters
            mlflow.log_params({
                "prop_type": self.prop_type,
                "max_models": self.max_models,
                "max_runtime_secs": self.max_runtime_secs,
                "n_features": len(self.feature_names)
            })
            
            # Initialize and train AutoML
            aml = H2OAutoML(
                max_models=self.max_models,
                max_runtime_secs=self.max_runtime_secs,
                seed=42
            )
            aml.train(x=x, y=y, training_frame=train)
            
            # Get best model
            self.model = aml.leader
            
            # Get model performance
            perf = self.model.model_performance()
            metrics = {
                "rmse": perf.rmse(),
                "mae": perf.mae(),
                "r2": perf.r2()
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            varimp = self.model.varimp()
            if varimp is not None:
                for i, row in enumerate(varimp):
                    if i < len(self.feature_names):
                        mlflow.log_metric(
                            f"importance_{row[0]}", 
                            row[1]
                        )
            
            logger.info(f"\nMetrics for {self.prop_type}:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.3f}")
            
            return metrics
    
    def predict_player(self, player_id: int) -> Dict[str, Any]:
        """Predict props for a player.
        
        Args:
            player_id: NBA API player ID
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Get player features
        df = self._get_player_features(player_id)
        
        if len(df) == 0:
            raise ValueError(f"No data found for player {player_id}")
        
        # Convert to H2O frame
        test = h2o.H2OFrame(df)
        
        # Make prediction
        pred = self.model.predict(test)
        pred_value = pred.as_data_frame()['predict'][0]
        
        # Get prediction interval
        pred_interval = self.model.predict_interval(test)
        lower = pred_interval.as_data_frame()['lower'][0]
        upper = pred_interval.as_data_frame()['upper'][0]
        
        # Calculate confidence score (0-1)
        range_width = upper - lower
        confidence = 1 - (range_width / (2 * pred_value)) if pred_value > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))  # Clip to 0-1
        
        return {
            'predicted_value': float(pred_value),
            'lower_bound': float(max(0, lower)),
            'upper_bound': float(upper),
            'confidence': float(confidence)
        }
    
    def __del__(self):
        """Cleanup H2O on deletion."""
        try:
            h2o.cluster().shutdown()
        except:
            pass
