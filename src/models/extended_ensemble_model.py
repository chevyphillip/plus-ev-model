"""Extended ensemble model supporting all prop types."""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import sklearn.ensemble
import sklearn.linear_model
import xgboost
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import duckdb
from src.data.db_config import get_db_connection
from src.models.prop_features_config import PROP_FEATURES_CONFIG

logger = logging.getLogger(__name__)

class ExtendedEnsembleModel:
    """Enhanced ensemble model supporting all prop types with specialized ranges."""
    
    def __init__(
        self,
        prop_type: str,
        db_path: str = 'data/nba_stats.duckdb',
        random_state: int = 42
    ) -> None:
        """Initialize ensemble model.
        
        Args:
            prop_type: Type of prop to predict
            db_path: Path to DuckDB database
            random_state: Random seed for reproducibility
        """
        if prop_type not in PROP_FEATURES_CONFIG:
            raise ValueError(f"Unsupported prop type: {prop_type}")
            
        self.prop_type = prop_type
        self.db_path = db_path
        self.random_state = random_state
        self.config = PROP_FEATURES_CONFIG[prop_type]
        
        # Initialize base models
        self.models: Dict[str, Union[
            sklearn.ensemble.RandomForestRegressor,
            sklearn.ensemble.GradientBoostingRegressor,
            xgboost.XGBRegressor,
            sklearn.linear_model.LassoCV
        ]] = {
            'rf': sklearn.ensemble.RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state
            ),
            'gb': sklearn.ensemble.GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=random_state
            ),
            'xgb': xgboost.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=random_state
            ),
            'lasso': sklearn.linear_model.LassoCV(
                cv=5,
                random_state=random_state
            )
        }
        
        # Initialize range-specific models
        self.range_models: Dict[str, Dict[str, Any]] = {}
        
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.model_weights: Optional[Dict[str, float]] = None
        self.range_thresholds: Optional[List[float]] = None
        
    def _get_prop_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get prop data from database.
        
        Returns:
            X: Feature matrix
            y: Target values
        """
        conn = get_db_connection(use_motherduck=False)
        try:
            # Get features from config
            features = (
                self.config['features'] + 
                self.config['engineered_features']
            )
            feature_str = ', '.join(features)
            
            query = f"""
                SELECT 
                    {feature_str},
                    {self.config['primary_stat']} as target
                FROM player_stats
                WHERE game_date IS NOT NULL
                ORDER BY player_id, game_date
            """
            
            df = conn.execute(query).fetchdf()
            
            # Split into features and target
            X = df[features].values
            y = df['target'].values
            
            return X, y
            
        finally:
            conn.close()
    
    def _determine_range_thresholds(self, y: np.ndarray) -> List[float]:
        """Determine value ranges for specialized models.
        
        Args:
            y: Target values
            
        Returns:
            List of range threshold values
        """
        # Use percentiles to create ranges
        percentiles = [20, 40, 60, 80]
        thresholds = [np.percentile(y, p) for p in percentiles]
        return thresholds
    
    def _get_range_label(self, value: float) -> str:
        """Get range label for a value.
        
        Args:
            value: Value to get range for
            
        Returns:
            Range label string
        """
        if self.range_thresholds is None:
            raise ValueError("Range thresholds not set")
            
        for i, threshold in enumerate(self.range_thresholds):
            if value <= threshold:
                return f"range_{i}"
        return f"range_{len(self.range_thresholds)}"
    
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
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = w
        
        return dict(zip(predictions.keys(), best_weights))
    
    def train(self) -> Dict[str, float]:
        """Train ensemble model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Get data
        X, y = self._get_prop_data()
        
        # Determine range thresholds
        self.range_thresholds = self._determine_range_thresholds(y)
        
        # Create time-based splits
        tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=5)
        
        # Track metrics
        metrics: Dict[str, List[float]] = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        # Train and evaluate on each fold
        for train_idx, test_idx in tscv.split(X):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train range-specific models
            range_labels = [self._get_range_label(val) for val in y_train]
            self.range_models = {}
            
            for range_label in set(range_labels):
                range_mask = np.array(range_labels) == range_label
                if np.sum(range_mask) < 50:  # Skip if too few samples
                    continue
                    
                # Train models for this range
                range_predictions = {}
                
                for name, model in self.models.items():
                    # Train model
                    model.fit(
                        X_train_scaled[range_mask],
                        y_train[range_mask]
                    )
                    
                    # Get predictions
                    pred = model.predict(X_train_scaled[range_mask])
                    range_predictions[name] = pred
                
                # Optimize weights for this range
                weights = self._optimize_weights(
                    range_predictions,
                    y_train[range_mask]
                )
                
                self.range_models[range_label] = {
                    'models': {
                        name: model.fit(
                            X_train_scaled[range_mask],
                            y_train[range_mask]
                        )
                        for name, model in self.models.items()
                    },
                    'weights': weights
                }
            
            # Make predictions using range-specific models
            y_pred = np.zeros_like(y_test)
            
            for i, val in enumerate(y_test):
                range_label = self._get_range_label(val)
                if range_label not in self.range_models:
                    # Use average of adjacent ranges
                    y_pred[i] = np.mean(y_train)
                    continue
                
                range_info = self.range_models[range_label]
                range_preds = []
                
                for name, weight in range_info['weights'].items():
                    model_pred = range_info['models'][name].predict(
                        X_test_scaled[i:i+1]
                    )
                    range_preds.append(weight * model_pred[0])
                
                y_pred[i] = sum(range_preds)
            
            # Calculate metrics
            metrics['rmse'].append(np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))
            metrics['mae'].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            metrics['r2'].append(sklearn.metrics.r2_score(y_test, y_pred))
        
        # Calculate final metrics
        final_metrics = {
            'rmse': float(np.mean(metrics['rmse'])),
            'mae': float(np.mean(metrics['mae'])),
            'r2': float(np.mean(metrics['r2']))
        }
        
        # Log results
        logger.info(f"\n{self.prop_type.title()} Model Performance:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.3f}")
        
        return final_metrics
    
    def predict(
        self,
        features: np.ndarray,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """Make prediction for new data.
        
        Args:
            features: Feature array
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Dictionary with prediction and metadata
        """
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make initial prediction to determine range
        base_pred = np.mean([
            model.predict(features_scaled)[0]
            for model in self.models.values()
        ])
        
        # Get appropriate range model
        range_label = self._get_range_label(base_pred)
        
        if range_label not in self.range_models:
            return {
                'prediction': float(base_pred),
                'confidence': 0.5 if return_confidence else None
            }
        
        # Make range-specific prediction
        range_info = self.range_models[range_label]
        range_preds = []
        
        for name, weight in range_info['weights'].items():
            model_pred = range_info['models'][name].predict(features_scaled)
            range_preds.append(weight * model_pred[0])
        
        final_pred = sum(range_preds)
        
        # Calculate prediction confidence if requested
        confidence = None
        if return_confidence:
            # Use standard deviation of model predictions as confidence metric
            pred_std = np.std([
                model.predict(features_scaled)[0]
                for model in range_info['models'].values()
            ])
            confidence = 1.0 / (1.0 + pred_std)
        
        return {
            'prediction': float(final_pred),
            'confidence': confidence
        }

def main() -> None:
    """Train and evaluate extended ensemble models."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train models for each prop type
    for prop_type in PROP_FEATURES_CONFIG.keys():
        logger.info(f"\nTraining {prop_type} model...")
        model = ExtendedEnsembleModel(prop_type=prop_type)
        metrics = model.train()
        
        print(f"\n{prop_type.title()} Model Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()
