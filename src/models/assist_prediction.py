import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import duckdb
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AssistPredictionModel:
    def __init__(self, db_path: str = 'data/nba_stats.duckdb') -> None:
        self.db_path = db_path
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        
    def _get_player_recent_games(self) -> pd.DataFrame:
        """Get game-by-game data and calculate rolling averages"""
        with duckdb.connect(self.db_path) as conn:
            # Get game-by-game stats
            df = conn.execute("""
                SELECT 
                    player_id,
                    player_name,
                    season,
                    ast,
                    pts,
                    min,
                    fg_pct,
                    plus_minus
                FROM player_stats
                ORDER BY season DESC, player_id
            """).fetchdf()
            
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable
        Features include:
        - Rolling averages of last 5 games for key stats
        - Season averages
        - Recent performance indicators
        """
        # Group by player and calculate rolling stats
        df = df.sort_values(['player_id', 'season'])
        rolling_stats = df.groupby('player_id').rolling(window=5, min_periods=1).agg({
            'ast': 'mean',
            'pts': 'mean',
            'min': 'mean',
            'fg_pct': 'mean',
            'plus_minus': 'mean'
        }).reset_index()
        
        # Rename columns to indicate they are rolling averages
        rolling_stats.columns = ['player_id', 'index'] + [f'{col}_rolling_5' for col in ['ast', 'pts', 'min', 'fg_pct', 'plus_minus']]
        
        # Merge rolling stats back with original data
        df = df.reset_index().merge(rolling_stats, on=['player_id', 'index'])
        
        # Create target variable (1 if next game has 5+ assists, 0 otherwise)
        df['target'] = df.groupby('player_id')['ast'].shift(-1) >= 5.0
        
        # Drop rows where we don't have a next game to predict
        df = df.dropna(subset=['target'])
        
        # Drop the last game for each player since we can't predict the next game
        df = df.groupby('player_id').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        # Select features
        features = [
            'ast_rolling_5',
            'pts_rolling_5',
            'min_rolling_5',
            'fg_pct_rolling_5',
            'plus_minus_rolling_5'
        ]
        
        X = df[features]
        y = df['target']
        
        return X, y
    
    def train(self, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the model and return evaluation metrics
        """
        # Get data
        df = self._get_player_recent_games()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature Importances:\n%s", feature_importance)
        logger.info("Model Metrics:\n%s", pd.Series(metrics))
        
        return metrics
    
    def predict_player(self, player_id: int) -> Dict[str, Any]:
        """
        Predict if a player will have 5+ assists in their next game
        """
        df = self._get_player_recent_games()
        player_data = df[df['player_id'] == player_id].copy()
        
        if player_data.empty:
            raise ValueError(f"No data found for player_id {player_id}")
        
        # Get latest 5 games
        recent_data = player_data.tail(5)
        
        # Calculate features
        features = pd.DataFrame({
            'ast_rolling_5': [recent_data['ast'].mean()],
            'pts_rolling_5': [recent_data['pts'].mean()],
            'min_rolling_5': [recent_data['min'].mean()],
            'fg_pct_rolling_5': [recent_data['fg_pct'].mean()],
            'plus_minus_rolling_5': [recent_data['plus_minus'].mean()]
        })
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prob = self.model.predict_proba(features_scaled)[0, 1]
        prediction = prob >= 0.5
        
        return {
            'player_name': player_data['player_name'].iloc[0],
            'probability_5plus_assists': float(prob),
            'prediction': bool(prediction),
            'recent_assist_avg': float(recent_data['ast'].mean())
        }

def train_assist_model(db_path: str = 'data/nba_stats.duckdb') -> Dict[str, float]:
    """
    Train the assist prediction model and return metrics
    """
    model = AssistPredictionModel(db_path)
    metrics = model.train()
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metrics = train_assist_model()
    print("\nModel Training Complete!")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
