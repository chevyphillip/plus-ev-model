"""Feature configuration for different prop types."""

from typing import Dict, List, Any

PROP_FEATURES_CONFIG: Dict[str, Dict[str, Any]] = {
    'points': {
        'primary_stat': 'pts',
        'rolling_windows': [5, 10, 20],
        'features': [
            # Direct performance metrics
            'pts_rolling_5',
            'pts_rolling_10',
            'pts_rolling_20',
            'pts_home',
            'pts_away',
            'min',  # minutes played
            
            # Shooting efficiency
            'fg_pct',
            'fg3_pct',
            'ft_pct',
            
            # Game context
            'is_home',
            'plus_minus',
            
            # Opponent metrics
            'opp_pts_allowed_avg',
            'opp_ast_allowed_avg',
            'opp_reb_rate'
        ],
        'engineered_features': [
            'pts_per_minute',
            'scoring_trend'  # Linear regression on last 5 games
        ]
    },
    
    'rebounds': {
        'primary_stat': 'reb',
        'rolling_windows': [5, 10, 20],
        'features': [
            # Direct performance metrics
            'reb_rolling_5',
            'reb_rolling_10',
            'reb_rolling_20',
            'reb_home',
            'reb_away',
            'min',
            
            # Game context
            'is_home',
            'plus_minus',
            
            # Opponent metrics
            'opp_pts_allowed_avg',
            'opp_reb_rate'
        ],
        'engineered_features': [
            'reb_per_minute',
            'rebound_trend'
        ]
    },
    
    'assists': {
        'primary_stat': 'ast',
        'rolling_windows': [5, 10, 20],
        'features': [
            # Direct performance metrics
            'ast_rolling_5',
            'ast_rolling_10',
            'ast_rolling_20',
            'ast_home',
            'ast_away',
            'min',
            
            # Game context
            'is_home',
            'plus_minus',
            
            # Opponent metrics
            'opp_pts_allowed_avg',
            'opp_ast_allowed_avg'
        ],
        'engineered_features': [
            'ast_per_minute',
            'playmaking_trend'
        ]
    },
    
    'threes': {
        'primary_stat': 'fg3m',
        'rolling_windows': [5, 10, 20],
        'features': [
            # Direct performance metrics
            'fg3m_rolling_5',
            'fg3m_rolling_10',
            'fg3m_rolling_20',
            'fg3_pct',  # Keep percentage as a feature
            'min',
            
            # Game context
            'is_home',
            'plus_minus',
            
            # Opponent metrics
            'opp_pts_allowed_avg',
            'opp_ast_allowed_avg'
        ],
        'engineered_features': [
            'shooting_trend'
        ]
    }
}

def get_feature_importance_metrics(prop_type: str) -> Dict[str, float]:
    """Get baseline feature importance weights for a prop type.
    
    Args:
        prop_type: Type of prop to get feature importance for
        
    Returns:
        Dictionary mapping features to importance weights
    """
    base_weights = {
        # Primary stat rolling averages
        f"{PROP_FEATURES_CONFIG[prop_type]['primary_stat']}_rolling_5": 1.0,
        f"{PROP_FEATURES_CONFIG[prop_type]['primary_stat']}_rolling_10": 0.8,
        f"{PROP_FEATURES_CONFIG[prop_type]['primary_stat']}_rolling_20": 0.6,
        
        # Minutes played
        'min': 0.7,
        
        # Home/Away
        'is_home': 0.3,
        
        # Game context
        'plus_minus': 0.2,
        
        # Opponent metrics get moderate weight
        'opp_pts_allowed_avg': 0.4,
        'opp_ast_allowed_avg': 0.4,
        'opp_reb_rate': 0.4,
        
        # Efficiency metrics
        'fg_pct': 0.5,
        'fg3_pct': 0.5,
        'ft_pct': 0.5
    }
    
    # Filter to only relevant features for this prop type
    prop_features = set(PROP_FEATURES_CONFIG[prop_type]['features'])
    return {k: v for k, v in base_weights.items() if k in prop_features}

def get_engineered_feature_sql(feature: str) -> str:
    """Get SQL expression for calculating an engineered feature.
    
    Args:
        feature: Name of engineered feature
        
    Returns:
        SQL expression for calculating the feature
    """
    sql_expressions = {
        # Per-minute features
        'pts_per_minute': 'CAST(pts AS FLOAT) / NULLIF(min, 0)',
        'reb_per_minute': 'CAST(reb AS FLOAT) / NULLIF(min, 0)',
        'ast_per_minute': 'CAST(ast AS FLOAT) / NULLIF(min, 0)',
        
        # Trend features will be calculated in Python using rolling windows
        'scoring_trend': 'NULL::FLOAT',
        'rebound_trend': 'NULL::FLOAT',
        'playmaking_trend': 'NULL::FLOAT',
        'shooting_trend': 'NULL::FLOAT'
    }
    
    return sql_expressions.get(feature, 'NULL::FLOAT')
