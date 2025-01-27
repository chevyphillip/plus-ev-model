# NBA Player Props Prediction Model

A machine learning system for predicting NBA player prop bets and identifying edges against sportsbook lines.

## Features

- Historical NBA stats collection via NBA API
- Advanced feature engineering including:
  - Rolling averages across multiple windows
  - Home/Away splits
  - Opponent strength metrics
  - Player trend analysis
- Ridge regression model for prop predictions
- Edge calculation against market lines
- Kelly criterion bet sizing
- DuckDB + MotherDuck for efficient data storage

## Project Structure

```
plus-ev-model/
├── data/                  # Data storage
│   ├── raw/              # Raw API data
│   ├── processed/        # Cleaned data
│   ├── interim/          # Intermediate processing
│   └── external/         # Third-party data
├── docs/                 # Documentation
├── src/                  # Source code
│   ├── data/            # Data collection
│   │   ├── nba_stats.py           # NBA API data pipeline
│   │   ├── nba_stats_career.py    # Career stats collection
│   │   └── db_config.py           # Database configuration
│   ├── models/          # ML models
│   │   ├── player_props_model.py  # Base props prediction model
│   │   └── predict_props.py       # Prediction script
│   └── core/            # Core utilities
│       ├── devig.py               # Odds processing
│       ├── monte_carlo.py         # Simulation tools
│       └── edge_calculator.py     # Betting edge analysis
└── tests/               # Test suite
```

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Set up environment variables in `.env`:

```env
LOCAL_DB_PATH=data/nba_stats.duckdb
MOTHERDUCK_TOKEN=your_token_here
NBA_API_DELAY=1.0
```

3. Initialize database:

```bash
python -m src.data.nba_stats
```

## Usage

### Fetch NBA Stats

```python
from src.data.nba_stats import update_player_stats

# Update stats in database
update_player_stats()
```

### Predict Player Props

```python
from src.models.predict_props import analyze_player_prop

# Analyze a single prop
result = analyze_player_prop(
    player_id=2544,  # LeBron James
    prop_type='points',
    line=25.5,
    over_odds=-110,
    under_odds=-110
)

# Print recommendation
if result['recommendation']:
    rec = result['recommendation']
    print(f"Bet: {rec['bet_type']} {rec['line']}")
    print(f"Edge: {rec['edge']:.1%}")
    print(f"Kelly: {rec['kelly_bet']:.1%}")
```

### Analyze Multiple Props

```python
from src.models.predict_props import analyze_multiple_props

props = [
    {
        'player_id': 2544,
        'prop_type': 'points',
        'line': 25.5
    },
    {
        'player_id': 2544,
        'prop_type': 'assists',
        'line': 7.5
    }
]

results = analyze_multiple_props(props)
```

## Model Details

### Feature Engineering

The model uses several types of features:

- Rolling averages (5, 10, 20 game windows)
- Home/Away performance splits
- Opponent strength indicators
- Recent performance trends
- Player consistency metrics

### Edge Calculation

Edges are calculated by:

1. Converting model predictions to probabilities
2. Comparing to market implied probabilities
3. Applying Kelly criterion for optimal bet sizing
4. Filtering by confidence threshold

### Performance Metrics

The model is evaluated on:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2)
- Prediction vs. Actual correlation
- ROI on historical bets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a pull request

## License

MIT License - see LICENSE file for details
