# NBA Player Props Prediction Model

A machine learning model for predicting NBA player performance, focusing on assist props and using historical data for accurate predictions.

## Project Structure

```
plus-ev-model/
├── data/                  # Data storage
│   └── nba_stats.duckdb  # DuckDB database with NBA stats
├── docs/                  # Documentation
├── src/                  # Source code
│   ├── data/            # Data ingestion and processing
│   ├── models/          # ML models
│   └── core/            # Core utilities
└── tests/               # Test suite
```

## Features

- NBA stats data pipeline with DuckDB storage
- Logistic regression model for predicting 5+ assists
- Rolling statistics and feature engineering
- Model evaluation and performance metrics
- Player-specific predictions with probability scores

## Model Performance

- Accuracy: 92.2%
- Precision: 80.0%
- ROC AUC: 97.6%
- Feature importance analysis shows rolling 5-game assist average as the strongest predictor

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plus-ev-model.git
cd plus-ev-model
```

2. Install dependencies using Poetry:

```bash
poetry install
```

## Usage

### Data Pipeline

Update NBA stats data:

```bash
python src/data/nba_stats.py
```

### Assist Prediction

Make predictions for specific players:

```bash
python src/models/predict_assists.py
```

Example output:

```
Predictions Summary:
-------------------
Trae Young:
  Probability of 5+ assists: 99.9%
  Recent assist average: 10.6
  Prediction: OVER 5 assists
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Project Organization

- `src/data/nba_stats.py`: NBA stats data pipeline
- `src/models/assist_prediction.py`: Core prediction model
- `src/models/predict_assists.py`: Prediction interface
- `tests/`: Unit tests and integration tests

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Setup Guide](docs/setup.md)
- [Technical Requirements](docs/technical_requirements.md)
- [Methodology](docs/methodology.md)

## License

MIT License - see LICENSE file for details.
