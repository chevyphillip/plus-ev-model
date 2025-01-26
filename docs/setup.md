# Setup Guide

## Prerequisites

1. Python 3.10 or higher
2. Poetry (Python package manager)
3. Git

## Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plus-ev-model.git
cd plus-ev-model
```

2. Install dependencies using Poetry:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. Create necessary directories:

```bash
mkdir -p data
```

## Initial Data Setup

1. Initialize the NBA stats database:

```bash
poetry run python src/data/nba_stats.py
```

This will:

- Create the DuckDB database
- Fetch historical NBA player stats
- Calculate rolling averages
- Store processed data

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following configurations:

```env
# Database settings
DB_PATH=data/nba_stats.duckdb

# NBA API settings
NBA_API_DELAY=1.0  # Rate limiting delay in seconds
```

## Verify Installation

1. Run tests:

```bash
poetry run pytest
```

2. Try a sample prediction:

```bash
poetry run python src/models/predict_assists.py
```

## Project Structure

```
plus-ev-model/
├── data/                  # Data storage
│   └── nba_stats.duckdb  # DuckDB database
├── docs/                  # Documentation
├── src/                  # Source code
│   ├── data/            # Data ingestion
│   │   ├── __init__.py
│   │   └── nba_stats.py
│   ├── models/          # ML models
│   │   ├── __init__.py
│   │   ├── assist_prediction.py
│   │   └── predict_assists.py
│   └── core/            # Core utilities
└── tests/               # Test suite
```

## Development Setup

### IDE Configuration

#### VSCode

1. Install recommended extensions:
   - Python
   - Pylance
   - Python Test Explorer

2. Configure settings.json:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

### Pre-commit Hooks

1. Install pre-commit:

```bash
poetry run pre-commit install
```

2. Run against all files:

```bash
poetry run pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. Database Connection Errors
   - Verify DB_PATH in .env
   - Check directory permissions
   - Ensure DuckDB is properly installed

2. NBA API Rate Limiting
   - Increase NBA_API_DELAY in .env
   - Check for API service status
   - Verify network connectivity

3. Missing Dependencies
   - Run `poetry install` again
   - Check Poetry environment activation
   - Verify Python version compatibility

### Getting Help

1. Check existing issues on GitHub
2. Review error logs in `data/logs/`
3. Create a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details

## Updates and Maintenance

1. Update dependencies:

```bash
poetry update
```

2. Update NBA stats:

```bash
poetry run python src/data/nba_stats.py
```

3. Rebuild models:

```bash
poetry run python src/models/assist_prediction.py
