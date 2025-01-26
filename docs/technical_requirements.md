# Technical Requirements

## System Requirements

1. Python 3.10 or higher
2. Poetry for dependency management
3. DuckDB for local data storage
4. MotherDuck account for cloud database

## Dependencies

### Core Dependencies

- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.6.1
- duckdb>=0.9.0
- nba_api>=1.4.0
- python-dotenv>=1.0.0

### Development Dependencies

- pytest>=7.4.0
- pytest-mock>=3.14.0
- black>=23.0.0
- isort>=5.12.0
- mypy>=1.7.0
- pylint>=3.0.0

## Database Setup

### Local DuckDB

- File location: data/nba_stats.duckdb
- Schema includes:
  - player_stats: Individual game statistics
  - player_season_averages: Calculated season averages
  - data_metadata: Pipeline metadata

### MotherDuck Cloud Database

- Database name: nba-ml-model-db
- Requires MOTHERDUCK_TOKEN in .env
- Automatic sync with local database
- Same schema as local database

## Data Pipeline

### NBA Stats Collection

- Uses NBA API for official statistics
- Rate limited with configurable delay
- Fetches per-game statistics

### Data Processing

- Rolling averages calculation
- Feature engineering
- Data validation and cleaning

### Model Requirements

- Logistic regression for binary predictions
- Feature standardization
- Cross-validation support

## Development Setup

### Environment Variables

```env
# Database Configuration
LOCAL_DB_PATH=data/nba_stats.duckdb
MOTHERDUCK_TOKEN=your_token_here

# NBA API Configuration
NBA_API_DELAY=1.0

# Logging
LOG_LEVEL=INFO
```

### Directory Structure

```
plus-ev-model/
├── data/                  # Data storage
│   ├── raw/              # Raw API data
│   ├── processed/        # Cleaned data
│   ├── interim/          # Intermediate processing
│   └── external/         # Third-party data
├── docs/                 # Documentation
├── src/                  # Source code
│   ├── data/            # Data ingestion
│   ├── models/          # ML models
│   └── core/            # Core utilities
└── tests/               # Test suite
```

### Code Style

- Black for formatting
- isort for import sorting
- Pylint for linting
- Mypy for type checking

## Testing Requirements

### Unit Tests

- pytest for test framework
- Coverage reporting
- Mock external APIs

### Integration Tests

- Database operations
- API interactions
- End-to-end pipeline

## Performance Requirements

### Data Pipeline

- Rate limiting for API calls
- Efficient database operations
- Incremental updates

### Model Performance

- Minimum 80% precision
- Sub-second prediction time
- Regular retraining capability

## Monitoring and Logging

### Logging

- Structured logging
- Different log levels
- File and console output

### Metrics

- Model performance tracking
- Data pipeline statistics
- API call monitoring

## Security Requirements

### API Keys

- Secure storage in .env
- No keys in version control
- Regular key rotation

### Database Access

- MotherDuck token management
- Connection string security
- Access control

## Deployment Requirements

### Local Development

- Poetry for dependency management
- Pre-commit hooks
- Development database

### Production

- MotherDuck cloud database
- Automated data updates
- Model versioning
