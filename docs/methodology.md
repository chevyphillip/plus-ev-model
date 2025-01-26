# NBA Assist Prediction Model Methodology

## Overview

The NBA Assist Prediction Model uses logistic regression to predict the probability of a player recording 5 or more assists in their next game. The model leverages historical player statistics and rolling averages to make predictions.

## Data Pipeline

### Data Collection

- Source: NBA official statistics via NBA API
- Frequency: Daily updates
- Storage: DuckDB database for efficient querying and storage

### Feature Engineering

1. Rolling Statistics (5-game window):
   - Assists
   - Points
   - Minutes played
   - Field goal percentage
   - Plus/minus

2. Feature Importance (normalized coefficients):

   ```
   ast_rolling_5:        2.23
   pts_rolling_5:        0.65
   min_rolling_5:        0.64
   fg_pct_rolling_5:     0.27
   plus_minus_rolling_5: 0.01
   ```

## Model Architecture

### Logistic Regression

- Binary classification (5+ assists vs. <5 assists)
- Standardized features using StandardScaler
- Random state: 42 for reproducibility

### Data Split

- Training set: 80% of data
- Test set: 20% of data
- Stratified by target variable

## Model Performance

### Metrics

- Accuracy: 92.2%
  - Overall correct predictions
- Precision: 80.0%
  - When model predicts 5+ assists, it's correct 80% of the time
- Recall: 61.5%
  - Model identifies 61.5% of actual 5+ assist games
- F1 Score: 69.6%
  - Harmonic mean of precision and recall
- ROC AUC: 97.6%
  - Excellent ability to rank predictions

### Validation Results

Example predictions for top playmakers:

```
Trae Young:        99.9% (10.6 avg)
Tyrese Haliburton: 99.3% (9.5 avg)
James Harden:      99.1% (9.4 avg)
Nikola Jokić:      99.7% (9.2 avg)
Chris Paul:        97.0% (8.7 avg)
```

## Implementation Details

### Feature Processing

1. Data sorting by player and season
2. Rolling window calculations (5 games)
3. Feature standardization
4. Target variable creation (binary: ≥5 assists)

### Model Training

1. Data preparation and splitting
2. Feature scaling
3. Model fitting with logistic regression
4. Performance evaluation on test set

### Prediction Pipeline

1. Fetch player's recent games
2. Calculate rolling averages
3. Scale features
4. Generate probability prediction
5. Return prediction with confidence score

## Limitations and Considerations

1. Data Dependencies
   - Requires at least 5 games of history
   - Sensitive to missing data

2. Model Assumptions
   - Linear relationship between features
   - Independence between observations

3. External Factors Not Considered
   - Injuries
   - Team matchups
   - Back-to-back games
   - Home/away splits

## Future Improvements

1. Feature Engineering
   - Incorporate opponent defensive ratings
   - Add team pace factors
   - Include rest days between games

2. Model Enhancements
   - Experiment with non-linear models
   - Add ensemble methods
   - Implement cross-validation

3. Prediction Refinements
   - Dynamic threshold adjustment
   - Confidence interval calculations
   - Uncertainty quantification

## Usage Guidelines

1. Data Freshness
   - Update stats daily
   - Verify data quality
   - Check for missing games

2. Prediction Interpretation
   - Consider probability scores
   - Review recent performance
   - Account for external factors

3. Model Monitoring
   - Track prediction accuracy
   - Monitor feature distributions
   - Validate assumptions regularly
