# Devigging Process

## Step 1: Convert Odds to Implied Probability

- For positive odds: `100 / (Line + 100)`
- For negative odds: `Line / (Line + 100)`

Example:

- Odds: +110
- Calculation: 100 / (110 + 100)
- Result: 47.62%

## Step 2: Remove Vig

Formula: Individual Probability / Sum of All Probabilities

Example:

- Team 1 odds: -110 (52.38%)
- Team 2 odds: -110 (52.38%)
- Total probability: 104.76%
- True probability:
  - Team 1: 50%
  - Team 2: 50%

# Expected Value (EV) Calculation

Formula: (Probability of Win × Profit) - (Probability of Loss × Stake)

Example:

- Stake: 100
- Odds: +120
- True probability: 50%
- Calculation: (0.5 × 120) - (0.5 × 100)
- EV result: +10

# Positive EV Threshold

Rule: Bet has positive EV when true probability > implied probability

Example:

- Book odds: +150 (40% implied)
- True probability: 45%
- Result: Positive EV bet
