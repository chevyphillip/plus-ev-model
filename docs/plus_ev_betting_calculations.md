# +EV Betting Guide: Formulas, Line Deviation, and Market-Consensus Averages

This guide outlines how to evaluate **+EV (positive expected value)** bets, **deviate a line**, and create a **market-consensus average** using sharp books like Pinnacle and BetOnline. These steps are crucial for finding the true probability and identifying value bets.

---

## 1. Evaluating +EV (Positive Expected Value)

### Formula for Expected Value (EV)
The expected value of a bet tells you how much you can expect to win or lose on average per bet. The formula is:

\[
EV = (Probability \, of \, Winning \times Potential \, Profit) - (Probability \, of \, Losing \times Amount \, Lost)
\]

#### Steps to Calculate EV
1. **Convert Odds to Implied Probability**:
   - For **American odds**:
     - **Positive Odds (e.g., +150)**:  
       \[
       \text{Implied Probability} = \frac{100}{\text{Odds} + 100}
       \]
     - **Negative Odds (e.g., -110)**:  
       \[
       \text{Implied Probability} = \frac{\text{Odds}}{\text{Odds} + 100}
       \]
   - For **Decimal odds**:  
     \[
     \text{Implied Probability} = \frac{1}{\text{Decimal Odds}}
     \]

2. **Compare to Your Model’s Probability**:
   - If your model predicts a higher probability than the implied probability, the bet is +EV.

3. **Calculate EV**:
   - Example: You bet $100 on a player to score Over 22.5 points at -110 odds.
     - Implied Probability:  
       \[
       \frac{110}{110 + 100} = 52.38\%
       \]
     - Your Model’s Probability: 60%.
     - Potential Profit: $90.91 (for a $100 bet).
     - EV Calculation:  
       \[
       EV = (0.60 \times 90.91) - (0.40 \times 100) = 54.55 - 40 = +14.55
       \]
     - This is a +EV bet because the EV is positive.

---

## 2. Deviating a Line
Deviating a line means adjusting the bookmaker’s odds to account for factors like injuries, lineup changes, or market inefficiencies.

### Steps to Deviate a Line
1. **Identify Key Factors**:
   - Injuries to key players.
   - Changes in starting lineups.
   - Weather conditions (for outdoor sports).
   - Recent performance trends.

2. **Adjust the Line**:
   - Use your model to estimate how these factors affect the player’s performance.
   - Example: If a star player is injured, reduce the team’s expected points by a certain percentage.

3. **Recalculate the Implied Probability**:
   - Use the adjusted line to calculate the new implied probability.

---

## 3. Creating a Market-Consensus Average
Sharp books like **Pinnacle** and **BetOnline** are known for their accurate odds. By averaging their lines, you can get closer to the "true probability."

### Steps to Create a Market-Consensus Average
1. **Collect Odds from Multiple Books**:
   - Gather odds for the same player prop from sharp books (e.g., Pinnacle, BetOnline) and other books.

2. **Convert Odds to Implied Probabilities**:
   - Use the formulas above to convert the odds from each book into implied probabilities.

3. **Calculate the Average Implied Probability**:
   - Average the implied probabilities from the sharp books to get the market-consensus probability.
   - Example:
     - Pinnacle: -110 → 52.38%
     - BetOnline: -105 → 51.22%
     - Market-Consensus Probability:  
       \[
       \frac{52.38\% + 51.22\%}{2} = 51.8\%
       \]

4. **Compare to Your Model’s Probability**:
   - If your model predicts a higher probability than the market-consensus, it’s a +EV bet.

---

## 4. Combining Everything to Find True Probability
Here’s how to combine these steps to find the true probability and identify +EV bets:

### Workflow
1. **Collect Odds**:
   - Gather odds from sharp books (Pinnacle, BetOnline) and other books.

2. **Calculate Market-Consensus Probability**:
   - Average the implied probabilities from the sharp books.

3. **Adjust for Deviations**:
   - Use your model to adjust the market-consensus probability based on factors like injuries or lineup changes.

4. **Compare to Your Model’s Probability**:
   - If your model’s probability is higher than the adjusted market-consensus, it’s a +EV bet.

5. **Calculate EV**:
   - Use the EV formula to confirm the bet is +EV.

---

## Example
Let’s say you’re analyzing a player prop for **Over 22.5 points**:

1. **Collect Odds**:
   - Pinnacle: -110 → 52.38%
   - BetOnline: -105 → 51.22%
   - Market-Consensus Probability: 51.8%.

2. **Adjust for Deviations**:
   - Your model accounts for an injury to the opposing team’s best defender, increasing the player’s expected points.
   - Adjusted Probability: 55%.

3. **Compare to Your Model’s Probability**:
   - Your model predicts a 60% chance of the player scoring Over 22.5 points.
   - Since 60% > 55%, this is a +EV opportunity.

4. **Calculate EV**:
   - Bet $100 at -110 odds.
   - Potential Profit: $90.91.
   - EV Calculation:  
     \[
     EV = (0.60 \times 90.91) - (0.40 \times 100) = 54.55 - 40 = +14.55
     \]
   - This is a +EV bet.

---

## Tools and Tips
- **Odds Conversion Tools**: Use online tools or Python libraries to convert odds to probabilities.
- **Automation**: Use Windsurf to automate data collection, odds comparison, and EV calculations.
- **Line Shopping**: Always compare odds across multiple books to find the best value.

---

By following these steps, you can evaluate +EV bets, adjust lines for key factors, and use market-consensus averages to find the true probability. Let me know if you need help implementing any of this in Python or Windsurf! 🚀