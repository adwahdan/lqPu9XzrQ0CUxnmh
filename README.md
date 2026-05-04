# Customer Happiness Prediction

## Problem
Predict binary customer happiness (Y) from 6 survey responses (X1–X6, Likert 1–5).

## Approach
Descriptive stats → correlation → VIF → t-tests → factor analysis → logistic regression / LDA / random forest.

## Result
65% accuracy with logistic regression on X1 alone. Multiple models converge at this ceiling, suggesting the limit is data, not algorithm.

## Bonus — Feature selection
Only X1 (on-time delivery) is a consistent predictor. All other questions can be dropped from future surveys without loss of predictive accuracy.

## Files
- `analysis.ipynb` — full workflow
- `data.csv` — dataset
