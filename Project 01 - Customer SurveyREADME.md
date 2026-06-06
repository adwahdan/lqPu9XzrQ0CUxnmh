# ACME Happiness Survey

Predicting customer happiness from a short post-delivery survey, and identifying which
questions are worth keeping.

## Problem

ACME runs a six-question survey after each delivery. Each question is answered on a 1 to 5
scale, and the customer is separately recorded as happy or unhappy. The task is twofold:

1. Build a classifier that predicts happiness (`Y`) from the six survey questions.
2. Find the smallest set of questions that preserves predictive value, so the survey can be
   shortened.

## Data

`ACME-HappinessSurvey2020.csv`, 126 responses, 7 columns, no missing values.

| Column | Meaning | Type |
|---|---|---|
| `customer_happiness` | Target, 1 = happy, 0 = unhappy | binary |
| `delivery_timeliness` | Order arrived on time | 1 to 5 |
| `order_accuracy` | Everything ordered was in the package | 1 to 5 |
| `product_availability` | Wanted products were in stock | 1 to 5 |
| `price_value_perception` | Paid a good price | 1 to 5 |
| `courier_service_rating` | Satisfied with the courier | 1 to 5 |
| `app_usability` | App made ordering easy | 1 to 5 |

The original column names (`Y`, `X1` to `X6`) were renamed to the descriptive labels above for
readability. Classes are close to balanced (about 55% happy). Sixteen rows exactly repeat
another response. These are kept rather than dropped: two customers giving identical answers
are legitimate independent observations, and survey answers naturally cluster on common
patterns, so removing repeats would distort how frequent each pattern really is.

## Approach

1. **Inspection and cleaning.** Checked shape, types, nulls, duplicates, and value ranges.
   Confirmed every feature sits inside 1 to 5 and the target is binary.
2. **Exploratory analysis.** Per-feature distributions split by class, a correlation matrix,
   and variance inflation factors (all below 1.5, so no multicollinearity).
3. **Statistical testing.** Welch's t-test and Mann-Whitney U on each feature to test whether
   happy and unhappy customers answer differently.
4. **Modelling.** Logistic Regression, Decision Tree, Random Forest, and XGBoost, with a
   stratified 75/25 train/test split and class weighting. Hyperparameters tuned with
   `GridSearchCV` over stratified 5-fold cross-validation.
5. **Feature selection.** Random Forest importance (Gini and permutation) plus `RFECV` to find
   the minimal feature set.

## Results

Random Forest on the full data reaches **81.3% accuracy (0.84 F1)**. On deduplicated data the
ceiling drops to about 64%, where the best single split was a depth-2 Decision Tree:

| Model | Test accuracy | Test F1 |
|---|---|---|
| Random Forest (full data) | 81.3% | 0.84 |
| Decision Tree (deduplicated) | 64.3% | 0.64 |
| Random Forest (deduplicated) | 60.7% | 0.61 |
| Logistic Regression (deduplicated) | 57.1% | 0.57 |
| XGBoost (deduplicated) | 53.6% | 0.54 |

- `RFECV` reduced the survey to three features and matched the deduplicated full-model accuracy,
  confirming the survey can be shortened.
- `delivery_timeliness` and `courier_service_rating` are the only statistically significant
  predictors (Mann-Whitney p = 0.0022 and 0.0197). `order_accuracy` shows no relationship to
  happiness at all.

> Most of the gap between 81% and 64% is the exact duplicates. Under a random split, some
> identical (answers and label) rows land on both sides, so the model gets test credit for
> rows it effectively saw in training. That makes 81% an optimistic read and the deduplicated
> 64% a pessimistic one; the true out-of-sample number sits between them. Cross-validated
> accuracy is the cleaner way to report it. The feature conclusions hold either way.

See `CONCLUSION.md` for the full write-up and business interpretation.

## Repository

```
.
├── Happy_Customers.ipynb          # full analysis notebook
├── ACME-HappinessSurvey2020.csv   # raw data
├── CONCLUSION.md                  # findings and interpretation
└── README.md
```

## Running it

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost statsmodels missingno
jupyter notebook Happy_Customers.ipynb
```

Run the cells top to bottom. The notebook reads the CSV from the working directory, so keep
the data file alongside the notebook.

## Key takeaways

- Delivery speed and courier service are the levers most associated with customer happiness.
- The survey can be cut to roughly three questions without losing predictive value.
- Sample size is the main ceiling. With duplicates kept the model reaches 81%, but the honest
  generalization estimate is lower, and more responses would help more than further tuning.


Apziva UID: lqPu9XzrQ0CUxnmh
