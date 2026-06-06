# ACME Happiness Survey

Predicting customer happiness from a six-question post-delivery survey, and identifying which questions are actually worth asking.

## Problem

A logistics company runs a short survey after each delivery. Customers answer six questions on a 1–5 scale, and a separate label records whether the customer was happy (1) or unhappy (0). Two things are needed:

1. A model that predicts happiness from the six answers.
2. A reading of which questions carry the signal, so the survey can be shortened without losing predictive power.

The survey questions, renamed for readability, cover: delivery timeliness, order accuracy, product availability, price/value perception, courier service rating, and app usability.

## Data

126 survey responses, six features plus one binary target. No missing values, and the two classes are roughly balanced. Sixteen rows exactly duplicate another response.

The duplicates are the central judgement call in this project. Removing them is the tidy default, but two customers giving identical 1–5 answers are legitimate, independent observations, and dropping them distorts how common each response pattern really is. The final analysis keeps them, and the difference this makes is measured directly rather than assumed (see Results).

## Approach

The notebook moves from data integrity, to understanding the signal, to modelling, to feature reduction.

**1. Inspection and quality checks**
Loaded the data, renamed the cryptic survey columns to meaningful names, and confirmed shape, dtypes, null counts, and duplicate counts. Asserted that every feature stays within its valid 1–5 range and that the target is strictly binary, so any out-of-range value would halt the notebook rather than slip through.

**2. Multicollinearity**
Built a Pearson correlation matrix and computed Variance Inflation Factors. All VIFs came in under 1.5, meaning the six questions measure largely distinct things and none is redundant on statistical grounds alone.

**3. Univariate analysis**
Plotted each feature's distribution split by happy versus unhappy, to see which questions visibly separate the two groups before formal testing.

**4. Statistical significance testing**
For each feature, compared happy and unhappy groups using both Welch's t-test (unequal variances) and the Mann-Whitney U test (non-parametric, robust to the discrete 1–5 scale). The Mann-Whitney result was used for the significance decision, since the Likert data is ordinal rather than normally distributed.

**5. Model comparison**
Split the data 75/25 with stratification, then trained four classifiers with class weighting to handle any imbalance: Logistic Regression, a shallow Decision Tree, Random Forest, and XGBoost. Compared them on weighted F1 and accuracy.

**6. Hyperparameter tuning**
Tuned XGBoost and Random Forest with `GridSearchCV` over 5-fold stratified cross-validation, searching depth, estimator count, learning rate / split criteria, and regularisation parameters.

**7. Feature importance and selection**
Read Random Forest importance two ways: Gini importance (impurity reduction) and permutation importance (drop in test performance when a feature is shuffled), since Gini alone can be misleading. Then ran `RFECV` to find the smallest feature set that holds cross-validated accuracy.

**8. Duplicate sensitivity check**
Re-ran the Random Forest on the full dataset with duplicates retained, using identical split settings, to quantify exactly how much the duplicate-handling decision moves the headline number.

## Results

**Which questions matter.** Only two features significantly separate happy from unhappy customers: delivery timeliness (p = 0.0022) and courier service rating (p = 0.0197). Order accuracy shows essentially no relationship with happiness (p = 0.93), making it the clearest candidate to drop from the survey.

**Model performance.** Random Forest was the strongest model, reaching 81.3% accuracy (0.84 F1) on the full data. On the deduplicated data the accuracy ceiling is roughly 64%, and RFECV reaches that same ceiling using just three features.

**Reading the gap honestly.** Most of the distance between 64% and 81% comes from the exact duplicates: under a random split, some identical rows can land in both train and test, so 81% is optimistic. The true generalization estimate sits between the two numbers. Crucially, the feature conclusions are stable either way; the duplicate handling moves the accuracy figure, not the story about which questions matter.

## Takeaway

Delivery speed and courier service are the levers tied to customer happiness. Order accuracy carries no signal and is the first question to cut. The six-question survey can be shortened to roughly three questions without sacrificing predictive value, which is the actionable result for the business.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, SciPy, matplotlib, seaborn.

## Repository contents

- `Happy_Customers.ipynb` — full analysis notebook, from data checks through modelling and conclusions.
- `ACME-HappinessSurvey2020.csv` — the survey dataset.

---

### Notes for reviewers

This project was completed as part of the Apziva AI Residency. The work it is meant to demonstrate is not just a fitted classifier but the reasoning around a small, messy dataset: deciding what to do about duplicate rows and then *measuring* the consequence of that decision rather than hiding it, choosing a significance test appropriate to ordinal survey data, and reading feature importance through more than one lens before recommending that a question be removed. On a 126-row dataset, the discipline of separating a stable conclusion (which features matter) from a fragile one (the exact accuracy number) is the point.

UID for Apziva Instructors: lqPu9XzrQ0CUxnmh
