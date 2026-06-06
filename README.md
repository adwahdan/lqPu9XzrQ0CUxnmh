# ACME Happiness Survey

Predicting whether a customer is happy from a short post-delivery survey, and working out which questions are actually worth asking.

## The problem

ACME sends a six-question survey after each delivery. Every question is scored 1 to 5, and the customer is separately marked happy or unhappy. I wanted to do two things: build a model that predicts happiness from the six answers, and find the smallest set of questions that keeps most of the predictive value so the survey can be cut down.

## The data

The file is ACME-HappinessSurvey2020.csv: 126 responses, 7 columns, nothing missing. The target is customer_happiness (1 happy, 0 unhappy). The six features are delivery_timeliness, order_accuracy, product_availability, price_value_perception, courier_service_rating and app_usability, all on a 1 to 5 scale. The original columns were named Y and X1 to X6; I renamed them so the analysis reads more clearly.

Classes are roughly balanced at about 55 percent happy. Sixteen rows exactly repeat another response. I kept those instead of dropping them. Two customers giving the same answers are still two real customers, and survey answers naturally bunch up on the same few patterns, so deleting repeats would misrepresent how common each pattern actually is.

## What I did

Started with the usual checks: shape, types, nulls, duplicates, and confirming every feature sits inside 1 to 5 with a binary target. Then exploratory work, with per-feature distributions split by class, a correlation matrix, and VIF (all under 1.5, so no multicollinearity to worry about).

To test which features matter I ran Welch's t-test and Mann-Whitney U comparing happy and unhappy customers on each question. For modelling I tried Logistic Regression, a Decision Tree, Random Forest and XGBoost on a stratified 75/25 split, then tuned with GridSearchCV over 5-fold cross-validation. Finally I used Random Forest importance and RFECV to find the minimal feature set.

## Results

Random Forest on the full data gets to about 81 percent accuracy. On deduplicated data the best score drops to roughly 64 percent. Most of that gap is the duplicates: with a random split some identical rows end up in both train and test, so the model gets credit for rows it already saw. 81 is the optimistic read and 64 the pessimistic one, and the honest number sits somewhere in between. Cross-validation is the cleaner way to report it. Either way the feature story doesn't change.

The two questions that actually separate happy from unhappy customers are delivery_timeliness (p = 0.0022) and courier_service_rating (p = 0.0197). The rest aren't significant, and order_accuracy shows no relationship at all. RFECV cut the survey down to three features and held the same accuracy, which answers the second goal: the survey can be shortened without losing much.

## Running it

```
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost statsmodels missingno
jupyter notebook Happy_Customers.ipynb
```

Keep the CSV in the same folder as the notebook and run the cells top to bottom.

## Takeaways

Delivery speed and courier service are the things tied to whether a customer ends up happy. order_accuracy is a candidate to drop or rewrite. And the real limit here is sample size, not the model, so more responses would do more than any extra tuning.
