# Stage 1 - Modeling Impact Review
Group: G009
Problem: MP244 - Bakery Bread Sales
Model Lead: Vishwas Bhargava

## 1. Leakage
Status: PRESENT

loaves_in_closing_register has extremely high
correlation with the target (0.998712) and is
unavailable in holdout data.

Impact:
Using this feature can cause data leakage and
unrealistically low training MAE.

## 2. Unit Mixup
Status: ABSENT

Impact:
No identified unit inconsistencies. No
additional modeling risk observed.

## 3. Sentinel Values
Status: ABSENT

Impact:
No identified placeholder values that could
distort numerical feature distributions.

## 4. Duplicate Rows
Status: PRESENT

Evidence: 55 duplicate rows.

Impact:
Duplicates can overweight certain observations
and bias model training and evaluation.

## 5. Stuck Sensor
Status: PRESENT

Evidence:
footfall_index remained 23 for 23 consecutive
days from April 1 to April 23.

Impact:
Incorrect sensor readings can distort feature
distributions and weaken model predictions.

## 6. Target Typos
Status: ABSENT

Evidence:
Target ranges from 43 to 561.
Mean: 295.11
Median: 290
Standard deviation: 94.08

No convincing 10x typographical outliers found.

Impact:
No confirmed target typo requiring correction.
Unverified target modifications could introduce
additional errors into regression training.

## 7. Spelling Variants
Status: PRESENT

Evidence:
day_type contains 17 raw representations
compared with 3 clean categories.

Impact:
Unnecessary categorical levels can increase
feature dimensionality and create inconsistent
category encoding.

## Conclusion

Target typo investigation found no convincing
evidence of 10x errors.

No raw data was modified.

Stage 1 investigation completed.
Model training deferred to Stage 2.