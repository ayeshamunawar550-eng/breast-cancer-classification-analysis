# Breast Cancer Wisconsin (Diagnostic): Classification Analysis

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting whether a breast tumor is malignant or benign from 30 numeric
features taken from a digitized image of a fine needle aspirate (FNA) of a
breast mass, the classic **Breast Cancer Wisconsin (Diagnostic)** data set.

## What's in here

The notebook covers exploratory data analysis, a data-quality pass,
preprocessing, a comparison of five classification algorithms under
stratified cross-validation, hyperparameter tuning for the strongest
candidates, evaluation with bootstrap confidence intervals and a
statistical test for whether the top two models actually differ, feature
importance (both impurity-based and permutation-based), and a sensitivity
check on the data set's redundant, highly-correlated features.

## Results

The data set has 569 patients and 30 features, with no missing values and
no duplicate rows. A majority-class baseline gets 63.2% accuracy just by
always predicting "benign", so every model below needs to clear that bar
by a wide margin to be worth the added complexity.

Five models (Logistic Regression, Decision Tree, Random Forest, SVM with
an RBF kernel, and KNN) were compared with 5-fold stratified cross-validation
on ROC-AUC. Logistic Regression, SVM, and Random Forest came out on top and
were carried forward and tuned. The lead model is picked by that same
cross-validated score rather than by the held-out test set, since ranking
three closely-matched models on just 114 test patients would mostly be
reading noise; the test set is then used only to report performance
honestly for the selected model:

| Model | CV ROC-AUC | Test Accuracy | Precision | Recall | F1-score | Test ROC-AUC (95% CI) | Brier score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.996 | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 (0.986–1.000) | 0.021 |
| SVM (RBF, tuned) | 0.995 | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 (0.982–1.000) | 0.020 |
| Random Forest (tuned) | 0.989 | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 (0.984–1.000) | 0.032 |

(Exact figures, including the Brier scores, are in the notebook. A fixed
random seed makes every number reproducible run to run; two independent
executions of the notebook produced identical output down to the last
decimal place.)

Logistic Regression, SVM, and Random Forest all separate the classes very
well, which isn't surprising given how clean the underlying measurements
are and how differently malignant and benign tumors are shaped. Precision,
recall, F1, and ROC-AUC are all reported rather than just accuracy, since
in a diagnostic context a missed malignant case is a far more expensive
mistake than a false alarm. McNemar's exact test on the top two models'
test-set disagreements gives p = 1.0; with only 3 disagreements out of 114
patients, the small remaining gap between them isn't something you can
call statistically real.

By both impurity-based and permutation feature importance, `worst
perimeter`, `worst concave points`, `worst radius`, `mean concave points`,
and `worst area` are the features that consistently matter most, all
measures of tumor size and shape irregularity, and all recognized
indicators in the clinical literature this data set comes from. Several of
these features are highly correlated with each other (radius, perimeter,
and area are geometrically related), so a sensitivity check retrains
Logistic Regression after dropping the redundant ones: cross-validated
ROC-AUC moves by less than 0.001, meaning the correlation is real but
doesn't bias the results reported here.

## Scientific review

| Aspect | Assessment | Notes |
|---|---|---|
| Sample size | Weak point | 569 patients from a single institution's imaging pipeline is small by modern ML standards. Cross-validation and bootstrap confidence intervals are reported alongside the plain test-set numbers specifically so a single 80/20 split isn't mistaken for a precise estimate. |
| Data cleaning | Strong point | Verified directly rather than assumed: zero missing values across all 30 features, zero exact duplicate rows. The non-informative `id` column is dropped before any modeling so it can't leak into feature importance or distance-based models like KNN. |
| Class balance | Strong point | About 63% benign / 37% malignant, imbalanced enough that a majority-class baseline reaches 63.2% accuracy on its own. That baseline is reported explicitly, and precision, recall, F1, and ROC-AUC are tracked throughout rather than leaning on accuracy alone. |
| Preprocessing / leakage | Strong point | Every scaled model wraps `StandardScaler` inside a `Pipeline`, fit only on each training fold. This means cross-validation, grid search, and the final test evaluation all scale features without ever peeking at data outside the current fold. |
| Model comparison | Strong point | Five algorithms are screened with the same stratified 5-fold cross-validation and the same scoring metric (ROC-AUC) before any tuning happens, so the shortlist isn't based on which model happened to look best on a single split. |
| Model selection | Fixed | The lead model is chosen by cross-validated ROC-AUC (the same objective used for tuning) rather than by re-ranking on the 114-patient test set. Picking a "winner" using the test set is a subtle form of data snooping; using the test set only to *report* the selected model's performance keeps that number honest. |
| Hyperparameter tuning | Reasonable | Random Forest and SVM go through a grid search under the same cross-validation scheme as the initial screening; the grids are deliberately small given how little training data (455 rows) there is to search over. |
| Uncertainty reporting | Fixed | ROC-AUC is now reported with a 95% bootstrap confidence interval (1,000 resamples of the test set) for every final model, not as a single unqualified number. |
| Statistical comparison | Fixed | McNemar's exact test is run on the test-set disagreements between the top two models by cross-validated score, to check whether the gap between them is distinguishable from chance rather than just reporting whichever scored marginally higher. |
| Calibration | Reasonable | SVM's probability outputs go through `CalibratedClassifierCV`, and Brier scores are reported for every final model as a quick, standard check that predicted probabilities are meaningful, not just a good ranking. |
| Multicollinearity | Fixed | Several features (radius, perimeter, area, and their "worst" counterparts) are geometrically related and highly correlated. Rather than only noting this in the EDA, a sensitivity check drops the redundant ones and retrains Logistic Regression; cross-validated ROC-AUC moves by under 0.001, confirming the correlation doesn't distort the results reported here. |
| Feature importance method | Strong point | Both impurity-based `feature_importances_` and true out-of-sample permutation importance are computed for the Random Forest, and they agree, a useful cross-check since impurity-based scores alone are known to be biased toward high-cardinality numeric features. |
| Reproducibility | Strong point | `random_state` is fixed everywhere a split or model is created, and `requirements.txt` pins exact package versions. Two independent executions of the notebook were compared cell-by-cell and produced identical output. |
| Clinical framing | Strong point | The write-up is explicit that this is a benchmark exercise on one data set, not a validated diagnostic tool, and says so again in the Limitations section rather than only once at the top. |

## Data

`breast_cancer.csv` is a standard redistribution of the [UCI Machine
Learning Repository's Breast Cancer Wisconsin (Diagnostic) data
set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
569 rows, 30 numeric predictors, 1 categorical target (`M`/`B`). The
notebook maps the target to a numeric column (`malignant = 1`, `benign =
0`) and writes the result to `breast_cancer_clean.csv` as part of its first
processing step, so that file is a build artifact of running the notebook
rather than a separately maintained data source.

## Getting started

```bash
git clone https://github.com/ayeshamunawar550-eng/breast-cancer-classification-analysis.git
cd breast-cancer-classification-analysis
pip install -r requirements.txt
jupyter notebook breast-cancer-classification-analysis.ipynb
```

Then run all cells top to bottom (Kernel → Restart & Run All). The full
notebook, including the grid searches, takes a couple of minutes on a
laptop.

## Repository structure

```
.
├── breast-cancer-classification-analysis.ipynb   # main analysis notebook
├── breast_cancer.csv                             # raw data set
├── breast_cancer_clean.csv                       # cleaned data set (generated by the notebook)
├── requirements.txt                              # pinned Python dependencies
├── LICENSE                                       # MIT license
└── README.md
```

## Limitations

This is a benchmark/educational classification exercise, not a validated
diagnostic tool. The data set is a single, well-studied sample of 569
patients from one source; using anything like these models in an actual
clinical setting would require external validation on independent patient
populations and imaging pipelines first.

## License / attribution

Code in this repository is released under the [MIT License](LICENSE).

Data set: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995).
Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning
Repository. Used here for educational/benchmark purposes.
