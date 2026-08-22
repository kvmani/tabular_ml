# User Guide

Tabular ML Workbench turns a CSV or Excel table into a reproducible,
leakage-safe classification or regression experiment. The interface is staged
so that every automatic recommendation can be inspected and changed.

## 1. Choose data

Select one of the six included datasets or upload a `.csv` or `.xlsx` file.
The initial profile reports row and column counts, inferred semantic types,
missingness, unique values, likely identifiers, target suggestions, and data
quality warnings. Uploaded data is held in memory and is not sent elsewhere.

## 2. Choose a target

Select the outcome column. The workbench recommends classification for
categorical or low-cardinality outcomes and regression for continuous numeric
outcomes. You can override the recommendation. Identifier-like, constant, and
very high-cardinality columns are highlighted because they can leak or obscure
the signal.

## 3. Prepare features

Each preparation choice is fitted only on the training folds.

- Missing numeric values: median, mean, or K-nearest-neighbour imputation.
- Missing categorical values: most-frequent filling, with optional missingness
  indicator columns.
- Categories: one-hot or ordinal encoding, plus optional rare-level grouping.
- Numeric scale: standard, min-max, robust, or unchanged.
- Dates: retain derived year/month/day/day-of-week components.
- Feature roles: exclude columns or revise an inferred semantic role.

The preview explains how many usable rows/features remain. A preparation plan
is stored with each model report so results can be interpreted and reproduced.

## 4. Inspect and treat outliers

Detection choices include IQR fences, z-score, modified z-score, Isolation
Forest, and Local Outlier Factor. Review the affected columns and rows before
choosing to winsorize, mask, drop, or reset. Dropping rows changes the training
population; winsorizing is often a safer exploratory default.

## 5. Explore visually

Available views include distributions with optional density curves,
correlation matrices, scatter plots with trend lines, grouped box plots, and
missingness maps. Every chart has a gear menu for palette, title, axis labels,
height, annotations, and export format. SVG is best for publication workflows;
PNG is convenient for presentations and documents.

## 6. Train and compare

The catalog provides CPU-only estimators appropriate to the detected task:

| Family | Classification | Regression |
|---|---|---|
| Linear | Logistic regression, SGD | Linear, SGD, Lasso, Elastic Net |
| Neighbours | KNN classifier | KNN regressor |
| Trees | Decision tree | Decision tree |
| Bagging | Random Forest, Extra Trees | Random Forest, Extra Trees |
| Boosting | Gradient Boosting, Histogram Gradient Boosting, AdaBoost | Same families |
| Kernel | Support Vector classifier | Support Vector regressor |
| Neural | scikit-learn MLP | scikit-learn MLP |
| Probabilistic | Gaussian Naive Bayes | — |

Parameter controls include inline explanations and bounded values. A fixed
holdout estimates final performance while cross-validation reports mean,
variation, and per-fold stability. Comparison uses the same split and primary
metric for every candidate.

Classification reports accuracy, balanced accuracy, precision, recall, F1,
ROC AUC where defined, log loss, confusion matrix, ROC and precision-recall
curves. Regression reports MAE, RMSE, R², explained variance, and residual
diagnostics. Native or permutation importance identifies influential features;
importance describes association with predictions, not causality.

## 7. Predict

The Predict stage creates an input form from the original schema. It returns a
class and probabilities when supported, or a numeric estimate for regression.
Models remain in the current server process. Downloaded joblib artifacts are
for trusted local use only: never load an untrusted Python model file.

## Practical guidance

- Keep a final holdout untouched while selecting models with cross-validation.
- Investigate warnings about target leakage before trusting a high score.
- Compare results across sensitive groups before any consequential use.
- Treat small datasets and rare classes cautiously; uncertainty will be high.
- Save the model report alongside exported figures to preserve configuration.
- The tool is educational and exploratory, not a substitute for domain review,
  monitoring, or a production model-governance process.
