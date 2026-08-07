# Hyperparameter Tuning with GridSearchCV

## Goal

Tune a Logistic Regression classifier by testing several values of the hyperparameter `C` with cross-validation.

The business problem is:

```text
Predict whether a support ticket will be escalated.
```

Target:

```text
0 = not escalated
1 = escalated
```

## Learning Objectives

After this project, you should be able to:

- distinguish parameters from hyperparameters;
- explain regularization;
- explain the role of `C`;
- define a parameter grid;
- use `GridSearchCV`;
- interpret `best_params_`;
- interpret `best_score_`;
- retrieve `best_estimator_`;
- inspect `cv_results_`;
- compare a default model with a tuned model;
- evaluate the tuned model on an untouched test set.

# 1. Parameter

A parameter is learned by the model during training.

Examples:

```text
coefficients
bias
```

In scikit-learn:

```python
model.coef_
model.intercept_
```

# 2. Hyperparameter

A hyperparameter is chosen before training.

Examples in Logistic Regression:

```text
C
max_iter
solver
penalty
```

The model does not automatically search for the best value during a normal `fit()`.

# 3. Regularization

Regularization constrains the model while it learns.

Its goal is to reduce unnecessarily large coefficients and help control overfitting.

Conceptually:

\[
Total\ Cost
=
Prediction\ Loss
+
Regularization\ Penalty
\]

Regularization does not add a new feature to the dataset.

# 4. Logistic Regression `C`

In scikit-learn, `C` is the inverse of regularization strength.

```text
small C
→ stronger regularization
→ coefficients are constrained more

large C
→ weaker regularization
→ coefficients are constrained less
```

`C` is not business data.

It controls how the model learns from the data.

# 5. Hyperparameter Tuning

Hyperparameter tuning means:

```text
test several settings
→ evaluate each setting with cross-validation
→ select the best setting
```

# 6. Parameter Grid

The project tests:

```python
parameter_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ],
}
```

`model__C` means:

```text
model
→ pipeline step named "model"

__
→ access a parameter inside that step

C
→ Logistic Regression hyperparameter
```

# 7. GridSearchCV

`GridSearchCV` tests every value in the parameter grid using cross-validation.

In this project:

```text
5 values of C
×
5 stratified folds
```

Each candidate value is evaluated on the five folds.

The best value is selected using:

```text
F1 score
```

# 8. `best_params_`

```python
grid_search.best_params_
```

returns the hyperparameter configuration with the best mean cross-validation score.

Example:

```text
{'model__C': 1}
```

Interpretation:

> `C = 1` produced the best average F1 score across the cross-validation folds.

# 9. `best_score_`

```python
grid_search.best_score_
```

returns the best mean validation score.

Example:

```text
0.82
```

Interpretation:

> The best configuration obtained an average validation F1 score of 0.82.

# 10. `best_estimator_`

```python
grid_search.best_estimator_
```

returns the complete pipeline using the best hyperparameter values.

By default, `GridSearchCV` refits this best configuration on the complete training set.

# 11. `cv_results_`

```python
grid_search.cv_results_
```

contains the results of every tested value.

Useful columns:

```text
param_model__C
mean_train_score
mean_test_score
std_test_score
rank_test_score
```

Interpretation:

```text
mean_test_score
→ average validation F1

std_test_score
→ variation between folds

rank_test_score
→ ranking of the C values
```

# 12. Final Test Evaluation

The untouched test set is used only after tuning.

Correct workflow:

```text
training data
→ GridSearchCV
→ best C
→ best model
→ untouched test set
```

The test set must not be used to choose `C`.

# 13. Default Model vs Tuned Model

The project compares:

```text
default Logistic Regression
versus
tuned Logistic Regression
```

Interpretation:

```text
tuned F1 > default F1
→ tuning improved final F1

tuned F1 = default F1
→ no improvement on the test set

tuned F1 < default F1
→ tuned settings performed worse on the test set
```

A tuned model is not automatically better. Statistics refuses to provide guarantees just because we asked politely.

# Project Workflow

```text
Load data
→ inspect data
→ train-test split
→ preprocessing pipeline
→ Logistic Regression
→ default model
→ define C values
→ GridSearchCV
→ inspect all C results
→ retrieve best model
→ final test evaluation
→ coefficients and bias
→ compare default and tuned F1
```

# Project Structure

```text
08-hyperparameter-tuning/
├── support_escalation_tuning.csv
├── support_escalation_hyperparameter_tuning.py
└── README.md
```

# Run

```bash
uv run python support_escalation_hyperparameter_tuning.py
```

# Important Limitation

The dataset is synthetic.

The objective is to understand hyperparameter tuning and interpretation, not to claim production-level performance.
