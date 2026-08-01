# Support Escalation Cross-Validation

## Goal

Evaluate a support-ticket escalation classifier with five-fold stratified cross-validation.

## Target

```text
0 = not escalated
1 = escalated
```

## Cross-Validation

Cross-validation trains and validates the model several times using different parts of the training data.

It provides:

- several validation scores;
- an average score;
- a measure of score stability.

## Data Roles

```text
Training data
→ used during cross-validation

Validation data
→ one fold used to evaluate each iteration

Test data
→ kept untouched for final evaluation
```

## Five-Fold Process

```text
4 folds → training
1 fold  → validation
```

The process repeats five times.

## StratifiedKFold

`StratifiedKFold` keeps approximately the same class proportions in every fold.

## Project Structure

```text
07-cross-validation-classification/
├── support_escalation.csv
├── support_escalation_cross_validation.py
└── README.md
```

## Run

```bash
uv run python support_escalation_cross_validation.py
```

## Current Progress

Completed:

- loading the dataset;
- selecting features and target;
- creating an untouched test set;
- building the preprocessing pipeline;
- building the Logistic Regression pipeline;
- creating five stratified folds.

Next:

- define evaluation metrics;
- run cross-validation;
- interpret the fold scores.
