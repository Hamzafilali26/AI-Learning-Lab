# Support Escalation Cross-Validation

## Goal

Evaluate a support-ticket escalation classifier more reliably using cross-validation.

The target is:

```text
escalated
```

Classes:

```text
0 = not escalated
1 = escalated
```

# Main Definitions

## Cross-Validation

Cross-validation is a model evaluation method.

Instead of evaluating the model with only one validation split, the training data is divided into several parts called folds.

The model is trained and validated several times.

Each time:

```text
some folds are used for training
one fold is used for validation
```

This gives several validation scores instead of only one.

Cross-validation helps estimate:

- average model performance;
- model stability;
- possible overfitting;
- how sensitive the model is to the selected data split.

## Fold

A fold is one subset of the training data.

For example, with five-fold cross-validation:

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Every fold is used once as validation data.

## K-Fold Cross-Validation

K-fold cross-validation divides the training data into `k` folds.

If:

```text
k = 5
```

the model is trained five times.

```text
Iteration 1:
Training   = Folds 2, 3, 4, 5
Validation = Fold 1

Iteration 2:
Training   = Folds 1, 3, 4, 5
Validation = Fold 2

Iteration 3:
Training   = Folds 1, 2, 4, 5
Validation = Fold 3

Iteration 4:
Training   = Folds 1, 2, 3, 5
Validation = Fold 4

Iteration 5:
Training   = Folds 1, 2, 3, 4
Validation = Fold 5
```

Each observation is:

```text
used for training four times
used for validation one time
```

## Stratified K-Fold

Stratified k-fold cross-validation is used mainly for classification.

It tries to preserve approximately the same class proportions in every fold.

Example:

```text
60% class 0
40% class 1
```

Each fold should contain approximately:

```text
60% class 0
40% class 1
```

This project uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

### `n_splits=5`

Divide the training data into five folds.

### `shuffle=True`

Shuffle the rows before creating the folds.

### `random_state=42`

Create the same fold assignment every time the program runs.

The number `42` is arbitrary. Any fixed integer can be used.

# Data Roles

## Training Data

Training data is used to learn:

- Logistic Regression coefficients;
- the model bias;
- imputation values;
- feature means;
- feature standard deviations;
- categorical values used by the encoder.

## Validation Data

Validation data is used during model development.

It evaluates the model on observations that were not used during that training iteration.

In cross-validation, each fold becomes the validation data once.

## Test Data

The test set is kept completely separate from cross-validation.

It is used only after model development to measure final performance on unseen data.

The workflow is:

```text
Complete dataset
├── Training set
│   └── Cross-validation
└── Untouched test set
```

Cross-validation uses:

```text
X_train
y_train
```

It does not use:

```text
X_test
y_test
```

until the final evaluation.

# Why Keep the Test Set Untouched?

If the test set is used during model development, it is no longer truly unseen.

This can produce an overly optimistic performance estimate.

The test set must remain untouched so it can provide a more honest final evaluation.

# Data Leakage

Data leakage happens when information from validation or test data influences model training.

An incorrect workflow would be:

```text
complete dataset
→ imputation
→ scaling
→ encoding
→ cross-validation
```

This allows validation data to influence:

- medians;
- means;
- standard deviations;
- encoded categories.

The correct workflow is:

```text
Pipeline
├── SimpleImputer
├── StandardScaler
├── OneHotEncoder
└── LogisticRegression
```

Then cross-validation evaluates the complete pipeline.

During every fold, preprocessing is learned only from that fold's training portion.

# Validation Score

A validation score measures model performance on the fold excluded from training during one iteration.

With five folds, we obtain five validation scores.

Example:

```text
Fold 1 F1 = 0.80
Fold 2 F1 = 0.75
Fold 3 F1 = 0.85
Fold 4 F1 = 0.78
Fold 5 F1 = 0.82
```

# Mean Validation Score

The mean validation score summarizes the average performance across all folds.

For `k` folds:

\[
\bar{s}
=
\frac{s_1+s_2+\cdots+s_k}{k}
\]

Where:

- \(s_i\) is the validation score of fold \(i\);
- \(k\) is the number of folds;
- \(\bar{s}\) is the mean validation score.

For five folds:

\[
\bar{s}
=
\frac{s_1+s_2+s_3+s_4+s_5}{5}
\]

Example:

```text
mean F1 = 0.80
```

Interpretation:

> On average, the model obtained an F1 score of 0.80 across the five validation folds.

# Standard Deviation

Standard deviation measures how much the fold scores vary around their mean.

Interpretation:

```text
small standard deviation
→ performance is stable across folds

large standard deviation
→ performance changes significantly between folds
```

Examples:

```text
mean F1 = 0.82
standard deviation = 0.02
```

Interpretation:

> The model performs consistently across the folds.

```text
mean F1 = 0.82
standard deviation = 0.18
```

Interpretation:

> The average is good, but performance is unstable and strongly depends on the selected rows.

# Training Score

The training score measures performance on the data used to fit the model during one fold.

A high training score alone does not prove that the model generalizes.

# Training-Validation Gap

The training-validation gap is the difference between training performance and validation performance.

Example:

```text
training F1 = 0.98
validation F1 = 0.70
```

This large gap may indicate overfitting.

Example:

```text
training F1 = 0.84
validation F1 = 0.81
```

This smaller gap suggests better generalization.

# Overfitting

Overfitting happens when a model learns the training data too closely but performs poorly on unseen data.

Possible sign:

```text
very high training score
much lower validation score
```

Cross-validation helps detect overfitting by comparing performance across several validation folds.

# Underfitting

Underfitting happens when a model is too simple to learn useful patterns.

Possible sign:

```text
low training score
low validation score
```

The model performs poorly even on the training data.

# Metrics Used

## Accuracy

Accuracy measures the proportion of all correct predictions.

\[
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}
\]

## Precision

Precision answers:

```text
Among tickets predicted as escalated,
how many were actually escalated?
```

\[
Precision =
\frac{TP}
{TP + FP}
\]

## Recall

Recall answers:

```text
Among tickets actually escalated,
how many did the model detect?
```

\[
Recall =
\frac{TP}
{TP + FN}
\]

Recall is important here because a false negative means the model missed a ticket that was actually escalated.

## F1 Score

F1 combines precision and recall.

\[
F1 =
2
\times
\frac{Precision \times Recall}
{Precision + Recall}
\]

A high F1 score requires both precision and recall to be reasonably high.

# Current Project Structure

```text
07-cross-validation-classification/
├── support_escalation.csv
├── support_escalation_cross_validation.py
└── README.md
```

# Run

```bash
uv run python support_escalation_cross_validation.py
```

# Current Progress

Completed:

- dataset loading;
- feature and target selection;
- untouched train-test split;
- numerical preprocessing pipeline;
- categorical preprocessing pipeline;
- `ColumnTransformer`;
- Logistic Regression pipeline;
- five stratified folds.

Next:

- define evaluation metrics;
- run cross-validation;
- display fold scores;
- calculate mean and standard deviation;
- evaluate the final model on the untouched test set.

# Important Limitation

The dataset is small and synthetic.

Cross-validation gives a better evaluation than one train-test split, but it does not magically turn a tiny artificial dataset into production evidence. Statistics remain stubbornly non-magical.
