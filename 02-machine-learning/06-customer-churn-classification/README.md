# Customer Churn Classification

## Goal

Predict whether a customer will leave a service.

The target column is:

```text
churned
```

Its values are:

```text
0 = customer stays
1 = customer churns
```

## Why This Project Matters

Regression predicts a numerical value.

Classification predicts a category.

Customer churn is a binary classification problem because the model chooses between two classes:

```text
stay
churn
```

## Learning Objectives

After this project, you should be able to:

- distinguish regression from classification;
- train a binary classifier;
- obtain predicted classes;
- obtain class probabilities;
- interpret a confusion matrix;
- calculate accuracy, precision, recall, and F1 score;
- preprocess numerical and categorical features in one pipeline.

## New Concepts

### Binary Classification

Binary classification predicts one of two possible classes.

In this project:

```text
0 = stays
1 = churns
```

### Logistic Regression

Despite its name, Logistic Regression is a classification algorithm.

It estimates the probability that an observation belongs to the positive class.

In this project, the positive class is:

```text
churned = 1
```

### Predicted Probability

The model returns a probability between 0 and 1.

Example:

```text
0.82
```

means the model estimates an 82% probability that the customer will churn.

### Decision Threshold

A probability is converted into a class using a threshold.

The usual default threshold is:

```text
0.5
```

Therefore:

```text
probability below 0.5    → class 0
probability at least 0.5 → class 1
```

### Stratified Split

The split uses:

```python
stratify=y
```

This preserves approximately the same proportion of churned and non-churned customers in the training and test sets.

### Confusion Matrix

For binary classification, the confusion matrix is arranged as:

```text
[[true negatives, false positives],
 [false negatives, true positives]]
```

Definitions:

- true negative: correctly predicted that a customer stays;
- false positive: predicted churn, but the customer stays;
- false negative: predicted stay, but the customer churns;
- true positive: correctly predicted that a customer churns.

## Classification Metrics

### Accuracy

Accuracy is the proportion of all predictions that are correct.

```text
accuracy = correct predictions / all predictions
```

Accuracy can be misleading when one class is much more common than the other.

### Precision

Precision answers:

```text
Among customers predicted to churn,
how many actually churned?
```

```text
precision = true positives / (true positives + false positives)
```

### Recall

Recall answers:

```text
Among customers who actually churned,
how many did the model detect?
```

```text
recall = true positives / (true positives + false negatives)
```

### F1 Score

F1 combines precision and recall into one score.

```text
F1 = 2 × precision × recall / (precision + recall)
```

A high F1 score requires both precision and recall to be reasonably high.

## Complete Workflow

```text
CSV data
→ Train-test split with stratification
→ Numerical preprocessing
→ Categorical preprocessing
→ Logistic Regression
→ Class predictions
→ Probability predictions
→ Classification evaluation
```

## Project Structure

```text
06-customer-churn-classification/
├── customer_churn.csv
├── customer_churn_classification.py
└── README.md
```

## Installation

```bash
uv add pandas scikit-learn
```

## Run the Project

```bash
uv run python customer_churn_classification.py
```

## Important Limitation

The dataset is small and synthetic.

The project teaches the classification workflow, but its metrics must not be treated as evidence of real-world performance.
