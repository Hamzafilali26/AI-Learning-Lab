# Exercise 02: Employee Attrition

## Subject

A company wants to identify employees who may leave.

## Goal

Predict:

```text
left_company
```

Classes:

```text
0 = stays
1 = leaves
```

## Requirements

- Handle numerical and categorical features.
- Handle missing values.
- Use a stratified train-test split.
- Train `LogisticRegression`.
- Predict classes and probabilities.
- Calculate accuracy, precision, recall, and F1.
- Display the confusion matrix.
- Display feature coefficients.

## Reflection Questions

1. Which is more costly here: a false positive or a false negative?
2. Which metric deserves the most attention?
3. What does a probability of `0.78` mean?
4. What is the difference between `predict()` and `predict_proba()`?
5. What does a positive coefficient generally indicate?
