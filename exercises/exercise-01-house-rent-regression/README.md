# Exercise 01: House Rent Prediction

## Subject

A property platform wants to estimate apartment rent.

## Goal

Predict:

```text
monthly_rent
```

## Requirements

- Inspect missing values and duplicates.
- Separate numerical and categorical features.
- Use a train-test split.
- Impute missing numerical values with the median.
- Impute missing categorical values with the most frequent value.
- Standardize numerical features.
- One-hot encode categorical features.
- Train `LinearRegression`.
- Calculate MAE, MSE, RMSE, and R².
- Display predictions and residuals.
- Display transformed feature names, coefficients, and bias.

## Reflection Questions

1. Which metric is expressed directly in rent units?
2. What does a negative residual mean?
3. Why must preprocessing stay inside the pipeline?
4. Which category becomes the reference category?
