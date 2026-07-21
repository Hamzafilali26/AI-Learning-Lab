# Customer Engagement Regression with Pandas

This project loads customer data from a CSV file with pandas and trains a Linear Regression model with scikit-learn.

## Goal

Predict a customer engagement score using:

- age;
- monthly income;
- previous purchases.

## Main Concepts

### Pandas DataFrame

A `DataFrame` is a table with rows, named columns, data types, and an index.

It is used to load, inspect, clean, and select data before Machine Learning.

### CSV File

A CSV file stores tabular data as text.

Each row represents one observation, and each column represents one variable.

### Features

The input features are:

- `age`
- `monthly_income`
- `previous_purchases`

These values are used by the model to make predictions.

### Target

The target is:

- `engagement_score`

This is the numerical value the model learns to predict.

### Train-Test Split

The dataset is divided into:

- training data, used to train the model;
- test data, used to evaluate predictions on unseen customers.

### Pipeline

The pipeline performs two steps in order:

1. standardizes the numerical features;
2. trains the Linear Regression model.

### Residual

A residual is the difference between the actual and predicted score:

```text
residual = actual_score - predicted_score
```

A positive residual means the model predicted too low.

A negative residual means the model predicted too high.

## Evaluation Metrics

### MAE

Mean Absolute Error measures the average absolute prediction error.

### MSE

Mean Squared Error measures the average squared prediction error and penalizes large errors more strongly.

### RMSE

Root Mean Squared Error is the square root of MSE and uses the same unit as the target.

### R²

R² compares the model with a baseline that always predicts the mean target value.

## Project Structure

```text
03-pandas-csv-regression/
├── customer_engagement.csv
├── customer_engagement_pandas.py
└── README.md
```

## Installation

```bash
uv add pandas scikit-learn
```

## Run the Project

```bash
uv run python customer_engagement_pandas.py
```

## Important Limitation

The dataset contains only 10 customers.

It is useful for learning the workflow, but it is too small for reliable real-world evaluation.
