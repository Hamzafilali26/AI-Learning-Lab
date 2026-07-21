import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Columns:
# 0 -> Age
# 1 -> Monthly income
# 2 -> Previous purchases

X = np.array([
    [22, 3000, 2],
    [45, 7000, 8],
    [31, 5000, 6],
    [52, 4500, 3],
    [27, 9000, 12],
    [40, 6000, 7],
    [36, 3500, 4],
    [29, 8000, 10],
    [48, 10000, 9],
    [33, 5500, 5],
], dtype=float)


actual_scores = np.array([
    2.0,
    6.0,
    4.3,
    4.1,
    7.2,
    5.3,
    3.3,
    6.5,
    7.3,
    4.1,
], dtype=float)


# Separate training data and test data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    actual_scores,
    test_size=0.2,
    random_state=42,
)


# The pipeline applies the steps in this order:
#
# 1. StandardScaler standardizes the features.
# 2. LinearRegression learns the weights and bias.

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LinearRegression(),
    ),
])


# fit() first trains the scaler on X_train,
# then trains LinearRegression using the standardized data.

pipeline.fit(
    X_train,
    y_train,
)


# predict() automatically standardizes X_test
# before applying the trained regression model.

test_predictions = pipeline.predict(X_test)


# Residual = actual value - predicted value

test_residuals = (
    y_test - test_predictions
)


# Calculate evaluation metrics

test_mae = mean_absolute_error(
    y_test,
    test_predictions,
)

test_mse = mean_squared_error(
    y_test,
    test_predictions,
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    test_predictions,
)


# Access the trained model inside the pipeline

trained_model = pipeline.named_steps["model"]

weights = trained_model.coef_
bias = trained_model.intercept_


print("Weights:")
print(weights)

print("Bias:")
print(bias)

print("Actual test scores:")
print(y_test)

print("Predicted test scores:")
print(test_predictions)

print("Residuals:")
print(test_residuals)

print("MAE:")
print(test_mae)

print("MSE:")
print(test_mse)

print("RMSE:")
print(test_rmse)

print("R2:")
print(test_r2)