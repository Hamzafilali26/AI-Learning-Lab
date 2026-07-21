import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import                  Role
# train_test_split  ==>   Separates training and test data
# StandardScaler   ==>   Standardizes numerical features
# LinearRegression ==>   Trains the regression model
# mean_absolute_error ==> Calculates the average absolute error
# mean_squared_error  ==> Calculates the average squared error
# r2_score            ==> Measures the variation explained by the model

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    actual_scores,
    test_size=0.2,
    random_state=42,  # Controls the random split
)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

scaler = StandardScaler()  # Standardizes the features

# Calculates the training means and standard deviations,
# then standardizes the training data
X_train_scaled = scaler.fit_transform(X_train)

# Standardizes the test data using the training statistics
X_test_scaled = scaler.transform(X_test)

# print(X_train_scaled)

# print("\nTraining means:")
# print(scaler.mean_)

# print("\nTraining standard deviations:")
# print(scaler.scale_)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

weights = model.coef_
bias = model.intercept_

age_weight = weights[0]
income_weight = weights[1]
purchases_weight = weights[2]

print("\nLearned weights:")
print("Age weight:", age_weight)
print("Income weight:", income_weight)
print("Purchases weight:", purchases_weight)

print("\nLearned bias:")
print(bias)

# Applies the learned equation to every training row
training_predictions = model.predict(X_train_scaled)

# Applies the learned equation to every test row
test_predictions = model.predict(X_test_scaled)

print("\nActual test scores:")
print(y_test)

print("\nPredicted test scores:")
print(test_predictions)

# Residual = actual value - predicted value
# Residuals allow us to inspect each prediction individually
test_residuals = y_test - test_predictions

test_mae = mean_absolute_error(
    y_test,
    test_predictions,
)
# Receives the actual values first and the predicted values second,
# then calculates the average absolute difference between them

# Mean Squared Error, MSE
test_mse = mean_squared_error(
    y_test,
    test_predictions,
)
# Calculates the average squared difference between
# the actual and predicted values

# Root Mean Squared Error, RMSE
test_rmse = np.sqrt(test_mse)

# MAE versus RMSE

# R², called the coefficient of determination, compares the model
# with a simple baseline that always predicts the mean of the actual values.
# The best possible value is 1.
# A value of 0 means that the model performs like the mean baseline.
# A negative value means that the model performs worse than the baseline.

test_r2 = r2_score(
    y_test,
    test_predictions,
)

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

# Residuals:
# [-0.2753, -0.1440]

# MAE:
# 0.2096

# MSE:
# 0.0483

# RMSE:
# 0.2197

# R²:
# 0.8858

# Residuals

# Both predictions are slightly too high.

# MAE

# The model misses the actual score by approximately
# 0.21 points on average.

# MSE

# The average squared error is approximately 0.0483.
# It penalizes larger errors more strongly,
# but its squared unit is not intuitive.

# RMSE

# The model’s typical error is approximately
# 0.22 engagement-score points,
# while larger errors receive more importance.

# R²

# The model performs substantially better than predicting
# the same mean score for both customers.
# However, this result is not reliable because
# the test set contains only two observations.

# | Tool      | Question answered                                              |
# | --------- | -------------------------------------------------------------- |
# | Residuals | What happened for each customer?                               |
# | MAE       | How wrong are the predictions on average?                      |
# | MSE       | How large are the errors when large mistakes are penalized?    |
# | RMSE      | What is the squared-error metric in the original target unit?  |
# | R²        | Is the model better than simply predicting the target mean?    |