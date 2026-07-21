import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load the CSV file into a pandas DataFrame

data = pd.read_csv(
    "customer_engagement.csv"
)


# Display the first rows of the dataset

print("Dataset:")
print(data.head())


# Display the number of rows and columns

print("Dataset shape:")
print(data.shape)


# Display the column names

print("Columns:")
print(data.columns)


# Check the data type of every column

print("Column data types:")
print(data.dtypes)


# Check the number of missing values in every column

print("Missing values:")
print(data.isna().sum())


# Select the input features

feature_columns = [
    "age",
    "monthly_income",
    "previous_purchases",
]

X = data[feature_columns]


# Select the numerical target

y = data["engagement_score"]


# Split the dataset into training data and test data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Create the Machine Learning pipeline

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


# Train the scaler and the Linear Regression model

pipeline.fit(
    X_train,
    y_train,
)


# Predict the engagement scores of unseen customers

test_predictions = pipeline.predict(
    X_test
)


# Calculate the prediction residuals

test_residuals = (
    y_test.to_numpy() - test_predictions
)


# Calculate the evaluation metrics

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


# Retrieve the trained Linear Regression model

trained_model = pipeline.named_steps["model"]


# Retrieve the learned weights and bias

weights = trained_model.coef_
bias = trained_model.intercept_


# Associate every feature with its learned weight

feature_weights = pd.Series(
    weights,
    index=feature_columns,
)


# Create a DataFrame containing the test results

results = pd.DataFrame({
    "actual_score": y_test.to_numpy(),
    "predicted_score": test_predictions,
    "residual": test_residuals,
})


# Display the learned parameters

print("Feature weights:")
print(feature_weights)

print("Bias:")
print(bias)


# Display actual values, predictions, and residuals

print("Test results:")
print(results)


# Display evaluation metrics

print("MAE:")
print(test_mae)

print("MSE:")
print(test_mse)

print("RMSE:")
print(test_rmse)

print("R2:")
print(test_r2)