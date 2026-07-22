from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# Load the CSV file from the same folder as this Python file

csv_path = (
    Path(__file__).parent
    / "customer_engagement_categorical.csv"
)

data = pd.read_csv(csv_path)


# Numerical features contain numbers with measurable quantities

numerical_features = [
    "age",
    "monthly_income",
    "previous_purchases",
]


# Categorical features contain labels or groups

categorical_features = [
    "membership_type",
]


# Select the input features and the target

X = data[
    numerical_features
    + categorical_features
]

y = data["engagement_score"]


# Separate training data and test data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)


# Numerical preprocessing:
# 1. Replace missing numbers with the training median
# 2. Standardize the numerical values

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


# Categorical preprocessing:
# 1. Replace missing categories with the most frequent category
# 2. Convert categories into numerical binary columns
#
# drop="first" removes the first category and uses it as
# the reference category.
#
# handle_unknown="ignore" prevents an error when a new category
# appears during prediction.

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        ),
    ),
])


# ColumnTransformer applies different preprocessing
# to different groups of columns.

preprocessor = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline,
        numerical_features,
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features,
    ),
])


# The complete pipeline:
# 1. Preprocess numerical and categorical columns
# 2. Train the Linear Regression model

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "model",
        LinearRegression(),
    ),
])


# Train the complete workflow

pipeline.fit(
    X_train,
    y_train,
)


# Predict engagement scores for unseen customers

test_predictions = pipeline.predict(X_test)


# Calculate residuals and evaluation metrics

test_residuals = (
    y_test.to_numpy()
    - test_predictions
)

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


# Retrieve transformed feature names and learned weights

trained_preprocessor = (
    pipeline.named_steps["preprocessor"]
)

trained_model = (
    pipeline.named_steps["model"]
)

transformed_feature_names = (
    trained_preprocessor.get_feature_names_out()
)

feature_weights = pd.Series(
    trained_model.coef_,
    index=transformed_feature_names,
)

bias = trained_model.intercept_


# Create a readable table with the test results

results = X_test.copy()

results["actual_score"] = (
    y_test.to_numpy()
)

results["predicted_score"] = (
    test_predictions
)

results["residual"] = (
    test_residuals
)


# Display the results

print("Dataset:")
print(data)

print("Feature data types:")
print(X.dtypes)

print("Transformed feature names:")
print(transformed_feature_names)

print("Learned feature weights:")
print(feature_weights)

print("Learned bias:")
print(bias)

print("Test results:")
print(results)

print("MAE:")
print(test_mae)

print("MSE:")
print(test_mse)

print("RMSE:")
print(test_rmse)

print("R2:")
print(test_r2)
