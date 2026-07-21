import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
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
from pathlib import Path

csv_path = Path(__file__).parent / "customer_engagement_missing.csv"

data = pd.read_csv(csv_path)



# Count missing values in each column

print("Missing values before cleaning:")
print(data.isna().sum())


# Count duplicate rows

print("Duplicate rows before cleaning:")
print(data.duplicated().sum())


# Remove duplicate rows

data = data.drop_duplicates()


# Select the input features

feature_columns = [
    "age",
    "monthly_income",
    "previous_purchases",
]

X = data[feature_columns]


# Select the target

y = data["engagement_score"]


# Split the data before fitting preprocessing steps

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)


# The pipeline performs these steps in order:
# 1. Replace missing feature values with the training median
# 2. Standardize the numerical features
# 3. Train the Linear Regression model

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LinearRegression(),
    ),
])


# Train all pipeline steps using only the training data

pipeline.fit(
    X_train,
    y_train,
)


# Predict engagement scores for unseen test customers

test_predictions = pipeline.predict(X_test)


# Calculate residuals

test_residuals = (
    y_test.to_numpy() - test_predictions
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


# Access the trained pipeline steps

trained_imputer = pipeline.named_steps["imputer"]
trained_model = pipeline.named_steps["model"]


# The imputer learned one median for each feature

learned_medians = pd.Series(
    trained_imputer.statistics_,
    index=feature_columns,
)


# The model learned one weight for each feature

feature_weights = pd.Series(
    trained_model.coef_,
    index=feature_columns,
)

bias = trained_model.intercept_


# Create a readable results table

results = pd.DataFrame({
    "actual_score": y_test.to_numpy(),
    "predicted_score": test_predictions,
    "residual": test_residuals,
})


# Display the results

print("Rows after removing duplicates:")
print(len(data))

print("Learned feature medians:")
print(learned_medians)

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