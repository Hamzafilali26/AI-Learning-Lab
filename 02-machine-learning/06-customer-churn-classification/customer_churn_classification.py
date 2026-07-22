from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    / "customer_churn.csv"
)

data = pd.read_csv(csv_path)


# Numerical features

numerical_features = [
    "age",
    "monthly_income",
    "previous_purchases",
    "months_inactive",
    "support_tickets",
]


# Categorical features

categorical_features = [
    "membership_type",
]


# Select the input features and the target

X = data[
    numerical_features
    + categorical_features
]

y = data["churned"]


# Split the data while preserving the class proportions

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


# Prepare numerical columns

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


# Prepare categorical columns

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


# Apply the correct preprocessing to each column group

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


# Build the complete classification pipeline

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
        ),
    ),
])


# Train the preprocessing steps and the classifier

pipeline.fit(
    X_train,
    y_train,
)


# Predict the class:
# 0 means the customer stays
# 1 means the customer churns

test_predictions = pipeline.predict(X_test)


# Predict the probability of class 1

churn_probabilities = (
    pipeline.predict_proba(X_test)[:, 1]
)


# Calculate classification metrics

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_confusion_matrix = confusion_matrix(
    y_test,
    test_predictions,
)


# Create a readable results table

results = X_test.copy()

results["actual_churn"] = (
    y_test.to_numpy()
)

results["predicted_churn"] = (
    test_predictions
)

results["churn_probability"] = (
    churn_probabilities
)


# Retrieve transformed feature names and model coefficients

trained_preprocessor = (
    pipeline.named_steps["preprocessor"]
)

trained_model = (
    pipeline.named_steps["model"]
)

transformed_feature_names = (
    trained_preprocessor.get_feature_names_out()
)

feature_coefficients = pd.Series(
    trained_model.coef_[0],
    index=transformed_feature_names,
)


# Display the results

print("Class distribution:")
print(y.value_counts())

print("Test results:")
print(results)

print("Accuracy:")
print(test_accuracy)

print("Precision:")
print(test_precision)

print("Recall:")
print(test_recall)

print("F1 score:")
print(test_f1)

print("Confusion matrix:")
print(test_confusion_matrix)

print("Classification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

print("Feature coefficients:")
print(
    feature_coefficients.sort_values(
        ascending=False
    )
)
