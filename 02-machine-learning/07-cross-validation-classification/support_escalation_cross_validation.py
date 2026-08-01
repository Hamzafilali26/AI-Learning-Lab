from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# Load the dataset

csv_path = Path(__file__).parent / "support_escalation.csv"
data = pd.read_csv(csv_path)


# Inspect the dataset

print("First rows:")
print(data.head())

print("Dataset shape:")
print(data.shape)

print("Missing values:")
print(data.isna().sum())

print("Duplicate rows:")
print(data.duplicated().sum())


# Define numerical features, categorical features, and target

numerical_features = [
    "waiting_time_minutes",
    "previous_contacts",
    "customer_tenure_months",
    "satisfaction_score",
]

categorical_features = [
    "channel",
    "plan_type",
    "issue_type",
]

X = data[numerical_features + categorical_features]
y = data["escalated"]


# Keep one test set untouched for the final evaluation

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Create the numerical preprocessing pipeline

numerical_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# Create the categorical preprocessing pipeline

categorical_pipeline = Pipeline(
    [
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
    ]
)


# Apply different preprocessing to numerical and categorical columns

preprocessor = ColumnTransformer(
    [
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
    ]
)


# Create the complete classification pipeline

model_pipeline = Pipeline(
    [
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
    ]
)


# Create five stratified folds

cross_validator = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# Define the metrics calculated for every fold

scoring = {
    "accuracy": make_scorer(
        accuracy_score,
    ),
    "precision": make_scorer(
        precision_score,
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        zero_division=0,
    ),
    "f1": make_scorer(
        f1_score,
        zero_division=0,
    ),
}


# Run cross-validation only on the training data

cross_validation_results = cross_validate(
    model_pipeline,
    X_train,
    y_train,
    cv=cross_validator,
    scoring=scoring,
    return_train_score=True,
)


# Create one result row for every fold

fold_results = pd.DataFrame(
    {
        "fold": range(
            1,
            cross_validator.get_n_splits() + 1,
        ),
        "train_accuracy": (cross_validation_results["train_accuracy"]),
        "validation_accuracy": (cross_validation_results["test_accuracy"]),
        "train_precision": (cross_validation_results["train_precision"]),
        "validation_precision": (cross_validation_results["test_precision"]),
        "train_recall": (cross_validation_results["train_recall"]),
        "validation_recall": (cross_validation_results["test_recall"]),
        "train_f1": (cross_validation_results["train_f1"]),
        "validation_f1": (cross_validation_results["test_f1"]),
    }
)


# Calculate the mean and standard deviation of validation scores

validation_summary = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ],
        "mean": [
            fold_results["validation_accuracy"].mean(),
            fold_results["validation_precision"].mean(),
            fold_results["validation_recall"].mean(),
            fold_results["validation_f1"].mean(),
        ],
        "standard_deviation": [
            fold_results["validation_accuracy"].std(),
            fold_results["validation_precision"].std(),
            fold_results["validation_recall"].std(),
            fold_results["validation_f1"].std(),
        ],
    }
)


# Train the final model on all training data

model_pipeline.fit(
    X_train,
    y_train,
)


# Predict the untouched test set

test_predictions = model_pipeline.predict(X_test)

escalation_probabilities = model_pipeline.predict_proba(X_test)[:, 1]


# Calculate final test metrics

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

test_matrix = confusion_matrix(
    y_test,
    test_predictions,
)


# Extract the confusion matrix values

tn, fp, fn, tp = test_matrix.ravel()


# Create the final test results table

test_results = X_test.copy()

test_results["actual"] = y_test.to_numpy()
test_results["predicted"] = test_predictions
test_results["escalation_probability"] = escalation_probabilities


# Retrieve transformed feature names, coefficients, and bias

trained_preprocessor = model_pipeline.named_steps["preprocessor"]

trained_model = model_pipeline.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()

coefficients = pd.Series(
    trained_model.coef_[0],
    index=feature_names,
).sort_values(
    ascending=False,
)

bias = trained_model.intercept_[0]


# Display cross-validation information

print("Training rows:")
print(len(X_train))

print("Untouched test rows:")
print(len(X_test))

print("Training class distribution:")
print(y_train.value_counts())

print("Cross-validation fold results:")
print(fold_results)

print("Validation summary:")
print(validation_summary)


# Display final test results

print("Final test results:")
print(test_results)

print("Final test accuracy:")
print(test_accuracy)

print("Final test precision:")
print(test_precision)

print("Final test recall:")
print(test_recall)

print("Final test F1 score:")
print(test_f1)

print("Final test confusion matrix:")
print(test_matrix)

print("True negatives:")
print(tn)

print("False positives:")
print(fp)

print("False negatives:")
print(fn)

print("True positives:")
print(tp)

print("Feature coefficients:")
print(coefficients)

print("Bias:")
print(bias)
