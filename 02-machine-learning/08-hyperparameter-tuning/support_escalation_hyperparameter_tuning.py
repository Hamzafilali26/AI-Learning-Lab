from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# Load the dataset

csv_path = (
    Path(__file__).parent
    / "support_escalation_tuning.csv"
)

data = pd.read_csv(csv_path)


# Inspect the dataset

print("Dataset shape:")
print(data.shape)

print("Missing values:")
print(data.isna().sum())

print("Class distribution:")
print(data["escalated"].value_counts())


# Define feature groups and target

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

X = data[
    numerical_features
    + categorical_features
]

y = data["escalated"]


# Keep one test set untouched

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Numerical preprocessing

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


# Categorical preprocessing

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
        ),
    ),
])


# Apply preprocessing to the correct columns

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


# Build the complete Logistic Regression pipeline

model_pipeline = Pipeline([
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


# Create five stratified folds

cross_validator = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# --------------------------------------------------
# DEFAULT MODEL
# --------------------------------------------------

model_pipeline.fit(
    X_train,
    y_train,
)

default_predictions = (
    model_pipeline.predict(X_test)
)

default_f1 = f1_score(
    y_test,
    default_predictions,
    zero_division=0,
)

print("Default model F1:")
print(default_f1)


# --------------------------------------------------
# HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# --------------------------------------------------

parameter_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ],
}


grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cross_validator,
    return_train_score=True,
)


grid_search.fit(
    X_train,
    y_train,
)


# Best hyperparameter and best mean CV score

print("Best parameters:")
print(grid_search.best_params_)

print("Best cross-validation F1:")
print(grid_search.best_score_)


# See the result of every C value

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_summary = grid_results[
    [
        "param_model__C",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
].sort_values(
    "rank_test_score"
)

print("Grid search results:")
print(grid_summary)


# Retrieve the best tuned model

best_model = grid_search.best_estimator_


# Evaluate the tuned model on the untouched test set

best_predictions = best_model.predict(
    X_test
)

best_accuracy = accuracy_score(
    y_test,
    best_predictions,
)

best_precision = precision_score(
    y_test,
    best_predictions,
    zero_division=0,
)

best_recall = recall_score(
    y_test,
    best_predictions,
    zero_division=0,
)

best_f1 = f1_score(
    y_test,
    best_predictions,
    zero_division=0,
)

best_matrix = confusion_matrix(
    y_test,
    best_predictions,
)


print("Best model accuracy:")
print(best_accuracy)

print("Best model precision:")
print(best_precision)

print("Best model recall:")
print(best_recall)

print("Best model F1:")
print(best_f1)

print("Best model confusion matrix:")
print(best_matrix)


# Retrieve coefficients and bias

trained_preprocessor = (
    best_model.named_steps["preprocessor"]
)

trained_model = (
    best_model.named_steps["model"]
)

feature_names = (
    trained_preprocessor.get_feature_names_out()
)

coefficients = pd.Series(
    trained_model.coef_[0],
    index=feature_names,
).sort_values(
    ascending=False
)

bias = trained_model.intercept_[0]


print("Best model coefficients:")
print(coefficients)

print("Best model bias:")
print(bias)


# Compare default and tuned model

print("Default model F1:")
print(default_f1)

print("Tuned model F1:")
print(best_f1)


from sklearn.model_selection import RandomizedSearchCV

# Define the possible hyperparameter values

parameter_distributions = {
    "model__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100,
        1000,
    ],
}


# Create RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=parameter_distributions,
    n_iter=4,
    scoring="f1",
    cv=cross_validator,
    random_state=42,
    return_train_score=True,
)


# Run the search

random_search.fit(
    X_train,
    y_train,
)


# Display the best result

print("Best RandomizedSearchCV parameters:")
print(random_search.best_params_)

print("Best RandomizedSearchCV cross-validation F1:")
print(random_search.best_score_)


# Display every randomly tested configuration

random_results = pd.DataFrame(random_search.cv_results_)

random_summary = random_results[
    [
        "param_model__C",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
].sort_values("rank_test_score")

print("Randomized search results:")
print(random_summary)


# Retrieve the best tuned model

best_random_model = random_search.best_estimator_


# Predict the untouched test set

random_predictions = random_search.predict(X_test)


# Calculate final test metrics

random_accuracy = accuracy_score(
    y_test,
    random_predictions,
)

random_precision = precision_score(
    y_test,
    random_predictions,
    zero_division=0,
)

random_recall = recall_score(
    y_test,
    random_predictions,
    zero_division=0,
)

random_f1 = f1_score(
    y_test,
    random_predictions,
    zero_division=0,
)

random_matrix = confusion_matrix(
    y_test,
    random_predictions,
)


# Display final test results

print("RandomizedSearchCV final accuracy:")
print(random_accuracy)

print("RandomizedSearchCV final precision:")
print(random_precision)

print("RandomizedSearchCV final recall:")
print(random_recall)

print("RandomizedSearchCV final F1:")
print(random_f1)

print("RandomizedSearchCV confusion matrix:")
print(random_matrix)


# Compare GridSearchCV and RandomizedSearchCV

print("GridSearchCV final F1:")
print(best_f1)

print("RandomizedSearchCV final F1:")
print(random_f1)
