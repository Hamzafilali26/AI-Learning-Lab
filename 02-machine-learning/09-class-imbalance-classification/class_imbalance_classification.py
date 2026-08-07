from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load data
csv_path = Path(__file__).parent / "employee_attrition_imbalanced.csv"
data = pd.read_csv(csv_path)

print("Class counts:")
print(data["left_company"].value_counts())

print("Class proportions:")
print(data["left_company"].value_counts(normalize=True))


# 2. Define features and target
numerical_features = [
    "age",
    "monthly_salary",
    "years_at_company",
    "satisfaction_score",
    "training_hours",
]

categorical_features = [
    "overtime",
    "department",
]

target = "left_company"

X = data[numerical_features + categorical_features]
y = data[target]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 4. Preprocessing
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

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
        ),
    ),
])

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


# 5. Majority-class baseline
dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)
dummy_predictions = dummy_model.predict(X_test)

print("\nDummy Classifier")

print("Accuracy:")
print(
    accuracy_score(
        y_test,
        dummy_predictions,
    )
)

print("Recall:")
print(
    recall_score(
        y_test,
        dummy_predictions,
        zero_division=0,
    )
)


# 6. Normal Logistic Regression
normal_pipeline = Pipeline([
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

normal_pipeline.fit(
    X_train,
    y_train,
)

normal_predictions = normal_pipeline.predict(
    X_test
)


# 7. Balanced Logistic Regression
balanced_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        ),
    ),
])

balanced_pipeline.fit(
    X_train,
    y_train,
)

balanced_predictions = balanced_pipeline.predict(
    X_test
)


# 8. Compare normal and balanced models
def print_metrics(name, actual, predicted):
    print("\n" + name)

    print("Accuracy:")
    print(
        accuracy_score(
            actual,
            predicted,
        )
    )

    print("Precision:")
    print(
        precision_score(
            actual,
            predicted,
            zero_division=0,
        )
    )

    print("Recall:")
    print(
        recall_score(
            actual,
            predicted,
            zero_division=0,
        )
    )

    print("F1:")
    print(
        f1_score(
            actual,
            predicted,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            actual,
            predicted,
        )
    )


print_metrics(
    "Normal Logistic Regression",
    y_test,
    normal_predictions,
)

print_metrics(
    "Balanced Logistic Regression",
    y_test,
    balanced_predictions,
)


# 9. Get class-1 probabilities
probabilities = balanced_pipeline.predict_proba(
    X_test
)[:, 1]


# 10. Compare threshold 0.5 and 0.3
predictions_05 = (
    probabilities >= 0.5
).astype(int)

predictions_03 = (
    probabilities >= 0.3
).astype(int)

print_metrics(
    "Balanced model with threshold 0.5",
    y_test,
    predictions_05,
)

print_metrics(
    "Balanced model with threshold 0.3",
    y_test,
    predictions_03,
)


# 11. ROC and ROC-AUC
fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    probabilities,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

print("\nROC-AUC:")
print(roc_auc)


# 12. Precision-Recall curve and Average Precision
precision_values, recall_values, pr_thresholds = precision_recall_curve(
    y_test,
    probabilities,
)

average_precision = average_precision_score(
    y_test,
    probabilities,
)

print("\nAverage Precision:")
print(average_precision)


# 13. Choose a threshold for recall >= 0.80
target_recall = 0.80

best_threshold = None
best_precision = -1
best_recall = None

for i in range(len(pr_thresholds)):
    current_precision = precision_values[i]
    current_recall = recall_values[i]
    current_threshold = pr_thresholds[i]

    if (
        current_recall >= target_recall
        and current_precision > best_precision
    ):
        best_threshold = current_threshold
        best_precision = current_precision
        best_recall = current_recall


if best_threshold is not None:
    print("\nChosen threshold:")
    print(best_threshold)

    print("Precision at chosen threshold:")
    print(best_precision)

    print("Recall at chosen threshold:")
    print(best_recall)

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print_metrics(
        "Final predictions",
        y_test,
        final_predictions,
    )
else:
    print("\nNo threshold reached the target recall.")
