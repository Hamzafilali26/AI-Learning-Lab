from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    csv_path = Path(__file__).parent / "support_escalation.csv"
    data = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)
else:
    print(data.head())

mission_values = data.isna().sum().sum()
duplicate_values = data.duplicated().sum()
print("mission values :", mission_values)
print("duplicate values : ", duplicate_values)

numerical_features = [
    "waiting_time_minutes",
    "previous_contacts",
    "customer_tenure_months",
    "satisfaction_score",
]
categorical_features = ["channel", "plan_type", "issue_type"]
target = "escalated"

X = data[numerical_features + categorical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))]
)


model_pipeline.fit(X_train, y_train)

test_predictions = model_pipeline.predict(X_test)

escalation_probabilities = model_pipeline.predict_proba(X_test)[
    :, 1
]  # probability that the ticket is escalated, meaning class 1.

print(escalation_probabilities)


accuracy = accuracy_score(
    y_test,
    test_predictions,
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
)

print("Accuracy:")
print(accuracy)

print("Precision:")
print(precision)

print("Recall:")
print(recall)

print("F1 score:")
print(f1)

print("Confusion matrix:")
print(matrix)


results = X_test.copy()

results["actual"] = y_test.to_numpy()
results["predicted"] = test_predictions
results["escalation_probability"] = escalation_probabilities

print(results)


trained_preprocessor = model_pipeline.named_steps["preprocessor"]
trained_model = model_pipeline.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()

coefficients = pd.Series(
    trained_model.coef_[0],
    index=feature_names,
)

print("Feature coefficients:")
print(coefficients.sort_values(ascending=False))
bias = trained_model.intercept_[0]

print("Bias:")
print(bias)

