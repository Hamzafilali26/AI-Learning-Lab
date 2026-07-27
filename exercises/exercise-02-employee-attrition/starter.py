from pathlib import Path
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    csv_path = Path(__file__).parent / "employee_attrition.csv"
    data = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)
else:
    print(data.head())


# print("First rows:")
# print(data.head())

# print("Shape:")
# print(data.shape)

# print("Data types:")
# print(data.dtypes)

print("Missing values:")
print(data.isna().sum().sum())

print("Duplicate rows:")
print(data.duplicated().sum())


numerical_features = ["age","monthly_salary","years_at_company","satisfaction_score","training_hours"]
categorical_features = ["overtime","department"]
target = "left_company"
X = data[numerical_features + categorical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
    ,
    stratify=y,
)

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        drop='first',
        handle_unknown='ignore'))
])  

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression()) # we used LogisticRegression because the target variable is binary (left_company: 0 or 1) ==> categorical
])


model_pipeline.fit(X_train, y_train)

test_predictions = model_pipeline.predict(X_test)

leave_probabilities = (
    model_pipeline.predict_proba(X_test)[:, 1]
)

print('leave_probabilities:', leave_probabilities)



accuracy = accuracy_score(
    y_test,
    test_predictions,
)

precision = precision_score(
    y_test,
    test_predictions,
)

recall = recall_score(
    y_test,
    test_predictions,
)

f1 = f1_score(
    y_test,
    test_predictions,
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
)

print("Accuracy:")
print(accuracy)
# accuracy = 1.0 ==> The model correctly classified 80% of all employees.

print("Precision:")
print(precision)
# precision = 1.0 ==> Among the employees predicted to leave, 100% actually left.

print("Recall:")
print(recall)
# recall = 1.0 ==> The model detected 100% of the employees who actually left.

print("F1 score:")
print(f1)
# f1 = 1.0 ==> The model has a balanced score of 100% between detecting employees who leave and avoiding false alerts.

print("Confusion matrix:")
print(matrix)
# Interpretation:

# TN: employee stayed, predicted stay. 3
# FP: employee stayed, predicted leave.
# FN: employee left, predicted stay.
# TP: employee left, predicted leave. 2



results = X_test.copy()

results["actual"] = y_test.to_numpy()
results["predicted"] = test_predictions
results["leave_probability"] = leave_probabilities

print(results)



trained_preprocessor = model_pipeline.named_steps["preprocessor"]
trained_model = model_pipeline.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()

coefficients = pd.Series(
    trained_model.coef_[0],
    index=feature_names,
)

print("Feature coefficients:")
print(coefficients)

bias = trained_model.intercept_[0]

print("Bias:")
print(bias)


# TODO 1: Inspect the dataset.
# TODO 2: Define features and target.
# TODO 3: Create a stratified train-test split.
# TODO 4: Build numerical and categorical pipelines.
# TODO 5: Build a ColumnTransformer.
# TODO 6: Build a Pipeline with LogisticRegression.
# TODO 7: Train the classifier.
# TODO 8: Predict classes and class-1 probabilities.
# TODO 9: Calculate accuracy, precision, recall, F1, and confusion matrix.
# TODO 10: Create a readable result table.
# TODO 11: Display transformed feature names and coefficients.
