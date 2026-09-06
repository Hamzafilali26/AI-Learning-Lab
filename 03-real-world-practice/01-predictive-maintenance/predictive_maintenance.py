import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

try:
    path = Path(__file__).parent / "data/ai4i2020.csv"
    df = pd.read_csv(path)
except Exception as e:
    print(e)
    exit()

# print(df.head())


# not necessary data : UDI, Product ID
# INPUT(X) : Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
# OUTPUT / TARGET(Y) : Machine failure

# ? not necessary data : TWF HDF PWF OSF RNF (but I should to know why)

# # Critical rule
# * Do NOT use the failure-mode target columns TWF, HDF, PWF, OSF or RNF as input features when predicting Machine failure. Machine failure is determined
# *     from those failure modes, so using them would give the model information derived directly from the target.
# ? let's make a test to see if it's true

failure_mode_mask = (
    (df["TWF"] == 1)
    | (df["HDF"] == 1)
    | (df["PWF"] == 1)
    | (df["OSF"] == 1)
    | (df["RNF"] == 1)
)
tst_result = df[failure_mode_mask & (df["Machine failure"] == 1)]
unexplained_failures = df[(df["Machine failure"] == 1) & (~failure_mode_mask)]

# print("Machine failures: ", end="")
# print((df["Machine failure"] == 1).sum())
# print("failure-mode columns: ", end="")
# print(len(tst_result))
# print("failure count without failure-mode columns: ", end="")
# print(len(unexplained_failures))

# TODO : Conclusion

# % Conclusion
# * After counting the machine failures, we found 339 failures in total.
# * With the failure-mode columns ["TWF", "HDF", "PWF", "OSF", "RNF"] detected 330 of them.
# * This means there are 9 machine failures that were not identified by any of the failure-mode columns ["TWF", "HDF", "PWF", "OSF", "RNF"].

selected_data = df[
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]
].copy()

# * changing fields
selected_data.columns = [
    "type",
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
]

# print(selected_data.head())

# @ Step 1: Separate between the categorical and numerical features
numerical_features = [
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
]
categorical_features = ["type"]
target = "machine_failure"

X = selected_data[numerical_features + categorical_features]
y = selected_data[target]

# @ Step 2: Separate training data and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# @ Step 3: Numerical and categorical preprocessing
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


pipeline = Pipeline(
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

pipeline.fit(
    X_train,
    y_train,
)


test_predictions = pipeline.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)


dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(
    X_train,
    y_train,
)

dummy_predictions = dummy_model.predict(X_test)

dummy_accuracy = accuracy_score(
    y_test,
    dummy_predictions,
)

print("Dummy accuracy: ", end="")
print(dummy_accuracy)

print("Logistic Regression accuracy: ", end="")
print(test_accuracy)

# * DummyClassifier
# @  → "stupid" baseline
# * Real model
# @  → must beat that baseline meaningfully

# * This shows that the real model is only marginally better than the majority-class baseline when using accuracy alone. But, accuracy 
# *      is not sufficient to evaluate the model, and we need to inspect recall, precision, F1-score, and the confusion matrix.


# TODO : # the columns that have the impact in the failer ??
