from pathlib import Path

import pandas as pd


csv_path = Path(__file__).parent / "employee_attrition.csv"
data = pd.read_csv(csv_path)


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
