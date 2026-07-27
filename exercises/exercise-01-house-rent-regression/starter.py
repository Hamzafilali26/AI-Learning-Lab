import pandas as pd
from pathlib import Path
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Inspect the dataset
try:
    csv_path = Path(__file__).parent / "house_rent.csv"
    data = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)
else:
    print(data.head())

# Step 2: Separate features and target
numerical_features = ["area_m2","bedrooms", "floor", "distance_center_km"]
categorical_features = ["furnished","city_zone"]
target = "monthly_rent"

X = data[numerical_features + categorical_features]
y = data[target]

# Step 3: Split training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Step 4: Numerical pipeline
    # replaces missing numerical values
    # → standardizes numerical features

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Step 5 : Pipeline catégorielle
    # replaces missing categories
    # → converts categories into numerical columns
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        drop='first',
        handle_unknown='ignore'))
])

# Step 6: Combine both preprocessing pipelines

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Step 7: Create the complete pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Step 8: Train the pipeline
    # This trains the preprocessing steps and the Linear Regression model using the training data.

model_pipeline.fit(X_train, y_train)

# Step 9: Make predictions
test_predictions = model_pipeline.predict(X_test)

results = pd.DataFrame({
    "actual": y_test,
    "predicted": test_predictions
})

print(results.head())
results["residual"] = results["actual"] - results["predicted"]
print(f"Mean Absolute Error: {results['residual'].abs().mean()}")

# Interpretation: the mean error is 238, which mean that the predicted monthly rent differs from the real monthly rent by about 238 rent units.

# Step 10: Calculate the evaluation metrics

mae = mean_absolute_error(y_test, test_predictions)

mse = mean_squared_error(y_test, test_predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, test_predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")  

# Mean Absolute Error (MAE): 238.69382724167122
# Interpretation: the mean error is 238, which mean that the predicted monthly rent differs from the real monthly rent by about 238 rent units.

# Mean Squared Error (MSE): 97671.55520525062
# Root Mean Squared Error (RMSE): 312.524487368991
# Interpretation: the model prediction error is typically around 312 rents units, while giving more importance to large errors.


# R-squared (R2): 0.94212055987837
    # Interpretation: the model explains 94.21% of the variance in the target variable.
    # the model explains approximately 94.21% of the variance in the monthly rent, indicating a strong fit to the data.
    # So about 5.79% of the rent variation is not explained by the model. Human housing prices still retain a little chaos, naturally.

# Step 11: Add residuals to the results table
print(results)
# positive residual → prediction is too low
# negative residual → prediction is too high
# zero residual     → perfect prediction


# Step 12: Display coefficients and bias

trained_preprocessor = model_pipeline.named_steps["preprocessor"]
trained_model = model_pipeline.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()

coefficients = pd.Series(
    trained_model.coef_,
    index=feature_names,
) 

print("Feature coefficients:")
print(coefficients)

print("Bias:")
print(trained_model.intercept_)

# positive coefficient → increases predicted rent
# negative coefficient → decreases predicted rent
# larger absolute value → stronger effect











































# TODO 1: Inspect the dataset.
# TODO 2: Define numerical features, categorical features, and target.
# TODO 3: Create a train-test split.
# TODO 4: Build the numerical preprocessing pipeline.
# TODO 5: Build the categorical preprocessing pipeline.
# TODO 6: Build a ColumnTransformer.
# TODO 7: Build a Pipeline with LinearRegression.
# TODO 8: Train the model.
# TODO 9: Predict the test values.
# TODO 10: Calculate MAE, MSE, RMSE, and R2.
# TODO 11: Display actual values, predictions, and residuals.
# TODO 12: Display transformed feature names, coefficients, and bias.
