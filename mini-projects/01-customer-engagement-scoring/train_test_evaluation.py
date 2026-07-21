import numpy as np


# ============================================================
# 1. COMPLETE DATASET
# ============================================================

# Columns:
# 0 -> Age
# 1 -> Monthly income
# 2 -> Previous purchases

X = np.array([
    [22, 3000, 2],
    [45, 7000, 8],
    [31, 5000, 6],
    [52, 4500, 3],
    [27, 9000, 12],
    [40, 6000, 7],
    [36, 3500, 4],
    [29, 8000, 10],
    [48, 10000, 9],
    [33, 5500, 5],
], dtype=float)

actual_scores = np.array([
    2.0,
    6.0,
    4.3,
    4.1,
    7.2,
    5.3,
    3.3,
    6.5,
    7.3,
    4.1,
], dtype=float)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

training_indices = np.array([
    0, 1, 2, 3, 4, 5, 6, 7
])

test_indices = np.array([
    8, 9
])

X_train = X[training_indices]
X_test = X[test_indices]

y_train = actual_scores[training_indices]
y_test = actual_scores[test_indices]


# ============================================================
# 3. STANDARDIZATION
# ============================================================

# Only the training data is used to calculate the means
# and standard deviations.

training_means = np.mean(
    X_train,
    axis=0,
)

training_standard_deviations = np.std(
    X_train,
    axis=0,
)

if np.any(training_standard_deviations == 0):
    raise ValueError(
        "A feature with zero standard deviation "
        "cannot be standardized."
    )

X_train_scaled = (
    X_train - training_means
) / training_standard_deviations

# The test data uses the training statistics.
# We do not calculate new test means or test deviations.

X_test_scaled = (
    X_test - training_means
) / training_standard_deviations


# ============================================================
# 4. TRAINING DESIGN MATRIX
# ============================================================

training_bias_column = np.ones(
    (X_train_scaled.shape[0], 1)
)

A_train = np.column_stack((
    X_train_scaled,
    training_bias_column,
))


# ============================================================
# 5. LEARN MODEL PARAMETERS
# ============================================================

# Parameter order:
# [
#     age_weight,
#     income_weight,
#     purchases_weight,
#     bias,
# ]

parameters = (
    np.linalg.pinv(A_train)
    @ y_train
)

weights = parameters[:-1]
bias = parameters[-1]

age_weight = weights[0]
income_weight = weights[1]
purchases_weight = weights[2]


# ============================================================
# 6. TRAINING PREDICTIONS
# ============================================================

training_predictions = A_train @ parameters


# ============================================================
# 7. TEST DESIGN MATRIX
# ============================================================

test_bias_column = np.ones(
    (X_test_scaled.shape[0], 1)
)

A_test = np.column_stack((
    X_test_scaled,
    test_bias_column,
))


# ============================================================
# 8. TEST PREDICTIONS
# ============================================================

test_predictions = A_test @ parameters


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

training_residuals = (
    y_train - training_predictions
)

test_residuals = (
    y_test - test_predictions
)

training_mse = np.mean(
    training_residuals ** 2
)

test_mse = np.mean(
    test_residuals ** 2
)

training_rmse = np.sqrt(
    training_mse
)

test_rmse = np.sqrt(
    test_mse
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

np.set_printoptions(
    precision=4,
    suppress=True,
)

print("Training feature means:")
print(training_means)

print("\nTraining feature standard deviations:")
print(training_standard_deviations)

print("\nLearned parameters:")
print(parameters)

print("\nLearned model:")
print(
    f"score = {age_weight:.4f} * standardized_age "
    f"+ {income_weight:.4f} * standardized_income "
    f"+ {purchases_weight:.4f} * standardized_purchases "
    f"+ {bias:.4f}"
)

print("\nTraining results:")
print("Customer | Actual | Predicted | Residual")

for index in range(len(y_train)):
    print(
        f"{index + 1:>8} | "
        f"{y_train[index]:>6.3f} | "
        f"{training_predictions[index]:>9.3f} | "
        f"{training_residuals[index]:>8.3f}"
    )

print("\nTest results:")
print("Customer | Actual | Predicted | Residual")

for index in range(len(y_test)):
    original_customer_number = (
        test_indices[index] + 1
    )

    print(
        f"{original_customer_number:>8} | "
        f"{y_test[index]:>6.3f} | "
        f"{test_predictions[index]:>9.3f} | "
        f"{test_residuals[index]:>8.3f}"
    )

print("\nTraining MSE:")
print(training_mse)

print("\nTraining RMSE:")
print(training_rmse)

print("\nTest MSE:")
print(test_mse)

print("\nTest RMSE:")
print(test_rmse)