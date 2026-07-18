import numpy as np


# ============================================================
# 1. HISTORICAL DATA
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

# Real engagement scores observed for the customers
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
# 2. DATA VALIDATION
# ============================================================

if X.shape[0] != actual_scores.shape[0]:
    raise ValueError(
        "X and actual_scores must contain the same number of customers."
    )

if np.any(~np.isfinite(X)):
    raise ValueError("X contains invalid values.")

if np.any(~np.isfinite(actual_scores)):
    raise ValueError("actual_scores contains invalid values.")


# ============================================================
# 3. FEATURE STANDARDIZATION
# ============================================================

# Formula:
# z = (x - mean) / standard_deviation

feature_means = np.mean(X, axis=0)
feature_standard_deviations = np.std(X, axis=0)

if np.any(feature_standard_deviations == 0):
    raise ValueError(
        "A feature with zero standard deviation cannot be standardized."
    )

X_scaled = (
    X - feature_means
) / feature_standard_deviations


# ============================================================
# 4. DESIGN MATRIX
# ============================================================

# The model is:
#
# predicted_score =
#     age_weight * standardized_age
#     + income_weight * standardized_income
#     + purchases_weight * standardized_purchases
#     + bias
#
# The last column contains ones because it multiplies the bias.

bias_column = np.ones(
    (X_scaled.shape[0], 1)
)

design_matrix = np.column_stack((
    X_scaled,
    bias_column,
))


# ============================================================
# 5. LEARN WEIGHTS AND BIAS
# ============================================================

# parameters = pseudoinverse(design_matrix) @ actual_scores
#
# Parameter order:
# [age_weight, income_weight, purchases_weight, bias]

parameters = (
    np.linalg.pinv(design_matrix)
    @ actual_scores
)

weights = parameters[:-1]
bias = parameters[-1]

age_weight = weights[0]
income_weight = weights[1]
purchases_weight = weights[2]


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

predicted_scores = design_matrix @ parameters


# ============================================================
# 7. CALCULATE ERRORS
# ============================================================

residuals = actual_scores - predicted_scores
squared_errors = residuals ** 2

mse = np.mean(squared_errors)
rmse = np.sqrt(mse)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

np.set_printoptions(
    precision=4,
    suppress=True,
)

print("Feature means:")
print(feature_means)

print("\nFeature standard deviations:")
print(feature_standard_deviations)

print("\nLearned weights:")
print(f"Age weight:       {age_weight:.4f}")
print(f"Income weight:    {income_weight:.4f}")
print(f"Purchases weight: {purchases_weight:.4f}")

print(f"\nLearned bias: {bias:.4f}")

print("\nLearned model:")
print(
    f"score = {age_weight:.4f} * standardized_age "
    f"+ {income_weight:.4f} * standardized_income "
    f"+ {purchases_weight:.4f} * standardized_purchases "
    f"+ {bias:.4f}"
)

print("\nCustomer results:")
print("Customer | Actual | Predicted | Residual")

number_of_customers = len(actual_scores)

for index in range(number_of_customers):
    customer_number = index + 1
    actual_score = actual_scores[index]
    predicted_score = predicted_scores[index]
    residual = residuals[index]

    print(
        f"{customer_number:>8} | "
        f"{actual_score:>6.3f} | "
        f"{predicted_score:>9.3f} | "
        f"{residual:>8.3f}"
    )

print(f"\nMSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")