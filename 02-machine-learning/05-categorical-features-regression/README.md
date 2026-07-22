# Categorical Features Regression

## Goal

Predict a customer engagement score using both numerical and categorical features.

The input features are:

- age;
- monthly income;
- previous purchases;
- membership type.

## New Concepts

### Categorical Feature

A categorical feature contains labels or groups instead of measurable numerical quantities.

In this project:

```text
membership_type
```

contains:

```text
Basic
Premium
VIP
```

The category names cannot be used directly by Linear Regression because the model requires numerical inputs.

### One-Hot Encoding

One-hot encoding converts each category into a numerical binary column.

For example:

```text
membership_type = VIP
```

can become:

```text
membership_type_Premium = 0
membership_type_VIP = 1
```

### Reference Category

The encoder uses:

```text
drop="first"
```

The first category is removed and becomes the reference category.

In this dataset, `Basic` is the reference category.

The learned Premium and VIP coefficients describe their difference from Basic, while the other features remain unchanged.

### Unknown Category

The encoder uses:

```text
handle_unknown="ignore"
```

This prevents prediction from failing when new data contains a category that was not present during training.

### ColumnTransformer

`ColumnTransformer` applies different preprocessing rules to different columns.

In this project:

```text
Numerical columns
→ missing-value imputation
→ standardization

Categorical columns
→ missing-value imputation
→ one-hot encoding
```

### Complete Pipeline

The final workflow is:

```text
Raw data
→ ColumnTransformer
→ Linear Regression
→ Predictions
```

## Project Structure

```text
05-categorical-features-regression/
├── customer_engagement_categorical.csv
├── customer_engagement_categorical.py
└── README.md
```

## Installation

```bash
uv add pandas scikit-learn
```

## Run the Project

```bash
uv run python customer_engagement_categorical.py
```

## What the Program Displays

- the dataset;
- feature data types;
- transformed feature names;
- learned feature weights;
- learned bias;
- actual and predicted test scores;
- residuals;
- MAE, MSE, RMSE, and R².

## Important Limitation

The dataset is small and synthetic.

It is intended to teach categorical preprocessing, not to produce a reliable business model.
