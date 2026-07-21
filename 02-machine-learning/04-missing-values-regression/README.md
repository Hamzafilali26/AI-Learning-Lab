# Missing Values Regression

## Goal

Train a Linear Regression model when the CSV dataset contains:

- missing numerical values;
- duplicate rows.

The target is the customer engagement score.

## New Concepts

### Missing Value

A missing value means that a piece of information was not recorded.

In this dataset, some customers have a missing:

- age;
- monthly income;
- number of previous purchases.

### Duplicate Row

A duplicate row is a repeated observation.

Duplicate rows are removed before training so the same customer record does not influence the model more than once.

### Imputation

Imputation means replacing missing values with calculated replacement values.

This project uses the median of each training feature.

### Median

The median is the middle value after sorting the observations.

The pipeline learns one median for:

- age;
- monthly income;
- previous purchases.

### SimpleImputer

`SimpleImputer` replaces missing values.

The strategy used in this project is:

```text
median
```

### Pipeline Order

The pipeline executes these steps:

```text
Missing-value imputation
→ Feature standardization
→ Linear Regression
```

The preprocessing steps are learned only from the training data.

## Project Structure

```text
04-missing-values-regression/
├── customer_engagement_missing.csv
├── customer_engagement_missing_values.py
└── README.md
```

## Installation

```bash
uv add pandas scikit-learn
```

## Run the Project

```bash
uv run python customer_engagement_missing_values.py
```

## What the Program Displays

- missing values before cleaning;
- duplicate-row count;
- rows remaining after duplicate removal;
- medians learned from the training data;
- learned feature weights;
- learned bias;
- test predictions;
- residuals;
- MAE, MSE, RMSE, and R².

## Important Limitation

The dataset is intentionally small and synthetic.

It is useful for learning the workflow, but it is not suitable for a real business model.
