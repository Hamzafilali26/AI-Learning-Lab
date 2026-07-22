# Customer Churn Classification

## Goal

Predict whether a customer will leave a service.

The target column is:

```text
churned
```

Its values are:

```text
0 = customer stays
1 = customer churns
```

## Why This Project Matters

Regression predicts a numerical value.

Classification predicts a category.

Customer churn is a binary classification problem because the model chooses between two classes:

```text
stay
churn
```

## Learning Objectives

After this project, you should be able to:

- distinguish regression from classification;
- explain what Logistic Regression does;
- understand how probabilities are converted into classes;
- train a binary classifier;
- obtain predicted classes;
- obtain class probabilities;
- interpret a confusion matrix;
- calculate accuracy, precision, recall, and F1 score;
- preprocess numerical and categorical features in one pipeline.

# Main Concepts

## Binary Classification

Binary classification predicts one of two possible classes.

In this project:

```text
0 = customer stays
1 = customer churns
```

The model does not directly predict a continuous score such as 5.7 or 8.2.

It predicts the probability that a customer belongs to class `1`.

---

## Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used mainly for binary classification.

Despite the word `Regression` in its name, it is used to predict categories.

In this project, Logistic Regression estimates:

```text
the probability that a customer will churn
```

The predicted probability is always between:

```text
0 and 1
```

Example:

```text
0.82
```

means:

```text
the model estimates an 82% probability that the customer will churn
```

---

## Linear Combination

Before calculating the probability, Logistic Regression calculates a linear score:

\[
z = b + w_1x_1 + w_2x_2 + \cdots + w_nx_n
\]

Where:

- \(z\) is the linear score;
- \(b\) is the bias or intercept;
- \(w_i\) is the learned coefficient of feature \(i\);
- \(x_i\) is the value of feature \(i\);
- \(n\) is the number of input features.

For this project, the equation can contain features such as:

```text
age
monthly_income
previous_purchases
months_inactive
support_tickets
membership_type
```

The linear score \(z\) can be any real number:

```text
negative
zero
positive
```

However, a probability must stay between 0 and 1.

For that reason, Logistic Regression applies the sigmoid function.

---

## Sigmoid Function

The sigmoid function converts the linear score \(z\) into a probability.

\[
P(y=1 \mid X) = rac{1}{1 + e^{-z}}
\]

Where:

- \(P(y=1 \mid X)\) is the probability that the customer belongs to class `1`;
- \(X\) represents all input features;
- \(e\) is Euler's number, approximately `2.718`;
- \(z\) is the linear combination calculated by the model.

Examples:

```text
z = -4  → probability close to 0
z = 0   → probability = 0.5
z = 4   → probability close to 1
```

The sigmoid function therefore transforms any real number into a value between 0 and 1.

---

## Predicted Probability

The method:

```python
pipeline.predict_proba(X_test)
```

returns the estimated probability of every class.

For binary classification, each row contains two probabilities:

```text
[probability of class 0, probability of class 1]
```

Example:

```text
[0.18, 0.82]
```

This means:

```text
18% probability that the customer stays
82% probability that the customer churns
```

In the code:

```python
pipeline.predict_proba(X_test)[:, 1]
```

`[:, 1]` selects only the probability of class `1`.

In this project:

```text
class 1 = churn
```

---

## Decision Threshold

A probability must be converted into a final class.

The default threshold is usually:

```text
0.5
```

The rule is:

```text
probability below 0.5     → class 0
probability at least 0.5  → class 1
```

Example:

```text
churn probability = 0.27 → predicted class = 0
churn probability = 0.74 → predicted class = 1
```

The method:

```python
pipeline.predict(X_test)
```

returns the final predicted classes.

The method:

```python
pipeline.predict_proba(X_test)
```

returns the probabilities.

---

## Logistic Regression Coefficients

Logistic Regression learns one coefficient for each transformed feature.

A positive coefficient increases the linear score \(z\).

This usually increases the predicted probability of class `1`.

A negative coefficient decreases the linear score \(z\).

This usually decreases the predicted probability of class `1`.

In this project:

```text
positive coefficient
→ tends to increase churn probability

negative coefficient
→ tends to decrease churn probability
```

The exact effect depends on the values of all other features.

The coefficients are available with:

```python
trained_model.coef_
```

The bias is available with:

```python
trained_model.intercept_
```

Because numerical features are standardized, their coefficients are easier to compare than coefficients learned from features with completely different scales.

---

## Stratified Train-Test Split

The split uses:

```python
stratify=y
```

This preserves approximately the same proportion of class `0` and class `1` in both:

```text
training data
test data
```

This is important because a random split could accidentally create a training or test set with an unrealistic class distribution.

---

## Confusion Matrix

A confusion matrix compares actual classes with predicted classes.

For binary classification:

```text
[[true negatives, false positives],
 [false negatives, true positives]]
```

### True Negative

The customer stays, and the model predicts that the customer stays.

```text
actual = 0
predicted = 0
```

### False Positive

The customer stays, but the model predicts churn.

```text
actual = 0
predicted = 1
```

### False Negative

The customer churns, but the model predicts that the customer stays.

```text
actual = 1
predicted = 0
```

### True Positive

The customer churns, and the model correctly predicts churn.

```text
actual = 1
predicted = 1
```

---

# Classification Metrics

## Accuracy

Accuracy measures the proportion of all predictions that are correct.

\[
Accuracy =
rac{TP + TN}
{TP + TN + FP + FN}
\]

Where:

- \(TP\) is the number of true positives;
- \(TN\) is the number of true negatives;
- \(FP\) is the number of false positives;
- \(FN\) is the number of false negatives.

Accuracy can be misleading when one class is much more frequent than the other.

---

## Precision

Precision answers:

```text
Among customers predicted to churn,
how many actually churned?
```

\[
Precision =
rac{TP}
{TP + FP}
\]

High precision means the model produces few false churn alerts.

---

## Recall

Recall answers:

```text
Among customers who actually churned,
how many did the model detect?
```

\[
Recall =
rac{TP}
{TP + FN}
\]

High recall means the model misses few customers who actually churn.

---

## F1 Score

F1 combines precision and recall.

\[
F1 =
2 	imes
rac{Precision 	imes Recall}
{Precision + Recall}
\]

A high F1 score requires both precision and recall to be reasonably high.

---

# Data Preprocessing

## Numerical Features

The numerical pipeline performs:

```text
missing-value imputation
→ standardization
```

It uses:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

## Categorical Features

The categorical pipeline performs:

```text
missing-value imputation
→ one-hot encoding
```

It uses:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
)
```

## ColumnTransformer

`ColumnTransformer` applies different preprocessing steps to different groups of columns.

```text
numerical columns
→ numerical pipeline

categorical columns
→ categorical pipeline
```

It then combines all transformed columns into one numerical matrix.

## Complete Pipeline

The final workflow is:

```text
Raw CSV data
→ Train-test split
→ Numerical preprocessing
→ Categorical preprocessing
→ Logistic Regression
→ Class predictions
→ Probability predictions
→ Classification evaluation
```

# Project Structure

```text
06-customer-churn-classification/
├── customer_churn.csv
├── customer_churn_classification.py
└── README.md
```

# Installation

```bash
uv add pandas scikit-learn
```

# Run the Project

```bash
uv run python customer_churn_classification.py
```

# Important Limitation

The dataset is small and synthetic.

The project teaches the classification workflow, but its metrics must not be treated as evidence of real-world performance.
