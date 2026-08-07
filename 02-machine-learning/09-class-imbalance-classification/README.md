# Class Imbalance Classification

This project studies **class imbalance** in a binary classification problem using employee attrition data.

The target is:

```text
left_company = 0
Employee stays

left_company = 1
Employee leaves
```

The dataset is intentionally imbalanced, so class `1` is much rarer than class `0`.

## Learning Objectives

After this project, you should understand:

- what class imbalance means
- why accuracy can be misleading
- how `DummyClassifier` creates a majority-class baseline
- what `class_weight="balanced"` changes during training
- how decision thresholds affect predictions
- the difference between False Positives and False Negatives
- ROC curve and ROC-AUC
- Precision-Recall curve
- Average Precision
- how to choose a threshold from a business requirement

# 1. Class Imbalance

Class imbalance means one target class contains many more observations than another.

Example:

```text
Class 0 = 90 observations
Class 1 = 10 observations
```

Class `0` is the **majority class** and class `1` is the **minority class**.

A classifier can obtain high accuracy simply by predicting the majority class most of the time. Accuracy alone can therefore be misleading.

Check the distribution with:

```python
print(y.value_counts())
print(y.value_counts(normalize=True))
```

# 2. DummyClassifier

With:

```python
DummyClassifier(strategy="most_frequent")
```

the classifier looks at `y_train`, finds the most frequent class, and predicts that same class for every test observation.

Example:

```text
y_train:
Class 0 = 73 observations
Class 1 = 27 observations
```

The most frequent class is `0`, so it predicts `0` for every row of `X_test`.

It does not use the feature values intelligently. It is only a baseline.

# 3. Logistic Regression Without Class Weighting

By default:

```python
LogisticRegression()
```

uses:

```text
class_weight = None
```

Conceptually, each class has weight `1`.

Example:

```text
Class 0 = 90 rows
Class 1 = 10 rows
```

Before balancing:

```text
Class 0 row weight = 1
Class 1 row weight = 1
```

So the total influence is roughly:

```text
Class 0 → 90 × 1 = 90
Class 1 → 10 × 1 = 10
```

The majority class can therefore dominate training.

# 4. class_weight="balanced"

Use:

```python
LogisticRegression(
    class_weight="balanced"
)
```

Scikit-learn calculates class weights from the training target frequencies.

The formula is:

\[
w_c = \frac{N}{K \times N_c}
\]

Where:

- \(N\) = total number of training observations
- \(K\) = number of classes
- \(N_c\) = number of observations in class \(c\)

Example:

```text
Class 0 = 90
Class 1 = 10
Total = 100
Number of classes = 2
```

For class `0`:

\[
w_0 = \frac{100}{2 \times 90} \approx 0.56
\]

For class `1`:

\[
w_1 = \frac{100}{2 \times 10} = 5
\]

So minority-class mistakes influence the training loss more strongly.

Important:

```text
class_weight="balanced"
does NOT duplicate rows
does NOT change X
does NOT change y
does NOT change the number of observations
```

It changes how strongly each class affects the training loss.

This can change:

```text
training loss
→ coefficients
→ intercept
→ probabilities
→ final predictions
```

# 5. Decision Threshold

Logistic Regression first predicts probabilities.

For binary classification:

```python
probabilities = model.predict_proba(X)[:, 1]
```

This extracts `P(class 1)`.

The default decision is approximately:

```text
P(class 1) < 0.5
→ class 0

P(class 1) >= 0.5
→ class 1
```

Example:

```text
P(class 1) = 0.32
threshold = 0.5
→ class 0
```

If the threshold becomes `0.3`:

```text
0.32 >= 0.3
→ class 1
```

Changing the threshold does not retrain the model. It only changes how probabilities become classes.

# 6. class_weight vs Threshold

These ideas act at different moments:

```text
class_weight="balanced"
→ changes training

decision threshold
→ changes the final probability-to-class decision
```

# 7. False Positive and False Negative

Suppose:

```text
Class 1 = employee leaves
Class 0 = employee stays
```

False Positive:

```text
Predicted = 1
Actual = 0
```

False Negative:

```text
Predicted = 0
Actual = 1
```

If False Negatives are more expensive, we normally care strongly about recall.

# 8. Precision

\[
Precision = \frac{TP}{TP + FP}
\]

Precision answers:

```text
Among all observations predicted as class 1,
how many were actually class 1?
```

High precision means fewer False Positives.

# 9. Recall

\[
Recall = \frac{TP}{TP + FN}
\]

Recall answers:

```text
Among all real class-1 observations,
how many did the model detect?
```

High recall means fewer False Negatives.

# 10. F1 Score

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

F1 summarizes the balance between precision and recall.

# 11. ROC Curve

The ROC curve evaluates many decision thresholds.

For each threshold:

\[
TPR = \frac{TP}{TP + FN}
\]

TPR is Recall.

And:

\[
FPR = \frac{FP}{FP + TN}
\]

In Python:

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities,
)
```

# 12. ROC-AUC

ROC-AUC means **Area Under the ROC Curve**.

Typical interpretation:

```text
1.0 → perfect separation
0.5 → similar to random ranking
```

Higher is better.

```python
roc_auc = roc_auc_score(
    y_test,
    probabilities,
)
```

# 13. Precision-Recall Curve

The Precision-Recall curve studies precision and recall across many thresholds.

It is especially useful when the positive class is rare.

```python
precision, recall, thresholds = precision_recall_curve(
    y_test,
    probabilities,
)
```

Lower thresholds normally increase recall but may reduce precision.

Higher thresholds normally increase precision but may reduce recall.

# 14. Average Precision

`average_precision_score` summarizes Precision-Recall performance into one score.

```python
average_precision = average_precision_score(
    y_test,
    probabilities,
)
```

Higher is better.

# 15. Choosing a Threshold

There is no universal rule saying the threshold must always be `0.5`.

Example business requirement:

```text
Recall must be at least 80%
```

Then we inspect the Precision-Recall results and choose a threshold satisfying:

```text
Recall >= 0.80
```

Among those thresholds, the code selects the one with the highest precision.

The workflow is:

```text
business requirement
→ choose target metric
→ inspect thresholds
→ select threshold
→ create final predictions
```

# Project Files

```text
09-class-imbalance-classification/
│
├── employee_attrition_imbalanced.csv
├── class_imbalance_classification.py
└── README.md
```

# Run

```bash
uv run python class_imbalance_classification.py
```

If required packages are missing:

```bash
uv add pandas scikit-learn
```
