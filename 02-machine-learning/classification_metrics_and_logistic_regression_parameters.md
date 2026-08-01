# Classification Metrics and Logistic Regression Parameters

This document explains the main classification metrics and Logistic Regression parameters independently from any exercise.

# 1. Confusion Matrix

## Definition

A confusion matrix compares the actual classes with the predicted classes.

For binary classification:

```text
[[TN, FP],
 [FN, TP]]
```

Where:

- `TN` = True Negative
- `FP` = False Positive
- `FN` = False Negative
- `TP` = True Positive

## Example

Suppose the model produces:

```text
[[50, 10],
 [ 5, 35]]
```

Interpretation:

```text
TN = 50
FP = 10
FN = 5
TP = 35
```

Meaning:

- 50 negative cases were correctly predicted as negative.
- 10 negative cases were incorrectly predicted as positive.
- 5 positive cases were incorrectly predicted as negative.
- 35 positive cases were correctly predicted as positive.

## Why It Matters

The confusion matrix shows exactly what kind of mistakes the model makes.

The other metrics are calculated from these four values.

---

# 2. Accuracy

## Definition

Accuracy measures the proportion of all predictions that are correct.

\[
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}
\]

## Example

Using:

```text
TN = 50
FP = 10
FN = 5
TP = 35
```

\[
Accuracy =
\frac{35 + 50}
{35 + 50 + 10 + 5}
=
\frac{85}{100}
=
0.85
\]

Interpretation:

> The model correctly classified 85% of all observations.

## When Accuracy Is Useful

Accuracy is useful when the classes are reasonably balanced.

## Limitation

Accuracy can be misleading when one class is much more common than the other.

Example:

```text
95 negative cases
5 positive cases
```

A model that always predicts negative obtains:

```text
95% accuracy
```

but detects none of the positive cases.

---

# 3. Precision

## Definition

Precision measures how many predicted positive cases were actually positive.

\[
Precision =
\frac{TP}
{TP + FP}
\]

Precision answers:

> Among all observations predicted as class 1, how many were truly class 1?

## Example

\[
Precision =
\frac{35}
{35 + 10}
=
\frac{35}{45}
\approx
0.778
\]

Interpretation:

> Among the observations predicted as positive, about 77.8% were actually positive.

## When Precision Is Important

Precision matters when false positives are costly.

Example:

A model predicts that an employee will leave.

A false positive means HR may intervene even though the employee was not going to leave.

High precision means fewer false alarms.

---

# 4. Recall

## Definition

Recall measures how many actual positive cases the model detected.

\[
Recall =
\frac{TP}
{TP + FN}
\]

Recall answers:

> Among all observations that were truly class 1, how many did the model detect?

## Example

\[
Recall =
\frac{35}
{35 + 5}
=
\frac{35}{40}
=
0.875
\]

Interpretation:

> The model detected 87.5% of all actual positive cases.

## When Recall Is Important

Recall matters when false negatives are costly.

Example:

A support ticket is actually escalated.

A false negative means the model predicts that it will not be escalated.

High recall means the model misses fewer important positive cases.

---

# 5. F1 Score

## Definition

F1 combines precision and recall into one score.

\[
F1 =
2
\times
\frac{Precision \times Recall}
{Precision + Recall}
\]

It is the harmonic mean of precision and recall.

## Example

Using:

```text
Precision = 0.778
Recall = 0.875
```

\[
F1 =
2
\times
\frac{0.778 \times 0.875}
{0.778 + 0.875}
\approx
0.824
\]

Interpretation:

> The model obtained an F1 score of about 82.4%, showing a reasonably strong balance between precision and recall.

## When F1 Is Useful

F1 is useful when:

- both false positives and false negatives matter;
- the classes are imbalanced;
- accuracy alone is not sufficient.

## Limitation

F1 does not include true negatives directly.

Therefore, it should not be the only metric used in every situation.

---

# 6. Relationship Between the Metrics

All four metrics come from the confusion matrix.

```text
Confusion matrix
→ raw prediction counts

Accuracy
→ overall correctness

Precision
→ quality of positive predictions

Recall
→ ability to detect actual positives

F1
→ balance between precision and recall
```

The most important metric depends on the business problem.

```text
False positives costly
→ focus on precision

False negatives costly
→ focus on recall

Both important
→ focus on F1

Balanced classes and equal mistake costs
→ accuracy can be useful
```

---

# 7. Logistic Regression Coefficients

## Definition

A Logistic Regression coefficient shows how one feature changes the model's linear score.

The model first calculates:

\[
z =
b +
w_1x_1 +
w_2x_2 +
\cdots +
w_nx_n
\]

Where:

- \(z\) = linear score
- \(b\) = bias
- \(w_i\) = coefficient of feature \(i\)
- \(x_i\) = value of feature \(i\)

The sigmoid function then converts \(z\) into a probability:

\[
P(y=1 \mid X)
=
\frac{1}
{1 + e^{-z}}
\]

## Positive Coefficient

A positive coefficient increases \(z\).

This usually increases the probability of class 1.

Example:

```text
overtime_Yes coefficient = 1.20
```

Interpretation:

> Working overtime tends to increase the probability of leaving, while the other features remain unchanged.

## Negative Coefficient

A negative coefficient decreases \(z\).

This usually decreases the probability of class 1.

Example:

```text
satisfaction_score coefficient = -0.80
```

Interpretation:

> Higher satisfaction tends to reduce the probability of leaving, while the other features remain unchanged.

## Coefficient Magnitude

A larger absolute coefficient usually means a stronger effect on the linear score.

Example:

```text
Feature A coefficient = 1.50
Feature B coefficient = 0.20
```

Feature A has a stronger influence on the model score than Feature B.

## Important Warning

A Logistic Regression coefficient does not represent a direct probability change.

Example:

```text
coefficient = 1.20
```

does not mean:

```text
probability increases by 120%
```

The coefficient changes the log-odds, and the final probability also depends on all other features and the bias.

---

# 8. Odds Ratio

## Definition

The odds ratio makes a Logistic Regression coefficient easier to interpret.

\[
Odds\ Ratio =
e^{coefficient}
\]

## Example

If:

```text
coefficient = 1.20
```

then:

\[
e^{1.20}
\approx
3.32
\]

Interpretation:

> A one-unit increase in the feature multiplies the odds of class 1 by about 3.32, while the other features remain unchanged.

If:

```text
coefficient = -0.80
```

then:

\[
e^{-0.80}
\approx
0.45
\]

Interpretation:

> A one-unit increase in the feature multiplies the odds of class 1 by about 0.45, which means the odds decrease.

---

# 9. Bias

## Definition

The bias is the model intercept.

It is the value of the linear score when all feature values equal zero.

\[
z = b
\]

In scikit-learn:

```python
bias = trained_model.intercept_[0]
```

## Example

Suppose:

```text
bias = -0.30
```

Then, when all feature values equal zero:

\[
z = -0.30
\]

The probability is:

\[
P(y=1)
=
\frac{1}
{1 + e^{0.30}}
\approx
0.426
\]

Interpretation:

> When all transformed feature values equal zero, the baseline probability of class 1 is about 42.6%.

## With Standardized Features

When numerical features are standardized:

```text
feature value = 0
```

means the original value is approximately equal to the training mean.

## With One-Hot Encoded Features

When all one-hot encoded categorical values equal zero, the observation belongs to the reference categories removed by:

```python
drop="first"
```

Therefore, the bias usually represents the baseline score for:

- numerical features at their training means;
- categorical features at their reference categories.

---

# 10. Complete Numerical Example

Suppose the model learned:

```text
bias = -0.30

overtime_Yes coefficient = 1.20
satisfaction_scaled coefficient = -0.80
tenure_scaled coefficient = -0.40
```

For one employee:

```text
overtime_Yes = 1
satisfaction_scaled = -1
tenure_scaled = 0.5
```

The linear score is:

\[
z =
-0.30
+
(1.20 \times 1)
+
(-0.80 \times -1)
+
(-0.40 \times 0.5)
\]

\[
z =
-0.30
+
1.20
+
0.80
-
0.20
=
1.50
\]

The probability is:

\[
P(y=1)
=
\frac{1}
{1 + e^{-1.50}}
\approx
0.818
\]

Interpretation:

> The model estimates an 81.8% probability that the employee belongs to class 1.

With the usual threshold:

```text
0.818 >= 0.5
```

the predicted class is:

```text
1
```

---

# 11. Quick Summary

```text
Confusion matrix
→ raw correct and incorrect prediction counts

Accuracy
→ percentage of all predictions that are correct

Precision
→ among predicted positives, percentage actually positive

Recall
→ among actual positives, percentage detected

F1
→ balance between precision and recall

Coefficient
→ effect of one feature on the Logistic Regression score

Bias
→ baseline score when transformed feature values equal zero
```
