# Linear Regression

## Definition

Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value.

It models the relationship between one or more input features and a numerical target using a linear equation.

Examples of continuous targets include:

- house price;
- salary;
- customer engagement score;
- sales revenue;
- temperature;
- delivery time.

---

## Supervised Learning

Supervised Learning is a type of Machine Learning in which a model learns from labeled data.

Each observation contains:

- input features;
- a known target value.

The model uses these examples to learn how to predict the target for new observations.

---

## Feature

A feature is an input variable used by the model to make a prediction.

In a customer engagement problem, possible features include:

- age;
- monthly income;
- previous purchases.

Features are commonly represented by the symbol \(x\).

---

## Target

The target is the value the model tries to predict.

In this project, the target is the customer engagement score.

The actual target is commonly represented by:

\[
y
\]

---

## Prediction

A prediction is the numerical value estimated by the model.

It is commonly represented by:

\[
\hat{y}
\]

The objective of the model is to make \(\hat{y}\) as close as possible to \(y\).

---

## Linear Model

A Linear Regression model has the general form:

\[
\hat{y}
=
w_1x_1
+
w_2x_2
+
\cdots
+
w_px_p
+
b
\]

Where:

- \(x_1, x_2, \ldots, x_p\) are the features;
- \(w_1, w_2, \ldots, w_p\) are the learned weights;
- \(b\) is the bias;
- \(\hat{y}\) is the predicted value.

---

## Weight

A weight is a parameter learned by the model.

It represents how strongly a feature influences the prediction.

A positive weight means the prediction tends to increase when the feature increases.

A negative weight means the prediction tends to decrease when the feature increases.

A weight close to zero means the feature has little linear influence on the prediction.

---

## Bias

The bias, also called the intercept, is the baseline value of the prediction.

It is the predicted value when all input features are equal to zero.

The bias allows the regression line or regression plane to shift instead of being forced to pass through the origin.

---

## Residual

A residual is the difference between the actual value and the predicted value.

\[
\text{Residual}
=
y-\hat{y}
\]

A positive residual means the model predicted too low.

A negative residual means the model predicted too high.

A residual close to zero means the prediction is close to the actual value.

---

## Ordinary Least Squares

Ordinary Least Squares is the standard method used to train Linear Regression.

It finds the weights and bias that minimize the sum of squared residuals:

\[
\sum_{i=1}^{n}
\left(y_i-\hat{y}_i\right)^2
\]

Squaring the residuals prevents positive and negative errors from cancelling each other and gives more importance to larger errors.

---

## Linear Relationship

A linear relationship means that a change in a feature produces an approximately constant change in the predicted target.

For example, if one additional purchase always increases the predicted engagement score by approximately the same amount, the relationship is approximately linear.

---

## Multiple Linear Regression

Multiple Linear Regression uses several features to predict one numerical target.

For example, a customer engagement score can be predicted from:

- age;
- income;
- previous purchases.

Each feature receives its own learned weight.

---

## When Linear Regression Is Used

Linear Regression is suitable when:

- the target is numerical;
- the output is continuous;
- the relationship is approximately linear;
- interpretability is important;
- a simple baseline model is needed.

---

## When Linear Regression Is Not Suitable

Linear Regression is not intended for categorical targets such as:

- yes or no;
- spam or not spam;
- fraud or normal;
- churn or no churn.

Those are classification problems.

It may also perform poorly when the relationship between features and target is strongly nonlinear.

---

## Main Advantages

Linear Regression is:

- simple;
- fast;
- interpretable;
- easy to train;
- useful as a baseline;
- effective for approximately linear relationships.

---

## Main Limitations

Linear Regression can be affected by:

- nonlinear relationships;
- outliers;
- highly correlated features;
- missing important variables;
- very small datasets;
- poor-quality data.

---

## Evaluation Metrics

### MAE

Mean Absolute Error is the average absolute difference between actual and predicted values.

Lower values indicate better predictions.

### MSE

Mean Squared Error is the average squared difference between actual and predicted values.

It gives more importance to large errors.

### RMSE

Root Mean Squared Error is the square root of MSE.

It is expressed in the same unit as the target.

### R²

R² measures how much of the target variation is explained by the model.

A value closer to 1 usually indicates a better fit.

---

## Summary

Linear Regression learns a linear relationship between numerical features and a continuous target.

It estimates:

- one weight for each feature;
- one bias;
- one predicted numerical value for each observation.

Its goal is to minimize the differences between actual and predicted values.
