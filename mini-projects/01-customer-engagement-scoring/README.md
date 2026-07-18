# Customer Engagement Scoring

This project uses NumPy to train a simple linear regression model from scratch.

## Goal

Predict a customer engagement score using:

- Age
- Monthly income
- Previous purchases

## Steps

1. Load the customer data.
2. Standardize the features.
3. Add a bias column.
4. Calculate the weights and bias with the pseudoinverse.
5. Make predictions.
6. Calculate residuals, MSE, and RMSE.

## Model

The model has this form:

```text
predicted_score =
    age_weight × standardized_age
    + income_weight × standardized_income
    + purchases_weight × standardized_purchases
    + bias
```

## Run the project

```bash
uv run python customer_engagement_scoring.py
```

## Output

The script displays:

- Feature means
- Feature standard deviations
- Learned weights
- Learned bias
- Actual scores
- Predicted scores
- Residuals
- MSE
- RMSE

## Note

The same data is used for training and evaluation. This project is for learning the basic mathematics of linear regression.