# Exercise 03: Support Ticket Escalation

## Subject

A support team wants to predict whether a ticket will be escalated.

## Goal

Predict:

```text
escalated
```

Classes:

```text
0 = not escalated
1 = escalated
```

## Requirements

Build the complete workflow yourself using:

- pandas;
- missing-value handling;
- numerical and categorical preprocessing;
- `ColumnTransformer`;
- `Pipeline`;
- `LogisticRegression`;
- stratified train-test split;
- classification metrics;
- class probabilities;
- feature coefficients.

## Reflection Questions

1. Which classification mistake is more damaging here?
2. Is accuracy sufficient?
3. Which features appear to increase escalation probability?
4. Could changing the threshold be useful? Explain without implementing it.
