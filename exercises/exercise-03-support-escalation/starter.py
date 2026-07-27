from pathlib import Path

import pandas as pd


csv_path = Path(__file__).parent / "support_escalation.csv"
data = pd.read_csv(csv_path)


# Build the complete classification workflow yourself.
#
# Required outputs:
# - class distribution
# - predicted classes
# - probability of escalation
# - confusion matrix
# - accuracy
# - precision
# - recall
# - F1 score
# - transformed feature names
# - feature coefficients
