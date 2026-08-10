# the columns that have the impact in the failer ??

import pandas as pd
from pathlib import Path

try:
    path = Path(__file__).parent / "data/ai4i2020.csv"
    df = pd.read_csv(path)
except Exception as e:
    print(e)
    exit()

# print(df.head())


# not necessary data : UDI, Product ID
# INPUT(X) : Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
# OUTPUT / TARGET(Y) : Machine failure

# ? not necessary data : TWF HDF PWF OSF RNF (but I should to know why)

# ! Critical rule
# + Critical rule
# * Do NOT use the failure-mode target columns TWF, HDF, PWF, OSF or RNF as input features when predicting Machine failure. Machine failure is determined
# *     from those failure modes, so using them would give the model information derived directly from the target.
# ? let's make a test to see if it's true

tst_result = df[
    (df["TWF"] == 1)
    | (df["HDF"] == 1)
    | (df["PWF"] == 1)
    | (df["OSF"] == 1)
    | (df["RNF"] == 1)
    & (df["Machine failure"] == 1)
]
only_failure = df.loc[df["Machine failure"] == 1, "Machine failure"].count()


print("failure-mode columns : ")
print(tst_result[["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]])

print("failure count: ", end="")
print(only_failure)

# TODO : Conclusion

# change naming fields
selected_data = df.iloc[:, 2 : 2 + 7].copy()
selected_data.columns = [
    "type",
    "air_temp",
    "precess_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
]
# print(selected_data.head())
