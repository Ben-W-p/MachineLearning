import numpy as np
import pandas as pd

df = pd.read_csv("materials.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_names = df.columns[:-1]

for i in range(X.shape[1]):
    r = np.corrcoef(X[:, i], y)[0, 1]
    print("Correlation between", feature_names[i], "and Strength:", r)

X_design = np.c_[np.ones(len(X)), X]
beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y

points = np.array([
    [32.1, 37.5, 128.95],
    [36.9, 35.37, 130.03]
])

for p in points:
    x_vec = np.insert(p, 0, 1)
    pred = 0
    for i in range(len(beta)):
        pred += beta[i] * x_vec[i]
    print("Predicted Strength:", pred)