import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([[1, 5],[3, 2],[8, 4],[7, 14]], dtype=float)

def my_mean(col):
    s = 0.0
    n = len(col)
    for v in col:
        s += v
    return s / n

def my_std(col):
    mu = my_mean(col)
    s = 0.0
    n = len(col)
    for v in col:
        s += (v - mu) ** 2
    return (s / n) ** 0.5

def standardize(X):
    rows, cols = X.shape
    means = [my_mean(X[:, j]) for j in range(cols)]
    stds  = [my_std(X[:, j])  for j in range(cols)]

    Z = np.zeros_like(X, dtype=float)
    for i in range(rows):
        for j in range(cols):
            Z[i, j] = (X[i, j] - means[j]) / stds[j]

    return Z, np.array(means), np.array(stds)

def inverse_standardize(Z, means, stds):
    rows, cols = Z.shape
    X_rec = np.zeros_like(Z, dtype=float)
    for i in range(rows):
        for j in range(cols):
            X_rec[i, j] = Z[i, j] * stds[j] + means[j]
    return X_rec

Z_my, means_my, stds_my = standardize(data)
data_back_my = inverse_standardize(Z_my, means_my, stds_my)
scaler = StandardScaler(with_mean=True, with_std=True)
Z_sk = scaler.fit_transform(data)
data_back_sk = scaler.inverse_transform(Z_sk)

print("Original data:\n", data)
print("\nMy standardized:\n", Z_my)
print("\nMy inverse standardized:\n", data_back_my)
print("\nSklearn standardized:\n", Z_sk)
print("\nSklearn inverse standardized:\n", data_back_sk)
print("\nMax abs diff (standardized):", np.max(np.abs(Z_my - Z_sk)))
print("Max abs diff (inverse):      ", np.max(np.abs(data_back_my - data_back_sk)))
