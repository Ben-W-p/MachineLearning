import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris.data.csv", header=None)
X = df.iloc[:, 0:4].astype(float).to_numpy()
y = df.iloc[:, 4].astype(str).to_numpy()

n = len(X)
kfolds = 5
fold_sizes = [30] * 5

indices = np.arange(n)
folds = []
start = 0
for fs in fold_sizes:
    folds.append(indices[start:start + fs])
    start += fs

accuracies = []
for i in range(kfolds):
    test_idx = folds[i]
    train_idx = np.hstack([folds[j] for j in range(kfolds) if j != i])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)
    print(f"Fold {i+1} accuracy:", acc)

print("Average accuracy:", float(np.mean(accuracies)))