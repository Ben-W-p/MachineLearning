import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def split_data(X, y, test_size):
    n = len(X)
    split = int(n * (1 - test_size))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    return X_train, X_test, y_train, y_test

def accuracy_score_manual(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_manual(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    m = np.zeros((len(labels), len(labels)), dtype=int)
    idx = {lab: i for i, lab in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return labels, m

df = pd.read_csv("Student-Pass-Fail.csv")
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values

test_size = float(input("Enter test size (e.g., 0.2): "))
X_train, X_test, y_train, y_test = split_data(X, y, test_size)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score_manual(y_test, y_pred)
labels, cm = confusion_matrix_manual(y_test, y_pred)

print("Accuracy:", acc)
print("Labels order:", labels.tolist())
print("Confusion Matrix:\n", cm)