import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("university_admissions_2features.csv")

X = df.iloc[:, 0:2].to_numpy(dtype=float)
y = df.iloc[:, 2].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)

model = LogisticRegression(solver="lbfgs", random_state=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

samples = np.array([
    [4.0, 8.5],
    [7.5, 3.0],
    [6.0, 6.0],
    [3.0, 3.0],
    [8.5, 8.5],
    [5.0, 5.0]
], dtype=float)

sample_preds = model.predict(samples)
print("\nPredicted samples (x1, x2) -> class:")
for s, p in zip(samples, sample_preds):
    print(s.tolist(), "->", int(p))

plt.figure()

m0 = y == 0
m1 = y == 1
plt.scatter(X[m0, 0], X[m0, 1], label="Not Admitted (0)")
plt.scatter(X[m1, 0], X[m1, 1], label="Admitted (1)")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
plt.contour(xx, yy, probs, levels=[0.5])

plt.scatter(samples[:, 0], samples[:, 1], marker="x", s=120, linewidths=3, label="Predicted samples (6 pts)")

plt.title("University Admission (2 Features) - Logistic Regression")
plt.xlabel("TEST (x1)")
plt.ylabel("GRADES (x2)")
plt.legend()
plt.show()