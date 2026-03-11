import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("university_admissions_3features.csv")

X = df.iloc[:, 0:3].to_numpy(dtype=float)
y = df.iloc[:, 3].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)

model = LogisticRegression(solver="lbfgs", random_state=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

samples = np.array([
    [8.5, 8.0, 2.0],
    [6.0, 6.0, 8.5],
    [4.0, 9.0, 4.0],
    [9.0, 4.0, 6.5],
    [3.0, 3.0, 9.0],
    [7.0, 7.0, 7.0]
], dtype=float)

sample_preds = model.predict(samples)
print("\nPredicted samples (x1, x2, x3) -> class:")
for s, p in zip(samples, sample_preds):
    print(s.tolist(), "->", int(p))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

m0 = y == 0
m1 = y == 1
ax.scatter(X[m0, 0], X[m0, 1], X[m0, 2], label="Not Admitted (0)")
ax.scatter(X[m1, 0], X[m1, 1], X[m1, 2], label="Admitted (1)")

ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], marker="x", s=90, linewidths=3, label="Predicted samples (6 pts)")

x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 30), np.linspace(x2_min, x2_max, 30))

w1, w2, w3 = model.coef_[0]
b = model.intercept_[0]
zz = (-b - w1 * xx - w2 * yy) / (w3 + 1e-12)

ax.plot_surface(xx, yy, zz, alpha=0.25)

ax.set_title("University Admission (3 Features) - Logistic Regression")
ax.set_xlabel("TEST (x1)")
ax.set_ylabel("GRADES (x2)")
ax.set_zlabel("CLASS RANK (x3)")
ax.legend()
plt.show()