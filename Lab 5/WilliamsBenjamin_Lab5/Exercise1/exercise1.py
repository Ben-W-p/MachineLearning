import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Student-Pass-Fail.csv")

X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values

model = LogisticRegression()
model.fit(X, y)

odds = np.exp(model.coef_)
print("Odds:", odds)

points = np.array([[7, 28], [10, 34], [2, 39]])
probs = model.predict_proba(points)

for p, pr in zip(points, probs):
    print("Data point:", p.tolist(), "Probabilities:", pr.tolist())