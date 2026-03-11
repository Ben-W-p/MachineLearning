import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Bank-data.csv")

df = df.iloc[:, 1:]
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

odds = np.exp(model.coef_)
print("Odds:", odds)

points = np.array([[1.335, 0, 1, 0, 0, 109], [1.25, 0, 0, 1, 0, 279]])
probs = model.predict_proba(points)

for p, pr in zip(points, probs):
    print("Data point:", p.tolist(), "Probabilities:", pr.tolist())