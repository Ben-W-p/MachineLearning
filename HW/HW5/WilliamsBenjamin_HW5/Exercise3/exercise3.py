import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("vehicles.csv")
if "make" in df.columns:
    df = df.drop(columns=["make"])

y = df["mpg"].values
X_raw = df.drop(columns=["mpg"])
X = pd.get_dummies(X_raw, drop_first=True)

model = LinearRegression()
model.fit(X, y)

coefs = pd.Series(model.coef_, index=X.columns)
top5_names = coefs.abs().sort_values(ascending=False).head(5).index.tolist()

x_new_full = np.array([[6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4]], dtype=float)

x_new = pd.DataFrame(columns=X.columns)
x_new.loc[0] = 0.0

vals = dict(zip(df.drop(columns=["mpg"]).columns, x_new_full.flatten()))
for k, v in vals.items():
    if k in X_raw.columns:
        if k in x_new.columns:
            x_new.loc[0, k] = float(v)

x_new_top5 = x_new[top5_names].values
pred = model.predict(x_new)[0]

print("Predicted mpg:", pred)