import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("vehicles.csv")
if "make" in df.columns:
    df = df.drop(columns=["make"])

y = df["mpg"].values
X = df.drop(columns=["mpg"])

X = pd.get_dummies(X, drop_first=True)

model = LinearRegression()
model.fit(X, y)

coefs = pd.Series(model.coef_, index=X.columns)
top5 = coefs.abs().sort_values(ascending=False).head(5)

print("Top 5 most important (by absolute weight):")
for name, val in top5.items():
    print(name, coefs[name])