import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("avgHigh_jan_1895-2018.csv")
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

model = LinearRegression()
model.fit(X, y)

years_pred = np.array([2019, 2023, 2024]).reshape(-1, 1)
preds = model.predict(years_pred)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

for yr, pr in zip(years_pred.flatten(), preds):
    print("Predicted temperature for Jan", yr, ":", pr)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.scatter(years_pred, preds)
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("Simple Linear Regression")
plt.show()