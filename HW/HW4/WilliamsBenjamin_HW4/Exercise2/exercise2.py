import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("avgHigh_jan_1895-2018.csv")
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

test_size = float(input("Enter test size (e.g., 0.2): "))
split = int(len(X) * (1 - test_size))

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for actual, pred in zip(y_test, y_pred):
    print("Actual:", actual, "Predicted:", pred)

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("RMSE:", rmse)