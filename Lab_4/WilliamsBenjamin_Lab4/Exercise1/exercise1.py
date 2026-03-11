import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

X10 = X[::10]
y10 = y[::10]

model = LinearRegression()
model.fit(X10, y10)

print("Features:", feature_names)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

x_new = np.array([[8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]])
pred = model.predict(x_new)[0]
print("Prediction (Median House Value):", pred)