import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = np.array(data.feature_names)

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

model = LinearRegression()
model.fit(Xz, y)

coefs = model.coef_
idx = np.argmax(np.abs(coefs))

print("All coefficients (standardized X):")
for n, c in zip(feature_names, coefs):
    print(n, c)

print("\nMost-weight coefficient (by absolute value):")
print(feature_names[idx], coefs[idx])