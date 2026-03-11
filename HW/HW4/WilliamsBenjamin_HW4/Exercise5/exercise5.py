import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression

df = pd.read_csv("materialsOutliers.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

mask = np.ones(len(df), dtype=bool)

for i in range(X.shape[1]):
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=15,
        stop_probability=1.0
    )
    ransac.fit(X[:, i].reshape(-1, 1), y)
    mask = mask & ransac.inlier_mask_

X_clean = X[mask]
y_clean = y[mask]

model = LinearRegression()
model.fit(X_clean, y_clean)

print("Final Coefficients:", model.coef_)
print("Intercept:", model.intercept_)