import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

X2 = X[::10, :2]
y2 = y[::10]

model = LinearRegression()
model.fit(X2, y2)

x0 = X2[:, 0]
x1 = X2[:, 1]

x0_grid, x1_grid = np.meshgrid(
    np.linspace(x0.min(), x0.max(), 40),
    np.linspace(x1.min(), x1.max(), 40)
)
grid = np.c_[x0_grid.ravel(), x1_grid.ravel()]
y_grid = model.predict(grid).reshape(x0_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(x0_grid, x1_grid, y_grid, alpha=0.55)
ax.scatter(x0, x1, y2, s=10)

ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel("Target (MedHouseVal)")
ax.set_title("Multiple Linear Regression (2 features): Plane + Scatter")

plt.show()