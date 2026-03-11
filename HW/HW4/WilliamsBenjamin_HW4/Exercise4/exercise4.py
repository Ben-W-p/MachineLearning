import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("materials.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

corrs = [abs(np.corrcoef(X.iloc[:, i], y)[0, 1]) for i in range(X.shape[1])]
top2_idx = np.argsort(corrs)[-2:]

X2 = X.iloc[:, top2_idx].values

model = LinearRegression()
model.fit(X2, y)

x1 = X2[:, 0]
x2 = X2[:, 1]

x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 30),
    np.linspace(x2.min(), x2.max(), 30)
)

grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
y_grid = model.predict(grid).reshape(x1_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5)
ax.scatter(x1, x2, y)

ax.set_xlabel(X.columns[top2_idx[0]])
ax.set_ylabel(X.columns[top2_idx[1]])
ax.set_zlabel("Strength")

plt.show()