import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

X = np.array([100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600, 634, 718, 750, 850, 903, 978, 1010, 1050, 1990], dtype=float).reshape(-1, 1)
y = np.array([12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989], dtype=float)

lr = LinearRegression()
lr.fit(X, y)
m1 = lr.coef_[0]
b1 = lr.intercept_

ransac = RANSACRegressor(estimator=LinearRegression(), random_state=0)
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

m2 = ransac.estimator_.coef_[0]
b2 = ransac.estimator_.intercept_

print("Before RANSAC: slope =", m1, ", y-intercept =", b1)
print("After  RANSAC: slope =", m2, ", y-intercept =", b2)

x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_line_before = lr.predict(x_line)
y_line_after = ransac.predict(x_line)

plt.figure()
plt.plot(x_line, y_line_before, label="Before RANSAC")
plt.plot(x_line, y_line_after, label="After RANSAC")
plt.scatter(X[inlier_mask], y[inlier_mask], s=25, label="Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask], s=25, label="Outliers")
plt.xlabel("Square Feet")
plt.ylabel("Price ($)")
plt.title("Linear Regression Before and After RANSAC")
plt.legend()
plt.show()