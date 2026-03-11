import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("wdbc.data.csv", header=None)

y = df.iloc[:,1].map({"M":1,"B":0}).to_numpy()
X = df.iloc[:,2:].to_numpy(dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance ratio:", pca.explained_variance_ratio_)

model = LogisticRegression()
model.fit(X_pca, y)

dataPoint = np.array([
7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,
0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,
9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039
])

dataPoint_scaled = scaler.transform(dataPoint.reshape(1,-1))
dataPoint_pca = pca.transform(dataPoint_scaled)

pred = model.predict(dataPoint_pca)[0]
print("Prediction:", pred)

plt.figure()

m0 = y==0
m1 = y==1

plt.scatter(X_pca[m0,0], X_pca[m0,1])
plt.scatter(X_pca[m1,0], X_pca[m1,1])

x_vals = np.linspace(X_pca[:,0].min(), X_pca[:,0].max(), 200)

w0 = model.intercept_[0]
w1, w2 = model.coef_[0]

y_vals = -(w0 + w1*x_vals)/w2

plt.plot(x_vals, y_vals)

plt.scatter(dataPoint_pca[:,0], dataPoint_pca[:,1], marker="x", s=150)

plt.title("PCA with Logistic Regression Decision Boundary")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()