import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("hsbdemo.csv")

X = df.drop(columns=["id", "prog", "cid"])
y = df["prog"]

cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

vr = pca.explained_variance_ratio_
print("Variance Ratios:")

for i in range(len(vr)):
    print(f"PC{i}: {vr[i]:.4f}")

plt.plot(np.cumsum(vr))
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.show()