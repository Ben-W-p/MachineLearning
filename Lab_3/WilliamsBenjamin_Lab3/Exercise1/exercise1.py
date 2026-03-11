import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("recipes_muffins_cupcakes_scones.csv")

num_df = df.select_dtypes(include=[np.number]).copy()

label_col = None
for c in df.columns:
    if c not in num_df.columns and df[c].nunique() <= max(2, min(10, len(df) // 5)):
        label_col = c
        break

X = num_df.values
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA(n_components=8, random_state=0)
Z = pca.fit_transform(Xz)

evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

print("Explained variance ratio (PC1..PC8):")
print(evr)
print("\nCumulative explained variance (PC1..PC8):")
print(cum)

plt.figure()
plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.xticks(range(1, len(cum) + 1))
plt.ylim(0, 1.05)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Sum of Explained Variance Ratio")
plt.grid(True, alpha=0.3)

plt.figure()
if label_col is not None:
    labels = df[label_col].astype(str).values
    for lab in np.unique(labels):
        m = labels == lab
        plt.scatter(Z[m, 0], Z[m, 1], s=35, alpha=0.85, label=lab)
    plt.legend(title=label_col, frameon=True)
else:
    plt.scatter(Z[:, 0], Z[:, 1], s=35, alpha=0.85)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scatter Plot: PC1 vs PC2")
plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 10))
cols = num_df.columns.tolist()
n = len(cols)
r = int(np.ceil(n / 2))
for i, c in enumerate(cols, 1):
    ax = plt.subplot(r, 2, i)
    ax.hist(df[c].dropna().values, bins=15)
    ax.set_title(c)
plt.tight_layout()

loadings = pd.DataFrame(pca.components_.T, index=cols, columns=[f"PC{i}" for i in range(1, 9)])

pc1 = loadings["PC1"]
pc2 = loadings["PC2"]

print("\nPC1 highest positive loading feature:", pc1.idxmax(), pc1.max())
print("PC1 most negative loading feature:", pc1.idxmin(), pc1.min())
print("PC2 highest positive loading feature:", pc2.idxmax(), pc2.max())
print("PC2 most negative loading feature:", pc2.idxmin(), pc2.min())

top_pc1 = pc1.abs().sort_values(ascending=False).head(5).index
top_pc2 = pc2.abs().sort_values(ascending=False).head(5).index
top_feats = list(dict.fromkeys(list(top_pc1) + list(top_pc2)))

plt.figure(figsize=(7, max(3, 0.5 * len(top_feats))))
sns.heatmap(loadings.loc[top_feats, ["PC1", "PC2"]], annot=True, cmap="coolwarm", center=0)
plt.title("Heatmap: Largest Variation (Loadings) in PC1 and PC2")
plt.xlabel("")
plt.ylabel("Feature")

corr = pd.DataFrame(Xz, columns=cols).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap (Standardized Features)")
plt.tight_layout()

plt.show()
