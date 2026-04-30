import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

columns = [
    "ID",
    "Clump_Thickness",
    "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
    "Class"
]

df = pd.read_csv(
    "breast-cancer-wisconsin-data.csv",
    header=None,
    names=columns,
    na_values="?"
)

print("Rows before cleaning:", len(df))
df = df.dropna().copy()
print("Rows after removing ?: ", len(df))

X = df.iloc[:, 1:-1].astype(float)
y = df.iloc[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model = SVC(kernel="linear")
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)

X_scaled_all = scaler.transform(X)
X_pca_all = pca.transform(X_scaled_all)

plt.figure(figsize=(8, 6))

for cls, color in zip(sorted(y.unique()), ["purple", "gold"]):
    mask = (y == cls)
    plt.scatter(
        X_pca_all[mask, 0],
        X_pca_all[mask, 1],
        label=str(cls),
        alpha=0.8
    )

w = model.coef_[0]
b = model.intercept_[0]

x_min, x_max = X_pca_all[:, 0].min() - 1, X_pca_all[:, 0].max() + 1
x_values = np.linspace(x_min, x_max, 300)

if abs(w[1]) > 1e-10:
    y_values = -(w[0] * x_values + b) / w[1]
    plt.plot(x_values, y_values, linewidth=2)
else:
    x_line = -b / w[0]
    plt.axvline(x=x_line, linewidth=2)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("SVC with PCA")
plt.legend(title="Class")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()