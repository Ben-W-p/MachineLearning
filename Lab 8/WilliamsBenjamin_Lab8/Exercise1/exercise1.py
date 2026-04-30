import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("speedLimits.csv")
df.columns = [col.strip() for col in df.columns]

speed_col = next((c for c in df.columns if "speed" in c.lower()), df.columns[0])
label_col = next((c for c in df.columns if "ticket" in c.lower()), df.columns[1])

df[label_col] = df[label_col].astype(str).str.strip().str.upper()


df = df[df[label_col].isin(["T", "NT"])].copy()

y_plot = df[label_col].map({"T": 0, "NT": 1})
colors = df[label_col].map({"T": "red", "NT": "green"})

plt.figure(figsize=(8, 5))
plt.scatter(df[speed_col], y_plot, c=colors, s=60, edgecolors="black")
plt.yticks([0, 1], ["T", "NT"])
plt.xlabel("Speed")
plt.ylabel("Ticket")
plt.title("Speed vs Ticket")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

X = df[[speed_col]]
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

kernels = ["linear", "poly", "rbf", "sigmoid"]
results = {}

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[kernel] = acc
    print(f"{kernel} kernel accuracy: {acc:.4f}")

best_kernel = max(results, key=results.get)
print(f"\nOptimal kernel: {best_kernel}")
print(f"Best accuracy: {results[best_kernel]:.4f}")