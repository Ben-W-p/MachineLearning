import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("hsbdemo.csv")

X = df.drop(columns=["id", "prog", "cid"])
y = df["prog"]

cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=9
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("Misclassified (predicted, actual):")
for i in range(len(y_pred)):
    if y_pred[i] != y_test.iloc[i]:
        print(f"{y_pred[i]}, {y_test.iloc[i]}")

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.colorbar()
plt.tight_layout()
plt.show()