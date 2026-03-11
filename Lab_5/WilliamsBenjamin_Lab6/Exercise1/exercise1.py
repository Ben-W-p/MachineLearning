import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")


# Since it's a greyscale image, I just normalized
# the pixel values to be between 0 and 1 by dividing by 255.0.
y_train = train_df.iloc[:, 0].to_numpy()
X_train = train_df.iloc[:, 1:].to_numpy() / 255.0

y_test = test_df.iloc[:, 0].to_numpy()
X_test = test_df.iloc[:, 1:].to_numpy() / 255.0
print("Data loaded successfully.")

model = LogisticRegression(solver='saga', max_iter=1000, tol = 1e-2)
print("Training the model...")
model.fit(X_train, y_train)

print("Model trained successfully.")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Text):")
print(cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()