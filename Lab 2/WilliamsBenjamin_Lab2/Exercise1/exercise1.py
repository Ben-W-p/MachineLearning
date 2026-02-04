import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# The handwritten digits dataset contains 1797 images where each image is 8x8
# Thus, we have 64 features (8x8)
# X: features (64)
# y: label (0-9)
# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target
print(f'Shape X: {X.shape}')
print(f'Shape y: {y.shape}')

# I wanted to keep track of the test indicies for later
index_list = np.arange(X.shape[0])

x_train, x_test, y_train, y_test, i_train, i_test = train_test_split(X, y, index_list, test_size=.2, random_state=42)

knn = KNeighborsClassifier(3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(matrix)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Accuracy: {accuracy:.4f}")
plt.show()

# Visualize some samples
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, idx in zip(axes, range(5)):
    ax.imshow(digits.images[i_test[idx]], cmap='gray')
    ax.set_title(f'Predicted Label: {y_pred[idx]}\n Actual Label: {y_test[idx]}')
    ax.axis('off')
plt.show()