import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


df = pd.read_csv("breast-cancer-wisconsin.data.csv", header=None)

df = df.replace("?", np.nan).dropna()

x = df.iloc[:, 1:-1].astype(float).to_numpy()
y = df.iloc[:, -1].astype(int).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3) 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

sns.heatmap(cm, annot=True)
plt.show()