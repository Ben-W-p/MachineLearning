import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns



names =  ["class", "Alcohol","Malic Acid","Ash","Acadlinity","Magnisium","Total Phenols",
               "Flavanoids","NonFlavanoid Phenols", "Proanthocyanins", "Color Intensity", 
               "Hue", "OD280/OD315", "Proline" ]
df = pd.read_csv("wine.data.csv", header=None, names = names)

x = np.array(df.iloc[:, 1:14])
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3) 

results = []
for i in range(1, 11):
    knn = KNeighborsClassifier(i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append([i, accuracy])


ks = [row[0] for row in results]
acc = [row[1] for row in results]

plt.plot(ks, acc)
plt.xlabel("k-value")
plt.ylabel("accuracy")
plt.title("k vs accuracy")
plt.show()
