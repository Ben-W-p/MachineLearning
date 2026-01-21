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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

x_2 = np.array([
    [14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065],
    [12.64,1.36,2.02,16.8,100,2.02,1.41,.53,.62,5.75,.98,1.59,450],
    [12.53,5.51,2.64,25,96,1.79,.6,.63,1.1,5,.82,1.69,515],
    [13.49,3.59,2.19,19.5,88,1.62,.48,.58,.88,5.7,.81,1.82,580]
])

predictions = knn.predict(x_2)
print("Predicted Classes:", predictions)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()