import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.data.csv")
df.columns = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class']

#gives the species a numerical value instead of string classifier
df['class'] = df['class'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})

color_map = {1: 'purple', 2: 'teal', 3: 'yellow'}
colors = df['class'].map(color_map)

#
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(df['sepalLength'], df['sepalWidth'], c = colors)
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')

ax2.scatter(df['petalLength'], df['petalWidth'], c = colors)
ax2.set_xlabel('Petal Length')
ax2.set_ylabel('Petal Width')

plt.show()