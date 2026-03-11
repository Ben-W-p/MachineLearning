import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

df2 = df.drop(columns=["Longitude", "Latitude"])

sns.pairplot(df2, diag_kind="hist")
plt.show()