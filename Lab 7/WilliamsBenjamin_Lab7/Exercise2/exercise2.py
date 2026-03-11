import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("golf.csv")

encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()

model = GaussianNB()
model.fit(X,y)

samples = [
["Rainy","Hot","High","True"],
["Sunny","Mild","Normal","False"],
["Sunny","Cool","High","False"]
]

samples_encoded = []

for row in samples:
    encoded = []
    for i,col in enumerate(df.columns[:-1]):
        encoded.append(encoders[col].transform([row[i]])[0])
    samples_encoded.append(encoded)

samples_encoded = np.array(samples_encoded)

preds = model.predict(samples_encoded)

print("Predictions:")
for s,p in zip(samples,preds):
    print(s,"->", encoders["PlayGolf"].inverse_transform([p])[0])