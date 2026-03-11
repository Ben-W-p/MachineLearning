import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

y_train = train_df.iloc[:, 0].to_numpy()
X_train = train_df.iloc[:, 1:].to_numpy() / 255.0

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    return img.reshape(1, -1)

img1 = preprocess_image("trousers.bmp")
img2 = preprocess_image("bag.jpg")

model = LogisticRegression(solver='saga', max_iter=1000, tol = 1e-2)
model.fit(X_train, y_train)

pred1 = model.predict(img1)[0]
pred2 = model.predict(img2)[0]

print("Prediction for trousers.bmp:", pred1)
print("Prediction for bag.jpg:", pred2)