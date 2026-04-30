import os
from tensorflow.keras.datasets import mnist
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation="relu"))
cnn.add(Dropout(0.3))
cnn.add(Dense(units=10, activation="softmax"))

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

def myPredict(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, (28, 28))
    image_gray = cv2.bitwise_not(image_gray)
    image_gray = image_gray.astype("float32") / 255
    image_gray = image_gray.reshape((1, 28, 28, 1))
    prediction = cnn.predict(image_gray, verbose=0)[0]
    return np.argmax(prediction)



image1 = cv2.imread("Five.png")
image2 = cv2.imread("Three.png")

pred1 = myPredict(image1)
pred2 = myPredict(image2)

print(f"Image 1 prediction: {pred1}")
print(f"Image 2 prediction: {pred2}")

plt.figure()
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {pred1}")
plt.axis("off")
plt.savefig("prediction1.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {pred2}")
plt.axis("off")
plt.savefig("prediction2.png", bbox_inches="tight")
plt.close()