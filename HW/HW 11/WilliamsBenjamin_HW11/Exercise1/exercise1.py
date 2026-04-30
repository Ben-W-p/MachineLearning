from tensorflow.keras.datasets import mnist
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

cnn.summary()

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

test_loss, test_accuracy = cnn.evaluate(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")