import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(42)

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

def plot_loss(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=[X_train_full.shape[1]]),
    tf.keras.layers.Dense(1)
])

model1.compile(optimizer="adam", loss="mse", metrics=["mae"])

history1 = model1.fit(
    X_train_full, y_train_full,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

test_loss1, test_mae1 = model1.evaluate(X_test, y_test, verbose=0)
print("Model 1 Test Loss:", test_loss1)
print("Model 1 Test MAE:", test_mae1)

plot_loss(history1, "Model 1: Training Loss vs Validation Loss")

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=[X_train_full.shape[1]]),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model2.compile(optimizer="adam", loss="mse", metrics=["mae"])

history2 = model2.fit(
    X_train_full, y_train_full,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

test_loss2, test_mae2 = model2.evaluate(X_test, y_test, verbose=0)
print("Model 2 Test Loss:", test_loss2)
print("Model 2 Test MAE:", test_mae2)

plot_loss(history2, "Model 2: Training Loss vs Validation Loss")