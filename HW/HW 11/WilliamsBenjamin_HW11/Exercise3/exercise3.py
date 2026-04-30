import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

script_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(script_dir, "train")
validation_dir = os.path.join(script_dir, "validation")
cat_image_path = os.path.join(script_dir, "my_cat.png")
dog_image_path = os.path.join(script_dir, "my_dog.png")

required_paths = [train_dir, validation_dir, cat_image_path, dog_image_path]
for path in required_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required path: {path}")

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 5

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

model = Sequential([
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=1,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(validation_generator, verbose=1)

best_train_acc = max(history.history["accuracy"])
best_val_acc = max(history.history["val_accuracy"])

print(f"Original source EPOCHS: 2")
print(f"New EPOCHS: {EPOCHS}")
print(f"Best Training Accuracy: {best_train_acc:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

model.save(os.path.join(script_dir, "cats_vs_dogs_model.h5"))

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss_values = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_values, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "training_history.png"))
plt.close()

class_labels = {v: k for k, v in train_generator.class_indices.items()}

def predict_external_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = class_labels[int(prediction >= 0.5)]

    out_path = os.path.splitext(img_path)[0] + "_prediction.png"

    plt.figure()
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({prediction:.4f})")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"{os.path.basename(img_path)} -> {predicted_class} ({prediction:.4f})")
    print(f"Saved: {out_path}")

predict_external_image(cat_image_path)
predict_external_image(dog_image_path)

print("Saved: training_history.png")