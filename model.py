import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

data = pd.read_csv('train.csv')
X_train = data.drop("label", axis=1).values.astype(np.float32)
Y_train = data["label"]

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=3)

image = cv.imread(r'C:\Users\viola\Downloads\Python\ai_project_numbers\digit.png', cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28, 28))
image = np.invert(image)
image = image / 255.0
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
predicted_label = np.argmax(prediction)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.show()

model.save("model.keras")