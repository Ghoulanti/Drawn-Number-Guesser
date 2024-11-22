import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

data = pd.read_csv('train.csv')
X_train = data.drop("label", axis=1).values.astype(np.float32)
Y_train = data["label"]

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0

model = tf.keras.models.load_model('model.keras', custom_objects={'softmax_v2': tf.nn.softmax})

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=3)

p = input()

if p == 0:
    model.save("model.keras")
