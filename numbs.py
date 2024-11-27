import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import tensorflow as tf
import pickle
from functools import partial

model = tf.keras.models.load_model('model.keras', custom_objects={'softmax_v2': tf.nn.softmax})

grid_size = 28
data = np.ones((grid_size, grid_size))
drawing = False

def on_press(event):
    global drawing
    if event.inaxes == ax:
        drawing = True
        draw(event)

def on_release(event):
    global drawing
    drawing = False
    predict_digit()

def on_motion(event):
    if drawing and event.inaxes == ax:
        draw(event)

def draw(event):
    x, y = int(event.xdata), int(event.ydata)
    if 0 <= x < grid_size and 0 <= y < grid_size:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    distance = np.sqrt(dx**2 + dy**2)
                    intensity = max(0, 1 - distance / 3)
                    data[ny, nx] = min(data[ny, nx], 1 - intensity)
        update_display()

def update_display():
    ax.clear()
    ax.imshow(data, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Draw a digit and release to predict")
    plt.draw()

def preprocess_data(data):
    return np.expand_dims(data, axis=(0, -1))

def predict_digit():
    input_data = preprocess_data(data)
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)
    ax.set_title(f"Prediction: {predicted_label}")
    plt.draw()

def reset_canvas(event):
    global data
    data = np.ones((grid_size, grid_size))
    update_display()

def rossz(event, button):
    model.train_on_batch(preprocess_data(data), np.array([button.value]))
    reset_canvas(event)
class Numbs:
    def __init__(self, n, ax, label):
        self.value = n
        self.button = Button(ax, label)
        self.button.on_clicked(partial(rossz, button=self))

fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

reset_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
reset_button = Button(reset_ax, 'Reset')
reset_button.on_clicked(reset_canvas)

button_positions = [
    [0.1, 0.05, 0.05, 0.05],
    [0.2, 0.05, 0.05, 0.05],
    [0.3, 0.05, 0.05, 0.05],
    [0.4, 0.05, 0.05, 0.05],
    [0.5, 0.05, 0.05, 0.05],
    [0.6, 0.05, 0.05, 0.05],
    [0.7, 0.05, 0.05, 0.05],
    [0.8, 0.15, 0.05, 0.05],
    [0.9, 0.15, 0.05, 0.05],
    [0.1, 0.15, 0.05, 0.05],
]

numeric_buttons = []
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i, pos in enumerate(button_positions):
    ax_button = plt.axes(pos)
    num_button = Numbs(i, ax_button, labels[i])
    numeric_buttons.append(num_button)

update_display()
plt.show()

