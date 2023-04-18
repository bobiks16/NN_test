import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

size_val = 5000
x_val_split = x_train[:size_val]
y_val_split = y_train_cat[:size_val]

x_train_split = x_train[size_val:size_val*2]
y_train_split = y_train_cat[size_val:size_val*2]

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(300, activation="relu"),
    #Dropout(0.8),
    BatchNormalization(),
    Dense(10, activation="softmax")
])
print(model.summary())

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

history = model.fit(x_train_split, y_train_split, batch_size=32, epochs=16, validation_data=(x_val_split, y_val_split))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.show()