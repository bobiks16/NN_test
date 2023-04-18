import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

size_val = 10000
x_val_split = x_train[:size_val]
y_val_split = y_train_cat[:size_val]

x_train_split = x_train[size_val:]
y_train_split = y_train_cat[size_val:]

from sklearn.model_selection import train_test_split
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2) 

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

myAdam = keras.optimizers.Adam(learning_rate=0.001)
mySGD = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=True)

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
print(model.summary())

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])


#model.fit(x_train, y_train_cat, batch_size=32, validation_split=0.2)
model.fit(x_train_split, y_train_split, batch_size=32, epochs=5, validation_data=(x_val_split, y_val_split))

model.evaluate(x_test, y_test_cat)

n = 10
x = np.expand_dims(x_test[n], axis=0)
result = model.predict(x)

print(result)
print(f"Распознана цифра: {np.argmax(result)}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()