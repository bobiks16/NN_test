import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
import pandas as pd

data = pd.read_csv("data/weatherHistory.csv")

model = keras.Sequential()

model.add(Dense(units=1, input_shape=(1,), activation="linear")) 
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=500)
plt.plot(history.history["loss"])
plt.grid(True)
plt.show()

print(model.predict([100]))
print(model.get_weights())