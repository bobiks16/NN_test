import yaml

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array


labels = []
for folder in ('data/test/labels/', 'data/train/labels/', 'data/valid/labels/'):
    for filename in os.listdir(folder):
        with open(folder + filename, 'r') as label_file:
            label_content = label_file.readline().split()
            label_content = list(map(float, label_content))
            label_content[0] = int(label_content[0])
            labels.append(label_content)
            
classes = []
coordinates = []


for label in labels:
    classes.append(int(label[0]))
    coordinates.append(label[1:])

classes = np.array(classes)
coordinates = np.array(coordinates)


images = []
for folder in img_dirs:
    for img in os.listdir(folder):
        img = load_img(folder + img, target_size=(208, 208))
        img_array = img_to_array(img)
        img_array = tf.image.rgb_to_grayscale(img_array)
        img_array /= 255.0
        images.append(img_array)

images = np.array(images)


X_train, X_test, y_train_classes, y_test_classes, y_train_coordinates, y_test_coordinates = \
    train_test_split(images, classes, coordinates, test_size=0.2, random_state=42)

y_train_classes_encoded = to_categorical(y_train_classes, 29)
y_test_classes_encoded = to_categorical(y_test_classes, 29)