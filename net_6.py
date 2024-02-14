from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


inputs = Input(shape=(208, 208, 1)) 

x = Conv2D(24, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(96, activation='relu')(x)

class_output = Dense(29, activation='softmax', name='class_output')(x)
bbox_output = Dense(4, name='bbox_output')(x) 

model = Model(inputs=inputs, outputs=[class_output, bbox_output])

model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mse'},
              metrics={'class_output': 'categorical_accuracy', 'bbox_output': 'mse'}) 

model.summary()


history = model.fit(
    X_train,
    {'class_output': y_train_classes_encoded, 'bbox_output': y_train_coordinates},
    validation_data=(X_test, {'class_output': y_test_classes_encoded, 'bbox_output': y_test_coordinates}),
    epochs=4, batch_size=32
)