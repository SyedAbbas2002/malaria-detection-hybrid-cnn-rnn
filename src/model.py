import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(img_size=(32,32,3)):
    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=img_size))
    model.add(layers.Conv2D(64, (7,7), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Reshape((128, 64)))  # Reshaping for RNN layers

    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
