import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_single(img_path):
    model = tf.keras.models.load_model('models/malaria_model.h5')
    img = image.load_img(img_path, target_size=(32,32))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    if class_idx == 0:
        print("Prediction: Parasitized")
    else:
        print("Prediction: Uninfected")

# Example
# predict_single('path_to_image.png')
