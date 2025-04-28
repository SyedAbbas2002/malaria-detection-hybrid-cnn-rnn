import tensorflow as tf
import numpy as np
from data_loader import load_data
from sklearn.metrics import classification_report, confusion_matrix

def main():
    model = tf.keras.models.load_model('models/malaria_model.h5')
    _, val_data = load_data('dataset')

    val_labels = val_data.classes
    pred_probs = model.predict(val_data)
    pred_labels = np.argmax(pred_probs, axis=1)

    print(confusion_matrix(val_labels, pred_labels))
    print(classification_report(val_labels, pred_labels, target_names=['Parasitized', 'Uninfected']))

if __name__ == "__main__":
    main()
