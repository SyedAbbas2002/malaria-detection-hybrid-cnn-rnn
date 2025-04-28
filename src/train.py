import os
import tensorflow as tf
from data_loader import load_data
from model import build_model

def main():
    data_dir = 'dataset'  # Place dataset folder here
    model_save_path = 'models/malaria_model.h5'

    train_data, val_data = load_data(data_dir)

    model = build_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        train_data,
        epochs=100,
        validation_data=val_data,
        callbacks=[early_stopping]
    )

    os.makedirs('models', exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
