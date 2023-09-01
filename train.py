from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
import argparse

import prepare_data as prep_data


def build_model():
    # Define a CNN model for digit classification
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(9, activation="softmax")]
    )
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model


def main(args):
    data_choice = args['data']
    batch_size = args['batch_size']
    epochs = args['epochs']
    model_save_fpath = args['model_save_fpath']
    exclude_fonts = args['exclude_fonts']

    # Load data depending on user choice
    x_train, x_val, x_test, y_train, y_val, y_test = prep_data.get_data(data_choice=data_choice,
                                                                        exclude=exclude_fonts)
        
    
    # Get a model instance
    model = build_model()
    # Train the model
    print("Starting training...")
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=batch_size,
              epochs=epochs)
    print("Training complete")
    
    # Save the model
    if os.path.exists(model_save_fpath):
        # Append the current date and time to the filepath so we don't overwrite a model
        now = datetime.now()
        suffix = now.strftime("%d_%m_%Y_%H_%M_%S")
        model_save_fpath = f"models/model_{suffix}.keras"

    model.save(model_save_fpath)

    print(f"Model saved at: {model_save_fpath}")


if __name__ == '__main__':
    # Construct an argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="both", type=str, help="Choose data to use ('mnist', 'fonts', 'both')")
    ap.add_argument("--exclude_fonts", default=True, type=bool, help="Whether or not to exclude fonts like those in 'data/font_exclude/'")
    ap.add_argument("--model_save_fpath", default="models/model.keras", type=str)
    ap.add_argument("--batch_size", default="128", type=int)
    ap.add_argument("--epochs", default="10", type=int)
    
    args = vars(ap.parse_args())

    main(args)