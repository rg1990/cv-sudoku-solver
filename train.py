from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
import cv2
from datetime import datetime
import argparse


def load_mnist_images():

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Remove the zero images - zero is an invalid sudoku entry
    non_zero_train_indices = np.where(y_train != 0)[0]
    non_zero_test_indices = np.where(y_test != 0)[0]
    
    x_train, y_train = x_train[non_zero_train_indices], y_train[non_zero_train_indices]
    x_test, y_test = x_test[non_zero_test_indices], y_test[non_zero_test_indices]
    
    # Create a validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      train_size=0.85,
                                                      random_state=2023)
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_val = x_val.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)
        
    # Convert class vectors to binary class matrices
    # Remove the first element of each one-hot array because we exclude zeros
    y_train = keras.utils.to_categorical(y_train, num_classes=10)[:, 1:]
    y_val = keras.utils.to_categorical(y_val, num_classes=10)[:, 1:]
    y_test = keras.utils.to_categorical(y_test, num_classes=10)[:, 1:]

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_font_image_arrays(image_dict):
    # Create x array with all image data
    x = np.array([v for v in image_dict.values()])
    x = np.reshape(x, newshape=(-1, 28, 28, 1))
    # Create label array
    y = np.array([np.repeat(k, len(image_dict[k])) for k in image_dict])
    y = np.reshape(y, newshape=(-1, 1))
    # Convert y to one-hot labels. Exclude zeros - invalid sudoku entry
    y = keras.utils.to_categorical(y, num_classes=10)[:, 1:]
    
    # Split into train, validation and test sets with shuffle
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.15,
                                                        shuffle=True,
                                                        random_state=0)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.18,
                                                      shuffle=True,
                                                      random_state=33)
    
    # Invert images so backgrounds are black and fonts are white
    x_train = np.array(list(map(cv2.bitwise_not, x_train)))
    x_val = np.array(list(map(cv2.bitwise_not, x_val)))
    x_test = np.array(list(map(cv2.bitwise_not, x_test)))
    
    # Scale images to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_val = x_val.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Add the "channels" dimension
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_font_image_dict(excluded_names=None):
    '''
    Load the font images and store in a dictionary whose keys are the digit
    and the values are the image pixel arrays converted to grayscale and resized
    to match the MNIST digit shape.
    
    '''
    
    folder_names = glob.glob("data/digit_images/*")
    digit_image_filepaths = [glob.glob(folder + "/*.png") for folder in folder_names]
    
    if excluded_names:
        # Exclude the unwanted files
        inclusion_list_indices = list(np.where([not any(elem in fpath for elem in excluded_names) for fpath in digit_image_filepaths[0]])[0])
        digit_image_filepaths = [[fpath_list[i] for i in inclusion_list_indices] for fpath_list in digit_image_filepaths]
    
    # Create dictionary and load images from file
    img_dict = {i+1: None for i in range(9)}
    for k in img_dict:
        img_dict[k] = [cv2.imread(fpath) for fpath in digit_image_filepaths[k-1]]
    
    # Convert all images to grayscale and resize to 28x28 px
    for k, v in img_dict.items():
        gray = [cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) for arr in v]
        img_dict[k] = [cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA) for arr in gray]
        # Add "channels" dimension
        img_dict[k] = np.expand_dims(img_dict[k], -1)
    
    return img_dict


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


def get_data(data_choice, exclude=True):
    
    data_choice = data_choice.lower()
    if data_choice not in ['mnist', 'fonts', 'both']:
        raise ValueError("Invalid value for data_choice: {data_choice}. Valid options are: 'mnist', 'fonts', or 'both'")
    
    # Load MNIST data
    if data_choice == "mnist" or data_choice == "both":
        mnist_data = list(load_mnist_images())
    
    # Load font data
    if data_choice == "fonts" or data_choice == "both":
        if exclude:
            # Find the fonts to be excluded
            to_exclude = glob.glob("data/font_exclude/*.png")
            to_exclude = [fpath.split("\\")[-1] for fpath in to_exclude]
            to_exclude = [fpath.split("-")[-1] for fpath in to_exclude]
        else:
            to_exclude=None
        # Load the font images from file
        img_dict = get_font_image_dict(excluded_names=to_exclude)
        font_data = list(load_font_image_arrays(img_dict))
    
    # Assign x and y sets based on data_choice
    if data_choice == "mnist":
        x_train, x_val, x_test, y_train, y_val, y_test = mnist_data
    
    elif data_choice == "fonts":
        x_train, x_val, x_test, y_train, y_val, y_test = font_data
    
    elif data_choice == "both":
        # Combine MNIST and font digit images
        x_train = np.concatenate((font_data[0], mnist_data[0]))
        x_val = np.concatenate((font_data[1], mnist_data[1]))
        x_test = np.concatenate((font_data[2], mnist_data[2]))
        y_train = np.concatenate((font_data[3], mnist_data[3]))
        y_val = np.concatenate((font_data[4], mnist_data[4]))
        y_test = np.concatenate((font_data[5], mnist_data[5]))

    return x_train, x_val, x_test, y_train, y_val, y_test


def main(args):
    data_choice = args['data']
    batch_size = args['batch_size']
    epochs = args['epochs']
    model_save_fpath = args['model_save_fpath']
    exclude_fonts = args['exclude_fonts']

    # Load data depending on user choice
    x_train, x_val, x_test, y_train, y_val, y_test = get_data(data_choice=data_choice,
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