import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)
