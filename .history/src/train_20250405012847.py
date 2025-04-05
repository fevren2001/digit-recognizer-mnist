# train.py
from src.model import build_model
from src.utils import load_and_preprocess_data
from tensorflow.keras.utils import to_categorical

def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save("model/mnist_cnn.h5")

if __name__ == "__main__":
    main()
