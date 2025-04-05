from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the model (make sure to adjust the path to your model file)
model = load_model('model_name.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = image.load_img(img, target_size=(28, 28))  # Adjust based on your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Preprocess the image as needed

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
