from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the trained model once when server starts
model = tf.keras.models.load_model("model/mnist_cnn.h5")

def preprocess_image(image_bytes):
    # Load image and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(img)

    # img.save("debug_input.png")

    # Invert colors (MNIST digits are white on black background)
    img_array = 255 - img_array
    # Normalize
    img_array = img_array / 255.0
    # Reshape to (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # The image comes as a base64 string "data:image/png;base64,....."
    img_b64 = data['image'].split(',')[1]
    image_bytes = base64.b64decode(img_b64)

    img_array = preprocess_image(image_bytes)

    # Predict digit
    prediction = model.predict(img_array)
    digit = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    return jsonify({'digit': int(digit), 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
    
