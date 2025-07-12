from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("model/mnist_cnn.h5")

img = Image.open("debug_input.png").convert('L')
img_array = np.array(img)
img_array = 255 - img_array
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

pred = model.predict(img_array)
digit = np.argmax(pred)
confidence = np.max(pred)

print(f"Model prediction: {digit} with confidence {confidence*100:.2f}%")
