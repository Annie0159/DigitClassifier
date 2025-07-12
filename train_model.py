import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to match CNN input: (batch, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build a simple CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=7, validation_data=(x_test, y_test))

# preds = model.predict(x_test)
# y_pred = np.argmax(preds, axis=1)

# # See incorrect predictions
# wrong = np.where(y_pred != y_test)[0]

# for i in wrong[:5]:
#     plt.imshow(x_test[i].reshape(28,28), cmap='gray')
#     plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
#     plt.show()

# Save model
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/mnist_cnn.h5")
print("✅ Model trained and saved.")



