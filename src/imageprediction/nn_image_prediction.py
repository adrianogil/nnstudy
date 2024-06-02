# Importing PlaidML. Make sure you follow this order
import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the image
print("Loading image...")
image_path = 'nolan.jpg'
img = Image.open(image_path)
img = img.convert('RGB')  # Ensure image is in RGB format

# Get image dimensions
width, height = img.size

# Extract pixel data
pixel_data = np.array(img)

# Normalize UV coordinates
U = np.linspace(0, 1, width)
V = np.linspace(0, 1, height)

# Create a grid of UV coordinates
UV = np.array([(u, v) for v in V for u in U])

# Extract RGB values
RGB = pixel_data.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]

# Print shapes
print("UV coordinates shape:", UV.shape)
print("RGB values shape:", RGB.shape)

# Build the model
model = Sequential([
    Dense(64, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Training model...")
model.fit(UV, RGB, epochs=100, batch_size=8)

# Predict RGB values for the entire UV grid
predicted_RGB = model.predict(UV)
predicted_RGB = (predicted_RGB * 255).astype(np.uint8)  # Denormalize RGB values

# Reshape predicted RGB values to match image dimensions
predicted_image = predicted_RGB.reshape(height, width, 3)

# Convert to PIL Image and save
predicted_img = Image.fromarray(predicted_image)
predicted_img.save('predicted_image.png')
predicted_img.show()