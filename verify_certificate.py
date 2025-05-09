import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match training size
    img = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

def verify_certificate(image_path):
    # Load the trained model
    model = tf.keras.models.load_model('medical_certificate_verifier.h5')
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Get result (assuming 0 is fake and 1 is real)
    result = "verified" if prediction[0][0] >= 0.5 else "not-verified"
    confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
    
    print(f"Result: {result}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_certificate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    verify_certificate(image_path) 