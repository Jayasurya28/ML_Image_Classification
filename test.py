import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Update the test folder path to a valid path on your system
test_folder = r"C:\Users\Jayasurya\OneDrive\Pictures\test_image_4.jpg"

# Load the trained model
model = load_model('trained_model.h5')

# Load class names correctly
class_file = 'class_names.json'
if os.path.exists(class_file):
    with open(class_file, 'r') as f:
        categories = json.load(f)

    # If it's a list, create a dictionary mapping indices to names
    if isinstance(categories, list):
        categories = {i: name for i, name in enumerate(categories)}
else:
    categories = {i: f"Class {i}" for i in range(model.output_shape[1])}  # Default class names

# Function to predict the image class
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"⚠️ File '{img_path}' not found.")
        return
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.  # Normalize
    predictions = model.predict(img_array)[0]  # Get the first (only) result

    # Get highest probability and corresponding class
    max_prob = np.max(predictions)  # Highest probability
    predicted_class_index = np.argmax(predictions)  # Class with highest probability

    # Always consider the highest percentage as the final output
    print(f"✅ '{os.path.basename(img_path)}' is classified as '{categories[predicted_class_index]}' with {max_prob:.2%} confidence.")

    # Print probabilities for all classes
    print("\nClass Probabilities:")
    for i, prob in enumerate(predictions):
        print(f"{categories[i]}: {float(prob):.2%}")

# Path to test image or folder
print(f"🔍 Checking path: {test_folder}")  # Debugging information
if os.path.isfile(test_folder):  # Single file case
    print(f"✅ Test image found: {test_folder}")  # Debugging information
    predict_image(test_folder)
elif os.path.isdir(test_folder):  # Directory case
    print(f"✅ Test directory found: {test_folder}")  # Debugging information
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        predict_image(img_path)
else:
    print(f"⚠️ Test folder or image not found at path: {test_folder}. Please provide a valid image or directory.")
