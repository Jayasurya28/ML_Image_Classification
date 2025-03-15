from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# ✅ Fix: Disable GPU if not needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Load class names
class_file = 'class_names.json'
if os.path.exists(class_file):
    with open(class_file, 'r') as f:
        categories = json.load(f)
    if isinstance(categories, list):
        categories = {i: name for i, name in enumerate(categories)}
else:
    categories = {i: f"Class {i}" for i in range(model.output_shape[1])}

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.  # Normalize

    predictions = model.predict(img_array)[0]
    max_prob = np.max(predictions)
    predicted_class_index = np.argmax(predictions)

    response = {
        'filename': file.filename,
        'prediction': categories.get(predicted_class_index, "Unknown"),
        'confidence': f"{max_prob:.2%}",
        'probabilities': {categories[i]: float(prob) for i, prob in enumerate(predictions)}
    }

    os.remove(filepath)  # Delete file after processing
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
