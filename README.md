



# **ML-Image-Classification**  

## 📌 **Key Features**  
✅ Accepts image uploads for classification  
✅ Uses a **pre-trained deep learning model** (VGG16)  
✅ Provides **confidence scores** for each prediction  
✅ Implements a **threshold-based prediction** (≥70%)  
✅ **Deployable as a web application** using Flask  

---

## 🛠 **Technologies Used**  
- **TensorFlow & Keras** - Model training & prediction  
- **Flask** - Web framework for deployment  
- **Python** - Backend logic & scripting  
- **NumPy & OpenCV** - Image preprocessing  
- **HTML, CSS, JavaScript** - Frontend (optional for UI)  

---

## 📁 **Project Structure**  
```
ML-Image-Classification/
├── static/                 # Stores static assets (CSS, JS, images)
├── templates/              # HTML templates (if using a web interface)
├── train-images/           # Training dataset (if retraining)
├── validation-images/      # Validation dataset
├── app.py                  # Flask backend (for web app deployment)
├── train_model.py          # Main script for training
├── image_classification.py # Script for model inference
├── class_names.json        # Class labels for predictions
├── trained_model.h5        # Saved TensorFlow model
├── test_image1.jpeg        # Sample test image
├── test_image2.jpeg        # Sample test image
├── requirements.txt        # Required dependencies
├── README.md               # Documentation
```

---

## 🚀 **Setup & Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Jayasurya28/ML_Image_Classification.git
cd ML_Image_Classification
```

### **2️⃣ Create & Activate a Virtual Environment**  
```bash
python3 -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate     # On Windows
```

### **3️⃣ Install Required Dependencies**  
```bash
pip install -r requirements.txt
```

---

## 🏋️ **Training the Model**  
If you want to train the model from scratch, use `train_model.py`:  
```bash
python train_model.py
```
This will:  
✅ Load the dataset  
✅ Train a convolutional neural network (CNN) using **VGG16**  
✅ Save the trained model as **trained_model.h5**  

---

## 🖼 **Running Image Classification**  
Use `image_classification.py` to classify an image:  
```bash
python image_classification.py --image test_image1.jpeg
```
### **Example Output:**  
```plaintext
'test_image2.jpeg' is classified as 'Laptop' with 99.88% confidence.
Backpack: 0.02%
Charger: 0.10%
Headphones: 0.00%
Laptop: 99.88%
Lock.and.key: 0.00%
Water bottle: 0.01%
```

---
🌐 **Running the Flask Web Application**  
To run the web app, use:  
```bash
python app.py
```
Then, open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

---

## 🔗 **GitHub Repository**  
[**ML-Image-Classification**](https://github.com/Jayasurya28/ML_Image_Classification)  

