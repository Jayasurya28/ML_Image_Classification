import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import json
import warnings
import subprocess

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure Pillow is installed
try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

# Disable GPU (to avoid CUDA errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)  # Flatten the features from VGG16
x = Dense(128, activation='relu')(x)  # Apply Dense layer correctly
x = Dense(6, activation='softmax')(x)  # Make sure softmax is applied to `x`

# Create model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define dataset paths (Windows format)
train_dir = os.path.join(r"D:\Surya Respositories\ML_Image_Classification\train-images")
validation_dir = os.path.join(r"D:\Surya Respositories\ML_Image_Classification\validation-images")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=10
)

# Save the trained model
model.save('trained_model.h5')
print("Model saved as 'trained_model.h5'")

# Save class names
class_names = list(train_generator.class_indices.keys())
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Class names saved in 'class_names.json'")
