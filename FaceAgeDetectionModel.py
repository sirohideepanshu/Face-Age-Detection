import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import get_file
import zipfile

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 8  # Assuming 8 age groups (e.g., 0-10, 11-20, etc.)

# Download and extract dataset
def download_and_prepare_dataset():
    dataset_url = "https://data.deepai.org/UTKFace.zip"
    dataset_path = get_file("UTKFace.zip", origin=dataset_url, extract=False)
    extract_folder = os.path.splitext(dataset_path)[0]
    
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
    
    return os.path.join(extract_folder, "UTKFace")

# Prepare dataset for training
def prepare_dataset(image_folder):
    images = []
    labels = []
    
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        try:
            age = int(image_file.split("_")[0])  # Extract age from filename
            age_group = min(age // 10, NUM_CLASSES - 1)  # Categorize into age groups
            image = Image.open(image_path).resize(IMG_SIZE).convert("RGB")
            images.append(np.array(image) / 255.0)  # Normalize pixel values
            labels.append(age_group)
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Load dataset
image_folder = "C:\Users\Deepanshu Sirohi\Downloads\data set"
images, labels = prepare_dataset(image_folder)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model creation
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

data_gen.fit(X_train)

# Training the model
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the model
model.save("face_age_detection_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")