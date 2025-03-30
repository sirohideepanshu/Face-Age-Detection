# Face-Age-Detection
// main file 


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





// webcam file


import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Load the trained regression model
model = tf.keras.models.load_model("Face_Age_Detection_Model.h5")

# Constants
IMG_SIZE = (128, 128)  # Input size for the model
MOVING_AVERAGE_WINDOW = 30  # Number of frames for smoothing (increased for better stability)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Queue to store the last few predictions for smoothing
age_predictions = deque(maxlen=MOVING_AVERAGE_WINDOW)

# Function to preprocess the face and predict the age
def preprocess_and_predict(face):
    try:
        # Resize and preprocess the face image
        face = cv2.resize(face, IMG_SIZE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_array = np.array(face) / 255.0  # Normalize pixel values
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

        # Predict the exact age using the regression model
        predicted_age = model.predict(face_array)[0][0]  # Single output for regression
        return predicted_age
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Open the webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Predict age for the detected face
        predicted_age = preprocess_and_predict(face)

        if predicted_age is not None:
            # Add the prediction to the queue
            age_predictions.append(predicted_age)

            # Calculate the moving average for stable output
            smoothed_age = np.mean(age_predictions)

            # Display the smoothed predicted age on the video feed
            label = f"Predicted Age: {smoothed_age:.1f} years"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the video feed with the prediction overlay
    cv2.imshow("Live Age Prediction", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
