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