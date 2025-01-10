import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model_path = 'trained-models/nasal_endoscope_classifier.h5'
model = load_model(model_path)

# Define preprocessing function
def preprocess_frame(frame, img_height=224, img_width=224):
    frame_resized = cv2.resize(frame, (img_width, img_height))
    frame_array = img_to_array(frame_resized)
    frame_array = frame_array / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

# Open video file
video_path = 'raw-data/video1.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video.")
        break

    # Preprocess frame
    processed_frame = preprocess_frame(frame)

    # Make prediction
    prediction = model.predict(processed_frame)

    # Interpret prediction
    if prediction[0] > 0.5:
        label = 'Out'
        color = (0, 0, 255)
    else:
        label = 'In'
        color = (0, 255, 0)

    # Overlay label
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display frame
    cv2.imshow('Nasal Endoscope Classification', frame)

    # Exit on 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
