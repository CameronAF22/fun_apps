import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from fastai.vision.all import load_learner
from fastai.vision.all import *
import patched_pickle
learner = load_learner('card_classifier_update.pkl', pickle_module=patched_pickle)

# Load the object detection model from TF Hub
detector_model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(detector_model_url)

# Load your FastAI card classifier model (ensure 'card_classifier_update.pkl' is in your working directory)
# learner = load_learner('card_classifier_update.pkl')

# Define card categories (if needed later)
categories = (
    'Ace of clubs', 'Ace of diamonds', 'ace of hearts', 'ace of spades',
    'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
    'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
    'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
    'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
    'joker', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
    'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades',
    'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades',
    'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
    'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades',
    'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades',
    'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades',
    'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades'
)

# Function to classify an image of a card using the FastAI learner
def classify_image(img):
    # Resize the image to 224x224 as expected by the classifier
    img = img.resize((224, 224))
    pred, idx, probs = learner.predict(img)
    return str(pred)

# Function to determine count change based on card name
def card_value(card):
    card_lower = card.lower()
    if any(sub in card_lower for sub in ['two', 'three', 'four', 'five', 'six']):
        print(f"Card: {card} Count change: 1")
        return 1
    elif any(sub in card_lower for sub in ['seven', 'eight', 'nine']):
        print(f"Card: {card} Count change: 0")
        return 0
    else:
        print(f"Card: {card} Count change: -1")
        return -1

# Try different camera indices to find a working camera
cap = None
for i in range(5):  # Try indices 0 to 4
    cap_temp = cv2.VideoCapture(i)
    if cap_temp.isOpened():
        print(f"Using camera index: {i}")
        cap = cap_temp
        break
    else:
        cap_temp.release()

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

# Initialize counting variables
count = 0
last_prediction_time = time.time()
last_card = None

print("Starting camera feed. Press 'q' in the window to quit.")

while True:
    # Capture a frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to tensor and add batch dimension for the TF Hub model
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run object detection
    output_dict = detector(input_tensor)

    # Get the number of detections (the model outputs a dict with detection info)
    num_detections = int(output_dict.pop('num_detections'))

    # Loop over each detection
    for i in range(num_detections):
        score = output_dict['detection_scores'][0][i].numpy()
        if score > 0.22:
            # Get bounding box coordinates and scale to frame size
            bbox = output_dict['detection_boxes'][0][i].numpy()
            ymin, xmin, ymax, xmax = bbox
            ymin = int(ymin * frame.shape[0])
            xmin = int(xmin * frame.shape[1])
            ymax = int(ymax * frame.shape[0])
            xmax = int(xmax * frame.shape[1])

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Crop the detected region and convert to PIL Image for classification
            cropped_img = frame[ymin:ymax, xmin:xmax]
            if cropped_img.size == 0:
                continue  # Skip if crop is empty
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            cropped_img = Image.fromarray(cropped_img)

            # Classify the card every 4 seconds
            if time.time() - last_prediction_time > 4:
                card = classify_image(cropped_img)
                if card != last_card:
                    print(f"Detected card: {card}")
                    count_change = card_value(card)
                    count += count_change
                    last_card = card
                last_prediction_time = time.time()

    # Display the current count on the frame
    cv2.putText(frame, f"Count: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame in a window
    cv2.imshow('Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release the camera and close all windows
cap.release()
cv2.destroyAllWindows()