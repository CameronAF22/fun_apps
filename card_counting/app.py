# app.py
import streamlit as st
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw
import numpy as np
import time
from fastai.vision.all import load_learner

# Function to load models (cached to improve performance)
@st.cache_resource
def load_models():
    # Load object detection model from TensorFlow Hub
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    detection_model = hub.load(model_url)

    # Load card classification model (assuming 'card_classifier_update.pkl' is in the same directory)
    classification_model = load_learner('card_classifier_update.pkl')
    return detection_model, classification_model

# Function to classify card image
def classify_image(img, learner):
    img_resized = img.resize((224, 224))
    pred, idx, probs = learner.predict(img_resized)
    return pred

# Function to calculate card value for count
def card_value(card):
    if any(substring in card for substring in ['two','three','four','five','six']):
        return 1
    elif any(substring in card for substring in ['seven','eight','nine']):
        return 0
    else:
        return -1

# Function to process image and perform card counting
def process_image(image, detection_model, classification_model):
    frame = np.array(image)
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = detection_model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    detection_scores = output_dict['detection_scores'][0].numpy()
    detection_boxes = output_dict['detection_boxes'][0].numpy()
    detection_classes = output_dict['detection_classes'][0].numpy()

    detected_cards = []
    count_change_total = 0
    image_with_boxes = Image.fromarray(frame.copy()) # Create a PIL Image for drawing
    draw = ImageDraw.Draw(image_with_boxes)

    for i in range(num_detections):
        if detection_scores[i] > 0.22 and detection_classes[i] == 1: # Class ID 1 is 'person' in this model, adjust if needed, or ideally fine-tune model for cards directly.
            ymin, xmin, ymax, xmax = detection_boxes[i]
            xmin_real, xmax_real, ymin_real, ymax_real = int(xmin * frame.shape[1]), int(xmax * frame.shape[1]), int(ymin * frame.shape[0]), int(ymax * frame.shape[0])

            # Draw bounding box on PIL Image
            draw.rectangle([(xmin_real, ymin_real), (xmax_real, ymax_real)], outline="green", width=3)

            cropped_img = frame[ymin_real:ymax_real, xmin_real:xmax_real]
            if cropped_img.size > 0: # Check if cropped_img is not empty
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                card_prediction = classify_image(cropped_img_pil, classification_model)
                detected_cards.append(str(card_prediction))

                count_change = card_value(str(card_prediction))
                count_change_total += count_change

    return image_with_boxes, detected_cards, count_change_total

# --- Streamlit App ---
st.title("🃏 Card Counting Computer Vision App")
st.write("Upload an image of playing cards and let the AI identify them and calculate the count!")

# Load models
detection_model, classification_model = load_models()

uploaded_file = st.file_uploader("Upload an image of cards", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    st.write("Processing image...")
    processed_image, detected_cards, count_change_total = process_image(image, detection_model, classification_model)

    st.image(processed_image, caption="Processed Image with Detected Cards", use_column_width=True)

    if detected_cards:
        st.write("### Detected Cards:")
        for card in detected_cards:
            st.write(f"- {card}")
        st.write(f"### Count Change: {count_change_total}")
    else:
        st.write("No cards detected in the image or confidence below threshold.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses computer vision to identify playing cards in an image and calculate a running count based on card counting strategies.\n\n"
    "**How to use:**\n"
    "1. Upload an image containing playing cards.\n"
    "2. The app will identify the cards and display bounding boxes around them.\n"
    "3. The detected cards and the total count change will be shown below the image.\n\n"
    "**Note:** This is for demonstration and educational purposes. Card counting in casinos may be restricted."
)