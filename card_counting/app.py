#!/usr/bin/env python
# Disable Streamlit's file watcher and set environment variables before importing anything else
import os
import sys

# Disable Streamlit's module watching completely to avoid PyTorch class conflicts
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "none"

# Import other libraries after setting environment variables
import streamlit as st
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

# Import FastAI and PyTorch after environment variables have been set
# This helps prevent the RuntimeError with torch.classes.__path__._path
from fastai.vision.all import load_learner
import patched_pickle
import torch
torch.classes.__path__ = []
# Function to load models (cached to improve performance)
@st.cache_resource
def load_models():
    # Load object detection model from TensorFlow Hub
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    detection_model = hub.load(model_url)

    # Load card classification model using patched_pickle to handle PosixPath on Windows
    classification_model = load_learner('card_classifier_update.pkl', pickle_module=patched_pickle)
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

# Function to process image and perform card counting with improved detection
def process_image(image, detection_model, classification_model, deck_status, recognized_cards):
    frame = np.array(image)
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = detection_model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    detection_scores = output_dict['detection_scores'][0].numpy()
    detection_boxes = output_dict['detection_boxes'][0].numpy()
    detection_classes = output_dict['detection_classes'][0].numpy()

    # Create a copy of the image for drawing
    image_with_boxes = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image_with_boxes)
    
    # Prepare to track detections
    detected_cards = []
    count_change_total = 0
    cards_counted = 0
    last_card = None

    # Process each detection
    for i in range(num_detections):
        if detection_scores[i] > 0.30:  # Slightly higher threshold for better precision
            # Get bounding box coordinates and scale to frame size
            ymin, xmin, ymax, xmax = detection_boxes[i]
            xmin_real = int(xmin * frame.shape[1])
            xmax_real = int(xmax * frame.shape[1])
            ymin_real = int(ymin * frame.shape[0])
            ymax_real = int(ymax * frame.shape[0])
            
            # Draw bounding box on PIL Image - thinner line for less intrusion
            draw.rectangle([(xmin_real, ymin_real), (xmax_real, ymax_real)], 
                           outline="green", width=2)
            
            # Add "CARD" label above the detection box
            try:
                label_font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                label_font = ImageFont.load_default()
                
            draw.text((xmin_real, ymin_real - 20), "CARD", fill="green", font=label_font)

            # Extract the detected region
            cropped_img = frame[ymin_real:ymax_real, xmin_real:xmax_real]
            if cropped_img.size == 0:  # Skip if crop is empty
                continue
                
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            card_prediction = classify_image(cropped_img_pil, classification_model)
            
            # Only add this card if it's not already recognized
            card_name = str(card_prediction)
            if card_name not in recognized_cards:
                detected_cards.append(card_name)
                recognized_cards.add(card_name)
                last_card = card_name
                
                # Calculate count change
                count_change = card_value(card_name)
                count_change_total += count_change
                
                # Update deck status
                try:
                    if "of" in card_name.lower():
                        parts = card_name.lower().split(" of ")
                        rank_name = parts[0].strip()
                        suit_name = parts[1].strip()
                        
                        # Map rank name to rank symbol
                        rank_map = {
                            'ace': 'A', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
                            'jack': 'J', 'queen': 'Q', 'king': 'K'
                        }
                        
                        # Map suit name to suit symbol
                        suit_map = {
                            'hearts': 'H', 'diamonds': 'D', 'clubs': 'C', 'spades': 'S'
                        }
                        
                        if rank_name in rank_map and suit_name in suit_map:
                            mapped_card = f"{rank_map[rank_name]}{suit_map[suit_name]}"
                            deck_status[mapped_card] = True
                            cards_counted += 1
                except:
                    # If mapping fails, just count without updating deck
                    cards_counted += 1

    return image_with_boxes, detected_cards, count_change_total, cards_counted, last_card

# Initialize deck representation
suits = ['S', 'H', 'D', 'C']  # Using simpler suit symbols
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
full_deck = []
for suit in suits:
    for rank in ranks:
        full_deck.append(f"{rank}{suit}")  # More compact card names

# Function to generate deck visualization using PIL for Streamlit
def draw_deck_pil(deck_status, cards_counted, width=800, height=400):
    # Create a new PIL image for the deck visualization
    img = Image.new('RGB', (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    
    # Card dimensions and layout
    card_width = 35
    card_height = 50
    padding_x = 8
    padding_y = 12
    start_x = 20
    start_y = 70
    
    # Try to load a font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        card_font = ImageFont.truetype("arial.ttf", 14)
        stats_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
        card_font = ImageFont.load_default()
        stats_font = ImageFont.load_default()
    
    # Define colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 200, 0)
    gold = (255, 215, 0)
    gray = (200, 200, 200)
    
    # Draw title
    draw.text((20, 20), "CARD COUNTING DASHBOARD", fill=gold, font=title_font)
    
    # Calculate statistics
    cards_remaining = 52 - cards_counted
    
    # Draw statistics
    draw.text((20, 45), f"Cards Counted: {cards_counted}", fill=white, font=stats_font)
    draw.text((250, 45), f"Cards Remaining: {cards_remaining}", fill=white, font=stats_font)
    
    # Draw each card in the deck
    for i, suit in enumerate(suits):
        # Draw suit header
        suit_labels = {'S': 'SPADES', 'H': 'HEARTS', 'D': 'DIAMONDS', 'C': 'CLUBS'}
        suit_y = start_y + i * (card_height + padding_y + 10)
        
        suit_color = red if suit in ['H', 'D'] else black
        draw.text((start_x, suit_y - 25), suit_labels[suit], fill=white, font=stats_font)
        
        for j, rank in enumerate(ranks):
            card_name = f"{rank}{suit}"
            pos_x = start_x + j * (card_width + padding_x)
            pos_y = suit_y
            
            # Draw card background
            if not deck_status.get(card_name, False):  # Card is in deck
                # Draw white card
                draw.rectangle([(pos_x, pos_y), (pos_x + card_width, pos_y + card_height)], fill=white, outline=gray)
                
                # Draw rank and suit
                text_color = red if suit in ['H', 'D'] else black
                draw.text((pos_x + 5, pos_y + 5), rank, fill=text_color, font=card_font)
                draw.text((pos_x + card_width//2, pos_y + card_height//2), suit, fill=text_color, font=card_font)
            else:  # Card removed from deck
                # Draw gray card with X
                draw.rectangle([(pos_x, pos_y), (pos_x + card_width, pos_y + card_height)], fill=gray, outline=gray)
                # Draw X
                draw.line([(pos_x, pos_y), (pos_x + card_width, pos_y + card_height)], fill=red, width=2)
                draw.line([(pos_x + card_width, pos_y), (pos_x, pos_y + card_height)], fill=red, width=2)
    
    return img

# Function to draw count display for Streamlit
def draw_count_display_pil(count, width=300, height=150):
    # Create a new PIL image for the count display
    img = Image.new('RGB', (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        count_font = ImageFont.truetype("arial.ttf", 72)
        trend_font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        title_font = ImageFont.load_default()
        count_font = ImageFont.load_default()
        trend_font = ImageFont.load_default()
    
    # Draw title
    draw.text((20, 15), "CURRENT COUNT", fill=(255, 215, 0), font=title_font)
    
    # Determine count color based on value
    if count > 0:
        count_color = (0, 255, 0)  # Green for positive
        trend_text = "↑ HIGH CARDS DEPLETED"
    elif count < 0:
        count_color = (255, 0, 0)  # Red for negative
        trend_text = "↓ LOW CARDS DEPLETED"
    else:
        count_color = (255, 255, 255)  # White for zero
        trend_text = "― NEUTRAL"
    
    # Draw count
    draw.text((width//2 - 40, 50), f"{count}", fill=count_color, font=count_font)
    
    # Draw trend indicator
    draw.text((20, 120), trend_text, fill=count_color, font=trend_font)
    
    return img

# Function to draw last detected card for Streamlit
def draw_last_card_pil(card_name, width=300, height=200):
    if not card_name:
        return None  # No card to display
        
    # Create new PIL image
    img = Image.new('RGB', (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    
    # Try to load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        card_font = ImageFont.truetype("arial.ttf", 32)
        info_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
        card_font = ImageFont.load_default()
        info_font = ImageFont.load_default()
    
    # Draw title
    draw.text((20, 15), "LAST DETECTED CARD", fill=(255, 215, 0), font=title_font)
    
    # Try to parse card name and create visual
    try:
        if "of" in card_name.lower():
            parts = card_name.lower().split(" of ")
            rank = parts[0].strip().title()
            suit = parts[1].strip().title()
            
            # Set color based on suit
            if "heart" in suit or "diamond" in suit:
                color = (255, 0, 0)  # Red
            else:
                color = (0, 0, 0)  # Black
                
            # Draw card representation
            card_x = width//2 - 50
            card_y = 50
            card_width = 100
            card_height = 140
            
            # Draw card background
            draw.rectangle([(card_x, card_y), (card_x + card_width, card_y + card_height)], 
                          fill=(255, 255, 255), outline=(150, 150, 150))
            
            # Draw suit and rank
            suit_symbol = ""
            if "heart" in suit:
                suit_symbol = "H"
            elif "diamond" in suit:
                suit_symbol = "D"
            elif "club" in suit:
                suit_symbol = "C"
            elif "spade" in suit:
                suit_symbol = "S"
                
            # Draw rank at top-left
            draw.text((card_x + 10, card_y + 10), rank[0].upper(), fill=color, font=card_font)
            
            # Draw suit symbol in center
            draw.text((card_x + 40, card_y + 60), suit_symbol, fill=color, font=card_font)
            
            # Determine count value
            if any(val in rank.lower() for val in ["two", "three", "four", "five", "six", "2", "3", "4", "5", "6"]):
                count_value = 1
                count_color = (0, 255, 0)  # Green
                count_text = "+1"
            elif any(val in rank.lower() for val in ["seven", "eight", "nine", "7", "8", "9"]):
                count_value = 0
                count_color = (255, 255, 255)  # White
                count_text = "0"
            else:
                count_value = -1
                count_color = (255, 0, 0)  # Red
                count_text = "-1"
                
            # Draw count value
            draw.text((width//2, card_y + card_height + 10), f"Count: {count_text}", 
                     fill=count_color, font=info_font)
    except:
        # If parsing fails, just display the card name
        draw.text((20, height//2), card_name, fill=(255, 255, 255), font=info_font)
    
    return img

# Helper function to calculate IoU for card detection
def calculate_iou(box1, box2):
    # Calculate intersection of union for detection boxes
    y1_max = max(box1[0], box2[0])
    x1_max = max(box1[1], box2[1])
    y2_min = min(box1[2], box2[2])
    x2_min = min(box1[3], box2[3])
    
    inter_area = max(0, y2_min - y1_max) * max(0, x2_min - x1_max)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# --- Streamlit App ---
st.title("🃏 Card Counting Computer Vision App")
st.write("Upload an image of playing cards and let the AI identify them and calculate the count!")

# Initialize session state to maintain persistent card counting
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'deck_status' not in st.session_state:
    st.session_state.deck_status = {card: False for card in full_deck}
if 'recognized_cards' not in st.session_state:
    st.session_state.recognized_cards = set()
if 'total_cards_counted' not in st.session_state:
    st.session_state.total_cards_counted = 0
if 'last_card' not in st.session_state:
    st.session_state.last_card = None

# Add a reset button
if st.button("Reset Card Counting"):
    st.session_state.total_count = 0
    st.session_state.deck_status = {card: False for card in full_deck}
    st.session_state.recognized_cards = set()
    st.session_state.total_cards_counted = 0
    st.session_state.last_card = None
    st.success("Card counting has been reset!")

# Load models
detection_model, classification_model = load_models()

# Display the current count status
col1, col2 = st.columns(2)
with col1:
    # Display current count visualization
    count_display = draw_count_display_pil(st.session_state.total_count)
    st.image(count_display, caption="Current Count Status")

with col2:
    # Display last detected card if available
    if st.session_state.last_card:
        last_card_display = draw_last_card_pil(st.session_state.last_card)
        if last_card_display:
            st.image(last_card_display, caption="Last Detected Card")

# Display deck visualization
deck_viz = draw_deck_pil(st.session_state.deck_status, st.session_state.total_cards_counted)
st.image(deck_viz, caption="Deck Status", use_container_width=True)

# Upload image section
uploaded_file = st.file_uploader("Upload an image of cards", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create a collapsible section for the uploaded image
    with st.expander("View Uploaded Image", expanded=False):
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing cards..."):
        # Process the image with our improved detection
        processed_image, detected_cards, count_change, cards_counted, last_card = process_image(
            image, 
            detection_model, 
            classification_model, 
            st.session_state.deck_status, 
            st.session_state.recognized_cards
        )
        
        # Update session state
        st.session_state.total_count += count_change
        st.session_state.total_cards_counted += cards_counted
        if last_card:
            st.session_state.last_card = last_card

    # Show the processed image with detection boxes
    st.image(processed_image, caption="Detected Cards", use_container_width=True)

    # Display detected cards information
    if detected_cards:
        with st.expander("View Detected Cards", expanded=True):
            st.write("### Newly Detected Cards:")
            for card in detected_cards:
                count_val = card_value(card)
                count_symbol = "+1" if count_val > 0 else ("-1" if count_val < 0 else "0")
                st.write(f"- **{card}** (Count: {count_symbol})")
            
            st.write(f"### Count Change: {count_change}")
            st.write(f"### Total Running Count: {st.session_state.total_count}")
    else:
        st.info("No new cards detected in this image.")
    
    # Update visualizations after processing
    st.rerun()

st.sidebar.header("About")
st.sidebar.info(
    "This app uses computer vision to identify playing cards in an image and calculate a running count based on card counting strategies.\n\n"
    "**How to use:**\n"
    "1. Upload an image containing playing cards.\n"
    "2. The app will identify the cards and display bounding boxes around them.\n"
    "3. The detected cards and the count change will be shown.\n"
    "4. The deck visualization shows which cards have been counted.\n\n"
    "**Card Values:**\n"
    "- 2-6: +1\n"
    "- 7-9: 0\n"
    "- 10-A: -1\n\n"
    "**Note:** This is for demonstration and educational purposes. Card counting in casinos may be restricted."
)