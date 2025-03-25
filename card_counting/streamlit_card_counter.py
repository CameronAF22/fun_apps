import streamlit as st
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from fastai.vision.all import load_learner
from fastai.vision.all import *
import patched_pickle  # Ensure patched_pickle is installed if needed
import torch
torch.classes.__path__ = []

st.set_page_config(
    page_title="Card Counting Vision System",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global deck variables
suits = ['S', 'H', 'D', 'C']  # Using simpler suit symbols
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
full_deck = []
for suit in suits:
    for rank in ranks:
        full_deck.append(f"{rank}{suit}")  # More compact card names

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    """Loads the card classifier and object detection models."""
    learner = load_learner('card_classifier_update.pkl', pickle_module=patched_pickle)
    detector_model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    detector = hub.load(detector_model_url)
    return learner, detector

learner, detector = load_models()

# Card categories (for reference, not directly used in core logic but can be helpful for expansion)
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

# --- Functions from your script ---
def classify_image(img):
    """Classifies a card image using the FastAI learner."""
    img = img.resize((224, 224))
    pred, idx, probs = learner.predict(img)
    return str(pred)

def card_value(card):
    """Determines the card counting value of a card."""
    card_lower = card.lower()
    if any(sub in card_lower for sub in ['two', 'three', 'four', 'five', 'six']):
        return 1
    elif any(sub in card_lower for sub in ['seven', 'eight', 'nine']):
        return 0
    else:
        return -1

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    y1_max = max(box1[0], box2[0])
    x1_max = max(box1[1], box2[1])
    y2_min = min(box1[2], box2[2])
    x2_min = min(box1[3], box2[3])
    inter_area = max(0, y2_min - y1_max) * max(0, x2_min - x1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box1[1]) # Corrected typo here
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# --- Deck and Display Functions ---
def initialize_deck():
    """Initializes the deck status."""
    suits = ['S', 'H', 'D', 'C']
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    full_deck = [f"{rank}{suit}" for suit in suits for rank in ranks]
    return {card: False for card in full_deck}

def draw_deck(frame, deck_status, cards_counted):
    """Draws the deck visualization on the frame."""
    panel_start_x, panel_start_y = 450, 20
    panel_width, panel_height = 200, 125
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_start_x, panel_start_y),
                 (panel_start_x + panel_width, panel_start_y + panel_height),
                 (40, 40, 40), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    card_width = 9
    card_height = 13
    padding_x = 2
    padding_y = 3
    cards_per_row = 13
    start_x = panel_start_x + 5
    start_y = panel_start_y + 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.3
    card_font_scale = 0.2
    stat_font_scale = 0.25
    font_thickness = 1
    card_font_thickness = 1
    line_type = cv2.LINE_AA

    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 165, 0)
    gold = (0, 215, 255)

    cards_remaining = 52 - cards_counted

    cv2.putText(frame, "CARD COUNTING DASHBOARD",
                (panel_start_x + 5, panel_start_y + 10),
                font, title_font_scale, gold, font_thickness, line_type)

    cv2.putText(frame, f"Counted: {cards_counted}",
                (panel_start_x + 5, panel_start_y + 18),
                font, stat_font_scale, white, 1, line_type)

    cv2.putText(frame, f"Remaining: {cards_remaining}",
                (panel_start_x + 100, panel_start_y + 18),
                font, stat_font_scale, white, 1, line_type)

    suits_ascii = {'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
    suit_colors = {'S': black, 'H': red, 'D': red, 'C': black}

    for i, suit in enumerate(suits):
        for j, rank in enumerate(ranks):
            card_name = f"{rank}{suit}"
            pos_x = start_x + j * (card_width + padding_x)
            pos_y = start_y + i * (card_height + padding_y + 2)

            if not deck_status[card_name]:
                cv2.rectangle(frame, (pos_x, pos_y),
                             (pos_x + card_width, pos_y + card_height),
                             white, -1)
                text_color = suit_colors[suit]
                cv2.putText(frame, rank,
                           (pos_x + 1, pos_y + 8),
                           font, card_font_scale, text_color,
                           card_font_thickness, line_type)
            else:
                cv2.rectangle(frame, (pos_x, pos_y),
                             (pos_x + card_width, pos_y + card_height),
                             (200, 200, 200), -1)
                cv2.line(frame, (pos_x, pos_y),
                        (pos_x + card_width, pos_y + card_height),
                        red, 1)
                cv2.line(frame, (pos_x + card_width, pos_y),
                        (pos_x, pos_y + card_height),
                        red, 1)

def draw_count_display(frame, count):
    """Draws the count display on the frame."""
    count_x, count_y = 50, 20
    count_width, count_height = 50, 25
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (count_x, count_y),
                 (count_x + count_width, count_y + count_height),
                 (40, 40, 40), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "COUNT",
               (count_x + 5, count_y + 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 215, 255),
               1, cv2.LINE_AA)

    if count > 0:
        count_color = (0, 255, 0)
        trend_text = "+"
    elif count < 0:
        count_color = (0, 0, 255)
        trend_text = "=" # Corrected to "=" to indicate negative trend
    else:
        count_color = (255, 255, 255)
        trend_text = "-"

    cv2.putText(frame, f"{count}",
               (count_x + 10, count_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, count_color,
               1, cv2.LINE_AA)

    cv2.putText(frame, trend_text,
               (count_x + 30, count_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, count_color,
               1, cv2.LINE_AA)

def draw_last_detected_card(frame, card_name):
    """Draws the last detected card display on the frame."""
    if not card_name:
        return

    panel_x, panel_y = 50, 50
    panel_width, panel_height = 50, 45
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (40, 40, 40), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "LAST CARD",
               (panel_x + 3, panel_y + 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 215, 255),
               1, cv2.LINE_AA)

    try:
        if "of" in card_name.lower():
            parts = card_name.lower().split(" of ")
            rank = parts[0].strip().title()
            suit = parts[1].strip().title()

            if "heart" in suit or "diamond" in suit:
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)

            card_x = panel_x + 5
            card_y = panel_y + 12
            card_width = 25
            card_height = 25

            cv2.rectangle(frame,
                         (card_x, card_y),
                         (card_x + card_width, card_y + card_height),
                         (255, 255, 255), -1)

            rank_abbr = rank[0].upper()

            suit_symbol = ""
            if "heart" in suit:
                suit_symbol = "H"
            elif "diamond" in suit:
                suit_symbol = "D"
            elif "club" in suit:
                suit_symbol = "C"
            elif "spade" in suit:
                suit_symbol = "S"

            cv2.putText(frame, f"{rank_abbr}{suit_symbol}",
                       (card_x + 5, card_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
                       1, cv2.LINE_AA)

            if any(val in rank.lower() for val in ["two", "three", "four", "five", "six", "2", "3", "4", "5", "6"]):
                count_value = 1
                count_color = (0, 255, 0)
                count_text = "+1"
            elif any(val in rank.lower() for val in ["seven", "eight", "nine", "7", "8", "9"]):
                count_value = 0
                count_color = (255, 255, 255)
                count_text = "0"
            else:
                count_value = -1
                count_color = (0, 0, 255)
                count_text = "-1"

            cv2.putText(frame, count_text,
                       (panel_x + 35, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, count_color,
                       1, cv2.LINE_AA)
    except:
        short_name = card_name[:10] if len(card_name) > 10 else card_name
        cv2.putText(frame, short_name,
                   (panel_x + 3, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                   1, cv2.LINE_AA)


# --- Streamlit App ---
st.title("🃏 Real-Time Card Counting with Computer Vision")
st.markdown("Point your camera at playing cards to track the count in real-time. This app uses computer vision to identify cards and assist in card counting strategies.")

# Add an improved About section in the sidebar
st.sidebar.markdown("## About This App")
st.sidebar.markdown("""
This application uses computer vision and machine learning to:
1. **Detect playing cards** through your camera
2. **Identify the rank and suit** of each card
3. **Track card counting values** in real-time
4. **Visualize deck state** showing which cards have been counted

### Card Counting Strategy
The app uses the Hi-Lo card counting system:
- **Low cards (2-6):** +1 (favorable for the player when removed)
- **Neutral cards (7-9):** 0 (no effect on odds)
- **High cards (10-A):** -1 (favorable for the dealer when removed)

### How To Use
1. Click "Start Card Counting"
2. Show cards to your camera
3. The app will identify cards and update the count
4. Use "Reset Counting" to start a new session

**Note:** This app is for educational purposes only. Card counting may be restricted in casinos.

### Created: March 2025
""")

# Improved Settings section
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
min_confidence = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.62, 
                                  help="Adjust how sensitive the card detector is. Higher values mean fewer false positives but might miss some cards.")

# Initialize session state variables more completely
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'last_card' not in st.session_state:
    st.session_state['last_card'] = None
if 'deck_status' not in st.session_state:
    st.session_state['deck_status'] = initialize_deck()
if 'recognized_cards' not in st.session_state:
    st.session_state['recognized_cards'] = set()
if 'previous_detections' not in st.session_state:
    st.session_state['previous_detections'] = []
if 'active_card_tracking' not in st.session_state:
    st.session_state['active_card_tracking'] = {}
if 'next_tracking_id' not in st.session_state:
    st.session_state['next_tracking_id'] = 0
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = False
if 'stop_button_pressed' not in st.session_state:
    st.session_state['stop_button_pressed'] = False
if 'cap' not in st.session_state:
    st.session_state['cap'] = None
if 'cards_counted' not in st.session_state:
    st.session_state['cards_counted'] = 0

# Calculate cards counted
st.session_state['cards_counted'] = len([card for card, removed in st.session_state['deck_status'].items() if removed])

# Add Reset button with proper functionality
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Reset Counting", key="reset_button", use_container_width=True):
        # Reset all counting state
        st.session_state['count'] = 0
        st.session_state['last_card'] = None
        st.session_state['deck_status'] = initialize_deck()
        st.session_state['recognized_cards'] = set()
        st.session_state['cards_counted'] = 0
        st.success("Card counting has been reset! All cards are back in the deck.")

# Display current count statistics
col1, col2 = st.columns(2)
with col1:
    st.metric("Current Count", st.session_state['count'], 
              delta="High Cards" if st.session_state['count'] > 0 else ("Low Cards" if st.session_state['count'] < 0 else "Neutral"),
              delta_color="normal")
with col2:
    st.metric("Cards Counted", st.session_state['cards_counted'], 
              delta=f"{52 - st.session_state['cards_counted']} remaining",
              delta_color="off")

# Display last detected card if available
if st.session_state['last_card']:
    st.info(f"**Last Card Detected:** {st.session_state['last_card']}") 

# Streamlit image placeholders
frame_placeholder = st.empty()

# Improved camera control buttons
camera_col1, camera_col2 = st.columns(2)
with camera_col1:
    start_button = st.button("🎬 Start Card Counting", type="primary", use_container_width=True, disabled=st.session_state['camera_running'])
with camera_col2:
    stop_button = st.button("🛑 Stop Card Counting", type="secondary", use_container_width=True, disabled=not st.session_state['camera_running'])

# Clean up existing camera if needed
def cleanup_camera():
    if st.session_state['cap'] is not None and st.session_state['cap'].isOpened():
        st.session_state['cap'].release()
        st.session_state['cap'] = None
        st.session_state['camera_running'] = False

# Handle Stop button
if stop_button:
    cleanup_camera()
    st.session_state['stop_button_pressed'] = True
    st.success("Camera stopped successfully.")
    st.rerun()

# Handle Start button with improved camera handling
if start_button:
    # Clean up any existing camera first
    cleanup_camera()
    
    # Initialize camera
    cap = None
    for i in range(5):
        cap_temp = cv2.VideoCapture(i)
        if cap_temp.isOpened():
            cap = cap_temp
            st.session_state['cap'] = cap
            st.session_state['camera_running'] = True
            break
        else:
            cap_temp.release()

    if cap is None or not cap.isOpened():
        st.error("Could not open camera. Please check camera connection.")
    else:
        st.success("Camera started. Counting cards...")
        count = st.session_state['count']
        last_card = st.session_state['last_card']
        recognized_cards = st.session_state['recognized_cards']
        previous_detections = st.session_state['previous_detections']
        active_card_tracking = st.session_state['active_card_tracking']
        next_tracking_id = st.session_state['next_tracking_id']
        deck_status = st.session_state['deck_status']
        cards_counted = st.session_state['cards_counted']

        st.session_state['stop_button_pressed'] = False

        while not st.session_state['stop_button_pressed']:
            try:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error reading frame from camera.")
                    cleanup_camera()
                    st.rerun()
                    break

                # Process frame for card detection
                input_tensor = tf.convert_to_tensor(frame)
                input_tensor = input_tensor[tf.newaxis, ...]
                output_dict = detector(input_tensor)
                num_detections = int(output_dict.pop('num_detections'))

                current_detections = []
                current_frame_time = time.time()

                # Clean up obsolete tracking entries
                obsolete_ids = []
                for tracking_id, (last_seen, _) in active_card_tracking.items():
                    if current_frame_time - last_seen > 5:
                        obsolete_ids.append(tracking_id)
                for tracking_id in obsolete_ids:
                    del active_card_tracking[tracking_id]

                # Process detections
                for i in range(num_detections):
                    score = output_dict['detection_scores'][0][i].numpy()
                    if score > min_confidence:
                        bbox = output_dict['detection_boxes'][0][i].numpy()
                        ymin, xmin, ymax, xmax = bbox
                        ymin_pixel = int(ymin * frame.shape[0])
                        xmin_pixel = int(xmin * frame.shape[1])
                        ymax_pixel = int(ymax * frame.shape[0])
                        xmax_pixel = int(xmax * frame.shape[1])

                        detection_box = [ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel]
                        current_detections.append(detection_box)

                        # Check if this matches an existing tracked card
                        matched_id = None
                        for tracking_id, (last_seen, card_name) in active_card_tracking.items():
                            is_match = False
                            for prev_box in previous_detections:
                                iou = calculate_iou(detection_box, prev_box)
                                if iou > 0.5:
                                    is_match = True
                                    matched_id = tracking_id
                                    break

                        # Handle new card detection
                        if matched_id is None:
                            # Extract and classify the card
                            cropped_img = frame[ymin_pixel:ymax_pixel, xmin_pixel:xmax_pixel]
                            if cropped_img.size == 0:
                                continue
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                            cropped_img = Image.fromarray(cropped_img)
                            card = classify_image(cropped_img)

                            # Track the new card
                            tracking_id = next_tracking_id
                            next_tracking_id += 1
                            active_card_tracking[tracking_id] = (current_frame_time, card)

                            # Process new card if not seen before
                            if card not in recognized_cards:
                                count_change = card_value(card)
                                count += count_change
                                last_card = card
                                recognized_cards.add(card)

                                # Update deck status with the new card
                                try:
                                    if "of" in card.lower():
                                        parts = card.lower().split(" of ")
                                        rank_name = parts[0].strip()
                                        suit_name = parts[1].strip()
                                        
                                        # Map card name to deck representation
                                        rank_map = {
                                            'ace': 'A', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                                            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
                                            'jack': 'J', 'queen': 'Q', 'king': 'K'
                                        }
                                        suit_map = {
                                            'hearts': 'H', 'diamonds': 'D', 'clubs': 'C', 'spades': 'S'
                                        }

                                        if rank_name in rank_map and suit_name in suit_map:
                                            mapped_card = f"{rank_map[rank_name]}{suit_map[suit_name]}"
                                            deck_status[mapped_card] = True
                                            cards_counted += 1
                                    else:
                                        # Handle jokers or other special cards
                                        cards_counted += 1
                                except:
                                    # If mapping fails, just count the card
                                    cards_counted += 1
                        else:
                            # Update tracking data for existing card
                            last_seen, card_name = active_card_tracking[matched_id]
                            active_card_tracking[matched_id] = (current_frame_time, card_name)

                        # Draw bounding box for the card
                        card_color = (0, 200, 0)
                        cv2.rectangle(frame, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), card_color, 1)
                        label_text = "CARD"
                        cv2.putText(frame, label_text, (xmin_pixel, ymin_pixel - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, card_color, 1, cv2.LINE_AA)

                # Update previous detections for next frame
                previous_detections = current_detections

                # Draw dashboard overlay
                dashboard_overlay = frame.copy()
                cv2.rectangle(dashboard_overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
                cv2.addWeighted(dashboard_overlay, 0.2, frame, 0.8, 0, frame)

                # Draw title bar
                title_height = 20
                cv2.rectangle(frame, (0, 0), (frame.shape[1], title_height), (40, 40, 40), -1)
                cv2.putText(frame, "CARD COUNTING VISION SYSTEM", (frame.shape[1]//2 - 150, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)

                # Draw visualizations
                draw_count_display(frame, count)
                draw_last_detected_card(frame, last_card)
                draw_deck(frame, deck_status, cards_counted)

                # Draw footer
                footer_y = frame.shape[0] - 15
                cv2.putText(frame, "Press 'Stop Card Counting' to quit", (20, footer_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

                # Display the frame
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                # Update session state
                st.session_state['count'] = count
                st.session_state['last_card'] = last_card
                st.session_state['deck_status'] = deck_status
                st.session_state['recognized_cards'] = recognized_cards
                st.session_state['previous_detections'] = previous_detections
                st.session_state['active_card_tracking'] = active_card_tracking
                st.session_state['next_tracking_id'] = next_tracking_id
                st.session_state['cards_counted'] = cards_counted
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                cleanup_camera()
                st.rerun()
                break

        # Clean up after stopping
        cleanup_camera()

# Add helpful instructions panel
with st.expander("📋 Instructions for Best Results", expanded=False):
    st.markdown("""
    ### Tips for Better Card Detection
    
    1. **Lighting**: Ensure good, consistent lighting on the cards
    2. **Placement**: Hold cards flat and fully visible to the camera
    3. **Movement**: Move cards slowly into view for better detection
    4. **Background**: Use a solid, contrasting background behind cards
    5. **Distance**: Keep cards within 1-2 feet of your camera
    
    If cards aren't being detected properly, try adjusting the confidence threshold in the settings panel.
    """)

# Improved footer with version info
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Developed by Cameron Cubra")
    st.markdown("Version 1.1.0 | March 2025")
with col2:
    st.markdown("### Technologies Used")
    st.markdown("TensorFlow, OpenCV, FastAI, Streamlit")