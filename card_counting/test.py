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

# Initialize deck representation
suits = ['S', 'H', 'D', 'C'] # Using simpler suit symbols
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
full_deck = []
for suit in suits:
    for rank in ranks:
        full_deck.append(f"{rank}{suit}") # More compact card names

deck_status = {card: False for card in full_deck} # False means card is in deck, True means removed
cards_counted = 0

# Function to draw the deck on the frame
def draw_deck(frame, deck_status, cards_counted):
    # Background panel for the deck display - reduced to 1/4 size
    panel_start_x, panel_start_y = 450, 20
    panel_width, panel_height = 200, 125  # Reduced from 800x500
    
    # Create a semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_start_x, panel_start_y), 
                 (panel_start_x + panel_width, panel_start_y + panel_height), 
                 (40, 40, 40), -1)
    # Apply the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Card dimensions and layout - much smaller cards
    card_width = 9  # Reduced from 35
    card_height = 13  # Reduced from 50
    padding_x = 2  # Reduced from 8
    padding_y = 3  # Reduced from 12
    cards_per_row = 13  # Number of ranks
    start_x = panel_start_x + 5  # Reduced margin
    start_y = panel_start_y + 20  # Reduced margin
    
    # Reduced visual elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 0.3  # Reduced from 0.8
    card_font_scale = 0.2  # Reduced from 0.5
    stat_font_scale = 0.25  # Reduced from 0.7
    font_thickness = 1  # Reduced from 2
    card_font_thickness = 1
    line_type = cv2.LINE_AA
    
    # Define colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 165, 0)
    gold = (0, 215, 255)
    
    # Calculate statistics
    cards_remaining = 52 - cards_counted
    
    # Draw title and statistics with better formatting - smaller text
    cv2.putText(frame, "CARD COUNTING DASHBOARD", 
                (panel_start_x + 5, panel_start_y + 10), 
                font, title_font_scale, gold, font_thickness, line_type)
    
    # Stats text - without box to save space
    cv2.putText(frame, f"Counted: {cards_counted}", 
                (panel_start_x + 5, panel_start_y + 18), 
                font, stat_font_scale, white, 1, line_type)
    
    cv2.putText(frame, f"Remaining: {cards_remaining}", 
                (panel_start_x + 100, panel_start_y + 18), 
                font, stat_font_scale, white, 1, line_type)
    
    # Define simple ASCII-based suit symbols
    suits_ascii = {'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}
    suit_colors = {'S': black, 'H': red, 'D': red, 'C': black}
    
    # Draw each card in the deck with improved visuals
    for i, suit in enumerate(suits):
        for j, rank in enumerate(ranks):
            card_name = f"{rank}{suit}"
            pos_x = start_x + j * (card_width + padding_x)
            pos_y = start_y + i * (card_height + padding_y + 2)  # Small extra space between suits
            
            # Draw card background (white rectangle)
            if not deck_status[card_name]:
                # Card in deck - draw normal card
                cv2.rectangle(frame, (pos_x, pos_y), 
                             (pos_x + card_width, pos_y + card_height), 
                             white, -1)
                # No border to save space
                
                # Draw rank and suit combined to save space - just show rank for brevity
                text_color = suit_colors[suit]
                
                # Draw rank only to save space
                cv2.putText(frame, rank, 
                           (pos_x + 1, pos_y + 8), 
                           font, card_font_scale, text_color, 
                           card_font_thickness, line_type)
            else:
                # Card removed - simple red X
                cv2.rectangle(frame, (pos_x, pos_y), 
                             (pos_x + card_width, pos_y + card_height), 
                             (200, 200, 200), -1)  # Light gray background
                
                # Draw crossed lines over removed cards
                cv2.line(frame, (pos_x, pos_y), 
                        (pos_x + card_width, pos_y + card_height), 
                        red, 1)  # Thinner line
                cv2.line(frame, (pos_x + card_width, pos_y), 
                        (pos_x, pos_y + card_height), 
                        red, 1)  # Thinner line

def draw_count_display(frame, count):
    # Create a smaller display for the current count value - reduced to 1/4 size
    count_x, count_y = 50, 20
    count_width, count_height = 50, 25  # Reduced from 200x100
    
    # Create a semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (count_x, count_y), 
                 (count_x + count_width, count_y + count_height), 
                 (40, 40, 40), -1)
    
    # Apply the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add title - smaller font
    cv2.putText(frame, "COUNT", 
               (count_x + 5, count_y + 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 215, 255), 
               1, cv2.LINE_AA)
    
    # Determine count color based on value
    if count > 0:
        count_color = (0, 255, 0)  # Green for positive count
        trend_text = "+"
    elif count < 0:
        count_color = (0, 0, 255)  # Red for negative count
        trend_text = "="
    else:
        count_color = (255, 255, 255)  # White for zero
        trend_text = "-"
    
    # Display count in smaller text
    cv2.putText(frame, f"{count}", 
               (count_x + 10, count_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, count_color, 
               1, cv2.LINE_AA)
    
    # Display mini trend indicator
    cv2.putText(frame, trend_text, 
               (count_x + 30, count_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, count_color, 
               1, cv2.LINE_AA)

def draw_last_detected_card(frame, card_name):
    if not card_name:
        return  # No card to display
        
    # Position for the last detected card display - reduced size
    panel_x, panel_y = 50, 50  # Moved up slightly to fit better
    panel_width, panel_height = 50, 45  # Reduced from 200x180
    
    # Create a semi-transparent panel for the last detected card
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (40, 40, 40), -1)
    
    # Apply the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add title - smaller font
    cv2.putText(frame, "LAST CARD", 
               (panel_x + 3, panel_y + 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 215, 255), 
               1, cv2.LINE_AA)
    
    # Try to parse the card name and display it more compactly
    try:
        # Check if card name is in the format from classification results
        if "of" in card_name.lower():
            parts = card_name.lower().split(" of ")
            rank = parts[0].strip().title()
            suit = parts[1].strip().title()
            
            # Set color based on suit
            if "heart" in suit or "diamond" in suit:
                color = (0, 0, 255)  # Red (BGR format)
            else:
                color = (0, 0, 0)    # Black
                
            # Draw smaller card representation
            card_x = panel_x + 5
            card_y = panel_y + 12
            card_width = 25  # Reduced from 100
            card_height = 25  # Reduced proportionally
            
            # Draw card background
            cv2.rectangle(frame, 
                         (card_x, card_y), 
                         (card_x + card_width, card_y + card_height), 
                         (255, 255, 255), -1)  # White card
            
            # No border to save space
            
            # Get abbreviated versions for compact display
            rank_abbr = rank[0].upper()  # Just first letter
            
            # Use ASCII suit symbol
            suit_symbol = ""
            if "heart" in suit:
                suit_symbol = "H"
            elif "diamond" in suit:
                suit_symbol = "D"
            elif "club" in suit:
                suit_symbol = "C"
            elif "spade" in suit:
                suit_symbol = "S"
            
            # Draw compact rank+suit in center
            cv2.putText(frame, f"{rank_abbr}{suit_symbol}", 
                       (card_x + 5, card_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 
                       1, cv2.LINE_AA)
            
            # Determine count value based on card name
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
                count_color = (0, 0, 255)  # Red
                count_text = "-1"
            
            # Display small count value
            cv2.putText(frame, count_text, 
                       (panel_x + 35, panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, count_color, 
                       1, cv2.LINE_AA)
    except:
        # If parsing fails, display abbreviated card name
        short_name = card_name[:10] if len(card_name) > 10 else card_name
        cv2.putText(frame, short_name, 
                   (panel_x + 3, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 
                   1, cv2.LINE_AA)

# Helper function to calculate IoU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: Each box is represented as [y1, x1, y2, x2] where (x1, y1) is the top-left corner
                and (x2, y2) is the bottom-right corner.
    
    Returns:
    iou: IoU value between 0 and 1
    """
    # Determine the coordinates of the intersection rectangle
    y1_max = max(box1[0], box2[0])
    x1_max = max(box1[1], box2[1])
    y2_min = min(box1[2], box2[2])
    x2_min = min(box1[3], box2[3])
    
    # Calculate area of intersection rectangle
    inter_area = max(0, y2_min - y1_max) * max(0, x2_min - x1_max)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

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

# Initialize detection and tracking variables
count = 0
last_card = None
recognized_cards = set() # Keep track of recognized cards to remove from deck display
previous_detections = [] # Track previous detection boxes
min_confidence = 0.53 # Minimum confidence threshold for object detection
active_card_tracking = {} # Dictionary to track active cards {tracking_id: (last_seen_time, card_name)}
next_tracking_id = 0 # Unique ID counter for tracked objects
card_cooldown_period = 20 # Don't recount the same card for this many seconds after it's been counted

print("Starting camera feed. Press 'q' in the window to quit.")

# Create a named window and set it to NORMAL to allow resizing
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
# Set a larger initial window size
cv2.resizeWindow('Frame', 1600, 900) # Increased from 1280x720 to 1600x900

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
    
    # Prepare to track detections in this frame
    current_detections = []
    current_frame_time = time.time()
    
    # Clean up old tracking entries 
    # (remove cards that haven't been seen for a while)
    obsolete_ids = []
    for tracking_id, (last_seen, _) in active_card_tracking.items():
        if current_frame_time - last_seen > 5:  # If not seen for 5 seconds
            obsolete_ids.append(tracking_id)
    
    for tracking_id in obsolete_ids:
        del active_card_tracking[tracking_id]

    # Loop over each detection
    for i in range(num_detections):
        score = output_dict['detection_scores'][0][i].numpy()
        if score > min_confidence:
            # Get bounding box coordinates and scale to frame size
            bbox = output_dict['detection_boxes'][0][i].numpy()
            ymin, xmin, ymax, xmax = bbox
            ymin_pixel = int(ymin * frame.shape[0])
            xmin_pixel = int(xmin * frame.shape[1])
            ymax_pixel = int(ymax * frame.shape[0])
            xmax_pixel = int(xmax * frame.shape[1])
            
            # Store the detection for tracking
            detection_box = [ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel]
            current_detections.append(detection_box)
            
            # Try to match this detection with previously tracked cards
            matched_id = None
            for tracking_id, (last_seen, card_name) in active_card_tracking.items():
                # Find the previous detection for this tracking ID
                is_match = False
                
                # Calculate IoU with all previous detections to find matches
                for prev_box in previous_detections:
                    iou = calculate_iou(detection_box, prev_box)
                    if iou > 0.5:  # If IoU is high enough, consider it the same card
                        is_match = True
                        matched_id = tracking_id
                        break
            
            # If this is a new detection, classify it and assign a new tracking ID
            if matched_id is None:
                # Crop the detected region and convert to PIL Image for classification
                cropped_img = frame[ymin_pixel:ymax_pixel, xmin_pixel:xmax_pixel]
                if cropped_img.size == 0:
                    continue  # Skip if crop is empty
                
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                cropped_img = Image.fromarray(cropped_img)
                
                # Classify the new card
                card = classify_image(cropped_img)
                
                # Create a new tracking ID for this card
                tracking_id = next_tracking_id
                next_tracking_id += 1
                active_card_tracking[tracking_id] = (current_frame_time, card)
                
                # Update count if this is a new card
                if card not in recognized_cards:
                    print(f"Detected new card: {card}")
                    count_change = card_value(card)
                    count += count_change
                    last_card = card
                    recognized_cards.add(card)
                    
                    # Map the card name to the deck representation format
                    try:
                        if "of" in card.lower():
                            parts = card.lower().split(" of ")
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
                        else:
                            # If it's a joker or other special card
                            cards_counted += 1
                    except:
                        # If mapping fails, just count the card without updating deck status
                        cards_counted += 1
            else:
                # Update the last seen time for this tracked card
                last_seen, card_name = active_card_tracking[matched_id]
                active_card_tracking[matched_id] = (current_frame_time, card_name)
            
            # Define card color for the detection box
            card_color = (0, 200, 0)  # Default green for all cards
            
            # Draw a smaller rectangle around the detected object
            cv2.rectangle(frame, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), card_color, 1)  # Thinner line
            
            # Add a smaller label above the detection box
            label_text = "CARD"
            cv2.putText(frame, label_text, (xmin_pixel, ymin_pixel - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, card_color, 1, cv2.LINE_AA)  # Smaller font
    
    # Update the previous detections list for the next frame
    previous_detections = current_detections

    # First draw a dark semi-transparent background for the entire frame
    dashboard_overlay = frame.copy()
    cv2.rectangle(dashboard_overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    # Add a slight darkening effect to highlight the visualizations
    cv2.addWeighted(dashboard_overlay, 0.2, frame, 0.8, 0, frame)

    # Create a smaller title bar at the top
    title_height = 20  # Reduced from 40
    cv2.rectangle(frame, (0, 0), (frame.shape[1], title_height), (40, 40, 40), -1)
    cv2.putText(frame, "CARD COUNTING VISION SYSTEM", (frame.shape[1]//2 - 150, 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)  # Smaller font

    # Display the current count using our visualization
    draw_count_display(frame, count)
    
    # Display the last detected card
    draw_last_detected_card(frame, last_card)
    
    # Draw the deck visualization
    draw_deck(frame, deck_status, cards_counted)
    
    # Add a smaller footer with instructions
    footer_y = frame.shape[0] - 15  # Moved up from 30
    cv2.putText(frame, "Press 'q' to quit", (20, footer_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)  # Smaller font

    # Show the frame in a window
    cv2.imshow('Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release the camera and close all windows
cap.release()
cv2.destroyAllWindows()