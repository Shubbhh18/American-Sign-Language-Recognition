import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def preprocess_image(image, output_size=(224, 224), padding=20):
    """
    Enhanced preprocessing for hand detection and segmentation:
    1. Use MediaPipe for more accurate hand detection
    2. Apply adaptive thresholding for better segmentation
    3. Implement ROI extraction with proper padding
    4. Apply normalization for better model input
    """
    # Create a copy of the image for processing
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape
    
    # Process the image with MediaPipe Hands
    results = hands.process(img_rgb)
    
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Get the bounding box of the hand
        hand_landmarks = results.multi_hand_landmarks[0]
        x_min, y_min = width, height
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Add padding to the bounding box
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        # Ensure the bounding box has some minimum size
        if x_max - x_min < 20 or y_max - y_min < 20:
            x_min, y_min = 0, 0
            x_max, y_max = width, height
        
        # Crop the image to the hand region
        cropped = img_rgb[y_min:y_max, x_min:x_max]
    else:
        # Fallback to traditional contour-based method if MediaPipe fails
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use adaptive thresholding for better segmentation
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding to the bounding box
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2 * padding, width - x)
            h = min(h + 2 * padding, height - y)
            
            # Crop the image to the hand region
            cropped = img_rgb[y:y+h, x:x+w]
        else:
            # If no contour is detected, use the full image
            cropped = img_rgb
    
    # Resize the cropped image to the desired output size
    processed = cv2.resize(cropped, output_size)
    
    # Convert back to BGR for OpenCV compatibility if needed
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    
    return processed_bgr