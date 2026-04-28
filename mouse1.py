import cv2
import keyboard
import pyautogui
import time
import numpy as np


# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully!")

    mp_drawing = mp.solutions.drawing_utils # 🌟 Added for drawing landmarks
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0, # Changed from 0 to 1 for slightly better accuracy, though 0 is faster
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available. Using OpenCV fallback.")
    print("💡 For full features, install Python 3.11/3.12 and mediapipe.")

    # ❌ CORRECTION: The original code used 'haarcascade_frontalface_default.xml'
    # for the fallback. This is for face detection, not hands. 
    # Since OpenCV does not have a reliable hand cascade, the fallback 
    # is fundamentally flawed for "hand tracking".
    # I've kept the face cascade as a placeholder to avoid an error, but 
    # added a note that this section is NOT for hands.
    hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Screen size
screen_w, screen_h = pyautogui.size()

# Smooth movement
prev_x, prev_y = screen_w / 2, screen_h / 2 # 🌟 Initialized to center screen
smooth = 5

# Click sensitivity/delay
CLICK_DISTANCE_THRESHOLD = 35
CLICK_COOLDOWN = 0.3 # 🌟 Increased cooldown to prevent multiple accidental clicks

# Start camera
cap = cv2.VideoCapture(0)
# 🌟 CORRECTION: Set a fixed resolution for reliable frame processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not found or cannot be accessed.")
    exit()

print("👉 Virtual Mouse running. Press ESC or Q to exit.")

while True:
    # 🌟 CORRECTION: Use cv2.waitKey() for both exit conditions for better flow
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Exit key 'q' pressed.")
        break
    if keyboard.is_pressed("esc"):
        print("Exiting via 'esc'...")
        break

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    # ❌ REMOVED: Redundant resize, as the camera resolution is set to 640x480
    small = frame 

    if MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Set frame as not writeable for optimization
        small.flags.writeable = False 
        results = hands.process(rgb)
        small.flags.writeable = True

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = small.shape

            # Index fingertip (landmark 8) for cursor position
            ix, iy = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w), \
                     int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            
            # Thumb fingertip (landmark 4) for click detection
            tx, ty = int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w), \
                     int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)
            
            # 1. Cursor Movement (Mapping to screen coordinates)
            scr_x = np.interp(ix, [w // 4, w * 3 // 4], [0, screen_w]) # 🌟 Improved mapping
            scr_y = np.interp(iy, [h // 4, h * 3 // 4], [0, screen_h]) # 🌟 to a central zone

            cur_x = prev_x + (scr_x - prev_x) / smooth
            cur_y = prev_y + (scr_y - prev_y) / smooth
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # 2. Click Detection (Distance between Index and Thumb)
            dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5
            
            # 🌟 CORRECTION: Use the drawing utility for clearer visualization
            mp_drawing.draw_landmarks(small, hand, mp_hands.HAND_CONNECTIONS)
            cv2.circle(small, (ix, iy), 10, (255, 0, 255), -1) # Highlight index tip

            if dist < CLICK_DISTANCE_THRESHOLD:
                # 🌟 Added visual feedback for click
                cv2.circle(small, (ix, iy), 15, (0, 0, 255), 2)
                pyautogui.click()
                time.sleep(CLICK_COOLDOWN) # 🌟 Use cooldown for debouncing

    else: # OpenCV Fallback (Face Detection only)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        objects = hand_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)) # 🌟 Adjusted parameters

        if len(objects) > 0:
            # Sort by area and take the largest (assumed to be the most prominent 'object')
            x, y, w_obj, h_obj = max(objects, key=lambda a: a[2] * a[3])
            cx, cy = x + w_obj // 2, y + h_obj // 2

            scr_x = int(screen_w * cx / small.shape[1])
            scr_y = int(screen_h * cy / small.shape[0])

            cur_x = prev_x + (scr_x - prev_x) / smooth
            cur_y = prev_y + (scr_y - prev_y) / smooth
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            cv2.rectangle(small, (x, y), (x + w_obj, y + h_obj), (255, 0, 0), 2)
            cv2.circle(small, (cx, cy), 5, (0, 0, 255), -1)
        
        # ⚠️ Note: Click detection is missing in the fallback since it relies on hand landmarks.

    cv2.imshow("Virtual Mouse", small)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🎯 Virtual Mouse session ended!")