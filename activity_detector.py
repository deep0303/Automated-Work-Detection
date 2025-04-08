import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_activity(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    debug_frame = small_frame.copy()

    # Detect hands
    hand_results = hands.process(rgb_frame)
    working_detected = False
    wrist_y = None

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(debug_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * 480
            index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 480
            print(f"Wrist Y: {wrist_y}, Index Tip Y: {index_tip_y}")  # Debug

            # Check if wrist or fingertips are in the lower part of the frame (typing area)
            if wrist_y > 300 or index_tip_y > 300:  # Lower 37.5% of frame
                working_detected = True
                break

    if working_detected:
        return "Working", debug_frame, wrist_y

    # Check for hand on head
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        if (abs(left_wrist.y * 480 - nose.y * 480) < 50 or 
            abs(right_wrist.y * 480 - nose.y * 480) < 50):
            return "Not Working (Hand on Head)", debug_frame, wrist_y

    # Basic phone detection
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 0.7 and y < 240:
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                return "Not Working (Using Phone)", debug_frame, wrist_y

    # Default case
    return "Not Working", debug_frame, wrist_y

# Cleanup
def cleanup():
    hands.close()
    pose.close()