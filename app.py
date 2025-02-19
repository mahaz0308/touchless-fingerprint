import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Finger mapping for MediaPipe landmarks
finger_map = {
    "thumb": mp_hands.HandLandmark.THUMB_TIP,
    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp_hands.HandLandmark.PINKY_TIP
}

# Circle settings
circle_center = None
circle_radius = 100

# Function to guide and capture the correct finger
def capture_single_finger(finger_name, frame, result):
    global circle_center
    frame_height, frame_width, _ = frame.shape
    circle_center = (frame_width // 2, frame_height // 2)

    # Draw the guide circle
    cv2.circle(frame, circle_center, circle_radius, (0, 255, 0), 2)

    # Process hand landmarks
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Get the coordinates of the required finger tip
            required_finger_tip = landmarks.landmark[finger_map[finger_name]]
            required_finger_coords = (int(required_finger_tip.x * frame_width),
                                      int(required_finger_tip.y * frame_height))

            # Check if the finger is inside the circle
            distance = np.sqrt((required_finger_coords[0] - circle_center[0]) ** 2 +
                               (required_finger_coords[1] - circle_center[1]) ** 2)

            if distance <= circle_radius:
                cv2.putText(frame, f"{finger_name.capitalize()} detected!", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(f"{finger_name}_captured.png", frame)
                return True

    # Feedback if the correct finger is not detected
    cv2.putText(frame, f"Place your {finger_name} in the circle", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return False

# Function to capture all fingers one by one
def capture_fingerprints():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    for finger_name in finger_names:
        print(f"Please place your {finger_name} in the circle.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            # Flip the frame for a mirror effect
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Check and capture the correct finger
            is_captured = capture_single_finger(finger_name, frame, result)

            # Show the frame
            cv2.imshow("Fingerprint Capture", frame)

            if is_captured:
                print(f"{finger_name.capitalize()} captured successfully!")
                break

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("All fingerprints captured!")

# Main function
def main():
    capture_fingerprints()

if __name__ == "__main__":
    main()
