import cv2
import mediapipe as mp
import time
import numpy as np

class GestureDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture detection parameters
        self.gesture_history = {
            "raised_single_hand": [],
            "both_hands_raised": [],
            "clenched_fist": [],
            "crossed_arms": [],
            "waving_hand": [],
            "hand_covering_mouth": [],
            "pointing_down": [],
            "l_shape_frame": []
        }
        
        self.gesture_alerts = {
            "raised_single_hand": "🚨 ALERT: Hand raised gesture detected! Assistance needed! 🚨",
            "both_hands_raised": "🚨 ALERT: Both hands raised! Possible surrender/threat situation! 🚨",
            "clenched_fist": "🚨 ALERT: Clenched fist detected! Possible health emergency! 🚨",
            "crossed_arms": "🚨 ALERT: Arms crossed over chest! SOS/immediate danger! 🚨",
            "waving_hand": "🚨 ALERT: Rapid hand waving detected! Panic situation! 🚨",
            "hand_covering_mouth": "🚨 ALERT: Hand covering mouth! Unable to speak/possible threat! 🚨",
            "pointing_down": "🚨 ALERT: Pointing down detected! Emergency below! 🚨",
            "l_shape_frame": "🚨 ALERT: L-shape frame detected! Requesting CCTV attention! 🚨"
        }
        
        self.gesture_colors = {
            "raised_single_hand": (0, 255, 0),     # Green
            "both_hands_raised": (0, 0, 255),      # Red
            "clenched_fist": (0, 165, 255),        # Orange
            "crossed_arms": (255, 0, 0),           # Blue
            "waving_hand": (255, 0, 255),          # Magenta
            "hand_covering_mouth": (255, 255, 0),  # Cyan
            "pointing_down": (0, 255, 255),        # Yellow
            "l_shape_frame": (180, 105, 255)       # Purple
        }
        
        self.gesture_start_times = {gesture: None for gesture in self.gesture_history.keys()}
        self.detected_gestures = {gesture: False for gesture in self.gesture_history.keys()}
        
        self.REQUIRED_DURATION = 3.0  # seconds required to hold gesture
        self.HISTORY_FRAMES = 10      # frames to keep in history for smoothing
        self.DETECTION_THRESHOLD = 0.8  # percentage of frames that must show gesture
        self.WAVE_THRESHOLD = 5       # number of direction changes for wave detection
        
        # For wave detection
        self.prev_hand_x = None
        self.direction_changes = 0
        self.last_direction = None
        self.wave_start_time = None
        
    def is_palm_open_and_raised(self, hand_landmarks):
        """Check if a single hand is raised with palm open facing the camera"""
        # Extract key points
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        
        # Check if hand is raised (middle finger is higher than wrist)
        is_raised = middle_finger_tip.y < wrist.y - 0.2  # Y-axis is inverted
        
        # Check if fingers are spread out (palm open)
        finger_spread_x = abs(thumb_tip.x - pinky_tip.x) > 0.2
        
        # Check if all fingers are extended
        fingers_extended = (
            middle_finger_tip.y < wrist.y and
            index_tip.y < wrist.y and
            ring_tip.y < wrist.y and
            pinky_tip.y < wrist.y
        )
        
        return is_raised and finger_spread_x and fingers_extended
    
    def is_clenched_fist(self, hand_landmarks):
        """Check if the hand forms a clenched fist"""
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # For clenched fist, finger tips should be close to palm
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Calculate 3D distances between fingertips and MCP (knuckles)
        index_bent = self.calculate_3d_distance(index_tip, index_mcp) < 0.1
        middle_bent = self.calculate_3d_distance(middle_tip, middle_mcp) < 0.1
        
        # All fingers should be bent and close to palm
        fingers_bent = index_bent and middle_bent
        
        # Hand should be facing the camera (z-coordinate of palm closer to camera than fingertips)
        facing_camera = wrist.z < index_tip.z
        
        return fingers_bent and facing_camera
    
    def is_hand_covering_mouth(self, hand_landmarks):
        """Check if hand is positioned over mouth area (center-ish of frame)"""
        # This is an approximation - we don't have face landmarks
        # Assuming the person's face is centered in the frame
        
        palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Check if palm is in the center region of the frame
        # We assume mouth would be roughly in center of frame
        is_center_x = 0.4 < palm.x < 0.6
        is_center_y = 0.4 < palm.y < 0.7
        
        # Check if palm is facing the camera
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        palm_facing = wrist.z > middle_tip.z
        
        return is_center_x and is_center_y and palm_facing
    
    def is_pointing_down(self, hand_landmarks):
        """Check if the index finger is pointing downward"""
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Index finger should be extended (straighter than other fingers)
        index_extended = index_tip.y > index_pip.y and index_pip.y > index_mcp.y
        
        # Index should be pointing down (y-axis is inverted, so tip should have larger y value)
        pointing_down = index_tip.y > index_mcp.y
        
        # Other fingers should be more bent
        other_fingers_bent = middle_tip.y < index_tip.y
        
        return index_extended and pointing_down and other_fingers_bent
    
    def is_l_shape_frame(self, hand_landmarks):
        """Check if hands are forming an L-shape/frame (like a director's frame)"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # For L-shape, thumb should be extended perpendicular to index finger
        # Other fingers should be bent
        
        # Check if thumb and index are extended
        thumb_extended = thumb_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y
        index_extended = index_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        
        # Check if they form a rough right angle
        middle_bent = middle_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        
        # Approximate L-shape check
        thumb_index_perpendicular = abs(thumb_tip.x - index_tip.x) > 0.1 and abs(thumb_tip.y - index_tip.y) > 0.1
        
        return thumb_extended and index_extended and middle_bent and thumb_index_perpendicular
    
    def detect_waving_hand(self, hand_landmarks):
        """Detect side-to-side waving motion"""
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        current_x = wrist.x
        
        if self.prev_hand_x is not None:
            # Detect direction change
            if current_x > self.prev_hand_x and (self.last_direction is None or self.last_direction == "left"):
                self.last_direction = "right"
                self.direction_changes += 1
            elif current_x < self.prev_hand_x and (self.last_direction is None or self.last_direction == "right"):
                self.last_direction = "left"
                self.direction_changes += 1
        
        self.prev_hand_x = current_x
        
        # Reset detection after some time to avoid false positives
        current_time = time.time()
        if self.wave_start_time is None:
            self.wave_start_time = current_time
        elif current_time - self.wave_start_time > 2.0:  # Reset after 2 seconds
            self.direction_changes = 0
            self.wave_start_time = current_time
        
        # If enough direction changes in the time window, consider it a wave
        return self.direction_changes >= self.WAVE_THRESHOLD
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two landmark points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)
    
    def process_frame(self, frame):
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = self.hands.process(rgb_frame)
        
        # Initialize current frame's gestures
        current_frame_gestures = {gesture: 0 for gesture in self.gesture_history.keys()}
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            # Check if both hands are detected for two-hand gestures
            both_hands_detected = len(results.multi_hand_landmarks) >= 2
            
            if both_hands_detected:
                # Process both hands raised gesture
                hands_raised_count = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.is_palm_open_and_raised(hand_landmarks):
                        hands_raised_count += 1
                
                if hands_raised_count >= 2:
                    current_frame_gestures["both_hands_raised"] = 1
            
            # Process single hand gestures
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Single hand gestures detection
                if self.is_palm_open_and_raised(hand_landmarks):
                    current_frame_gestures["raised_single_hand"] = 1
                
                if self.is_clenched_fist(hand_landmarks):
                    current_frame_gestures["clenched_fist"] = 1
                
                if self.is_hand_covering_mouth(hand_landmarks):
                    current_frame_gestures["hand_covering_mouth"] = 1
                
                if self.is_pointing_down(hand_landmarks):
                    current_frame_gestures["pointing_down"] = 1
                    
                if self.is_l_shape_frame(hand_landmarks):
                    current_frame_gestures["l_shape_frame"] = 1
                
                if self.detect_waving_hand(hand_landmarks):
                    current_frame_gestures["waving_hand"] = 1
                
                # For crossed arms (X shape), we need to analyze hand positions further
                # This is a simplified approximation
                if both_hands_detected and len(results.multi_hand_landmarks) == 2:
                    hand1 = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST]
                    hand2 = results.multi_hand_landmarks[1].landmark[self.mp_hands.HandLandmark.WRIST]
                    
                    # Check if hands are crossing each other
                    # This is a simplified check - ideally we would check more points
                    if (hand1.x < 0.5 and hand2.x > 0.5) or (hand1.x > 0.5 and hand2.x < 0.5):
                        if abs(hand1.y - hand2.y) < 0.2:  # Approximately at same height
                            current_frame_gestures["crossed_arms"] = 1
        
        # Update gesture histories
        for gesture, detected in current_frame_gestures.items():
            self.gesture_history[gesture].append(detected)
            # Keep only the last N frames in history
            if len(self.gesture_history[gesture]) > self.HISTORY_FRAMES:
                self.gesture_history[gesture].pop(0)
        
        # Process each gesture's detection status
        current_time = time.time()
        
        # Track which gestures to display
        active_gestures = []
        
        for gesture in self.gesture_history.keys():
            # Calculate detection ratio
            if self.gesture_history[gesture]:
                detection_ratio = sum(self.gesture_history[gesture]) / len(self.gesture_history[gesture])
            else:
                detection_ratio = 0
            
            # Check if gesture is consistently detected
            if detection_ratio >= self.DETECTION_THRESHOLD:
                # Start timing if we haven't already
                if self.gesture_start_times[gesture] is None:
                    self.gesture_start_times[gesture] = current_time
                    
                # If the gesture has been held long enough
                elapsed_time = current_time - self.gesture_start_times[gesture]
                if elapsed_time >= self.REQUIRED_DURATION and not self.detected_gestures[gesture]:
                    self.detected_gestures[gesture] = True
                    print(self.gesture_alerts[gesture])
                
                # Add to active gestures for display
                active_gestures.append((gesture, elapsed_time))
            else:
                # Reset timing if gesture is not detected consistently
                self.gesture_start_times[gesture] = None
                self.detected_gestures[gesture] = False
        
        # Display active gestures (limit to 2 most progressed for clarity)
        active_gestures.sort(key=lambda x: x[1], reverse=True)
        
        for i, (gesture, elapsed_time) in enumerate(active_gestures[:2]):
            y_offset = 40 + i * 60
            
            # If gesture is detected
            if self.detected_gestures[gesture]:
                display_text = f"ALERT: {gesture.replace('_', ' ').title()}"
                cv2.putText(frame, display_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.gesture_colors[gesture], 2)
            else:
                # Show progress bar
                progress = min(elapsed_time / self.REQUIRED_DURATION, 1.0)
                bar_width = 200
                filled_width = int(bar_width * progress)
                
                cv2.rectangle(frame, (10, y_offset + 10), (10 + bar_width, y_offset + 30), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(frame, (10, y_offset + 10), (10 + filled_width, y_offset + 30), 
                             self.gesture_colors[gesture], cv2.FILLED)
                
                status_text = f"Detecting {gesture.replace('_', ' ').title()}: {int(progress*100)}%"
                cv2.putText(frame, status_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.gesture_colors[gesture], 2)
        
        # Display overall status at bottom
        if any(self.detected_gestures.values()):
            triggered_gestures = [g.replace('_', ' ').title() for g, v in self.detected_gestures.items() if v]
            status = f"ALERT TRIGGERED: {', '.join(triggered_gestures)}"
            cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Status: Monitoring", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return frame
        
    def reset(self):
        """Reset detection state"""
        for gesture in self.gesture_history.keys():
            self.gesture_history[gesture] = []
            self.gesture_start_times[gesture] = None
            self.detected_gestures[gesture] = False
        
        self.prev_hand_x = None
        self.direction_changes = 0
        self.last_direction = None
        self.wave_start_time = None
        
    def release(self):
        """Release resources"""
        self.hands.close()


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from camera. Check camera index.")
            break
            
        # Process the frame with our detector
        processed_frame = detector.process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('Emergency Gesture Detection', processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


