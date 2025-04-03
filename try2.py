import cv2
import mediapipe as mp
import time
import numpy as np

class SimpleGestureDetector:
    def __init__(self):
        # Initialize MediaPipe Pose with lower thresholds for better detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use medium complexity for better performance
            smooth_landmarks=True,
            min_detection_confidence=0.5,  # Lower threshold to detect more poses
            min_tracking_confidence=0.5    # Lower threshold for better tracking
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Focus on the most important gestures
        self.gesture_history = {
            "raised_hand": [],          # Single hand raised
            "both_hands_raised": [],    # Both hands raised
            "fall_detected": []         # Person has fallen
        }
        
        self.gesture_alerts = {
            "raised_hand": "🚨 ALERT: Hand raised gesture detected! Assistance needed! 🚨",
            "both_hands_raised": "🚨 ALERT: Both hands raised! Attention requested! 🚨",
            "fall_detected": "🚨 ALERT: Fall detected! Person may be injured! 🚨"
        }
        
        self.gesture_colors = {
            "raised_hand": (0, 255, 0),       # Green
            "both_hands_raised": (0, 0, 255), # Red
            "fall_detected": (0, 165, 255)    # Orange
        }
        
        # Initialize tracking variables with shorter hold times and lower thresholds
        self.gesture_start_times = {gesture: None for gesture in self.gesture_history.keys()}
        self.detected_gestures = {gesture: False for gesture in self.gesture_history.keys()}
        
        # Use much more sensitive detection parameters
        self.REQUIRED_DURATION = 1.0     # Only 1 second to confirm gesture
        self.HISTORY_FRAMES = 5          # Fewer frames for quicker response
        self.DETECTION_THRESHOLD = 0.5   # Lower threshold for detection
        
        # Body landmarks dictionary for easy reference
        self.pose_landmarks_dict = {}
        
        # For fall detection
        self.prev_pose_landmarks = None
        
    def update_landmarks_dict(self, pose_landmarks):
        """Convert pose landmarks to dictionary for easier access"""
        self.pose_landmarks_dict = {}
        if pose_landmarks:
            for idx, landmark in enumerate(pose_landmarks.landmark):
                self.pose_landmarks_dict[idx] = landmark
    
    def is_visible(self, landmark_idx, visibility_threshold=0.5):
        """Check if a landmark is visible with sufficient confidence"""
        if landmark_idx in self.pose_landmarks_dict:
            return self.pose_landmarks_dict[landmark_idx].visibility > visibility_threshold
        return False
    
    def detect_raised_hand(self):
        """Simple raised hand detection - just check if either wrist is above shoulder"""
        # Check if we have the necessary landmarks
        left_visible = all(self.is_visible(idx) for idx in [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value
        ])
        
        right_visible = all(self.is_visible(idx) for idx in [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value
        ])
        
        if not (left_visible or right_visible):
            return False
            
        raised = False
        
        # Check left arm if visible
        if left_visible:
            left_wrist = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_shoulder = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            
            # Simple check: wrist is above shoulder
            if left_wrist.y < left_shoulder.y - 0.05:
                raised = True
        
        # Check right arm if visible
        if right_visible:
            right_wrist = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_shoulder = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Simple check: wrist is above shoulder
            if right_wrist.y < right_shoulder.y - 0.05:
                raised = True
        
        return raised
    
    def detect_both_hands_raised(self):
        """Detect both hands raised above shoulders"""
        # Check if we have the necessary landmarks
        if not all(self.is_visible(idx) for idx in [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value
        ]):
            return False
            
        left_wrist = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Simple check: both wrists above shoulders
        left_raised = left_wrist.y < left_shoulder.y - 0.05
        right_raised = right_wrist.y < right_shoulder.y - 0.05
        
        return left_raised and right_raised
    
    def detect_fall(self):
        """Simple fall detection - check if body orientation is more horizontal than vertical"""
        # Check if we have the necessary landmarks
        if not all(self.is_visible(idx, visibility_threshold=0.4) for idx in [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value
        ]):
            return False
            
        left_shoulder = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = self.pose_landmarks_dict[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate spine vector (from hips to shoulders)
        spine_x = ((left_shoulder.x + right_shoulder.x) / 2) - ((left_hip.x + right_hip.x) / 2)
        spine_y = ((left_shoulder.y + right_shoulder.y) / 2) - ((left_hip.y + right_hip.y) / 2)
        
        # Calculate angle with vertical axis
        angle = abs(np.degrees(np.arctan2(spine_x, spine_y)))
        
        # Body is more horizontal than vertical
        return angle > 45
    
    def process_frame(self, frame):
        """Process a video frame and detect gestures"""
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect body pose
        results = self.pose.process(rgb_frame)
        
        # Initialize current frame's gestures
        current_frame_gestures = {gesture: 0 for gesture in self.gesture_history.keys()}
        
        # Draw pose landmarks on the frame and detect gestures
        if results.pose_landmarks:
            # Update the landmarks dictionary for easy access
            self.update_landmarks_dict(results.pose_landmarks)
            
            # Draw landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Detect gestures
            if self.detect_raised_hand():
                current_frame_gestures["raised_hand"] = 1
                
            if self.detect_both_hands_raised():
                current_frame_gestures["both_hands_raised"] = 1
                
            if self.detect_fall():
                current_frame_gestures["fall_detected"] = 1
            
            # Store landmarks for next frame comparison
            self.prev_pose_landmarks = results.pose_landmarks
        
        # Update gesture histories
        for gesture, detected in current_frame_gestures.items():
            # Update history
            self.gesture_history[gesture].append(detected)
            if len(self.gesture_history[gesture]) > self.HISTORY_FRAMES:
                self.gesture_history[gesture].pop(0)
        
        # Process gesture alerts
        current_time = time.time()
        active_gestures = []
        
        for gesture in self.gesture_history.keys():
            # Calculate detection percentage
            if self.gesture_history[gesture]:
                detection_ratio = sum(self.gesture_history[gesture]) / len(self.gesture_history[gesture])
            else:
                detection_ratio = 0
            
            # Check if gesture exceeds threshold
            if detection_ratio >= self.DETECTION_THRESHOLD:
                # Start timing if we haven't already
                if self.gesture_start_times[gesture] is None:
                    self.gesture_start_times[gesture] = current_time
                
                # Calculate elapsed time
                elapsed_time = current_time - self.gesture_start_times[gesture]
                
                # If the gesture has been held long enough
                if elapsed_time >= self.REQUIRED_DURATION and not self.detected_gestures[gesture]:
                    self.detected_gestures[gesture] = True
                    print(self.gesture_alerts[gesture])
                
                # Add to active gestures for display
                active_gestures.append((gesture, elapsed_time, self.REQUIRED_DURATION, detection_ratio))
            else:
                # Reset timing if detection falls below threshold
                self.gesture_start_times[gesture] = None
                self.detected_gestures[gesture] = False
        
        # Display active gestures
        for i, (gesture, elapsed_time, required_duration, confidence) in enumerate(active_gestures):
            y_offset = 40 + i * 60
            
            # If gesture is detected
            if self.detected_gestures[gesture]:
                display_text = f"ALERT: {gesture.replace('_', ' ').title()}"
                cv2.putText(frame, display_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.gesture_colors[gesture], 2)
            else:
                # Show progress bar
                progress = min(elapsed_time / required_duration, 1.0)
                bar_width = 200
                filled_width = int(bar_width * progress)
                
                cv2.rectangle(frame, (10, y_offset + 10), (10 + bar_width, y_offset + 30), 
                             (100, 100, 100), cv2.FILLED)
                cv2.rectangle(frame, (10, y_offset + 10), (10 + filled_width, y_offset + 30), 
                             self.gesture_colors[gesture], cv2.FILLED)
                
                # Add gesture text
                status_text = f"Detecting {gesture.replace('_', ' ').title()}: {int(progress*100)}%"
                cv2.putText(frame, status_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.gesture_colors[gesture], 2)
        
        # Display overall status at bottom
        if any(self.detected_gestures.values()):
            triggered_gestures = [g.replace('_', ' ').title() for g, v in self.detected_gestures.items() if v]
            status = f"❗ ALERT TRIGGERED: {', '.join(triggered_gestures)}"
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
            
        self.prev_pose_landmarks = None
        
    def release(self):
        """Release resources"""
        self.pose.close()


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create detector
    detector = SimpleGestureDetector()
    
    # Process video stream
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from camera. Check camera index.")
            break
            
        # Process the frame with our detector
        processed_frame = detector.process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('Simple Gesture Detection', processed_frame)
        
        # Handle keypresses
        key = cv2.waitKey(5) & 0xFF
        
        # Exit on 'q' press
        if key == ord('q'):
            break
        # Reset detection on 'r' press
        elif key == ord('r'):
            detector.reset()
            
    detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()