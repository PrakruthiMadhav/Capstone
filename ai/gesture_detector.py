# gesture_detector.py - Main detector class
import cv2
import mediapipe as mp
import time
import numpy as np

from gesture_types import GestureType, GESTURE_ALERTS, GESTURE_COLORS
from detectors.crossed_wrists_detector import detect_crossed_wrists
from detectors.clutching_chest_detector import detect_clutching_chest
from detectors.one_knee_down_detector import detect_one_knee_down
from detectors.person_fallen_detector import detect_person_fallen
from detectors.surrender_pose_detector import detect_surrender_pose
from detectors.hand_over_mouth_detector import detect_hand_over_mouth
from detectors.waving_arm_detector import detect_waving_one_arm
from utils.conflict_resolver import resolve_gesture_conflicts
from utils.kalman_filter import apply_kalman_filtering

class EmergencyGestureDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Initialize MediaPipe Hands and Pose
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Gesture detection parameters
        self.gestures = {gesture: {"detected": False,
                                "start_time": None,
                                "history": [],
                                "confidence": 0.0,
                                "rejection_count": 0,
                                "confirmation_count": 0,
                                "positions": {}
                               }
                        for gesture in GestureType}

        # Detection parameters for better sensitivity
        self.REQUIRED_DURATION = 2.5     # Duration required for detection
        self.HISTORY_FRAMES = 10         # Consider last 10 frames for smoothing
        self.DETECTION_THRESHOLD = 0.45  # Confidence threshold for detection
        self.CONFIDENCE_HYSTERESIS = 0.1 # Hysteresis to prevent flickering
        self.MINIMUM_CONSECUTIVE_DETECTIONS = 3  # Required consecutive detections
        self.REJECTION_THRESHOLD = 3     # Consecutive rejections before resetting
        
        # Add hysteresis to prevent flickering
        self.HYSTERESIS = 0.1  # Once detected, allow 0.1 lower confidence to maintain

        # Debug mode for visualization
        self.DEBUG = True  # Set to True to see confidence values and detection regions

        # Previous frame data for comparison
        self.prev_landmarks = None
        self.frame_count = 0
        
        # Initialize kalman filters for confidence smoothing
        self.kalman_filters = {gesture: {"estimate": 0.0, 
                                        "error": 1.0, 
                                        "process_noise": 0.01, 
                                        "measurement_noise": 0.1} 
                              for gesture in GestureType}
        
        # Additional state for better gesture tracking
        self.currently_active_gesture = None
        self.active_gesture_timeout = 0

    def process_frame(self, frame):
        """
        Process a single frame and return the annotated frame
        Returns the frame with annotations
        """
        # Convert the image from BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Initialize confidences for current frame
        current_frame_confidences = {gesture: 0.0 for gesture in GestureType}

        # Increment frame counter
        self.frame_count += 1

        # Extract hand landmarks (if any hands detected)
        hand_landmarks_list = []
        if hand_results.multi_hand_landmarks:
            hand_landmarks_list = hand_results.multi_hand_landmarks
            
            # Draw hand landmarks
            for hand_landmarks in hand_landmarks_list:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        # Process the landmarks to detect gestures
        if pose_results.pose_landmarks:
            current_frame_confidences[GestureType.CROSSED_WRISTS] = detect_crossed_wrists(
                pose_results.pose_landmarks, self.mp_pose)
                
            current_frame_confidences[GestureType.CLUTCHING_CHEST] = detect_clutching_chest(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose, 
                self.mp_hands, self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.ONE_KNEE_DOWN] = detect_one_knee_down(
                pose_results.pose_landmarks, self.mp_pose)
                
            current_frame_confidences[GestureType.PERSON_FALLEN] = detect_person_fallen(
                pose_results.pose_landmarks, self.mp_pose, self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.SURRENDER_POSE] = detect_surrender_pose(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose, 
                self.mp_hands, self.gestures[GestureType.SURRENDER_POSE], self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.HAND_OVER_MOUTH] = detect_hand_over_mouth(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose, 
                self.mp_hands, self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.WAVING_ONE_ARM] = detect_waving_one_arm(
                pose_results.pose_landmarks, self.mp_pose, 
                self.gestures[GestureType.WAVING_ONE_ARM], self.DEBUG, self.frame_count)

            # Apply conflict resolution
            current_frame_confidences = resolve_gesture_conflicts(current_frame_confidences)

            # Store landmarks for next frame comparison
            self.prev_landmarks = pose_results.pose_landmarks

        # Draw debugging visualizations if enabled
        if self.DEBUG and pose_results.pose_landmarks:
            self._draw_debug_visualizations(frame, pose_results.pose_landmarks)

        # Update gesture histories and process detection
        self._update_gesture_states(frame, current_frame_confidences)

        return frame

    def _draw_debug_visualizations(self, frame, pose_landmarks):
        """Draw debugging visualizations on the frame"""
        # Draw chest area for clutching chest detection
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Define chest area boundaries (expanded)
        chest_top = min(left_shoulder.y, right_shoulder.y)
        chest_bottom = chest_top + 0.4
        chest_left = left_shoulder.x - 0.15
        chest_right = right_shoulder.x + 0.15

        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape
        chest_rect = (
            int(chest_left * w),
            int(chest_top * h),
            int((chest_right - chest_left) * w),
            int((chest_bottom - chest_top) * h)
        )
        cv2.rectangle(frame, (chest_rect[0], chest_rect[1]),
                    (chest_rect[0] + chest_rect[2],
                    chest_rect[1] + chest_rect[3]),
                    (0, 255, 255), 1)

        # Draw mouth area for hand-over-mouth detection
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]

        eye_y = (left_eye.y + right_eye.y) / 2
        eye_nose_distance = nose.y - eye_y
        mouth_y = nose.y + eye_nose_distance * 0.8

        mouth_x = int(nose.x * w)
        mouth_y = int(mouth_y * h)
        mouth_radius = int(0.15 * w)

        cv2.circle(frame, (mouth_x, mouth_y),
                mouth_radius, (255, 0, 255), 1)

    def _update_gesture_states(self, frame, current_frame_confidences):
        """Update gesture states based on the current frame confidences"""
        current_time = time.time()
        
        # Apply Kalman filtering for smoother confidence values
        smoothed_confidences = apply_kalman_filtering(current_frame_confidences, self.kalman_filters)

        # Track which gestures to display
        active_gestures = []

        for gesture in GestureType:
            # Update history
            self.gestures[gesture]["history"].append(smoothed_confidences[gesture])

            # Keep only the last N frames in history
            if len(self.gestures[gesture]["history"]) > self.HISTORY_FRAMES:
                self.gestures[gesture]["history"].pop(0)

            # Calculate average confidence with more weight on recent frames
            if self.gestures[gesture]["history"]:
                # Linear weights increasing for more recent frames
                weights = np.linspace(0.5, 1.0, len(self.gestures[gesture]["history"]))
                weighted_confidences = np.array(self.gestures[gesture]["history"]) * weights
                avg_confidence = np.sum(weighted_confidences) / np.sum(weights)
                self.gestures[gesture]["confidence"] = avg_confidence
            else:
                self.gestures[gesture]["confidence"] = 0.0

            # Apply hysteresis for detection threshold
            detection_threshold = self.DETECTION_THRESHOLD
            if self.gestures[gesture]["detected"]:
                # Lower threshold for already detected gestures
                detection_threshold -= self.HYSTERESIS

            # Check if gesture is consistently detected with high confidence
            if self.gestures[gesture]["confidence"] >= detection_threshold:
                # Start timing if we haven't already
                if self.gestures[gesture]["start_time"] is None:
                    self.gestures[gesture]["start_time"] = current_time

                # If the gesture has been held long enough
                elapsed_time = current_time - self.gestures[gesture]["start_time"]
                if elapsed_time >= self.REQUIRED_DURATION and not self.gestures[gesture]["detected"]:
                    self.gestures[gesture]["detected"] = True
                    print(GESTURE_ALERTS[gesture])

                # Add to active gestures for display
                active_gestures.append(
                    (gesture, elapsed_time, self.gestures[gesture]["confidence"]))
            else:
                # Reset timing if gesture is not detected consistently
                # Add a small delay before resetting to prevent flickering
                if self.gestures[gesture]["start_time"] is not None:
                    # Only reset if confidence has been low for a few frames
                    low_confidence_frames = sum(1 for conf in self.gestures[gesture]["history"][-3:] 
                                            if conf < detection_threshold)
                    
                    if low_confidence_frames >= 3:  # Reset after 3 consecutive low confidence frames
                        self.gestures[gesture]["start_time"] = None
                        self.gestures[gesture]["detected"] = False
                else:
                    self.gestures[gesture]["detected"] = False

        # Display all gestures with significant confidence
        # Sort by confidence
        active_gestures.sort(key=lambda x: x[2], reverse=True)

        # Display timestamp on frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display all gestures with at least 0.2 confidence
        significant_gestures = [g for g in active_gestures if g[2] >= 0.2]
        self._draw_gesture_indicators(frame, significant_gestures)

        # Display overall status at bottom
        self._draw_status_bar(frame)

    def _draw_gesture_indicators(self, frame, significant_gestures):
        """Draw indicators for each significant gesture"""
        for i, (gesture, elapsed_time, confidence) in enumerate(significant_gestures):
            y_offset = 70 + i * 60

            # If gesture is detected
            if self.gestures[gesture]["detected"]:
                gesture_name = gesture.name.replace('_', ' ').title()
                display_text = f"ALERT: {gesture_name} (Conf: {confidence:.2f})"
                cv2.putText(frame, display_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GESTURE_COLORS[gesture], 2)
            else:
                # Show progress bar for gesture detection duration
                progress = min(elapsed_time / self.REQUIRED_DURATION, 1.0)
                bar_width = 200
                filled_width = int(bar_width * progress)

                cv2.rectangle(frame, (10, y_offset + 10), (10 +
                            bar_width, y_offset + 30), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(frame, (10, y_offset + 10), (10 + filled_width, y_offset + 30),
                            GESTURE_COLORS[gesture], cv2.FILLED)

                gesture_name = gesture.name.replace('_', ' ').title()
                status_text = f"Detecting {gesture_name}: {int(progress*100)}% (Conf: {confidence:.2f})"
                cv2.putText(frame, status_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GESTURE_COLORS[gesture], 2)

    def _draw_status_bar(self, frame):
        """Draw overall status bar at the bottom of the frame"""
        if any(self.gestures[g]["detected"] for g in GestureType):
            triggered_gestures = [g.name.replace('_', ' ').title()
                                for g in GestureType
                                if self.gestures[g]["detected"]]
            status = f"EMERGENCY ALERT: {', '.join(triggered_gestures)}"
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Status: Monitoring for emergency gestures", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def reset(self):
        """Reset detection state"""
        for gesture in GestureType:
            self.gestures[gesture]["history"] = []
            self.gestures[gesture]["start_time"] = None
            self.gestures[gesture]["detected"] = False
            self.gestures[gesture]["confidence"] = 0.0
            self.gestures[gesture]["rejection_count"] = 0
            self.gestures[gesture]["confirmation_count"] = 0
            
            # Reset or initialize positions dictionaries
            if gesture == GestureType.WAVING_ONE_ARM:
                self.gestures[gesture]["positions"] = {
                    "left_wrist": [], "right_wrist": [],
                    "left_elbow": [], "right_elbow": []
                }
            elif gesture == GestureType.SURRENDER_POSE:
                self.gestures[gesture]["wrist_positions"] = {"left": [], "right": []}
            
            # Initialize any other gesture-specific state here if needed

        self.prev_landmarks = None
        self.frame_count = 0
        self.currently_active_gesture = None
        self.active_gesture_timeout = 0
        
        # Reset Kalman filters
        for gesture in GestureType:
            self.kalman_filters[gesture]["estimate"] = 0.0
            self.kalman_filters[gesture]["error"] = 1.0

    def release(self):
        """Release resources"""
        self.hands.close()
        self.pose.close()