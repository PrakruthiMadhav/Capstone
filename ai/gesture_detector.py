# gesture_detector.py - Updated to delay detection until full hold time
import cv2
import mediapipe as mp
import time
import numpy as np

from gesture_types import GestureType, GESTURE_ALERTS, GESTURE_COLORS
from detectors.crossed_wrists_detector import detect_crossed_wrists
from detectors.hand_over_mouth_detector import detect_hand_over_mouth
from detectors.surrender_pose_detector import detect_surrender_pose
from detectors.call_symbol_detector import detect_call_symbol
from detectors.l_shape_sign_detector import detect_l_shape_sign
from utils.conflict_resolver import resolve_gesture_conflicts
from utils.kalman_filter import apply_kalman_filtering

class EmergencyGestureDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Initialize MediaPipe Hands and Pose
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize hand detector with more hands
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

        # Gesture detection parameters - initialize without CIRCLE_IN_PALM
        self.gestures = {gesture: {"detected": False,
                                "start_time": None,
                                "history": [],
                                "confidence": 0.0,
                                "rejection_count": 0,
                                "confirmation_count": 0,
                                "positions": {},
                                "tracking_duration": 0.0,  # Add tracking duration field
                                "is_tracking": False       # Add tracking state field
                               }
                        for gesture in GestureType}

        # Detection parameters for better sensitivity
        self.REQUIRED_DURATION = 5.0     # Increased to 5 seconds - IMPORTANT
        self.HISTORY_FRAMES = 15         # Frames to keep for stability
        self.DETECTION_THRESHOLD = 0.55  # Threshold for detection
        self.CONFIDENCE_HYSTERESIS = 0.15 # For stability
        self.MINIMUM_CONSECUTIVE_DETECTIONS = 4  # Required frames for reliability
        self.REJECTION_THRESHOLD = 4     # Frames needed to reject
        
        # Add hysteresis to prevent flickering
        self.HYSTERESIS = 0.15  # For maintaining detection through variations

        # Debug mode for visualization
        self.DEBUG = True  # Debug mode on by default

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
            # Detect crossed wrists with emphasis on reliable detection
            current_frame_confidences[GestureType.CROSSED_WRISTS] = detect_crossed_wrists(
                pose_results.pose_landmarks, self.mp_pose)
                
            current_frame_confidences[GestureType.SURRENDER_POSE] = detect_surrender_pose(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose, 
                self.mp_hands, self.gestures[GestureType.SURRENDER_POSE], self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.HAND_OVER_MOUTH] = detect_hand_over_mouth(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose, 
                self.mp_hands, self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.CALL_SYMBOL] = detect_call_symbol(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose,
                self.mp_hands, self.DEBUG, self.frame_count)
                
            current_frame_confidences[GestureType.L_SHAPE_SIGN] = detect_l_shape_sign(
                pose_results.pose_landmarks, hand_landmarks_list, self.mp_pose,
                self.mp_hands, self.DEBUG, self.frame_count)

            # Apply conflict resolution - with CIRCLE_IN_PALM removed
            current_frame_confidences = resolve_gesture_conflicts(current_frame_confidences)

            # Store landmarks for next frame comparison
            self.prev_landmarks = pose_results.pose_landmarks

        # Draw debugging visualizations if enabled
        if self.DEBUG and pose_results.pose_landmarks:
            self._draw_debug_visualizations(frame, pose_results.pose_landmarks, hand_landmarks_list)

        # Update gesture histories and process detection
        self._update_gesture_states(frame, current_frame_confidences)

        return frame
        
    def _draw_debug_visualizations(self, frame, pose_landmarks, hand_landmarks_list):
        """Draw debugging visualizations on the frame"""
        h, w, _ = frame.shape
        
        # Draw face region for hand-over-mouth detection
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
        
        # Draw crossed wrists detection region
        # Get wrist positions for visualization
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate midline
        midline_x = int(((left_shoulder.x + right_shoulder.x) / 2) * w)
        top_y = int(((left_shoulder.y + right_shoulder.y) / 2) * h)
        bottom_y = int(h * 0.9)
        
        # Draw body midline
        cv2.line(frame, (midline_x, top_y), (midline_x, bottom_y), (0, 255, 255), 1)
        
        # Draw wrist positions
        left_wrist_x, left_wrist_y = int(left_wrist.x * w), int(left_wrist.y * h)
        right_wrist_x, right_wrist_y = int(right_wrist.x * w), int(right_wrist.y * h)
        
        # Draw wrist markers with labels
        cv2.circle(frame, (left_wrist_x, left_wrist_y), 8, (0, 255, 0), 2)
        cv2.putText(frame, "L", (left_wrist_x - 5, left_wrist_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        cv2.circle(frame, (right_wrist_x, right_wrist_y), 8, (0, 0, 255), 2)
        cv2.putText(frame, "R", (right_wrist_x - 5, right_wrist_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw line between wrists to visualize crossing
        cv2.line(frame, (left_wrist_x, left_wrist_y), (right_wrist_x, right_wrist_y), 
                (255, 255, 255), 1, cv2.LINE_AA)
                
        # Display crossed wrists detection info if wrists are close
        wrist_distance = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
        if wrist_distance < 0.5:  # If wrists are reasonably close
            # Check if wrists are crossing midline
            crossing_text = ""
            if left_wrist.x > midline_x / w and right_wrist.x < midline_x / w:
                crossing_text = "CROSSING: Both wrists crossed"
            elif left_wrist.x > midline_x / w:
                crossing_text = "PARTIAL: Left wrist right of midline"
            elif right_wrist.x < midline_x / w:
                crossing_text = "PARTIAL: Right wrist left of midline"
            else:
                crossing_text = "NO CROSSING: Wrists on proper sides"
                
            cv2.putText(frame, crossing_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _update_gesture_states(self, frame, current_frame_confidences):
        """Update gesture histories and process detection"""
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

            # Calculate average confidence with more weight on recent frames and outlier rejection
            if self.gestures[gesture]["history"]:
                # Exponential weights increasing for more recent frames (stronger recency bias)
                weights = np.exp(np.linspace(0, 1.0, len(self.gestures[gesture]["history"])))
                
                # Get history array
                history_array = np.array(self.gestures[gesture]["history"])
                
                # Apply outlier rejection - ignore extreme low values which could be detection errors
                sorted_history = np.sort(history_array)
                if len(sorted_history) > 5:
                    # Remove bottom 20% of values if we have enough history
                    filtered_history = sorted_history[int(len(sorted_history) * 0.2):]
                    if len(filtered_history) > 0:
                        filtered_weights = weights[-len(filtered_history):]
                        weighted_confidences = filtered_history * filtered_weights
                        avg_confidence = np.sum(weighted_confidences) / np.sum(filtered_weights)
                    else:
                        avg_confidence = 0.0
                else:
                    # Not enough history for filtering
                    weighted_confidences = history_array * weights
                    avg_confidence = np.sum(weighted_confidences) / np.sum(weights)
                
                self.gestures[gesture]["confidence"] = avg_confidence
            else:
                self.gestures[gesture]["confidence"] = 0.0

            # Apply hysteresis for tracking threshold
            tracking_threshold = self.DETECTION_THRESHOLD
            if self.gestures[gesture]["is_tracking"]:
                # Lower threshold once we start tracking
                tracking_threshold -= self.HYSTERESIS

            # CHANGED LOGIC: Track gesture consistently with time, but don't "detect" until duration met
            if self.gestures[gesture]["confidence"] >= tracking_threshold:
                # If not already tracking, start now
                if not self.gestures[gesture]["is_tracking"]:
                    self.gestures[gesture]["is_tracking"] = True
                    self.gestures[gesture]["start_time"] = current_time
                    self.gestures[gesture]["tracking_duration"] = 0.0
                    if self.DEBUG:
                        print(f"Started tracking {gesture.name} (conf: {self.gestures[gesture]['confidence']:.2f})")
                else:
                    # Update tracking duration
                    self.gestures[gesture]["tracking_duration"] = current_time - self.gestures[gesture]["start_time"]
                    
                # IMPORTANT: Only mark as detected if tracking duration meets required time
                if self.gestures[gesture]["tracking_duration"] >= self.REQUIRED_DURATION and not self.gestures[gesture]["detected"]:
                    self.gestures[gesture]["detected"] = True
                    print(GESTURE_ALERTS[gesture])
                
                # Add to active gestures for display (even if not officially detected yet)
                active_gestures.append(
                    (gesture, self.gestures[gesture]["tracking_duration"], self.gestures[gesture]["confidence"]))
            else:
                # Reset if confidence dropped below threshold
                if self.gestures[gesture]["is_tracking"]:
                    # Only reset if confidence has been low for a few frames
                    low_confidence_frames = sum(1 for conf in self.gestures[gesture]["history"][-3:] 
                                               if conf < tracking_threshold)
                    
                    if low_confidence_frames >= 3:  # Reset after 3 consecutive low confidence frames
                        if self.DEBUG:
                            print(f"Stopped tracking {gesture.name} after {self.gestures[gesture]['tracking_duration']:.1f}s")
                        self.gestures[gesture]["is_tracking"] = False
                        self.gestures[gesture]["start_time"] = None
                        self.gestures[gesture]["tracking_duration"] = 0.0
                        self.gestures[gesture]["detected"] = False
                else:
                    # Ensure not detected if not tracking
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
            
            # Get status information
            is_tracking = self.gestures[gesture]["is_tracking"]
            is_detected = self.gestures[gesture]["detected"]
            tracking_duration = self.gestures[gesture]["tracking_duration"]
            gesture_name = gesture.name.replace('_', ' ').title()

            # If gesture is officially detected (after full duration)
            if is_detected:
                display_text = f"ALERT: {gesture_name} (Conf: {confidence:.2f})"
                cv2.putText(frame, display_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GESTURE_COLORS[gesture], 2)
            
            # If gesture is being tracked but not yet confirmed
            elif is_tracking:
                # Show progress bar for gesture detection duration
                progress = min(tracking_duration / self.REQUIRED_DURATION, 1.0)
                bar_width = 200
                filled_width = int(bar_width * progress)

                cv2.rectangle(frame, (10, y_offset + 10), (10 +
                            bar_width, y_offset + 30), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(frame, (10, y_offset + 10), (10 + filled_width, y_offset + 30),
                            GESTURE_COLORS[gesture], cv2.FILLED)

                # Add time text to progress bar
                time_left = max(0, self.REQUIRED_DURATION - tracking_duration)
                status_text = f"Tracking {gesture_name}: {int(progress*100)}% ({time_left:.1f}s left, Conf: {confidence:.2f})"
                cv2.putText(frame, status_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GESTURE_COLORS[gesture], 2)
                            
                # For crossed wrists, add extra indicator for better visibility
                if gesture == GestureType.CROSSED_WRISTS and confidence > 0.4:
                    cv2.putText(frame, "KEEP HOLDING CROSSED WRISTS", (10, y_offset + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GESTURE_COLORS[gesture], 1)
            
            # Just show confidence if not tracking yet
            else:
                status_text = f"Possible {gesture_name}: Conf: {confidence:.2f}"
                cv2.putText(frame, status_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def _draw_status_bar(self, frame):
        """Draw overall status bar at the bottom of the frame"""
        detected_gestures = [g for g in GestureType if self.gestures[g]["detected"]]
        tracking_gestures = [g for g in GestureType if self.gestures[g]["is_tracking"] and not self.gestures[g]["detected"]]
        
        if detected_gestures:
            detected_names = [g.name.replace('_', ' ').title() for g in detected_gestures]
            status = f"EMERGENCY ALERT: {', '.join(detected_names)}"
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif tracking_gestures:
            tracking_names = [g.name.replace('_', ' ').title() for g in tracking_gestures]
            status = f"Tracking: {', '.join(tracking_names)} - hold for {self.REQUIRED_DURATION}s"
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(frame, f"Monitoring for emergency gestures (hold for {self.REQUIRED_DURATION}s)", 
                        (10, frame.shape[0] - 20),
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
            self.gestures[gesture]["is_tracking"] = False
            self.gestures[gesture]["tracking_duration"] = 0.0
            
            # Reset position dictionaries
            self.gestures[gesture]["positions"] = {}
            
            # Initialize gesture-specific states
            if gesture == GestureType.SURRENDER_POSE:
                self.gestures[gesture]["wrist_positions"] = {"left": [], "right": []}
        
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