import cv2
import mediapipe as mp
import time
import numpy as np
from enum import Enum, auto


class GestureType(Enum):
    """Enum to represent different emergency gestures"""
    CROSSED_WRISTS = auto()    # Captive/abuse/hostage signaling for police
    CLUTCHING_CHEST = auto()   # Hand on chest (health emergency)
    ONE_KNEE_DOWN = auto()     # About to collapse, weakness for medical team
    PERSON_FALLEN = auto()     # Horizontal body position
    SURRENDER_POSE = auto()    # Hands raised, palms forward
    HAND_OVER_MOUTH = auto()   # Hand covering mouth (can't speak)
    WAVING_ONE_ARM = auto()    # Help needed — outdoor signaling for search & rescue


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
                                "confidence": 0.0}
                        for gesture in GestureType}

        # Alert messages for each gesture
        self.gesture_alerts = {
            GestureType.CROSSED_WRISTS: "🚨 ALERT: Crossed wrists detected! Possible captive/hostage situation! 🚨",
            GestureType.CLUTCHING_CHEST: "🚨 ALERT: Hand clutching chest! Possible cardiac emergency! 🚨",
            GestureType.ONE_KNEE_DOWN: "🚨 ALERT: One knee down! Person about to collapse! Medical assistance needed! 🚨",
            GestureType.PERSON_FALLEN: "🚨 ALERT: Person fallen! Unconscious or injured! 🚨",
            GestureType.SURRENDER_POSE: "🚨 ALERT: Hands raised palms forward! Possible threat situation! 🚨",
            GestureType.HAND_OVER_MOUTH: "🚨 ALERT: Hand over mouth! Unable to speak/possible threat! 🚨",
            GestureType.WAVING_ONE_ARM: "🚨 ALERT: Waving one arm! Help needed - outdoor signaling! 🚨"
        }

        # Colors for visualization
        self.gesture_colors = {
            GestureType.CROSSED_WRISTS: (0, 255, 0),      # Green
            GestureType.CLUTCHING_CHEST: (0, 0, 255),     # Red
            GestureType.ONE_KNEE_DOWN: (255, 191, 0),     # Deep Sky Blue
            GestureType.PERSON_FALLEN: (255, 0, 0),       # Blue
            GestureType.SURRENDER_POSE: (255, 0, 255),    # Magenta
            GestureType.HAND_OVER_MOUTH: (255, 255, 0),   # Cyan
            GestureType.WAVING_ONE_ARM: (128, 0, 128)     # Purple
        }

        # IMPROVED: Detection parameters for better sensitivity
        self.REQUIRED_DURATION = 2.5     # Reduced from 3.0 for faster detection
        self.HISTORY_FRAMES = 10         # Consider last 10 frames for smoothing
        self.DETECTION_THRESHOLD = 0.45  # Reduced from 0.5 for better sensitivity
        
        # IMPROVED: Add hysteresis to prevent flickering
        self.HYSTERESIS = 0.1  # Once detected, allow 0.1 lower confidence to maintain

        # Debug mode for visualization
        self.DEBUG = True  # Set to True to see confidence values and detection regions

        # Previous frame data for comparison
        self.prev_landmarks = None
        self.frame_count = 0

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

        # Process the landmarks to detect gestures
        if pose_results.pose_landmarks:
            current_frame_confidences[GestureType.CROSSED_WRISTS] = self.detect_crossed_wrists(
                pose_results.pose_landmarks)
            current_frame_confidences[GestureType.CLUTCHING_CHEST] = self.detect_clutching_chest(
                pose_results.pose_landmarks, hand_landmarks_list)
            current_frame_confidences[GestureType.ONE_KNEE_DOWN] = self.detect_one_knee_down(
                pose_results.pose_landmarks)
            current_frame_confidences[GestureType.PERSON_FALLEN] = self.detect_person_fallen(
                pose_results.pose_landmarks)
            current_frame_confidences[GestureType.SURRENDER_POSE] = self.detect_surrender_pose(
                pose_results.pose_landmarks, hand_landmarks_list)
            current_frame_confidences[GestureType.HAND_OVER_MOUTH] = self.detect_hand_over_mouth(
                pose_results.pose_landmarks, hand_landmarks_list)
            current_frame_confidences[GestureType.WAVING_ONE_ARM] = self.detect_waving_one_arm(
                pose_results.pose_landmarks)

            # Apply conflict resolution - implement the resolve_gesture_conflicts method
            current_frame_confidences = self.resolve_gesture_conflicts(current_frame_confidences)

            # Store landmarks for next frame comparison
            self.prev_landmarks = pose_results.pose_landmarks

        # Draw hand landmarks
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        # Draw debugging visualizations if enabled
        if self.DEBUG and pose_results.pose_landmarks:
            # Draw chest area for clutching chest detection
            left_shoulder = pose_results.pose_landmarks.landmark[
                self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

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
            nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_eye = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]

            eye_y = (left_eye.y + right_eye.y) / 2
            eye_nose_distance = nose.y - eye_y
            mouth_y = nose.y + eye_nose_distance * 0.8

            mouth_x = int(nose.x * w)
            mouth_y = int(mouth_y * h)
            mouth_radius = int(0.15 * w)

            cv2.circle(frame, (mouth_x, mouth_y),
                    mouth_radius, (255, 0, 255), 1)

        # Update gesture histories and process detection
        current_time = time.time()

        # Track which gestures to display
        active_gestures = []

        for gesture in GestureType:
            # Update history
            self.gestures[gesture]["history"].append(
                current_frame_confidences[gesture])

            # Keep only the last N frames in history
            if len(self.gestures[gesture]["history"]) > self.HISTORY_FRAMES:
                self.gestures[gesture]["history"].pop(0)

            # IMPROVED: Calculate average confidence with more weight on recent frames
            if self.gestures[gesture]["history"]:
                # Linear weights increasing for more recent frames
                weights = np.linspace(0.5, 1.0, len(self.gestures[gesture]["history"]))
                weighted_confidences = np.array(self.gestures[gesture]["history"]) * weights
                avg_confidence = np.sum(weighted_confidences) / np.sum(weights)
                self.gestures[gesture]["confidence"] = avg_confidence
            else:
                self.gestures[gesture]["confidence"] = 0.0

            # IMPROVED: Apply hysteresis for detection threshold
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
                elapsed_time = current_time - \
                    self.gestures[gesture]["start_time"]
                if elapsed_time >= self.REQUIRED_DURATION and not self.gestures[gesture]["detected"]:
                    self.gestures[gesture]["detected"] = True
                    print(self.gesture_alerts[gesture])

                # Add to active gestures for display
                active_gestures.append(
                    (gesture, elapsed_time, self.gestures[gesture]["confidence"]))
            else:
                # Reset timing if gesture is not detected consistently
                # IMPROVED: Add a small delay before resetting to prevent flickering
                if self.gestures[gesture]["start_time"] is not None:
                    # Only reset if confidence has been low for a few frames
                    low_confidence_frames = sum(1 for conf in self.gestures[gesture]["history"][-3:] 
                                            if conf < detection_threshold)
                    
                    if low_confidence_frames >= 3:  # Reset after 3 consecutive low confidence frames
                        self.gestures[gesture]["start_time"] = None
                        self.gestures[gesture]["detected"] = False
                else:
                    self.gestures[gesture]["detected"] = False

        # Display all gestures with significant confidence (not just top 2)
        # Sort by confidence rather than elapsed time
        active_gestures.sort(key=lambda x: x[2], reverse=True)

        # Display timestamp on frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # IMPROVED: Display all gestures with at least 0.2 confidence (more feedback)
        significant_gestures = [g for g in active_gestures if g[2] >= 0.2]

        for i, (gesture, elapsed_time, confidence) in enumerate(significant_gestures):
            y_offset = 70 + i * 60

            # If gesture is detected
            if self.gestures[gesture]["detected"]:
                gesture_name = gesture.name.replace('_', ' ').title()
                display_text = f"ALERT: {gesture_name} (Conf: {confidence:.2f})"
                cv2.putText(frame, display_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.gesture_colors[gesture], 2)
            else:
                # Show progress bar for gesture detection duration
                progress = min(elapsed_time / self.REQUIRED_DURATION, 1.0)
                bar_width = 200
                filled_width = int(bar_width * progress)

                cv2.rectangle(frame, (10, y_offset + 10), (10 +
                            bar_width, y_offset + 30), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(frame, (10, y_offset + 10), (10 + filled_width, y_offset + 30),
                            self.gesture_colors[gesture], cv2.FILLED)

                gesture_name = gesture.name.replace('_', ' ').title()
                status_text = f"Detecting {gesture_name}: {int(progress*100)}% (Conf: {confidence:.2f})"
                cv2.putText(frame, status_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.gesture_colors[gesture], 2)

        # Display overall status at bottom
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

        return frame

    def update_detection_state(self, current_frame_confidences):
        """
        NEW: More robust state management with hysteresis and Kalman filtering
        """
        current_time = time.time()
        active_gestures = []
        
        # First, apply Kalman filtering to smooth confidence values
        smoothed_confidences = self.apply_kalman_filtering(current_frame_confidences)
        
        for gesture in GestureType:
            # Update history with smoothed confidence
            self.gestures[gesture]["history"].append(smoothed_confidences[gesture])
            
            # Keep only the most recent frames
            if len(self.gestures[gesture]["history"]) > self.HISTORY_FRAMES:
                self.gestures[gesture]["history"].pop(0)
            
            # Calculate average confidence (with more weight on recent frames)
            if self.gestures[gesture]["history"]:
                weights = np.linspace(0.5, 1.0, len(self.gestures[gesture]["history"]))
                weighted_confidences = np.array(self.gestures[gesture]["history"]) * weights
                avg_confidence = np.sum(weighted_confidences) / np.sum(weights)
                self.gestures[gesture]["confidence"] = avg_confidence
            else:
                self.gestures[gesture]["confidence"] = 0.0
            
            # Apply hysteresis to avoid flickering
            detection_threshold = self.DETECTION_THRESHOLD
            if self.gestures[gesture]["detected"]:
                # Lower threshold for already detected gestures (hysteresis)
                detection_threshold -= self.CONFIDENCE_HYSTERESIS
            
            # Check if gesture is consistently detected with sufficient confidence
            if self.gestures[gesture]["confidence"] >= detection_threshold:
                # Reset rejection counter and increment confirmation counter
                self.gestures[gesture]["rejection_count"] = 0
                self.gestures[gesture]["confirmation_count"] += 1
                
                # Start timing if we haven't already
                if self.gestures[gesture]["start_time"] is None:
                    self.gestures[gesture]["start_time"] = current_time
                
                # Require consecutive detections before activating
                if self.gestures[gesture]["confirmation_count"] >= self.MINIMUM_CONSECUTIVE_DETECTIONS:
                    # If the gesture has been held long enough
                    elapsed_time = current_time - self.gestures[gesture]["start_time"]
                    
                    if elapsed_time >= self.REQUIRED_DURATION and not self.gestures[gesture]["detected"]:
                        self.gestures[gesture]["detected"] = True
                        print(self.gesture_alerts[gesture])
                        
                        # Set as currently active gesture
                        self.currently_active_gesture = gesture
                        self.active_gesture_timeout = 15  # Give it 15 frames of priority
                    
                    # Add to active gestures for display
                    active_gestures.append(
                        (gesture, elapsed_time, self.gestures[gesture]["confidence"]))
            else:
                # Increment rejection counter and reset confirmation counter
                self.gestures[gesture]["rejection_count"] += 1
                self.gestures[gesture]["confirmation_count"] = 0
                
                # If we've seen enough consecutive low confidences, reset detection
                if self.gestures[gesture]["rejection_count"] >= self.REJECTION_THRESHOLD:
                    self.gestures[gesture]["start_time"] = None
                    self.gestures[gesture]["detected"] = False
        
        # Prioritize the currently active gesture to prevent rapid switching
        if self.currently_active_gesture is not None:
            self.active_gesture_timeout -= 1
            if self.active_gesture_timeout <= 0:
                self.currently_active_gesture = None
        
        return active_gestures

    def apply_kalman_filtering(self, current_confidences):
        """
        NEW: Apply Kalman filtering to smooth confidence values
        """
        smoothed_confidences = {}
        
        for gesture, confidence in current_confidences.items():
            # Get current Kalman filter state
            kalman = self.kalman_filters[gesture]
            
            # Prediction step
            prediction = kalman['estimate']
            prediction_error = kalman['error'] + kalman['process_noise']
            
            # Update step
            kalman_gain = prediction_error / (prediction_error + kalman['measurement_noise'])
            estimate = prediction + kalman_gain * (confidence - prediction)
            error = (1 - kalman_gain) * prediction_error
            
            # Update filter state
            kalman['estimate'] = estimate
            kalman['error'] = error
            
            # Store smoothed confidence
            smoothed_confidences[gesture] = estimate
        
        return smoothed_confidences


    def detect_one_knee_down(self, pose_landmarks):
        """
        Detect if person has one knee down (about to collapse/weakness)
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0

        # Get relevant landmarks
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_foot = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        # Calculate vectors for thigh and calf
        left_thigh_vec = [left_knee.x - left_hip.x, left_knee.y - left_hip.y]
        left_calf_vec = [left_ankle.x - left_knee.x, left_ankle.y - left_knee.y]
        right_thigh_vec = [right_knee.x - right_hip.x, right_knee.y - right_hip.y]
        right_calf_vec = [right_ankle.x - right_knee.x, right_ankle.y - right_knee.y]
        
        # Function to calculate angle between vectors (in degrees)
        def calc_angle(v1, v2):
            dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
            mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            # Prevent division by zero
            if mag_v1 * mag_v2 == 0:
                return 0
                
            cos_angle = dot_prod / (mag_v1 * mag_v2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Ensure within valid range
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        
        # Calculate knee angles
        left_knee_angle = calc_angle(left_thigh_vec, left_calf_vec)
        right_knee_angle = calc_angle(right_thigh_vec, right_calf_vec)
        
        # FIXED: More lenient knee bend threshold
        left_leg_bent = left_knee_angle < 140  # Increased from 120
        right_leg_bent = right_knee_angle < 140  # Increased from 120
        
        # Check if one leg is bent (knee down) and other is relatively straight
        one_knee_down = (left_leg_bent and not right_leg_bent) or (right_leg_bent and not left_leg_bent)
        
        # FIXED: Give some confidence even if both knees are slightly bent
        # As long as one is much more bent than the other
        if not one_knee_down:
            # If both legs are bent but one is much more bent than the other
            if left_leg_bent and right_leg_bent:
                angle_diff = abs(left_knee_angle - right_knee_angle)
                if angle_diff > 30:  # One knee is significantly more bent
                    one_knee_down = True
                else:
                    return 0.0  # Both knees similarly bent - not one knee down pose
            else:
                return 0.0  # No knee down
        
        # Calculate vertical difference between knees
        knee_y_diff = abs(left_knee.y - right_knee.y)
        
        # Check which knee is down
        down_knee = left_knee if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else right_knee
        down_foot = left_foot if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else right_foot
        straight_knee = right_knee if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else left_knee
        straight_foot = right_foot if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else left_foot
        
        # FIXED: More lenient position checks
        # The down knee should be lower in frame than the straight knee
        proper_knee_position = down_knee.y >= straight_knee.y - 0.05  # Added tolerance
        
        # FIXED: More lenient foot position check
        foot_knee_distance = np.sqrt((down_foot.x - down_knee.x)**2 + (down_foot.y - down_knee.y)**2)
        foot_close_to_knee = foot_knee_distance < 0.25  # Increased from 0.15
        
        # FIXED: More lenient ground detection
        # Calculate approx ground based on the down foot and knee
        estimated_ground_y = max(down_foot.y, down_knee.y)
        
        # Down knee should be close to the ground
        knee_ground_dist = abs(1.0 - down_knee.y)
        knee_near_ground = knee_ground_dist < 0.4  # Increased from 0.3
        
        # Calculate body position relative to knee
        hip_midpoint_y = (left_hip.y + right_hip.y) / 2
        
        # Body should be higher than down knee
        body_above_knee = hip_midpoint_y < down_knee.y + 0.1  # Added tolerance
        
        # FIXED: More lenient body distance check
        body_knee_dist = abs(hip_midpoint_y - down_knee.y)
        proper_body_dist = 0.05 < body_knee_dist < 0.6  # Widened range (was 0.1-0.5)
        
        # Calculate confidence components
        angle_conf = 0.0
        if left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle):
            angle_conf = max(0.0, min(1.0, (180 - left_knee_angle) / 90))
        else:
            angle_conf = max(0.0, min(1.0, (180 - right_knee_angle) / 90))
        
        # FIXED: More generous position confidence
        position_conf = 0.0
        if proper_knee_position:
            position_conf = min(1.0, knee_y_diff * 7.0)  # Increased multiplier from 5.0
        else:
            # Still give some confidence even if knees are at similar height
            position_conf = min(0.5, knee_y_diff * 3.0)
        
        ground_conf = 0.0
        if knee_near_ground:
            ground_conf = max(0.0, 1.0 - (knee_ground_dist * 2.5))  # Less harsh penalty (was 3.0)
        
        body_position_conf = 0.0
        if body_above_knee:
            body_position_conf = max(0.3, min(1.0, body_knee_dist * 2.5))  # Min 0.3 confidence
        
        foot_position_conf = 0.0
        if foot_close_to_knee:
            foot_position_conf = max(0.0, 1.0 - (foot_knee_distance * 4.0))  # Less harsh penalty (was 5.0)
        
        # FIXED: Adjusted weights to prioritize angle and position
        confidence = (
            angle_conf * 0.35 +           # Increased (was 0.30)
            position_conf * 0.25 +        # Increased (was 0.20)
            ground_conf * 0.15 +          # Decreased (was 0.20)
            body_position_conf * 0.15 +   # Unchanged
            foot_position_conf * 0.10     # Decreased (was 0.15)
        )
        
        # FIXED: Lower threshold for clear knee bends
        if angle_conf > 0.6 and position_conf > 0.4:
            confidence = max(confidence, 0.55)  # Lower threshold for minimum confidence
        
        return confidence

    def detect_waving_one_arm(self, pose_landmarks):

        if not pose_landmarks:
            return 0.0
        
    # Get current landmarks
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
    
    # Check if at least one arm is raised
        left_arm_raised = left_wrist.y < (left_shoulder.y - 0.05)
        right_arm_raised = right_wrist.y < (right_shoulder.y - 0.05)
    
    # CRITICAL: Differentiate from surrender pose - typically only ONE arm is waving
        both_arms_raised = left_arm_raised and right_arm_raised
    
    # Key difference: if BOTH arms are raised, require asymmetric movement
    # for wave detection (unlike surrender pose which is symmetric and stable)
    
    # Track wrist positions in history to detect wave pattern
        gesture = self.gestures[GestureType.WAVING_ONE_ARM]
    
    # Add wrist and elbow positions to tracking history
        if "positions" not in gesture:
            gesture["positions"] = {
            "left_wrist": [], "right_wrist": [],
            "left_elbow": [], "right_elbow": []
        }
    
    # Add current positions
        gesture["positions"]["left_wrist"].append((left_wrist.x, left_wrist.y))
        gesture["positions"]["right_wrist"].append((right_wrist.x, right_wrist.y))
        gesture["positions"]["left_elbow"].append((left_elbow.x, left_elbow.y))
        gesture["positions"]["right_elbow"].append((right_elbow.x, right_elbow.y))
    
    # Keep only recent positions
        max_history = 15  # Increased for better pattern detection
        for key in gesture["positions"]:
            if len(gesture["positions"][key]) > max_history:
                gesture["positions"][key] = gesture["positions"][key][-max_history:]
    
    # Need at least a few frames to detect movement
        if len(gesture["positions"]["left_wrist"]) < 4:
            return 0.0
    
    # Calculate movement for each joint
        movement_scores = {}
        for joint in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
            positions = gesture["positions"][joint]
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
        
            if len(x_positions) > 3:
            # Calculate variance in x and y directions
                x_variance = np.var(x_positions) 
                y_variance = np.var(y_positions)
            
            # Calculate mean absolute difference for consecutive points
                x_diffs = [abs(x_positions[i] - x_positions[i-1]) for i in range(1, len(x_positions))]
                y_diffs = [abs(y_positions[i] - y_positions[i-1]) for i in range(1, len(y_positions))]
            
            # Check for oscillation pattern (direction changes)
                x_direction_changes = sum(1 for i in range(2, len(x_positions)) if 
                                    (x_positions[i] - x_positions[i-1]) * 
                                    (x_positions[i-1] - x_positions[i-2]) <= 0)
            
                oscillation_factor = min(3.0, 1.0 + x_direction_changes / 3.0)
            
            # Combined movement score with oscillation boost
                if x_diffs and y_diffs:
                    x_mad = np.mean(x_diffs)
                    y_mad = np.mean(y_diffs)
                    movement_scores[joint] = ((x_variance + y_variance) * 1500 + 
                                        (x_mad + y_mad) * 500) * oscillation_factor
                else:
                    movement_scores[joint] = 0
    
    # Determine which arm is moving more
        left_arm_movement = max(movement_scores.get("left_wrist", 0), movement_scores.get("left_elbow", 0))
        right_arm_movement = max(movement_scores.get("right_wrist", 0), movement_scores.get("right_elbow", 0))
    
    # Critical difference from surrender pose: movement asymmetry
    # One arm should move much more than the other
        movement_asymmetry = max(0, abs(left_arm_movement - right_arm_movement))
    
    # Enhanced waving pattern detection - check for side-to-side movement
        side_to_side_pattern = False
        dominant_plane = "none"
    
    # Calculate dominant plane of movement (horizontal for waving)
        if left_arm_raised and "left_wrist" in movement_scores:
            left_x = [p[0] for p in gesture["positions"]["left_wrist"]]
            left_y = [p[1] for p in gesture["positions"]["left_wrist"]]
            x_range = max(left_x) - min(left_x)
            y_range = max(left_y) - min(left_y)
        
            if x_range > y_range * 1.5:  # Horizontal movement dominates
                side_to_side_pattern = True
                dominant_plane = "horizontal-left"
    
        if right_arm_raised and "right_wrist" in movement_scores:
            right_x = [p[0] for p in gesture["positions"]["right_wrist"]]
            right_y = [p[1] for p in gesture["positions"]["right_wrist"]]
            x_range = max(right_x) - min(right_x)
            y_range = max(right_y) - min(right_y)
        
            if x_range > y_range * 1.5:  # Horizontal movement dominates
                side_to_side_pattern = True
                dominant_plane = "horizontal-right"
    
    # Calculate confidence based on which arm is moving
        if both_arms_raised:
        # If both arms raised, require asymmetric movement (one arm waving, one still)
            movement_ratio = 0
            if left_arm_movement > 0 and right_arm_movement > 0:
                movement_ratio = max(left_arm_movement, right_arm_movement) / max(0.00001, min(left_arm_movement, right_arm_movement))
        
            if movement_ratio > 3.0:  # One arm moving at least 3x more than the other
                if left_arm_movement > right_arm_movement:
                    confidence = min(1.0, left_arm_movement)
                    moving_arm = "left"
                else:
                    confidence = min(1.0, right_arm_movement)
                    moving_arm = "right"
            else:
            # Both arms moving similarly - reduce confidence as this is more like surrender
                confidence = max(0.0, min(0.3, (left_arm_movement + right_arm_movement) / 10))
                moving_arm = "both"
        else:
        # One arm raised
            if left_arm_raised and left_arm_movement > 0.1:
                confidence = min(1.0, left_arm_movement)
                moving_arm = "left"
            elif right_arm_raised and right_arm_movement > 0.1:
                confidence = min(1.0, right_arm_movement)
                moving_arm = "right"
            else:
                confidence = 0.0
                moving_arm = "none"
    
    # Boost confidence if side-to-side pattern detected
        if side_to_side_pattern and confidence > 0.1:
            confidence = min(1.0, confidence * 1.5)
    
    # Debug output
        if self.DEBUG and self.frame_count % 3 == 0 and confidence > 0.1:
            print(f"Waving arm: left_mov={left_arm_movement:.6f}, right_mov={right_arm_movement:.6f}, " +
             f"both={both_arms_raised}, asym={movement_asymmetry:.2f}")
            print(f"Waving {moving_arm} arm: pattern={side_to_side_pattern}, " +
             f"plane={dominant_plane}, conf={confidence:.2f}")
    
        return min(1.0, confidence)  # Cap at 1.0

    def detect_clutching_chest(self, pose_landmarks, hand_landmarks_list):
        """
        Detect if a hand is clutching the chest (health emergency)
        Returns confidence score between 0-1
        """
        if not pose_landmarks or not hand_landmarks_list:
            return 0.0

        # Get chest area landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # CRITICAL: Check for crossed arms pattern - this is NOT clutching chest
        # In crossed arms, both wrists are typically close together and elbows are far apart
        
        # Calculate distances
        wrist_distance = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
        elbow_distance = np.sqrt((left_elbow.x - right_elbow.x)**2 + (left_elbow.y - right_elbow.y)**2)
        
        # In crossed arms, wrists are close but elbows are far apart
        wrists_close = wrist_distance < 0.2
        elbows_far = elbow_distance > 0.25
        
        # If crossed arms pattern detected, return very low confidence for clutching chest
        if wrists_close and elbows_far:
            # Check arm directions for crossing
            left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
            right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
            
            # If arms appear to be crossing (left arm going right, right arm going left)
            if left_arm_vector[0] > 0 and right_arm_vector[0] < 0:
                if self.DEBUG and self.frame_count % 15 == 0:
                    print("Crossed arms detected - NOT clutching chest")
                return 0.05  # Very low confidence
        
        # For true chest clutching, we expect only ONE hand on chest, not both
        # First identify if both hands are on chest area
        
        # Define chest area boundaries more precisely
        chest_top = min(left_shoulder.y, right_shoulder.y) + 0.05
        chest_bottom = chest_top + 0.25  
        chest_left = left_shoulder.x - 0.1
        chest_right = right_shoulder.x + 0.1
        
        # Check both wrists - if both are in chest area, it's less likely to be clutching
        left_wrist_on_chest = (chest_left <= left_wrist.x <= chest_right and 
                            chest_top <= left_wrist.y <= chest_bottom)
        right_wrist_on_chest = (chest_left <= right_wrist.x <= chest_right and 
                            chest_top <= right_wrist.y <= chest_bottom)
        
        # If both hands are on chest, reduce confidence - not typical clutching pattern
        both_hands_on_chest_penalty = 1.0
        if left_wrist_on_chest and right_wrist_on_chest:
            both_hands_on_chest_penalty = 0.4  # Significant penalty
        
        # Heart area is more specific for clutching
        heart_center_x = chest_left + (chest_right - chest_left) * 0.35  # Left side of chest
        heart_center_y = chest_top + (chest_bottom - chest_top) * 0.4    # Upper chest
        heart_radius = 0.08
        
        max_confidence = 0.0
        
        # Check each hand 
        for hand_landmarks in hand_landmarks_list:
            # Use multiple hand points for detection
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            
            # Determine if this is left or right hand (approximation)
            is_left_hand = wrist.x < (left_shoulder.x + right_shoulder.x) / 2
            
            # CRITICAL: Check hand cupping/grabbing posture
            # For clutching chest, fingers should be curled or flexed
            
            # Get fingertip positions
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            
            # Calculate average distance from fingertips to palm
            # In clutching, this distance is shorter (fingers bent)
            fingertip_distances = [
                np.sqrt((index_tip.x - palm.x)**2 + (index_tip.y - palm.y)**2),
                np.sqrt((middle_tip.x - palm.x)**2 + (middle_tip.y - palm.y)**2),
                np.sqrt((ring_tip.x - palm.x)**2 + (ring_tip.y - palm.y)**2)
            ]
            avg_fingertip_distance = sum(fingertip_distances) / len(fingertip_distances)
            
            # For clutching, fingers should be closer to palm than in normal hand pose
            fingers_curled = avg_fingertip_distance < 0.08
            
            # Check if palm is on chest
            palm_on_chest = (chest_left <= palm.x <= chest_right and 
                            chest_top <= palm.y <= chest_bottom)
            
            # Check if palm is in heart area (more specific for heart attack gesture)
            palm_in_heart_area = np.sqrt((palm.x - heart_center_x)**2 + 
                                        (palm.y - heart_center_y)**2) <= heart_radius
            
            # Base confidence calculation
            if palm_on_chest:
                # Palm on chest is the main indicator
                base_confidence = 0.5
                
                # Boost for palm in heart area
                heart_area_boost = 0.3 if palm_in_heart_area else 0.0
                
                # Boost for fingers curled (clutching posture)
                finger_posture_boost = 0.2 if fingers_curled else 0.0
                
                # Check if palm is facing the chest (inward)
                thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
                wrist_to_palm_z = palm.z - wrist.z
                palm_facing_inward = wrist_to_palm_z < 0  # Palm z closer to camera than wrist
                
                palm_orientation_boost = 0.2 if palm_facing_inward else 0.0
                
                # Calculate confidence
                confidence = base_confidence + heart_area_boost + finger_posture_boost + palm_orientation_boost
                confidence = min(1.0, confidence)  # Cap at 1.0
                
                # Apply penalties
                confidence *= both_hands_on_chest_penalty
                
                # Update max confidence
                max_confidence = max(max_confidence, confidence)
                
                # Debug output
                if self.DEBUG and self.frame_count % 15 == 0 and confidence > 0.3:
                    hand_str = "left" if is_left_hand else "right"
                    print(f"Clutching chest ({hand_str}): on_heart={palm_in_heart_area}, " +
                        f"fingers_curled={fingers_curled}, facing_in={palm_facing_inward}, " +
                        f"both_hands_penalty={both_hands_on_chest_penalty:.1f}, conf={confidence:.2f}")
        
        return max_confidence


    def detect_crossed_wrists(self, pose_landmarks):
        """
        Detect if person has wrists crossed (captive/abuse/hostage signal)
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0

        # Get relevant landmarks
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        # FIXED: More lenient wrist proximity check
        wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
        wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)

        # Calculate shoulder width for scaling
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        # FIXED: Increased thresholds for better detection
        max_horizontal_dist = shoulder_width * 0.35  # Increased from 0.25
        max_vertical_dist = shoulder_width * 0.35    # Increased from 0.25
        
        wrists_close = (wrist_horizontal_distance < max_horizontal_dist and 
                    wrist_vertical_distance < max_vertical_dist)
        
        # FIXED: Even if wrists aren't perfectly close, give more confidence
        proximity_conf = max(0.0, 1.0 - (
            wrist_horizontal_distance / max_horizontal_dist + 
            wrist_vertical_distance / max_vertical_dist) / 2)
        
        # Only apply strict crossing check if proximity is decent
        if proximity_conf < 0.3:
            return proximity_conf * 0.4  # Still return some confidence
        
        # FIXED: More lenient arm crossing detection
        left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
        right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
        
        # FIXED: Allow more flexibility in arm directions
        left_going_right = left_arm_vector[0] > -0.05  # Allow slight leftward movement
        right_going_left = right_arm_vector[0] < 0.05   # Allow slight rightward movement
        
        # Calculate crossing confidence
        crossing_conf = 0.0
        if left_going_right and right_going_left:
            crossing_conf = 1.0  # Perfect crossing
        elif left_going_right or right_going_left:
            crossing_conf = 0.6  # Increased from 0.4 for partial crossing
        else:
            # FIXED: Still give some confidence even if arms aren't perfectly crossing
            crossing_conf = 0.3
        
        # Position checks for wrists
        chest_center_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # FIXED: More lenient height check
        optimal_wrist_y = chest_center_y + (shoulder_width * 0.1)
        wrist_avg_y = (left_wrist.y + right_wrist.y) / 2
        
        # FIXED: Allow wider vertical range for detection
        wrist_too_high = wrist_avg_y < (nose.y - 0.05)  # More lenient
        
        # Get hip position to check if wrists are too low
        if hasattr(self.mp_pose.PoseLandmark, 'LEFT_HIP'):
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            hip_center_y = (left_hip.y + right_hip.y) / 2
            wrist_too_low = wrist_avg_y > hip_center_y + 0.1  # Added buffer
        else:
            # Approximate hip position if not available
            estimated_hip_y = chest_center_y + (shoulder_width * 1.0)
            wrist_too_low = wrist_avg_y > estimated_hip_y + 0.1  # Added buffer
        
        # FIXED: Less severe height penalties
        height_penalty = 1.0
        if wrist_too_high:
            height_penalty = 0.7  # Less severe (was 0.5)
        elif wrist_too_low:
            height_penalty = 0.8  # Less severe (was 0.7)
        
        # FIXED: More lenient centering requirement
        wrist_avg_x = (left_wrist.x + right_wrist.x) / 2
        horizontal_offset = abs(wrist_avg_x - chest_center_x)
        center_conf = max(0.0, 1.0 - (horizontal_offset / (shoulder_width * 0.7)))  # More lenient
        
        # FIXED: Adjusted weights to prioritize proximity and crossing
        confidence = (
            proximity_conf * 0.40 +    # Increased weight (was 0.30)
            crossing_conf * 0.40 +     # Increased weight (was 0.35)
            center_conf * 0.20         # Reduced weight (was combined 0.35)
        ) * height_penalty
        
        # FIXED: Boost confidence for clear cases with more lenient criteria
        if proximity_conf > 0.6 and crossing_conf > 0.6:
            confidence = max(confidence, 0.65)  # Lower threshold, higher minimum
        
        return confidence

    def detect_person_fallen(self, pose_landmarks):
        """
        Detect if person is in a horizontal position (fallen)
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0

        # Get key landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_foot = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        # Calculate the angle of the body with respect to vertical
        shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2,
                            (left_shoulder.y + right_shoulder.y) / 2]
        hip_midpoint = [(left_hip.x + right_hip.x) / 2,
                        (left_hip.y + right_hip.y) / 2]

        # Vector from hip to shoulder
        body_vector = [shoulder_midpoint[0] - hip_midpoint[0],
                    shoulder_midpoint[1] - hip_midpoint[1]]

        # Calculate angle with vertical (y-axis)
        vertical_vector = [0, -1]  # Negative because image coordinates increase downward

        # Calculate the angle using dot product
        dot_product = body_vector[0] * vertical_vector[0] + body_vector[1] * vertical_vector[1]
        body_magnitude = np.sqrt(body_vector[0]**2 + body_vector[1]**2)
        vertical_magnitude = np.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)

        # Prevent division by zero
        if body_magnitude * vertical_magnitude == 0:
            return 0.0

        cosine_angle = dot_product / (body_magnitude * vertical_magnitude)
        # Ensure within valid range
        cosine_angle = max(-1.0, min(1.0, cosine_angle))
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

        # CRITICAL CHECK: For fallen detection, body should be nearly horizontal
        # This means angle should be close to 90 degrees with vertical
        horizontal_confidence = 1.0 - (abs(angle_degrees - 90) / 90)
        
        # If body is not sufficiently horizontal, return very low confidence
        if horizontal_confidence < 0.7:  # Require at least 70% horizontality
            return max(0.0, horizontal_confidence * 0.1)  # Very low confidence
        
        # NEW: Check body length axis
        # For a person truly lying down, the body length (nose to feet) should be horizontal
        # Calculate foot midpoint
        foot_midpoint = [(left_foot.x + right_foot.x) / 2,
                        (left_foot.y + right_foot.y) / 2]
        
        # Vector from nose to feet
        body_length_vector = [foot_midpoint[0] - nose.x, 
                            foot_midpoint[1] - nose.y]
        
        # Calculate angle of body length with horizontal
        horizontal_vector = [1, 0]
        
        length_dot_product = body_length_vector[0] * horizontal_vector[0] + body_length_vector[1] * horizontal_vector[1]
        length_magnitude = np.sqrt(body_length_vector[0]**2 + body_length_vector[1]**2)
        
        # Prevent division by zero
        if length_magnitude == 0:
            length_angle_confidence = 0.0
        else:
            cosine_length_angle = length_dot_product / length_magnitude  # horizontal_magnitude is 1
            cosine_length_angle = max(-1.0, min(1.0, cosine_length_angle))
            length_angle_radians = np.arccos(cosine_length_angle)
            length_angle_degrees = np.degrees(length_angle_radians)
            
            # For true horizontal position, this angle should be small (body aligned with horizontal)
            # or close to 180 (body aligned but in opposite direction)
            min_angle = min(length_angle_degrees, 180 - length_angle_degrees)
            length_angle_confidence = 1.0 - (min_angle / 90)
        
        # NEW: Check for body flatness (all key points at similar height)
        key_y_values = [
            left_shoulder.y, right_shoulder.y,
            left_hip.y, right_hip.y,
            nose.y
        ]
        
        y_variance = np.var(key_y_values)
        flatness_confidence = max(0.0, 1.0 - (y_variance * 50))  # Lower variance = higher confidence
        
        # NEW: Check height in frame - fallen person is usually lower in frame
        body_center_y = (shoulder_midpoint[1] + hip_midpoint[1]) / 2
        height_in_frame_confidence = min(1.0, body_center_y * 1.2)  # Higher y = lower in frame
        
        # NEW: Check relative foot position for fallen person
        # For lying down, feet should be far from head horizontally
        head_foot_horizontal_dist = abs(foot_midpoint[0] - nose.x)
        relative_body_length = head_foot_horizontal_dist / body_magnitude
        length_confidence = min(1.0, relative_body_length * 2)
        
        # Check if person might be kneeling instead of fallen
        # For kneeling, one or both knees would be bent
        left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate knee angles
        left_knee_vector1 = [left_knee.x - left_hip.x, left_knee.y - left_hip.y]
        left_knee_vector2 = [left_knee.x - left_ankle.x, left_knee.y - left_ankle.y]
        
        right_knee_vector1 = [right_knee.x - right_hip.x, right_knee.y - right_hip.y]
        right_knee_vector2 = [right_knee.x - right_ankle.x, right_knee.y - right_ankle.y]
        
        # Function to calculate angle between vectors
        def vector_angle(v1, v2):
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 * mag2 == 0:
                return 0
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        
        left_knee_angle = vector_angle(left_knee_vector1, left_knee_vector2)
        right_knee_angle = vector_angle(right_knee_vector1, right_knee_vector2)
        
        # Penalize if knees are bent (typical for kneeling, not for fallen)
        knee_penalty = 1.0
        if left_knee_angle < 120 or right_knee_angle < 120:  # Bent knees
            knee_penalty = 0.5  # Strong penalty for bent knees
        
        # Combine all factors with weights
        # Horizontal alignment is most important
        confidence = (
            horizontal_confidence * 0.40 +      # Body angle near 90° to vertical
            length_angle_confidence * 0.20 +    # Body length alignment with horizontal
            flatness_confidence * 0.15 +        # Body points at similar height
            height_in_frame_confidence * 0.15 + # Position low in frame
            length_confidence * 0.10            # Head-to-foot stretch
        ) * knee_penalty                        # Penalty for bent knees
        
        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0:
            print(f"Fallen: horz={horizontal_confidence:.2f}, align={length_angle_confidence:.2f}, " +
                f"flat={flatness_confidence:.2f}, height={height_in_frame_confidence:.2f}, " +
                f"length={length_confidence:.2f}, knee_pen={knee_penalty:.1f}, final={confidence:.2f}")

        return confidence

    def detect_surrender_pose(self, pose_landmarks, hand_landmarks_list):
        if not pose_landmarks or not hand_landmarks_list or len(hand_landmarks_list) < 2:
            return 0.0
        
    # Get relevant pose landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Check if both hands are raised above shoulders
        both_hands_raised = False
        palms_forward = False
    
    # For each hand, check if it's raised and palm is forward
        left_hand_raised = False
        right_hand_raised = False
        left_palm_forward = False
        right_palm_forward = False
    
        left_wrist = None
        right_wrist = None
    
        for hand_landmarks in hand_landmarks_list:
        # Determine if it's left or right hand
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Check if hand is raised above shoulders
            is_raised = wrist.y < min(left_shoulder.y, right_shoulder.y)
        
        # Check if palm is facing forward
        # For palm forward, fingertips should be closer to camera than wrist
            middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Better palm forward detection using multiple finger points
            fingertips_z = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].z
        ]
            avg_fingertip_z = sum(fingertips_z) / len(fingertips_z)
            is_palm_forward = avg_fingertip_z < palm_center.z
        
        # Determine if it's left or right hand based on position relative to shoulders
            if wrist.x < (left_shoulder.x + right_shoulder.x) / 2:
                left_hand_raised = is_raised
                left_palm_forward = is_palm_forward
                left_wrist = wrist
            else:
                right_hand_raised = is_raised
                right_palm_forward = is_palm_forward
                right_wrist = wrist
    
    # Check for movement to differentiate from waving
        gesture = self.gestures[GestureType.SURRENDER_POSE]
    
    # Initialize position history if needed
        if "wrist_positions" not in gesture:
            gesture["wrist_positions"] = {"left": [], "right": []}
    
    # Add current positions if hands are detected
        if left_wrist:
            gesture["wrist_positions"]["left"].append((left_wrist.x, left_wrist.y))
        if right_wrist:
            gesture["wrist_positions"]["right"].append((right_wrist.x, right_wrist.y))
    
    # Keep only recent history
        max_history = 10
        for hand in ["left", "right"]:
            if len(gesture["wrist_positions"][hand]) > max_history:
                gesture["wrist_positions"][hand] = gesture["wrist_positions"][hand][-max_history:]
    
    # Calculate stability (low movement is good for surrender pose)
        stability_score = 1.0
        if left_hand_raised and len(gesture["wrist_positions"]["left"]) > 3:
            left_x = [p[0] for p in gesture["wrist_positions"]["left"]]
            left_y = [p[1] for p in gesture["wrist_positions"]["left"]]
            left_variance = np.var(left_x) + np.var(left_y)
        # Low variance = higher stability
            left_stability = max(0.0, 1.0 - (left_variance * 2000)) 
            stability_score = min(stability_score, left_stability)
    
        if right_hand_raised and len(gesture["wrist_positions"]["right"]) > 3:
            right_x = [p[0] for p in gesture["wrist_positions"]["right"]]
            right_y = [p[1] for p in gesture["wrist_positions"]["right"]]
            right_variance = np.var(right_x) + np.var(right_y)
        # Low variance = higher stability
            right_stability = max(0.0, 1.0 - (right_variance * 2000))
            stability_score = min(stability_score, right_stability)
    
    # Calculate confidence based on hand positions
        if left_hand_raised and right_hand_raised:
        # Get height above shoulders
            if left_wrist and right_wrist:
            # How high above shoulders
                left_height = left_shoulder.y - left_wrist.y
                right_height = right_shoulder.y - right_wrist.y
            
            # Normalize to a reasonable range (0-1)
                height_confidence = min(1.0, (left_height + right_height) / 0.8)
            
            # How symmetrical the pose is (both hands at similar height)
                symmetry = 1.0 - min(1.0, abs(left_height - right_height) * 5.0)
            
            # Check palm orientation - important for surrender pose
                palm_orientation = 0.0
                if left_palm_forward and right_palm_forward:
                    palm_orientation = 1.0
                elif left_palm_forward or right_palm_forward:
                    palm_orientation = 0.5
            
            # Arms should be somewhat apart for surrender pose
                hands_apart = 0.0
                if left_wrist and right_wrist:
                    wrist_distance = abs(left_wrist.x - right_wrist.x)
                # Ideal distance is around shoulder width
                    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                    if wrist_distance > shoulder_width * 0.7:
                        hands_apart = 1.0
                    else:
                        hands_apart = wrist_distance / (shoulder_width * 0.7)
            
            # Combine factors for overall confidence with stability as key factor
                confidence = (
                height_confidence * 0.2 +  # How high hands are raised
                symmetry * 0.2 +           # Symmetry of pose
                palm_orientation * 0.3 +   # Palm orientation
                hands_apart * 0.1 +        # Hands appropriately apart
                stability_score * 0.2      # Stability (not moving much)
            )
            
            # Debug visualization
                if self.DEBUG and self.frame_count % 10 == 0 and confidence > 0.3:
                    print(f"Surrender: height={height_confidence:.2f}, sym={symmetry:.2f}, " +
                     f"palms={palm_orientation:.1f}, apart={hands_apart:.2f}, stable={stability_score:.2f}, conf={confidence:.2f}")
            
                return confidence
    
        return 0.0

    def detect_hand_over_mouth(self, pose_landmarks, hand_landmarks_list):
        """
        Detect if a hand is covering the mouth area
        Returns confidence score between 0-1
        """
        if not pose_landmarks or not hand_landmarks_list:
            return 0.0

        # Estimate mouth position from face landmarks
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]

        # Better mouth position estimation using multiple face landmarks
        face_center_x = nose.x
        eye_y = (left_eye.y + right_eye.y) / 2

        # Mouth is below the nose and at a distance proportional to eye-nose distance
        estimated_mouth_x = face_center_x
        eye_nose_distance = nose.y - eye_y
        estimated_mouth_y = nose.y + eye_nose_distance * 0.8

        # Define mouth area (larger to improve detection)
        mouth_radius = 0.15  # Increased radius for better detection

        max_confidence = 0.0

        # Check if any hand is over the mouth area
        for hand_landmarks in hand_landmarks_list:
            # Check multiple points on the hand for better detection
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

            # Calculate distances from different parts of hand to estimated mouth
            palm_distance = np.sqrt((palm_center.x - estimated_mouth_x)**2 +
                                    (palm_center.y - estimated_mouth_y)**2)

            index_distance = np.sqrt((index_tip.x - estimated_mouth_x)**2 +
                                     (index_tip.y - estimated_mouth_y)**2)

            middle_distance = np.sqrt((middle_tip.x - estimated_mouth_x)**2 +
                                      (middle_tip.y - estimated_mouth_y)**2)

            # Use the shortest distance from any hand part to mouth
            min_distance = min(palm_distance, index_distance, middle_distance)

            # More lenient check for hand near mouth
            hand_near_mouth = min_distance < mouth_radius

            if hand_near_mouth:
                # Calculate proximity confidence - closer means higher confidence
                proximity_confidence = 1.0 - (min_distance / mouth_radius)

                # Z-axis check is optional - if palm is facing inward, boost confidence
                palm_facing_inward = palm_center.z < wrist.z
                z_boost = 1.2 if palm_facing_inward else 1.0

                # Calculate final confidence with higher minimum
                confidence = max(0.4, proximity_confidence) * z_boost
                max_confidence = max(max_confidence, min(1.0, confidence))

                # Debug visualization
                if self.DEBUG and self.frame_count % 10 == 0:
                    print(f"Hand over mouth: dist={min_distance:.3f}, prox_conf={proximity_confidence:.2f}, " +
                          f"z_boost={z_boost:.1f}, final_conf={confidence:.2f}")

        return max_confidence


    def resolve_gesture_conflicts(self, confidences):
        """
        Advanced conflict resolution between gestures that shouldn't be detected simultaneously.
        Returns modified confidences dictionary.
        """
        modified_confidences = confidences.copy()
        
        # First, identify the primary gesture candidate (highest confidence)
        primary_gesture = max(confidences.items(), key=lambda x: x[1])
        primary_type, primary_conf = primary_gesture
        
        # FIXED: Only apply conflict resolution if primary gesture has meaningful confidence
        if primary_conf < 0.3:  # Increased threshold to avoid early conflict resolution
            return modified_confidences
        
        # Define incompatible gesture groups (mutually exclusive gestures)
        # These are gestures that physically cannot occur together
        
        incompatible_groups = [
            # Group 1: Standing/upper body gestures - can't do these while fallen
            {
                GestureType.CROSSED_WRISTS, 
                GestureType.CLUTCHING_CHEST,
                GestureType.SURRENDER_POSE,
                GestureType.HAND_OVER_MOUTH,
                GestureType.WAVING_ONE_ARM
            },
            
            # Group 2: Lower body positions - can't do multiple at once
            {
                GestureType.ONE_KNEE_DOWN,
                GestureType.PERSON_FALLEN
            },
            
            # Group 3: Arm/hand specific conflicts
            {
                GestureType.CROSSED_WRISTS,
                GestureType.CLUTCHING_CHEST,
                GestureType.SURRENDER_POSE, 
                GestureType.WAVING_ONE_ARM
            },
            
            # Group 4: Hand over mouth vs other hand gestures
            {
                GestureType.HAND_OVER_MOUTH,
                GestureType.CLUTCHING_CHEST,
                GestureType.CROSSED_WRISTS
            }
        ]
        
        # FIXED: Less aggressive suppression for specific problematic pairs
        critical_conflicts = [
            # Your specific issue: crossed arms vs clutching chest
            (GestureType.CROSSED_WRISTS, GestureType.CLUTCHING_CHEST, 0.7),  # Reduced from 0.85
            
            # Fallen person vs kneeling 
            (GestureType.PERSON_FALLEN, GestureType.ONE_KNEE_DOWN, 0.8),  # Reduced from 0.95
            
            # Waving vs surrender (both involve raised arms)
            (GestureType.WAVING_ONE_ARM, GestureType.SURRENDER_POSE, 0.75),  # Reduced from 0.9
            
            # Hand over mouth vs clutching chest (hand can't be both places)
            (GestureType.HAND_OVER_MOUTH, GestureType.CLUTCHING_CHEST, 0.75)  # Reduced from 0.9
        ]
        
        # FIXED: Only apply critical conflicts if the primary confidence is high enough
        if primary_conf > 0.45:  # Only apply when primary is fairly confident
            for g1, g2, suppression_factor in critical_conflicts:
                if primary_type in (g1, g2):
                    other_type = g2 if primary_type == g1 else g1
                    
                    # FIXED: Less aggressive suppression
                    if confidences[other_type] > 0:
                        modified_confidences[other_type] *= (1.0 - suppression_factor * primary_conf)
        
        # FIXED: Only apply group conflicts when primary confidence is higher
        if primary_conf > 0.5:  # Increased from implicit 0.2
            for group in incompatible_groups:
                if primary_type in group:
                    # Find other gestures in this incompatible group
                    competing_gestures = [(g, confidences[g]) for g in group if g != primary_type]
                    
                    # FIXED: Less aggressive suppression for competing gestures
                    for competing_type, competing_conf in competing_gestures:
                        if competing_conf > 0.2:  # Only bother with more meaningful competitors
                            # The stronger the primary, the more we suppress competitors
                            # The closer the competitor, the less aggressive the suppression
                            confidence_gap = max(0.05, primary_conf - competing_conf)
                            suppression_strength = min(0.8, 0.4 + confidence_gap)  # Less aggressive
                            
                            modified_confidences[competing_type] *= (1.0 - suppression_strength)
        
        # FIXED: Less aggressive winner-takes-most approach
        if primary_conf > 0.65:  # Only apply for high confidence (increased from 0.5)
            # Get all gestures with significant confidence
            significant_gestures = [(g, c) for g, c in confidences.items() 
                                if c > 0.4 and g != primary_type]  # Higher threshold
            
            # Apply winner-take-most approach with less suppression
            for gesture, conf in significant_gestures:
                # Less aggressive suppression
                suppression = 0.5 + (primary_conf * 0.2)  # Reduced from 0.6 + 0.3
                modified_confidences[gesture] *= (1.0 - suppression)
        
        return modified_confidences

    def reset(self):
        """Reset detection state"""
        for gesture in GestureType:
            self.gestures[gesture]["history"] = []
            self.gestures[gesture]["start_time"] = None
            self.gestures[gesture]["detected"] = False
            self.gestures[gesture]["confidence"] = 0.0

        self.prev_landmarks = None
        self.frame_count = 0

    def release(self):
        """Release resources"""
        self.hands.close()
        self.pose.close()


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize detector with debug mode enabled
    detector = EmergencyGestureDetector()

    print("Emergency Gesture Detection System")
    print("----------------------------------")
    print("Detecting the following emergency gestures:")
    for gesture in GestureType:
        print(f"- {gesture.name.replace('_', ' ').title()}")
    print("\nGesture must be held for 5 seconds to trigger an alert")
    print("Debug mode is ON - detection regions will be shown")
    print("\nControls:")
    print("  q - Quit")
    print("  r - Reset detection state")
    print("  d - Toggle debug mode")

    # Main loop
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame from camera.")
                break

            # Display instructions on frame
            instruction_text = "Hold gesture for 5s | q:Quit | r:Reset | d:Debug"
            cv2.putText(frame, instruction_text,
                        (frame.shape[1] - 380, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Process the frame with our detector
            processed_frame = detector.process_frame(frame)

            # Display the processed frame
            cv2.imshow('Emergency Gesture Detection', processed_frame)

            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset()
                print("Detection state reset")
            elif key == ord('d'):
                detector.DEBUG = not detector.DEBUG
                print(f"Debug mode {'ON' if detector.DEBUG else 'OFF'}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        detector.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
