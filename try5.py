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
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Hands and Pose
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize hand detector with more lenient parameters for far distances
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # Use higher complexity model for better detection
        )

        # Initialize pose detector with more lenient parameters
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use highest complexity model for better distance detection
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,  # Disable segmentation to save processing
            smooth_landmarks=True       # Enable smoothing for better stability
        )

        # Gesture detection parameters
        self.gestures = {gesture: {"detected": False,
                                "start_time": None,
                                "history": [],
                                "confidence": 0.0,
                                "confirmation_count": 0,
                                "rejection_count": 0}
                        for gesture in GestureType}

        # Initialize Kalman filters for each gesture
        self.kalman_filters = {gesture: {
            "estimate": 0.0,
            "error": 1.0,
            "process_noise": 0.01,
            "measurement_noise": 0.1
        } for gesture in GestureType}

        # Currently active gesture (for prioritization)
        self.currently_active_gesture = None
        self.active_gesture_timeout = 0

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
        self.REQUIRED_DURATION = 2.0     # Reduced for faster detection, especially at distance
        self.HISTORY_FRAMES = 15         # Increased for smoother response
        self.DETECTION_THRESHOLD = 0.35  # Reduced threshold for distant subjects
        
        # Parameters for hysteresis and stability
        self.CONFIDENCE_HYSTERESIS = 0.15  # Larger hysteresis for more stable detection
        self.MINIMUM_CONSECUTIVE_DETECTIONS = 2  # Require 2 consecutive detections
        self.REJECTION_THRESHOLD = 5  # Number of low frames before resetting detection
        
        # Debug mode for visualization
        self.DEBUG = True  # Set to True to see confidence values and detection regions

        # Previous frame data for comparison
        self.prev_landmarks = None
        self.frame_count = 0
        
        # NEW: Frame dimensions for normalization
        self.frame_width = 640
        self.frame_height = 480

    def process_frame(self, frame):
        """
        Process a single frame and return the annotated frame
        Returns the frame with annotations
        """
        # Update frame dimensions for normalization
        self.frame_height, self.frame_width = frame.shape[:2]
        
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
            # Draw the bounding box for the detected pose to help with off-center detection
            if self.DEBUG:
                self.draw_pose_bounding_box(frame, pose_results.pose_landmarks)
                
            # NEW: Calculate visibility factor for each detection to handle partial visibility
            visibility_factor = self.calculate_pose_visibility(pose_results.pose_landmarks)
            
            # Detect gestures with improved distance handling
            current_frame_confidences[GestureType.CROSSED_WRISTS] = self.detect_crossed_wrists(
                pose_results.pose_landmarks, visibility_factor)
            current_frame_confidences[GestureType.CLUTCHING_CHEST] = self.detect_clutching_chest(
                pose_results.pose_landmarks, hand_landmarks_list, visibility_factor)
            current_frame_confidences[GestureType.ONE_KNEE_DOWN] = self.detect_one_knee_down(
                pose_results.pose_landmarks, visibility_factor)
            current_frame_confidences[GestureType.PERSON_FALLEN] = self.detect_person_fallen(
                pose_results.pose_landmarks, visibility_factor)
            current_frame_confidences[GestureType.SURRENDER_POSE] = self.detect_surrender_pose(
                pose_results.pose_landmarks, hand_landmarks_list, visibility_factor)
            current_frame_confidences[GestureType.HAND_OVER_MOUTH] = self.detect_hand_over_mouth(
                pose_results.pose_landmarks, hand_landmarks_list, visibility_factor)
            current_frame_confidences[GestureType.WAVING_ONE_ARM] = self.detect_waving_one_arm(
                pose_results.pose_landmarks, visibility_factor)

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
            chest_left = left_shoulder.x - 0.25  # Wider for off-center detection
            chest_right = right_shoulder.x + 0.25  # Wider for off-center detection

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
            mouth_radius = int(0.2 * w)  # Increased for distance detection

            cv2.circle(frame, (mouth_x, mouth_y),
                    mouth_radius, (255, 0, 255), 1)

        # Process detection with the new more robust state management
        active_gestures = self.update_detection_state(current_frame_confidences)

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
    
    def draw_pose_bounding_box(self, frame, pose_landmarks):
        """
        Draw bounding box around detected pose to help visualize off-center detection
        """
        # Initialize min/max values
        min_x, min_y = 1.0, 1.0
        max_x, max_y = 0.0, 0.0
        
        # Find bounding box around visible landmarks
        for landmark in pose_landmarks.landmark:
            if landmark.visibility > 0.5:  # Only consider visible landmarks
                min_x = min(min_x, landmark.x)
                min_y = min(min_y, landmark.y)
                max_x = max(max_x, landmark.x)
                max_y = max(max_y, landmark.y)
        
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape
        min_x_px, min_y_px = int(min_x * w), int(min_y * h)
        max_x_px, max_y_px = int(max_x * w), int(max_y * h)
        
        # Add padding to bounding box
        padding = 20
        min_x_px = max(0, min_x_px - padding)
        min_y_px = max(0, min_y_px - padding)
        max_x_px = min(w, max_x_px + padding)
        max_y_px = min(h, max_y_px + padding)
        
        # Draw the bounding box
        cv2.rectangle(frame, (min_x_px, min_y_px), (max_x_px, max_y_px), (0, 255, 0), 2)
        
        # Display distance estimate
        pose_width = max_x - min_x
        # Rough distance estimate based on width of pose bounding box
        # (The wider the bounding box, the closer the person)
        distance_estimate = 1.0 / max(0.1, pose_width)  # Avoid division by zero
        
        # Normalize to a more intuitive scale (1-10 where 1 is close, 10 is far)
        normalized_distance = min(10, max(1, int(distance_estimate * 5)))
        
        cv2.putText(frame, f"Distance: {normalized_distance}/10", 
                   (min_x_px, min_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def calculate_pose_visibility(self, pose_landmarks):
        """
        Calculate a visibility factor (0-1) based on how much of the pose is visible
        This helps adjust detection thresholds for partial visibility
        """
        # Count number of visible landmarks
        visible_landmarks = 0
        total_landmarks = 0
        
        for landmark in pose_landmarks.landmark:
            total_landmarks += 1
            if landmark.visibility > 0.5:
                visible_landmarks += 1
        
        visibility_factor = visible_landmarks / total_landmarks
        
        # For extreme cases, ensure at least a minimum visibility factor
        visibility_factor = max(0.6, visibility_factor)
        
        return visibility_factor

    def update_detection_state(self, current_frame_confidences):
        """
        More robust state management with hysteresis and Kalman filtering
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
                elif self.gestures[gesture]["start_time"] is not None:
                    # Still include in active gestures if we started timing
                    elapsed_time = current_time - self.gestures[gesture]["start_time"]
                    active_gestures.append(
                        (gesture, elapsed_time, self.gestures[gesture]["confidence"]))
        
        # Prioritize the currently active gesture to prevent rapid switching
        if self.currently_active_gesture is not None:
            self.active_gesture_timeout -= 1
            if self.active_gesture_timeout <= 0:
                self.currently_active_gesture = None
        
        return active_gestures

    def apply_kalman_filtering(self, current_confidences):
        """
        Apply Kalman filtering to smooth confidence values
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


    def detect_one_knee_down(self, pose_landmarks, visibility_factor=1.0):
        """
        Detect if person has one knee down (about to collapse/weakness)
        Now improved for distance and off-center detection
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

        # Check visibility of relevant landmarks
        key_landmarks = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot]
        key_visibility = [lm.visibility for lm in key_landmarks]
        avg_visibility = np.mean(key_visibility)
        
        # If key landmarks are not visible enough, reduce confidence
        if avg_visibility < 0.6:
            visibility_penalty = max(0.5, avg_visibility)
        else:
            visibility_penalty = 1.0

        # Calculate vectors for thigh and calf, adjusting for off-center issues
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
        
        # IMPROVED: Even more lenient knee bend threshold for distance cases
        left_leg_bent = left_knee_angle < 150  # Increased from 140
        right_leg_bent = right_knee_angle < 150  # Increased from 140
        
        # Check if one leg is bent (knee down) and other is relatively straight
        one_knee_down = (left_leg_bent and not right_leg_bent) or (right_leg_bent and not left_leg_bent)
        
        # IMPROVED: Give even more confidence for partial knee detection
        if not one_knee_down:
            # If both legs are bent but one is more bent than the other
            if left_leg_bent and right_leg_bent:
                angle_diff = abs(left_knee_angle - right_knee_angle)
                if angle_diff > 20:  # Less strict difference (was 30)
                    one_knee_down = True
                else:
                    return 0.1  # Still give minimal confidence
            else:
                return 0.0  # No knee down
        
        # Calculate vertical difference between knees
        knee_y_diff = abs(left_knee.y - right_knee.y)
        
        # Check which knee is down
        down_knee = left_knee if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else right_knee
        down_foot = left_foot if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else right_foot
        straight_knee = right_knee if (left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle)) else left_knee
        
        # IMPROVED: Much more lenient position checks for distance detection
        # The down knee should be lower in frame than the straight knee
        proper_knee_position = down_knee.y >= straight_knee.y - 0.1  # Added more tolerance
        
        # IMPROVED: Much more lenient foot position check
        foot_knee_distance = np.sqrt((down_foot.x - down_knee.x)**2 + (down_foot.y - down_knee.y)**2)
        foot_close_to_knee = foot_knee_distance < 0.35  # Increased from 0.25
        
        # IMPROVED: More lenient ground detection with dynamic threshold based on visibility
        # Calculate approx ground based on the down foot and knee
        estimated_ground_y = max(down_foot.y, down_knee.y)
        
        # Down knee should be close to the ground, but be more lenient for distant/partial visibility
        knee_ground_dist = abs(1.0 - down_knee.y)
        knee_near_ground = knee_ground_dist < (0.5 * (2.0 - visibility_factor))  # Dynamic threshold
        
        # Calculate body position relative to knee
        hip_midpoint_y = (left_hip.y + right_hip.y) / 2
        
        # Body should be higher than down knee
        body_above_knee = hip_midpoint_y < down_knee.y + 0.15  # More tolerance
        
        # IMPROVED: More lenient body distance check
        body_knee_dist = abs(hip_midpoint_y - down_knee.y)
        proper_body_dist = 0.03 < body_knee_dist < 0.7  # Widened range
        
        # Calculate confidence components
        angle_conf = 0.0
        if left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle):
            angle_conf = max(0.0, min(1.0, (180 - left_knee_angle) / 90))
        else:
            angle_conf = max(0.0, min(1.0, (180 - right_knee_angle) / 90))
        
        # IMPROVED: More generous position confidence
        position_conf = 0.0
        if proper_knee_position:
            position_conf = min(1.0, knee_y_diff * 8.0)  # Increased multiplier
        else:
            # Still give even more confidence even if knees are at similar height
            position_conf = min(0.7, knee_y_diff * 5.0)  # More generous scaling
        
        ground_conf = 0.0
        if knee_near_ground:
            ground_conf = max(0.0, 1.0 - (knee_ground_dist * 2.0))  # Less harsh penalty
        else:
            # Still give some confidence even if knee isn't perfectly near ground
            ground_conf = max(0.0, 0.4 - (knee_ground_dist * 0.5))
            
        body_position_conf = 0.0
        if body_above_knee:
            body_position_conf = max(0.3, min(1.0, body_knee_dist * 2.0))  # Min 0.3 confidence
        else:
            # More forgiving if body isn't perfectly above knee
            body_position_conf = max(0.0, 0.3 - (abs(hip_midpoint_y - down_knee.y) * 0.5))
        
        foot_position_conf = 0.0
        if foot_close_to_knee:
            foot_position_conf = max(0.0, 1.0 - (foot_knee_distance * 3.0))  # Less harsh penalty
        else:
            # Still give minimal confidence even if foot isn't close to knee
            foot_position_conf = 0.2
        
        # IMPROVED: Adjusted weights for far distance detection
        # Prioritize angle and position which are more clearly visible from distance
        confidence = (
            angle_conf * 0.40 +            # Increased (was 0.35)
            position_conf * 0.30 +         # Increased (was 0.25)
            ground_conf * 0.10 +           # Decreased (was 0.15)
            body_position_conf * 0.10 +    # Decreased (was 0.15)
            foot_position_conf * 0.10      # Unchanged
        )
        
        # Apply visibility factor
        confidence *= visibility_penalty
        
        # IMPROVED: Lower threshold for far-distance knee bends
        if angle_conf > 0.5 and position_conf > 0.3:
            confidence = max(confidence, 0.45)  # Lower threshold for minimum confidence
        
        return confidence

    def detect_waving_one_arm(self, pose_landmarks, visibility_factor=1.0):
        """
        Detect if person is waving one arm (outdoor emergency signal)
        Improved for distance and off-center detection
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0
        
        # Get current landmarks
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # Check visibility of key landmarks
        key_points = [left_wrist, right_wrist, left_shoulder, right_shoulder, left_elbow, right_elbow]
        key_visibilities = [lm.visibility for lm in key_points]
        avg_key_visibility = sum(key_visibilities) / len(key_visibilities)
        
        # If key landmarks are not visible enough, adjust confidence
        visibility_multiplier = min(1.0, avg_key_visibility * 1.3)  # Boost low visibility
    
        # Check if at least one arm is raised - more lenient for distance
        left_arm_raised = left_wrist.y < (left_shoulder.y - 0.03)  # Reduced threshold
        right_arm_raised = right_wrist.y < (right_shoulder.y - 0.03)  # Reduced threshold
    
        # IMPROVED: For off-center detection, account for potential perspective distortion
        # When off-center, one arm might appear more raised than it is
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        frame_center_x = 0.5
        
        # Calculate horizontal offset from center (0 = centered, 1 = far edge)
        center_offset = abs(shoulder_center_x - frame_center_x) * 2
        
        # Apply perspective adjustment - if person is off-center, be more lenient
        if center_offset > 0.3:  # If significantly off-center
            # Determine which side is closer to frame edge
            if shoulder_center_x < frame_center_x:  # Person is on left side
                # Be more lenient with left arm raised check
                left_arm_raised = left_arm_raised or (left_wrist.y < left_shoulder.y)
            else:  # Person is on right side
                # Be more lenient with right arm raised check
                right_arm_raised = right_arm_raised or (right_wrist.y < right_shoulder.y)
    
        # IMPROVED: Differentiate from surrender pose - typically only ONE arm is waving
        both_arms_raised = left_arm_raised and right_arm_raised
    
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
        max_history = 20  # Increased for better pattern detection at distance
        for key in gesture["positions"]:
            if len(gesture["positions"][key]) > max_history:
                gesture["positions"][key] = gesture["positions"][key][-max_history:]
    
        # Need at least a few frames to detect movement
        if len(gesture["positions"]["left_wrist"]) < 4:
            return 0.0
    
        # IMPROVED: Calculate movement for each joint with more sensitivity for distance detection
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
            
                # IMPROVED: More sensitive detection of oscillation patterns
                # Check for oscillation pattern (direction changes)
                x_direction_changes = sum(1 for i in range(2, len(x_positions)) if 
                                    (x_positions[i] - x_positions[i-1]) * 
                                    (x_positions[i-1] - x_positions[i-2]) <= 0)
            
                oscillation_factor = min(3.5, 1.0 + x_direction_changes / 2.5)  # More sensitive
            
                # Boosted movement score for better distance detection
                if x_diffs and y_diffs:
                    x_mad = np.mean(x_diffs)
                    y_mad = np.mean(y_diffs)
                    movement_scores[joint] = ((x_variance + y_variance) * 2000 + 
                                        (x_mad + y_mad) * 800) * oscillation_factor
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
        if left_arm_raised and len(gesture["positions"]["left_wrist"]) > 5:
            left_x = [p[0] for p in gesture["positions"]["left_wrist"]]
            left_y = [p[1] for p in gesture["positions"]["left_wrist"]]
            x_range = max(left_x) - min(left_x)
            y_range = max(left_y) - min(left_y)
        
            if x_range > y_range * 1.2:  # Less strict horizontal dominance (was 1.5)
                side_to_side_pattern = True
                dominant_plane = "horizontal-left"
    
        if right_arm_raised and len(gesture["positions"]["right_wrist"]) > 5:
            right_x = [p[0] for p in gesture["positions"]["right_wrist"]]
            right_y = [p[1] for p in gesture["positions"]["right_wrist"]]
            x_range = max(right_x) - min(right_x)
            y_range = max(right_y) - min(right_y)
        
            if x_range > y_range * 1.2:  # Less strict horizontal dominance
                side_to_side_pattern = True
                dominant_plane = "horizontal-right"
    
        # Calculate confidence based on which arm is moving
        if both_arms_raised:
            # If both arms raised, require asymmetric movement (one arm waving, one still)
            movement_ratio = 0
            if left_arm_movement > 0 and right_arm_movement > 0:
                movement_ratio = max(left_arm_movement, right_arm_movement) / max(0.00001, min(left_arm_movement, right_arm_movement))
        
            if movement_ratio > 2.5:  # Less strict ratio (was 3.0)
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
            # One arm raised - more lenient thresholds for distance
            if left_arm_raised and left_arm_movement > 0.05:  # Lower threshold
                confidence = min(1.0, left_arm_movement)
                moving_arm = "left"
            elif right_arm_raised and right_arm_movement > 0.05:  # Lower threshold
                confidence = min(1.0, right_arm_movement)
                moving_arm = "right"
            else:
                confidence = 0.0
                moving_arm = "none"
    
        # Boost confidence if side-to-side pattern detected
        if side_to_side_pattern and confidence > 0.05:  # Lower threshold
            confidence = min(1.0, confidence * 1.7)  # Higher boost
    
        # Apply visibility and distance adjustments
        confidence *= visibility_multiplier
        
        # IMPROVED: Boost confidence if off-center but clear waving
        if center_offset > 0.3 and confidence > 0.3:
            # Boost confidence for clear off-center waving
            confidence = min(1.0, confidence * (1.0 + center_offset * 0.3))
    
        # Debug output
        if self.DEBUG and self.frame_count % 5 == 0 and confidence > 0.1:
            print(f"Waving arm: left_mov={left_arm_movement:.6f}, right_mov={right_arm_movement:.6f}, " +
                f"both={both_arms_raised}, asym={movement_asymmetry:.2f}, vis={visibility_multiplier:.2f}")
            print(f"Waving {moving_arm} arm: pattern={side_to_side_pattern}, " +
                f"plane={dominant_plane}, off_center={center_offset:.2f}, conf={confidence:.2f}")
    
        return min(1.0, confidence)  # Cap at 1.0

    def detect_clutching_chest(self, pose_landmarks, hand_landmarks_list, visibility_factor=1.0):
        """
        Detect if a hand is clutching the chest (health emergency)
        Improved for distance and off-center detection
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
        
        # Check visibility of key points
        key_landmarks = [left_shoulder, right_shoulder, left_wrist, right_wrist, left_elbow, right_elbow]
        visible_count = sum(1 for lm in key_landmarks if lm.visibility > 0.5)
        key_visibility_ratio = visible_count / len(key_landmarks)
        
        # Adjust thresholds based on visibility and distance
        visibility_adjustment = max(0.7, key_visibility_ratio)
        
        # IMPROVED: Check for crossed arms pattern - this is NOT clutching chest
        # But make the check more lenient for distance and off-center
        
        # Calculate distances with more tolerance
        wrist_distance = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
        elbow_distance = np.sqrt((left_elbow.x - right_elbow.x)**2 + (left_elbow.y - right_elbow.y)**2)
        
        # IMPROVED: More lenient thresholds for crossed arms detection at distance
        wrists_close = wrist_distance < 0.25  # Increased from 0.2
        elbows_far = elbow_distance > 0.2  # Decreased from 0.25
        
        # Calculate shoulder width as a reference scale
        shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                                (left_shoulder.y - right_shoulder.y)**2)
        
        # Adjust thresholds based on apparent size (distance)
        # If shoulder width is small, person is likely far away
        distance_factor = max(0.5, min(1.0, shoulder_width / 0.3))
        
        # Adjust crossed arms detection based on distance
        crossed_arms_penalty = 1.0
        if wrists_close and elbows_far:
            # Check arm directions for crossing
            left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
            right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
            
            # Less strict crossing check for distance
            if (left_arm_vector[0] > -0.05 and right_arm_vector[0] < 0.05):
                crossed_arms_penalty = 0.1  # Stronger penalty for clear crossing
            else:
                crossed_arms_penalty = 0.3  # Lighter penalty for possible crossing
        
        # For true chest clutching, we expect only ONE hand on chest, not both
        # First identify if both hands are on chest area
        
        # Define chest area boundaries more generously for distance detection
        chest_top = min(left_shoulder.y, right_shoulder.y) + 0.05
        chest_bottom = chest_top + 0.3  # Larger area
        chest_left = left_shoulder.x - 0.15 * (2.0 - distance_factor)  # Wider for distance
        chest_right = right_shoulder.x + 0.15 * (2.0 - distance_factor)  # Wider for distance
        
        # Check both wrists - if both are in chest area, it's less likely to be clutching
        left_wrist_on_chest = (chest_left <= left_wrist.x <= chest_right and 
                            chest_top <= left_wrist.y <= chest_bottom)
        right_wrist_on_chest = (chest_left <= right_wrist.x <= chest_right and 
                            chest_top <= right_wrist.y <= chest_bottom)
        
        # If both hands are on chest, reduce confidence - not typical clutching pattern
        both_hands_on_chest_penalty = 1.0
        if left_wrist_on_chest and right_wrist_on_chest:
            both_hands_on_chest_penalty = 0.5  # Penalty but less severe for distance
        
        # Heart area is more specific for clutching - but wider for distance detection
        heart_center_x = chest_left + (chest_right - chest_left) * 0.4  # Left side of chest
        heart_center_y = chest_top + (chest_bottom - chest_top) * 0.4  # Upper chest
        heart_radius = 0.12 * (2.0 - distance_factor)  # Larger radius for distant detection
        
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
            
            # IMPROVED: Check hand cupping/grabbing posture with more leniency
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
            
            # IMPROVED: More lenient fingers curled check for distance
            fingers_curled = avg_fingertip_distance < (0.1 * (2.0 - distance_factor))
            
            # IMPROVED: Wider checks for palm on chest
            palm_on_chest = (chest_left <= palm.x <= chest_right and 
                            chest_top <= palm.y <= chest_bottom)
            
            # More lenient check for palm in heart area
            palm_in_heart_area = np.sqrt((palm.x - heart_center_x)**2 + 
                                        (palm.y - heart_center_y)**2) <= heart_radius
            
            # IMPROVED: Base confidence calculation with more leniency
            if palm_on_chest:
                # Palm on chest is the main indicator
                base_confidence = 0.5
                
                # Boost for palm in heart area
                heart_area_boost = 0.3 if palm_in_heart_area else 0.0
                
                # Boost for fingers curled (clutching posture)
                finger_posture_boost = 0.2 if fingers_curled else 0.1  # Still give partial boost
                
                # Check if palm is facing the chest (inward) - more lenient
                thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
                wrist_to_palm_z = palm.z - wrist.z
                palm_facing_inward = wrist_to_palm_z < 0.05  # More lenient
                
                palm_orientation_boost = 0.2 if palm_facing_inward else 0.1  # Partial boost
                
                # Calculate confidence
                confidence = base_confidence + heart_area_boost + finger_posture_boost + palm_orientation_boost
                confidence = min(1.0, confidence)  # Cap at 1.0
                
                # Apply penalties
                confidence *= both_hands_on_chest_penalty
                confidence *= crossed_arms_penalty
                
                # Apply visibility adjustment
                confidence *= visibility_adjustment
                
                # Adjust for distance - make detection easier at distance
                if distance_factor < 0.7:  # Person is far away
                    confidence = min(1.0, confidence * 1.2)  # Boost by 20%
                
                # Update max confidence
                max_confidence = max(max_confidence, confidence)
                
                # Debug output
                if self.DEBUG and self.frame_count % 15 == 0 and confidence > 0.2:
                    hand_str = "left" if is_left_hand else "right"
                    print(f"Clutching chest ({hand_str}): on_heart={palm_in_heart_area}, " +
                        f"fingers_curled={fingers_curled}, facing_in={palm_facing_inward}, " +
                        f"dist_factor={distance_factor:.2f}, vis={visibility_adjustment:.2f}, " +
                        f"penalties={both_hands_on_chest_penalty:.1f}/{crossed_arms_penalty:.1f}, conf={confidence:.2f}")
        
        return max_confidence


    def detect_crossed_wrists(self, pose_landmarks, visibility_factor=1.0):
        """
        Detect if person has wrists crossed (captive/abuse/hostage signal)
        Improved for distance and off-center detection
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

        # Check visibility of key landmarks
        key_landmarks = [left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder]
        visible_count = sum(1 for lm in key_landmarks if lm.visibility > 0.6)
        visibility_ratio = visible_count / len(key_landmarks)
        
        # Apply visibility factor
        visibility_multiplier = max(0.6, visibility_ratio)
        
        # IMPROVED: More generous threshold for wrist proximity for distance 
        wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
        wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)

        # Calculate shoulder width for scaling
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        # IMPROVED: Estimate distance based on shoulder width
        distance_factor = max(0.5, min(1.0, shoulder_width / 0.3))
        
        # IMPROVED: Adapt thresholds based on estimated distance
        max_horizontal_dist = shoulder_width * (0.4 + (1.0 - distance_factor) * 0.2)
        max_vertical_dist = shoulder_width * (0.4 + (1.0 - distance_factor) * 0.2)
        
        wrists_close = (wrist_horizontal_distance < max_horizontal_dist and 
                    wrist_vertical_distance < max_vertical_dist)
        
        # IMPROVED: More lenient proximity confidence
        proximity_conf = max(0.0, 1.0 - (
            wrist_horizontal_distance / max_horizontal_dist + 
            wrist_vertical_distance / max_vertical_dist) / 2)
        
        # Only apply strict crossing check if proximity is decent
        if proximity_conf < 0.2:  # Even more lenient threshold
            return proximity_conf * 0.5  # Still return some confidence
        
        # IMPROVED: Even more lenient arm crossing detection
        left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
        right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
        
        # IMPROVED: Much more flexible arm directions
        left_going_right = left_arm_vector[0] > -0.1  # Very lenient
        right_going_left = right_arm_vector[0] < 0.1   # Very lenient
        
        # Calculate crossing confidence
        crossing_conf = 0.0
        if left_going_right and right_going_left:
            crossing_conf = 1.0  # Perfect crossing
        elif left_going_right or right_going_left:
            crossing_conf = 0.7  # Increased for partial crossing
        else:
            # Still give some confidence even if arms aren't perfectly crossing
            crossing_conf = 0.4  # More generous
        
        # Position checks for wrists
        chest_center_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # IMPROVED: Much more lenient height check
        optimal_wrist_y = chest_center_y + (shoulder_width * 0.1)
        wrist_avg_y = (left_wrist.y + right_wrist.y) / 2
        
        # IMPROVED: Allow wider vertical range for detection
        wrist_too_high = wrist_avg_y < (nose.y - 0.02)  # Even more lenient
        
        # Get hip position to check if wrists are too low
        if hasattr(self.mp_pose.PoseLandmark, 'LEFT_HIP'):
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            hip_center_y = (left_hip.y + right_hip.y) / 2
            wrist_too_low = wrist_avg_y > hip_center_y + 0.15  # Added buffer
        else:
            # Approximate hip position if not available
            estimated_hip_y = chest_center_y + (shoulder_width * 1.2)  # More lenient
            wrist_too_low = wrist_avg_y > estimated_hip_y + 0.15  # Added buffer
        
        # IMPROVED: Even less severe height penalties
        height_penalty = 1.0
        if wrist_too_high:
            height_penalty = 0.8  # Less severe (was 0.7)
        elif wrist_too_low:
            height_penalty = 0.9  # Less severe (was 0.8)
        
        # IMPROVED: More lenient centering requirement
        wrist_avg_x = (left_wrist.x + right_wrist.x) / 2
        horizontal_offset = abs(wrist_avg_x - chest_center_x)
        center_conf = max(0.0, 1.0 - (horizontal_offset / (shoulder_width * 0.9)))  # More lenient
        
        # Adjust for off-center detection
        # If person is not centered, give extra leniency in center positioning
        frame_center_x = 0.5
        person_offset = abs(chest_center_x - frame_center_x)
        if person_offset > 0.2:  # Person is significantly off-center
            center_conf = max(center_conf, 0.6)  # Minimum confidence for off-center
        
        # IMPROVED: Adjusted weights to prioritize proximity and crossing
        confidence = (
            proximity_conf * 0.45 +    # Increased weight
            crossing_conf * 0.40 +     # Slightly reduced
            center_conf * 0.15         # Reduced weight
        ) * height_penalty
        
        # Apply visibility multiplier
        confidence *= visibility_multiplier
        
        # IMPROVED: Boost confidence for distance cases
        if distance_factor < 0.7:  # Person appears small (far away)
            confidence = min(1.0, confidence * 1.25)  # 25% boost
        
        # IMPROVED: Boost confidence for clear cases with more lenient criteria
        if proximity_conf > 0.5 and crossing_conf > 0.5:
            confidence = max(confidence, 0.55)  # Higher minimum confidence
        
        return confidence

    def detect_person_fallen(self, pose_landmarks, visibility_factor=1.0):
        """
        Detect if person is in a horizontal position (fallen)
        Improved for distance and off-center detection
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
        
        # Check visibility of key landmarks
        key_points = [left_shoulder, right_shoulder, left_hip, right_hip, nose]
        key_visibilities = [lm.visibility for lm in key_points]
        avg_visibility = sum(key_visibilities) / len(key_points)
        
        # Adjust for visibility issues at distance or partially visible
        visibility_multiplier = max(0.7, min(1.0, avg_visibility * 1.2))
        
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

        # IMPROVED: More lenient horizontal check for distance/off-center
        # This means angle should be closer to 90 degrees with vertical
        horizontal_confidence = 1.0 - (abs(angle_degrees - 90) / 90)
        
        # IMPROVED: Lower threshold for horizontal position at distance
        if horizontal_confidence < 0.6:  # Reduced from 0.7
            return max(0.0, horizontal_confidence * 0.2)  # Some minimal confidence
        
        # IMPROVED: Estimate distance based on body size
        shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                               (left_shoulder.y - right_shoulder.y)**2)
        distance_factor = max(0.5, min(1.0, shoulder_width / 0.3))
        
        # NEW: Check body length axis with more leniency
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
            
            # IMPROVED: More lenient angle check for distance
            # For true horizontal position, this angle should be small or close to 180
            min_angle = min(length_angle_degrees, 180 - length_angle_degrees)
            length_angle_confidence = 1.0 - (min_angle / 100)  # More lenient
        
        # IMPROVED: Check for body flatness with leniency for distance
        key_y_values = [
            left_shoulder.y, right_shoulder.y,
            left_hip.y, right_hip.y,
            nose.y
        ]
        
        y_variance = np.var(key_y_values)
        # More forgiving flatness check for distance
        flatness_confidence = max(0.0, 1.0 - (y_variance * 40))  # Lower multiplier
        
        # IMPROVED: More lenient height in frame - fallen person usually lower in frame
        body_center_y = (shoulder_midpoint[1] + hip_midpoint[1]) / 2
        height_in_frame_confidence = min(1.0, body_center_y * 1.1)  # Slightly less strict
        
        # Check relative foot position with more leniency for distance
        # For lying down, feet should be far from head horizontally
        head_foot_horizontal_dist = abs(foot_midpoint[0] - nose.x)
        relative_body_length = head_foot_horizontal_dist / max(0.2, body_magnitude)
        length_confidence = min(1.0, relative_body_length * 1.8)  # Less strict
        
        # IMPROVED: Check for knee bends with more leniency
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
        
        # IMPROVED: Less severe penalty for bent knees at distance
        knee_penalty = 1.0
        if left_knee_angle < 110 or right_knee_angle < 110:  # More lenient (was 120)
            knee_penalty = 0.7  # Less severe penalty (was 0.5)
        
        # IMPROVED: Adjust weights for distance detection
        # Horizontal alignment is still most important but with different weights
        confidence = (
            horizontal_confidence * 0.45 +      # Increased (was 0.40)
            length_angle_confidence * 0.15 +    # Decreased (was 0.20)
            flatness_confidence * 0.15 +        # Unchanged
            height_in_frame_confidence * 0.15 + # Unchanged
            length_confidence * 0.10            # Unchanged
        ) * knee_penalty * visibility_multiplier
        
        # Apply a distance boost for far-away subjects
        if distance_factor < 0.7:  # Person is far away
            confidence = min(1.0, confidence * 1.2)  # Boost confidence by 20%
        
        # Special boost for very horizontal bodies regardless of other factors
        if horizontal_confidence > 0.85:  # Very horizontal
            confidence = max(confidence, 0.6)  # Ensure at least 60% confidence
        
        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0:
            print(f"Fallen: horz={horizontal_confidence:.2f}, align={length_angle_confidence:.2f}, " +
                f"flat={flatness_confidence:.2f}, height={height_in_frame_confidence:.2f}, " +
                f"dist={distance_factor:.2f}, vis={visibility_multiplier:.2f}, " +
                f"final={confidence:.2f}")

        return confidence

    def detect_surrender_pose(self, pose_landmarks, hand_landmarks_list, visibility_factor=1.0):
        """
        Detect if person has hands raised in surrender pose
        Improved for distance and off-center detection
        Returns confidence score between 0-1
        """
        if not pose_landmarks or not hand_landmarks_list or len(hand_landmarks_list) < 2:
            return 0.0
        
        # Get relevant pose landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate shoulder width as distance reference
        shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                               (left_shoulder.y - right_shoulder.y)**2)
        
        # Estimate distance factor based on shoulder width
        distance_factor = max(0.5, min(1.0, shoulder_width / 0.3))
        
        # Check visibility of key landmarks
        key_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
        visibility_multiplier = max(0.7, min(1.0, key_visibility * 1.2))
        
        # Check if both hands are raised above shoulders - more lenient for distance
        both_hands_raised = False
        palms_forward = False
        
        # For each hand, check if it's raised and palm is forward
        left_hand_raised = False
        right_hand_raised = False
        left_palm_forward = False
        right_palm_forward = False
        
        left_wrist = None
        right_wrist = None
        
        # IMPROVED: Handle off-center position
        # Calculate person's offset from center of frame
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        frame_center_x = 0.5
        center_offset = abs(shoulder_center_x - frame_center_x)
        offset_direction = -1 if shoulder_center_x < frame_center_x else 1  # -1=left, 1=right
        
        for hand_landmarks in hand_landmarks_list:
            # Determine if it's left or right hand
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # More lenient raised hand check for distance/off-center
            # For distant subjects, hands may appear closer to shoulders
            raised_threshold = -0.03 * (2.0 - distance_factor)  # Dynamic threshold
            is_raised = wrist.y < min(left_shoulder.y, right_shoulder.y) + raised_threshold
            
            # IMPROVED: Off-center adjustment for raised hand detection
            # When person is not centered, perspective can affect apparent hand height
            if center_offset > 0.2:  # If significantly off-center
                # For left side of frame, left hand may appear less raised
                if offset_direction < 0 and wrist.x < shoulder_center_x:
                    is_raised = is_raised or (wrist.y < left_shoulder.y + 0.05)
                # For right side of frame, right hand may appear less raised
                elif offset_direction > 0 and wrist.x > shoulder_center_x:
                    is_raised = is_raised or (wrist.y < right_shoulder.y + 0.05)
            
            # Check if palm is facing forward - more lenient at distance
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
            
            # IMPROVED: More lenient palm forward detection for distance
            is_palm_forward = avg_fingertip_z < (palm_center.z + 0.03)  # Added tolerance
            
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
        # But be more lenient for distance detection
        stability_score = 1.0
        if left_hand_raised and len(gesture["wrist_positions"]["left"]) > 3:
            left_x = [p[0] for p in gesture["wrist_positions"]["left"]]
            left_y = [p[1] for p in gesture["wrist_positions"]["left"]]
            left_variance = np.var(left_x) + np.var(left_y)
            # Low variance = higher stability but more lenient for distance
            left_stability = max(0.0, 1.0 - (left_variance * 1500))  # Reduced from 2000
            stability_score = min(stability_score, left_stability)
        
        if right_hand_raised and len(gesture["wrist_positions"]["right"]) > 3:
            right_x = [p[0] for p in gesture["wrist_positions"]["right"]]
            right_y = [p[1] for p in gesture["wrist_positions"]["right"]]
            right_variance = np.var(right_x) + np.var(right_y)
            # Low variance = higher stability but more lenient
            right_stability = max(0.0, 1.0 - (right_variance * 1500))  # Reduced from 2000
            stability_score = min(stability_score, right_stability)
        
        # Calculate confidence based on hand positions
        if left_hand_raised and right_hand_raised:
            # Get height above shoulders
            if left_wrist and right_wrist:
                # How high above shoulders - more lenient for distance
                left_height = left_shoulder.y - left_wrist.y
                right_height = right_shoulder.y - right_wrist.y
                
                # Normalize to a reasonable range (0-1) with distance adjustment
                height_confidence = min(1.0, (left_height + right_height) / (0.6 * distance_factor))
                
                # How symmetrical the pose is (both hands at similar height)
                # More forgiving for off-center and distance
                symmetry = 1.0 - min(1.0, abs(left_height - right_height) * 3.0)  # Reduced from 5.0
                
                # Apply off-center adjustment to symmetry
                if center_offset > 0.2:
                    symmetry = max(symmetry, 0.7)  # Ensure minimum symmetry value for off-center
                
                # Check palm orientation - important for surrender pose but more lenient at distance
                palm_orientation = 0.0
                if left_palm_forward and right_palm_forward:
                    palm_orientation = 1.0
                elif left_palm_forward or right_palm_forward:
                    palm_orientation = 0.6  # Increased from 0.5 for leniency
                else:
                    # Still give minimal palm orientation score at distance
                    palm_orientation = 0.3
                
                # Arms should be somewhat apart for surrender pose
                hands_apart = 0.0
                if left_wrist and right_wrist:
                    wrist_distance = abs(left_wrist.x - right_wrist.x)
                    # Ideal distance is around shoulder width but be lenient
                    if wrist_distance > shoulder_width * 0.6:  # Reduced from 0.7
                        hands_apart = 1.0
                    else:
                        hands_apart = wrist_distance / (shoulder_width * 0.6)
                    
                    # Apply off-center adjustment to hands-apart
                    if center_offset > 0.2:
                        hands_apart = max(hands_apart, 0.6)  # Minimum score for off-center
                
                # Combine factors for overall confidence with stability as key factor
                # But reweight for distance detection
                confidence = (
                    height_confidence * 0.25 +     # How high hands are raised
                    symmetry * 0.15 +              # Symmetry of pose (reduced importance)
                    palm_orientation * 0.25 +      # Palm orientation (reduced importance)
                    hands_apart * 0.15 +           # Hands appropriately apart (increased)
                    stability_score * 0.20         # Stability (unchanged)
                ) * visibility_multiplier
                
                # Apply distance boost for far subjects
                if distance_factor < 0.7:  # Person is far away
                    confidence = min(1.0, confidence * 1.25)  # Boost by 25%
                
                # Special boost for clearly raised hands regardless of other factors
                if height_confidence > 0.8:
                    confidence = max(confidence, 0.5)  # Ensure minimum confidence
                
                # Debug visualization
                if self.DEBUG and self.frame_count % 10 == 0 and confidence > 0.2:
                    print(f"Surrender: height={height_confidence:.2f}, sym={symmetry:.2f}, " +
                        f"palms={palm_orientation:.1f}, apart={hands_apart:.2f}, " + 
                        f"stable={stability_score:.2f}, dist={distance_factor:.2f}, " +
                        f"off_center={center_offset:.2f}, conf={confidence:.2f}")
                
                return confidence
        
        return 0.0

    def detect_hand_over_mouth(self, pose_landmarks, hand_landmarks_list, visibility_factor=1.0):
        """
        Detect if a hand is covering the mouth area
        Improved for distance and off-center detection
        Returns confidence score between 0-1
        """
        if not pose_landmarks or not hand_landmarks_list:
            return 0.0

        # Estimate mouth position from face landmarks
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]
        
        # Check visibility of face landmarks
        face_landmarks = [nose, left_eye, right_eye]
        face_visibility = sum(lm.visibility for lm in face_landmarks) / len(face_landmarks)
        
        # If face is not visible enough, reduce confidence
        if face_visibility < 0.7:
            face_visibility_factor = max(0.5, face_visibility)
        else:
            face_visibility_factor = 1.0

        # Better mouth position estimation using multiple face landmarks
        face_center_x = nose.x
        eye_y = (left_eye.y + right_eye.y) / 2

        # Mouth is below the nose and at a distance proportional to eye-nose distance
        estimated_mouth_x = face_center_x
        eye_nose_distance = nose.y - eye_y
        estimated_mouth_y = nose.y + eye_nose_distance * 0.8
        
        # Calculate shoulder width as a distance reference
        if hasattr(pose_landmarks.landmark, '__getitem__'):
            try:
                left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + 
                                    (left_shoulder.y - right_shoulder.y)**2)
                distance_factor = max(0.5, min(1.0, shoulder_width / 0.3))
            except IndexError:
                distance_factor = 0.8  # Default if shoulders not detected
        else:
            distance_factor = 0.8  # Default
        
        # IMPROVED: Define mouth area (much larger for distance detection)
        mouth_radius = 0.2 * (2.0 - distance_factor)  # Dynamic radius based on distance
        
        # Check for off-center position
        frame_center_x = 0.5
        face_offset = abs(face_center_x - frame_center_x)
        
        # If face is off-center, adjust mouth position slightly
        if face_offset > 0.15:
            # Adjust estimated mouth position based on direction of offset
            if face_center_x < frame_center_x:  # Face is on left side
                estimated_mouth_x += 0.02  # Adjust slightly right
            else:  # Face is on right side
                estimated_mouth_x -= 0.02  # Adjust slightly left

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

            # IMPROVED: Much more lenient check for hand near mouth at distance
            hand_near_mouth = min_distance < mouth_radius

            if hand_near_mouth:
                # Calculate proximity confidence - closer means higher confidence
                proximity_confidence = 1.0 - (min_distance / mouth_radius)

                # Z-axis check is optional - if palm is facing inward, boost confidence
                # But be more lenient at distance
                palm_facing_inward = palm_center.z < (wrist.z + 0.03)  # Added tolerance
                z_boost = 1.2 if palm_facing_inward else 1.0

                # IMPROVED: Calculate final confidence with higher minimum for distance
                confidence = max(0.35, proximity_confidence) * z_boost
                
                # Apply face visibility factor
                confidence *= face_visibility_factor
                
                # Apply distance boost
                if distance_factor < 0.7:  # Person is far away
                    confidence = min(1.0, confidence * 1.2)  # Boost by 20%
                
                # Apply off-center boost
                if face_offset > 0.15:
                    confidence = min(1.0, confidence * (1.0 + face_offset * 0.5))
                
                max_confidence = max(max_confidence, min(1.0, confidence))

                # Debug visualization
                if self.DEBUG and self.frame_count % 10 == 0:
                    print(f"Hand over mouth: dist={min_distance:.3f}, radius={mouth_radius:.2f}, " +
                          f"prox_conf={proximity_confidence:.2f}, z_boost={z_boost:.1f}, " +
                          f"off_center={face_offset:.2f}, dist={distance_factor:.2f}, " +
                          f"final_conf={confidence:.2f}")

        return max_confidence


    def resolve_gesture_conflicts(self, confidences):
        """
        Advanced conflict resolution between gestures that shouldn't be detected simultaneously.
        Improved for distance and off-center detection.
        Returns modified confidences dictionary.
        """
        modified_confidences = confidences.copy()
        
        # First, identify the primary gesture candidate (highest confidence)
        primary_gesture = max(confidences.items(), key=lambda x: x[1])
        primary_type, primary_conf = primary_gesture
        
        # IMPROVED: Even more lenient threshold to avoid early conflict resolution
        if primary_conf < 0.25:  # Reduced from 0.3
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
        
        # IMPROVED: Even less aggressive suppression for specific problematic pairs
        critical_conflicts = [
            # Your specific issue: crossed arms vs clutching chest
            (GestureType.CROSSED_WRISTS, GestureType.CLUTCHING_CHEST, 0.6),  # Reduced
            
            # Fallen person vs kneeling 
            (GestureType.PERSON_FALLEN, GestureType.ONE_KNEE_DOWN, 0.7),  # Reduced
            
            # Waving vs surrender (both involve raised arms)
            (GestureType.WAVING_ONE_ARM, GestureType.SURRENDER_POSE, 0.65),  # Reduced
            
            # Hand over mouth vs clutching chest (hand can't be both places)
            (GestureType.HAND_OVER_MOUTH, GestureType.CLUTCHING_CHEST, 0.65)  # Reduced
        ]
        
        # IMPROVED: Only apply critical conflicts if the primary confidence is high enough
        # But with lower threshold for distance detection
        if primary_conf > 0.35:  # Reduced from 0.45
            for g1, g2, suppression_factor in critical_conflicts:
                if primary_type in (g1, g2):
                    other_type = g2 if primary_type == g1 else g1
                    
                    # IMPROVED: Even less aggressive suppression
                    if confidences[other_type] > 0:
                        modified_confidences[other_type] *= (1.0 - suppression_factor * primary_conf)
        
        # IMPROVED: Only apply group conflicts when primary confidence is higher
        # But with reduced threshold for distance detection
        if primary_conf > 0.4:  # Reduced from 0.5
            for group in incompatible_groups:
                if primary_type in group:
                    # Find other gestures in this incompatible group
                    competing_gestures = [(g, confidences[g]) for g in group if g != primary_type]
                    
                    # IMPROVED: Even less aggressive suppression for competing gestures
                    for competing_type, competing_conf in competing_gestures:
                        if competing_conf > 0.15:  # Lower threshold for meaningful competitors
                            # The stronger the primary, the more we suppress competitors
                            # The closer the competitor, the less aggressive the suppression
                            confidence_gap = max(0.05, primary_conf - competing_conf)
                            suppression_strength = min(0.7, 0.35 + confidence_gap)  # Less aggressive
                            
                            modified_confidences[competing_type] *= (1.0 - suppression_strength)
        
        # IMPROVED: Even less aggressive winner-takes-most approach
        if primary_conf > 0.55:  # Reduced from 0.65
            # Get all gestures with significant confidence
            significant_gestures = [(g, c) for g, c in confidences.items() 
                                if c > 0.3 and g != primary_type]  # Lower threshold
            
            # Apply winner-take-most approach with less suppression
            for gesture, conf in significant_gestures:
                # Less aggressive suppression
                suppression = 0.4 + (primary_conf * 0.2)  # Reduced
                modified_confidences[gesture] *= (1.0 - suppression)
        
        return modified_confidences

    def reset(self):
        """Reset detection state"""
        for gesture in GestureType:
            self.gestures[gesture]["history"] = []
            self.gestures[gesture]["start_time"] = None
            self.gestures[gesture]["detected"] = False
            self.gestures[gesture]["confidence"] = 0.0
            self.gestures[gesture]["confirmation_count"] = 0
            self.gestures[gesture]["rejection_count"] = 0

            # Reset Kalman filter
            self.kalman_filters[gesture] = {
                "estimate": 0.0,
                "error": 1.0,
                "process_noise": 0.01,
                "measurement_noise": 0.1
            }

        self.prev_landmarks = None
        self.frame_count = 0
        self.currently_active_gesture = None

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

    # Set resolution - increased resolution for better distance detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased from 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increased from 480

    # Initialize detector with more lenient parameters for distance
    detector = EmergencyGestureDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("🔍 Enhanced Emergency Gesture Detection System 🔍")
    print("----------------------------------------------")
    print("✅ Now optimized for detection at greater distances")
    print("✅ Improved detection for subjects not in center of frame")
    print("✅ More robust against partial visibility")
    print("\nDetecting the following emergency gestures:")
    for gesture in GestureType:
        print(f"- {gesture.name.replace('_', ' ').title()}")
    print(f"\nGesture must be held for {detector.REQUIRED_DURATION} seconds to trigger an alert")
    print("Debug mode is ON - detection regions will be shown")
    print("\nControls:")
    print("  q - Quit")
    print("  r - Reset detection state")
    print("  d - Toggle debug mode")
    print("  + - Increase detection sensitivity (for greater distances)")
    print("  - - Decrease detection sensitivity (for closer subjects)")

    # Main loop
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame from camera.")
                break

            # Display instructions on frame
            instruction_text = "Hold gesture for detection | q:Quit | r:Reset | d:Debug | +/-:Adjust sensitivity"
            cv2.putText(frame, instruction_text,
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Process the frame with our detector
            processed_frame = detector.process_frame(frame)

            # Display the processed frame
            cv2.imshow('Enhanced Emergency Gesture Detection', processed_frame)

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
            elif key == ord('+') or key == ord('='):
                # Increase sensitivity for distance detection
                detector.DETECTION_THRESHOLD = max(0.2, detector.DETECTION_THRESHOLD - 0.05)
                print(f"Sensitivity increased: threshold={detector.DETECTION_THRESHOLD:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease sensitivity
                detector.DETECTION_THRESHOLD = min(0.5, detector.DETECTION_THRESHOLD + 0.05)
                print(f"Sensitivity decreased: threshold={detector.DETECTION_THRESHOLD:.2f}")
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