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
                                "confidence": 0.0,
                                "rejection_count": 0,  # NEW: Track consecutive rejections
                                "confirmation_count": 0  # NEW: Track consecutive confirmations
                                }
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

        # Detection parameters
        self.REQUIRED_DURATION = 3.0    # Time required to confirm gesture
        self.HISTORY_FRAMES = 15        # Increased from 10 to 15 for more stable detection
        self.DETECTION_THRESHOLD = 0.55  # Increased from 0.5 to 0.55 for higher confidence requirement
        
        # NEW: Hysteresis parameters to prevent flickering and false positives
        self.CONFIDENCE_HYSTERESIS = 0.1  # Confidence buffer to prevent flickering
        self.MINIMUM_CONSECUTIVE_DETECTIONS = 3  # Require multiple consecutive detections
        self.REJECTION_THRESHOLD = 4  # Number of consecutive low confidences to reject a gesture

        # Debug mode for visualization
        self.DEBUG = True  # Set to True to see confidence values and detection regions

        # Previous frame data for comparison
        self.prev_landmarks = None
        self.frame_count = 0
        
        # NEW: Track the currently active gesture to prevent rapid switching
        self.currently_active_gesture = None
        self.active_gesture_timeout = 0  # Counter for maintaining active gesture

        # NEW: Kalman filtering for gesture confidence smoothing
        self.kalman_filters = {gesture: {'estimate': 0.0, 'error': 1.0, 'process_noise': 0.01, 'measurement_noise': 0.1}
                            for gesture in GestureType}

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

        # Check horizontal and vertical proximity of wrists
        wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
        wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)

        # NEW: Position check - wrists should be in front of body (not on chest)
        # Check if wrists are in a reasonable position for crossed wrists signal
        # (typically in front of chest, not ON the chest)
        chest_center_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_center_y = (left_shoulder.y + right_shoulder.y) / 2 + 0.2  # Slightly below shoulders
        
        # At least one wrist should be above waist level (not too low)
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        waist_y = (left_hip.y + right_hip.y) / 2 - 0.1  # Slightly above hips
        
        wrists_above_waist = (left_wrist.y < waist_y) or (right_wrist.y < waist_y)
        if not wrists_above_waist:
            return 0.0  # Wrists too low for the signal

        # Check if wrists are close to each other (crossed)
        wrists_close = wrist_horizontal_distance < 0.1 and wrist_vertical_distance < 0.1

        # NEW: Make sure wrists are actually crossing, not just close
        # Check if arms are crossed (not just wrists touching)
        left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
        right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]

        # Simple check if the directions are opposite
        left_right_dir = left_arm_vector[0] > 0  # Left arm going right
        right_left_dir = right_arm_vector[0] < 0  # Right arm going left

        # If arm directions don't indicate crossing, significantly reduce confidence
        if not (left_right_dir and right_left_dir):
            return max(0.0, 0.1)  # Almost zero confidence if arms aren't crossing

        # NEW: Check for X-pattern (true crossing)
        # Extend arm lines and check if they intersect
        left_arm_slope = left_arm_vector[1] / (left_arm_vector[0] + 1e-6)  # Avoid division by zero
        right_arm_slope = right_arm_vector[1] / (right_arm_vector[0] + 1e-6)
        
        # If slopes have the same sign, they don't form an X (parallel or same direction)
        slopes_indicate_crossing = (left_arm_slope * right_arm_slope <= 0)
        
        # Calculate intersection point of arm lines (using simple line equations)
        # This helps determine if the crossing is clear and in a reasonable position
        intersection_exists = abs(left_arm_slope - right_arm_slope) > 0.1  # Non-parallel
        
        # Calculate proximity confidence based on wrist proximity and arm directions
        proximity_conf = max(0.0, 1.0 - (wrist_horizontal_distance * 5 + wrist_vertical_distance * 5))
        
        # NEW: Position confidence - higher for wrists in front and center of body
        wrist_midpoint_x = (left_wrist.x + right_wrist.x) / 2
        wrist_midpoint_y = (left_wrist.y + right_wrist.y) / 2
        
        # Distance from chest center
        chest_distance = np.sqrt((wrist_midpoint_x - chest_center_x)**2 + 
                                (wrist_midpoint_y - chest_center_y)**2)
        
        # Ideal distance from chest (not too far, not too close)
        position_conf = max(0.0, 1.0 - chest_distance * 2.5)
        
        # Better crossing confidence calculation
        if slopes_indicate_crossing and intersection_exists:
            crossing_conf = 1.0
        elif slopes_indicate_crossing:
            crossing_conf = 0.7
        else:
            crossing_conf = 0.0
        
        # Compute final confidence with weighted factors
        confidence = proximity_conf * 0.3 + crossing_conf * 0.5 + position_conf * 0.2
        
        # NEW: Penalize if hands appear to be clutching chest
        # Hands should not be directly on the chest center
        on_chest_penalty = 1.0
        if chest_distance < 0.15:
            # Hands too close to chest center, likely clutching chest instead
            on_chest_penalty = 0.3
        
        confidence *= on_chest_penalty
        
        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0 and confidence > 0.3:
            print(f"Crossed wrists: prox={proximity_conf:.2f}, cross={crossing_conf:.2f}, " +
                f"pos={position_conf:.2f}, chest_pen={on_chest_penalty:.2f}, final={confidence:.2f}")

        return confidence


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

        # Check if one leg is bent (knee down) and other is relatively straight
        left_leg_bent = left_knee.y > left_hip.y + 0.15 and left_knee.y > left_ankle.y
        right_leg_bent = right_knee.y > right_hip.y + \
            0.15 and right_knee.y > right_ankle.y

        # Only one knee should be down
        one_knee_down = (left_leg_bent and not right_leg_bent) or (
            right_leg_bent and not left_leg_bent)

        if not one_knee_down:
            return 0.0

        # Calculate vertical difference between knees, higher difference means clearer signal
        knee_y_diff = abs(left_knee.y - right_knee.y)

        # Calculate distances to ground
        left_ground_dist = 1.0 - left_knee.y
        right_ground_dist = 1.0 - right_knee.y

        # Lower confidence if both knees are at similar heights
        knee_diff_conf = min(1.0, knee_y_diff * 5.0)

        # Determine which knee is down
        down_knee = left_knee if left_leg_bent else right_knee
        straight_knee = right_knee if left_leg_bent else left_knee

        # Higher confidence if the bent knee is lower in the frame
        height_conf = min(1.0, (down_knee.y - straight_knee.y) * 3.0)

        # Combined confidence
        confidence = max(0.0, (knee_diff_conf * 0.7 + height_conf * 0.3))

        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0 and confidence > 0.3:
            bent_leg = "left" if left_leg_bent else "right"
            print(
                f"One knee down: {bent_leg} bent, diff={knee_y_diff:.2f}, height={height_conf:.2f}, conf={confidence:.2f}")

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
        
        # NEW: Get more body landmarks to establish context
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # NEW: Check for crossed wrists first - if detected, reduce chest detection
        wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
        wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)
        wrists_close = wrist_horizontal_distance < 0.1 and wrist_vertical_distance < 0.1
        
        # If wrists are close to each other, this might be crossed wrists gesture
        wrists_crossed_penalty = 1.0
        if wrists_close:
            # Check if arms appear to be crossing
            left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
            right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
            
            # If arm directions indicate crossing, penalize chest detection
            if (left_arm_vector[0] > 0 and right_arm_vector[0] < 0):
                wrists_crossed_penalty = 0.3
        
        # Define chest area boundaries more precisely
        chest_top = min(left_shoulder.y, right_shoulder.y) + 0.05  # Start lower than shoulders
        chest_bottom = chest_top + 0.25  # Smaller chest height
        chest_left = max(left_shoulder.x - 0.1, 0.25)  # Less expanded, min x=0.25
        chest_right = min(right_shoulder.x + 0.1, 0.75)  # Less expanded, max x=0.75
        
        # NEW: Define a smaller "heart area" in center-left of chest for higher confidence
        heart_center_x = chest_left + (chest_right - chest_left) * 0.4  # Left side of chest
        heart_center_y = chest_top + (chest_bottom - chest_top) * 0.5
        heart_radius = 0.08  # Smaller focused area
        
        max_confidence = 0.0
        best_hand_points = 0
        total_hand_points = 6  # Total points we check
        
        # Check if any hand is on the chest
        for hand_landmarks in hand_landmarks_list:
            # Use multiple hand points for better detection
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Check if any part of the hand is in the chest area
            hand_points = [palm, wrist, index_mcp, pinky_mcp, thumb_tip, index_tip]
            points_in_chest = sum(1 for point in hand_points if
                                chest_left <= point.x <= chest_right and
                                chest_top <= point.y <= chest_bottom)
            
            # NEW: Check points in heart area for higher confidence
            points_in_heart = sum(1 for point in hand_points if
                                np.sqrt((point.x - heart_center_x)**2 + 
                                        (point.y - heart_center_y)**2) <= heart_radius)
            
            if points_in_chest > 0:
                # Calculate center of chest
                center_x = (chest_left + chest_right) / 2
                center_y = (chest_top + chest_bottom) / 2
                
                # Find distance to center of chest for primary hand points
                palm_distance = np.sqrt((palm.x - center_x)**2 + (palm.y - center_y)**2)
                wrist_distance = np.sqrt((wrist.x - center_x)**2 + (wrist.y - center_y)**2)
                index_distance = np.sqrt((index_mcp.x - center_x)**2 + (index_mcp.y - center_y)**2)
                
                # NEW: Check if palm is facing the chest (inward)
                # Palm facing chest is a stronger signal for clutching
                palm_facing_inward = False
                if hand_landmarks_list:
                    thumb_joint = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
                    pinky_joint = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
                    
                    # Calculate hand plane normal using cross product
                    v1 = [pinky_joint.x - palm.x, pinky_joint.y - palm.y, pinky_joint.z - palm.z]
                    v2 = [thumb_joint.x - palm.x, thumb_joint.y - palm.y, thumb_joint.z - palm.z]
                    
                    # Cross product to get normal vector (simplified)
                    normal = [v1[1]*v2[2] - v1[2]*v2[1], 
                            v1[2]*v2[0] - v1[0]*v2[2], 
                            v1[0]*v2[1] - v1[1]*v2[0]]
                    
                    # Check if z component is positive (palm facing body/inward)
                    palm_facing_inward = normal[2] > 0
                
                # Base confidence calculation
                base_confidence = 0.3 + (points_in_chest / total_hand_points) * 0.3
                
                # Boost for points in heart area
                heart_boost = (points_in_heart / total_hand_points) * 0.4
                
                # Distance factor - closer to center is better
                min_distance = min(palm_distance, wrist_distance, index_distance)
                distance_factor = max(0, 1.0 - (min_distance * 3.0))
                
                # Clutching chest usually involves the palm
                palm_importance = 0.0
                if chest_left <= palm.x <= chest_right and chest_top <= palm.y <= chest_bottom:
                    palm_importance = 0.2
                
                # Palm orientation bonus
                palm_orientation_bonus = 0.2 if palm_facing_inward else 0.0
                
                # Calculate confidence
                confidence = base_confidence + heart_boost + palm_importance + palm_orientation_bonus
                
                # Limit to reasonable range
                confidence = min(1.0, confidence)
                
                # Apply crossed wrists penalty if applicable
                confidence *= wrists_crossed_penalty
                
                # Keep track of best confidence
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_hand_points = points_in_chest
                
                # Debug visualization
                if self.DEBUG and self.frame_count % 5 == 0:
                    facing_str = "inward" if palm_facing_inward else "outward"
                    print(f"Hand on chest: points={points_in_chest}/{total_hand_points}, heart={points_in_heart}, " +
                        f"palm={facing_str}, dist={min_distance:.3f}, wrist_pen={wrists_crossed_penalty:.1f}, " +
                        f"conf={confidence:.3f}")
        
        return max_confidence

    def detect_person_fallen(self, pose_landmarks):
        """
        Detect if person is in a horizontal position (fallen)
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0

        # Get hip and shoulder landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        # Calculate the angle of the body with respect to vertical
        shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2,
                             (left_shoulder.y + right_shoulder.y) / 2]
        hip_midpoint = [(left_hip.x + right_hip.x) / 2,
                        (left_hip.y + right_hip.y) / 2]

        # Vector from hip to shoulder
        body_vector = [shoulder_midpoint[0] - hip_midpoint[0],
                       shoulder_midpoint[1] - hip_midpoint[1]]

        # Calculate angle with vertical (y-axis)
        vertical_vector = [0, 1]

        # Calculate the angle
        dot_product = body_vector[0] * vertical_vector[0] + \
            body_vector[1] * vertical_vector[1]
        body_magnitude = np.sqrt(body_vector[0]**2 + body_vector[1]**2)
        vertical_magnitude = np.sqrt(
            vertical_vector[0]**2 + vertical_vector[1]**2)

        # Prevent division by zero
        if body_magnitude * vertical_magnitude == 0:
            return 0.0

        cosine_angle = dot_product / (body_magnitude * vertical_magnitude)
        # Ensure within valid range
        cosine_angle = max(-1.0, min(1.0, cosine_angle))
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

        # Calculate how horizontal the body is (90 degrees would be perfectly horizontal)
        horizontal_confidence = 1.0 - (abs(angle_degrees - 90) / 90)

        # Alternative method: check if head is at similar height as torso
        # This helps detect when person is lying down
        head_torso_height_diff = abs(nose.y - shoulder_midpoint[1])
        head_aligned_horizontally = head_torso_height_diff < 0.15

        # Less strict requirement for position in frame
        body_center_y = (shoulder_midpoint[1] + hip_midpoint[1]) / 2
        is_low_in_frame = body_center_y > 0.5  # Lower 50% of frame

        # Calculate position confidence
        position_confidence = min(1.0, body_center_y)

        # Relax stability constraint or make it optional
        stability_confidence = 1.0
        if self.prev_landmarks:
            prev_shoulder = [(self.prev_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                              self.prev_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                             (self.prev_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                              self.prev_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2]

            movement = np.sqrt((prev_shoulder[0] - shoulder_midpoint[0])**2 +
                               (prev_shoulder[1] - shoulder_midpoint[1])**2)
            stability_confidence = max(0.5, 1.0 - (movement * 20))

        # Combined confidence calculation with higher weight on horizontal angle
        confidence = (horizontal_confidence * 0.6 +
                      position_confidence * 0.2 +
                      stability_confidence * 0.2)

        # Boost confidence if head is aligned horizontally with torso
        if head_aligned_horizontally:
            confidence = min(1.0, confidence * 1.3)

        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0:
            print(f"Fallen detection: angle={angle_degrees:.1f}°, h_conf={horizontal_confidence:.2f}, " +
                  f"p_conf={position_confidence:.2f}, s_conf={stability_confidence:.2f}, " +
                  f"final={confidence:.2f}")

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

            # NEW: Apply gesture conflict resolution
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

            # Calculate average confidence over history
            if self.gestures[gesture]["history"]:
                avg_confidence = sum(
                    self.gestures[gesture]["history"]) / len(self.gestures[gesture]["history"])
                self.gestures[gesture]["confidence"] = avg_confidence
            else:
                self.gestures[gesture]["confidence"] = 0.0

            # Check if gesture is consistently detected with high confidence
            if self.gestures[gesture]["confidence"] >= self.DETECTION_THRESHOLD:
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
                self.gestures[gesture]["start_time"] = None
                self.gestures[gesture]["detected"] = False

        # Display all gestures with significant confidence (not just top 2)
        # Sort by confidence rather than elapsed time
        active_gestures.sort(key=lambda x: x[2], reverse=True)

        # Display timestamp on frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display all gestures with at least 0.3 confidence (more transparent feedback)
        significant_gestures = [g for g in active_gestures if g[2] >= 0.3]

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

    def resolve_gesture_conflicts(self, confidences):
        """
        Resolve conflicts between gestures that shouldn't be detected simultaneously.
        Returns modified confidences dictionary.
        """
        modified_confidences = confidences.copy()

        # Define incompatible gesture pairs (gestures that shouldn't be detected together)
        incompatible_pairs = [
            # Crossed wrists vs Clutching chest (the issue you mentioned)
            (GestureType.CROSSED_WRISTS, GestureType.CLUTCHING_CHEST),
            
            # Surrender vs Waving one arm (both involve raised arms)
            (GestureType.SURRENDER_POSE, GestureType.WAVING_ONE_ARM),
            
            # One knee down vs Person fallen (can't be both kneeling and fallen)
            (GestureType.ONE_KNEE_DOWN, GestureType.PERSON_FALLEN),
            
            # Hand over mouth vs Clutching chest (hand can't be in both places)
            (GestureType.HAND_OVER_MOUTH, GestureType.CLUTCHING_CHEST)
        ]

        # Resolve conflicts for incompatible pairs
        for g1, g2 in incompatible_pairs:
            if confidences[g1] > 0.2 and confidences[g2] > 0.2:
                # If both gestures have significant confidence, keep only the higher one
                if confidences[g1] > confidences[g2]:
                    # Penalize the weaker gesture based on the stronger one's confidence
                    penalty_factor = min(0.95, confidences[g1])
                    modified_confidences[g2] *= (1.0 - penalty_factor)
                    
                    if self.DEBUG and self.frame_count % 15 == 0:
                        print(f"Conflict resolved: {g1.name} ({confidences[g1]:.2f}) vs {g2.name} ({confidences[g2]:.2f}) - reduced {g2.name} to {modified_confidences[g2]:.2f}")
                else:
                    # Penalize the weaker gesture based on the stronger one's confidence
                    penalty_factor = min(0.95, confidences[g2])
                    modified_confidences[g1] *= (1.0 - penalty_factor)
                    
                    if self.DEBUG and self.frame_count % 15 == 0:
                        print(f"Conflict resolved: {g2.name} ({confidences[g2]:.2f}) vs {g1.name} ({confidences[g1]:.2f}) - reduced {g1.name} to {modified_confidences[g1]:.2f}")

        # Special handling for Crossed Wrists vs Clutching Chest (your specific issue)
        # Improve the detection for crossed wrists by checking wrist positions more strictly
        if confidences[GestureType.CROSSED_WRISTS] > 0.3:
            # Penalize chest detection more aggressively when wrists are definitely crossed
            modified_confidences[GestureType.CLUTCHING_CHEST] *= 0.2

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
