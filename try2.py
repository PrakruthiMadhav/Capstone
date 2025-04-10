import cv2
import mediapipe as mp
import time
import numpy as np
from enum import Enum, auto


class GestureType(Enum):
    """Enum to represent different emergency gestures"""
    CROSSED_WRISTS = auto()    # Captive/abuse/hostage signaling for police
    ARMS_OVER_HEAD = auto()    # Danger from attack/disaster for emergency response
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
            GestureType.ARMS_OVER_HEAD: "🚨 ALERT: Arms over head! Danger from attack or disaster! 🚨",
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
            GestureType.ARMS_OVER_HEAD: (0, 165, 255),    # Orange
            GestureType.CLUTCHING_CHEST: (0, 0, 255),     # Red
            GestureType.ONE_KNEE_DOWN: (255, 191, 0),     # Deep Sky Blue
            GestureType.PERSON_FALLEN: (255, 0, 0),       # Blue
            GestureType.SURRENDER_POSE: (255, 0, 255),    # Magenta
            GestureType.HAND_OVER_MOUTH: (255, 255, 0),   # Cyan
            GestureType.WAVING_ONE_ARM: (128, 0, 128)     # Purple
        }
        
        # Detection parameters - lower threshold for easier detection
        self.REQUIRED_DURATION = 3.0    # Reduced from 5.0 to 3.0 seconds to confirm gesture
        self.HISTORY_FRAMES = 10        # Consider last 10 frames for smoothing
        self.DETECTION_THRESHOLD = 0.5  # Reduced from 0.6 to 0.5 - 50% of frames must show the gesture
        
        # Debug mode for visualization
        self.DEBUG = True  # Set to True to see confidence values and detection regions
        
        # Previous frame data for comparison
        self.prev_landmarks = None
        self.frame_count = 0

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
        
        # Check horizontal and vertical proximity of wrists
        wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
        wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)
        
        # Check if wrists are close to each other (crossed)
        wrists_close = wrist_horizontal_distance < 0.1 and wrist_vertical_distance < 0.1
        
        # Check if arms are somewhat crossed (not just wrists touching)
        left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
        right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
        
        # Simple check if the directions are opposite
        left_right_dir = left_arm_vector[0] > 0  # Left arm going right
        right_left_dir = right_arm_vector[0] < 0  # Right arm going left
        
        # Calculate confidence based on wrist proximity and arm directions
        proximity_conf = max(0.0, 1.0 - (wrist_horizontal_distance * 5 + wrist_vertical_distance * 5))
        crossing_conf = 1.0 if left_right_dir and right_left_dir else 0.0
        
        # Weighted confidence
        confidence = proximity_conf * 0.7 + crossing_conf * 0.3
        
        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0 and confidence > 0.3:
            print(f"Crossed wrists: prox={wrist_horizontal_distance:.3f}h,{wrist_vertical_distance:.3f}v, " +
                 f"cross={crossing_conf:.1f}, conf={confidence:.2f}")
            
        return confidence
        
    def detect_arms_over_head(self, pose_landmarks):
        """
        Detect if person has both arms over head (danger/disaster signal)
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0
            
        # Get relevant landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Calculate how high each arm is raised relative to shoulders
        left_raise_amount = left_shoulder.y - left_wrist.y
        right_raise_amount = right_shoulder.y - right_wrist.y
        
        # Check if arms are raised high above head
        left_arm_raised = left_wrist.y < left_shoulder.y - 0.15  # Higher threshold for over head
        right_arm_raised = right_wrist.y < right_shoulder.y - 0.15  # Higher threshold for over head
        
        # Check if arms are not spread out too much (should be over head, not Y-shape)
        arms_close_together = abs(left_wrist.x - right_wrist.x) < 0.4
        
        # Calculate individual component confidences
        left_height_conf = min(1.0, left_raise_amount * 2.5) if left_arm_raised else 0.0
        right_height_conf = min(1.0, right_raise_amount * 2.5) if right_arm_raised else 0.0
        proximity_conf = max(0.0, 1.0 - abs(left_wrist.x - right_wrist.x) * 1.5)
        
        # Weight the individual confidences
        confidence = (left_height_conf * 0.4 + right_height_conf * 0.4 + proximity_conf * 0.2)
        
        # Both arms must be raised for this gesture
        if not (left_arm_raised and right_arm_raised):
            confidence = 0.0
            
        # Debug output
        if self.DEBUG and self.frame_count % 15 == 0 and confidence > 0.3:
            print(f"Arms over head: left={left_raise_amount:.2f}, right={right_raise_amount:.2f}, " +
                 f"close={arms_close_together}, conf={confidence:.2f}")
        
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
        right_leg_bent = right_knee.y > right_hip.y + 0.15 and right_knee.y > right_ankle.y
        
        # Only one knee should be down
        one_knee_down = (left_leg_bent and not right_leg_bent) or (right_leg_bent and not left_leg_bent)
        
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
            print(f"One knee down: {bent_leg} bent, diff={knee_y_diff:.2f}, height={height_conf:.2f}, conf={confidence:.2f}")
            
        return confidence
        
    def detect_waving_one_arm(self, pose_landmarks):
        """
        Detect if person is waving one arm (outdoor signaling for help)
        Returns confidence score between 0-1
        """
        if not pose_landmarks:
            return 0.0
            
        # Get current landmarks
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Check if at least one arm is raised
        left_arm_raised = left_wrist.y < left_shoulder.y - 0.1
        right_arm_raised = right_wrist.y < right_shoulder.y - 0.1
        
        if not (left_arm_raised or right_arm_raised):
            return 0.0
        
        # Track wrist positions in history to detect wave pattern
        # Use self.gestures to maintain state
        gesture = self.gestures[GestureType.WAVING_ONE_ARM]
        
        # Add wrist positions to tracking history
        if "wrist_positions" not in gesture:
            gesture["wrist_positions"] = {"left": [], "right": []}
        
        # Add current positions
        gesture["wrist_positions"]["left"].append((left_wrist.x, left_wrist.y))
        gesture["wrist_positions"]["right"].append((right_wrist.x, right_wrist.y))
        
        # Keep only recent positions
        max_history = 10
        if len(gesture["wrist_positions"]["left"]) > max_history:
            gesture["wrist_positions"]["left"] = gesture["wrist_positions"]["left"][-max_history:]
            gesture["wrist_positions"]["right"] = gesture["wrist_positions"]["right"][-max_history:]
        
        # Need at least a few frames to detect movement
        if len(gesture["wrist_positions"]["left"]) < 3:
            return 0.0
        
        # Calculate movement variance for each wrist (larger variance = more movement)
        left_x_positions = [pos[0] for pos in gesture["wrist_positions"]["left"]]
        left_y_positions = [pos[1] for pos in gesture["wrist_positions"]["left"]]
        right_x_positions = [pos[0] for pos in gesture["wrist_positions"]["right"]]
        right_y_positions = [pos[1] for pos in gesture["wrist_positions"]["right"]]
        
        # Calculate variance in x and y directions
        if len(left_x_positions) > 1:  # Need at least 2 points for variance
            left_x_variance = np.var(left_x_positions)
            left_y_variance = np.var(left_y_positions)
            right_x_variance = np.var(right_x_positions)
            right_y_variance = np.var(right_y_positions)
            
            # Total movement variance (combine x and y)
            left_variance = left_x_variance + left_y_variance
            right_variance = right_x_variance + right_y_variance
            
            # Determine which arm is moving more
            left_moving_more = left_variance > right_variance * 1.5
            right_moving_more = right_variance > left_variance * 1.5
            
            # Calculate movement score - scale up to make more sensitive
            movement_multiplier = 1000  # Increase sensitivity
            left_movement = min(1.0, left_variance * movement_multiplier)
            right_movement = min(1.0, right_variance * movement_multiplier)
            
            # Determine confidence based on which arm is moving and raised
            confidence = 0.0
            
            if left_arm_raised and left_moving_more:
                confidence = max(0.5, left_movement)  # Minimum 0.5 confidence
                moving_arm = "left"
            elif right_arm_raised and right_moving_more:
                confidence = max(0.5, right_movement)  # Minimum 0.5 confidence
                moving_arm = "right"
            elif left_arm_raised and left_movement > 0.3:  # Lower threshold for any movement
                confidence = max(0.4, left_movement)
                moving_arm = "left"
            elif right_arm_raised and right_movement > 0.3:  # Lower threshold for any movement
                confidence = max(0.4, right_movement)
                moving_arm = "right"
            else:
                moving_arm = "none"
            
            # Debug output
            if self.DEBUG and self.frame_count % 5 == 0 and (left_movement > 0.2 or right_movement > 0.2):
                print(f"Waving arm: left_var={left_variance:.6f}, right_var={right_variance:.6f}")
                print(f"Waving {moving_arm} arm: movement={confidence:.2f}, raised={left_arm_raised or right_arm_raised}")
                
            return confidence
        
        return 0.0

    def detect_clutching_chest(self, pose_landmarks, hand_landmarks_list):
        """
        Detect if a hand is clutching the chest area
        Returns confidence score between 0-1
        """
        if not pose_landmarks or not hand_landmarks_list:
            return 0.0
            
        # Get chest area landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Define chest area boundaries with significantly expanded region
        chest_top = min(left_shoulder.y, right_shoulder.y)
        chest_bottom = chest_top + 0.4  # Much larger chest height
        chest_left = left_shoulder.x - 0.15  # Expanded left boundary
        chest_right = right_shoulder.x + 0.15  # Expanded right boundary
        
        max_confidence = 0.0
        
        # Check if any hand is on the chest
        for hand_landmarks in hand_landmarks_list:
            # Use multiple hand points for better detection
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            
            # Check if any part of the hand is in the chest area
            palm_in_chest = (
                chest_left <= palm.x <= chest_right and
                chest_top <= palm.y <= chest_bottom
            )
            
            wrist_in_chest = (
                chest_left <= wrist.x <= chest_right and
                chest_top <= wrist.y <= chest_bottom
            )
            
            index_in_chest = (
                chest_left <= index_mcp.x <= chest_right and
                chest_top <= index_mcp.y <= chest_bottom
            )
            
            pinky_in_chest = (
                chest_left <= pinky_mcp.x <= chest_right and
                chest_top <= pinky_mcp.y <= chest_bottom
            )
            
            # More relaxed criteria - any part of hand near chest
            hand_near_chest = palm_in_chest or wrist_in_chest or index_in_chest or pinky_in_chest
            
            if hand_near_chest:
                # Calculate center of chest
                center_x = (chest_left + chest_right) / 2
                center_y = (chest_top + chest_bottom) / 2
                
                # Calculate distance from palm to center
                distance = np.sqrt((palm.x - center_x)**2 + (palm.y - center_y)**2)
                
                # Much more lenient calculation with higher base confidence
                confidence = max(0.7, 1.0 - (distance * 1.5))  # Start at 0.7 confidence
                max_confidence = max(max_confidence, confidence)
                
                # Debug visualization - show more frequent updates
                if self.DEBUG and self.frame_count % 5 == 0:
                    print(f"Hand near chest: distance={distance:.3f}, confidence={confidence:.3f}")
        
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
        shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
        hip_midpoint = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
        
        # Vector from hip to shoulder
        body_vector = [shoulder_midpoint[0] - hip_midpoint[0], shoulder_midpoint[1] - hip_midpoint[1]]
        
        # Calculate angle with vertical (y-axis)
        vertical_vector = [0, 1]
        
        # Calculate the angle
        dot_product = body_vector[0] * vertical_vector[0] + body_vector[1] * vertical_vector[1]
        body_magnitude = np.sqrt(body_vector[0]**2 + body_vector[1]**2)
        vertical_magnitude = np.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
        
        # Prevent division by zero
        if body_magnitude * vertical_magnitude == 0:
            return 0.0
            
        cosine_angle = dot_product / (body_magnitude * vertical_magnitude)
        cosine_angle = max(-1.0, min(1.0, cosine_angle))  # Ensure within valid range
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
        """
        Detect hands raised palms forward (surrender/threat position)
        Returns confidence score between 0-1
        """
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
        
        for hand_landmarks in hand_landmarks_list:
            # Determine if it's left or right hand
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Check if hand is raised above shoulders
            is_raised = wrist.y < min(left_shoulder.y, right_shoulder.y)
            
            # Check if palm is facing forward
            # For palm forward, fingertips should be closer to camera than wrist
            middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            is_palm_forward = middle_finger_tip.z < palm_center.z
            
            # Determine if it's left or right hand based on position relative to shoulders
            if wrist.x < (left_shoulder.x + right_shoulder.x) / 2:
                left_hand_raised = is_raised and is_palm_forward
            else:
                right_hand_raised = is_raised and is_palm_forward
        
        # Calculate confidence based on hand positions
        if left_hand_raised and right_hand_raised:
            # Get height above shoulders
            left_wrist = None
            right_wrist = None
            
            for hand_landmarks in hand_landmarks_list:
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                if wrist.x < (left_shoulder.x + right_shoulder.x) / 2:
                    left_wrist = wrist
                else:
                    right_wrist = wrist
            
            if left_wrist and right_wrist:
                # How high above shoulders
                left_height = left_shoulder.y - left_wrist.y
                right_height = right_shoulder.y - right_wrist.y
                
                # Normalize to a reasonable range (0-1)
                height_confidence = min(1.0, (left_height + right_height) / 0.8)
                
                # How symmetrical the pose is (both hands at similar height)
                symmetry = 1.0 - abs(left_height - right_height)
                
                # Combine factors for overall confidence
                confidence = height_confidence * 0.7 + symmetry * 0.3
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
            current_frame_confidences[GestureType.CROSSED_WRISTS] = self.detect_crossed_wrists(pose_results.pose_landmarks)
            current_frame_confidences[GestureType.ARMS_OVER_HEAD] = self.detect_arms_over_head(pose_results.pose_landmarks)
            current_frame_confidences[GestureType.CLUTCHING_CHEST] = self.detect_clutching_chest(pose_results.pose_landmarks, hand_landmarks_list)
            current_frame_confidences[GestureType.ONE_KNEE_DOWN] = self.detect_one_knee_down(pose_results.pose_landmarks)
            current_frame_confidences[GestureType.PERSON_FALLEN] = self.detect_person_fallen(pose_results.pose_landmarks)
            current_frame_confidences[GestureType.SURRENDER_POSE] = self.detect_surrender_pose(pose_results.pose_landmarks, hand_landmarks_list)
            current_frame_confidences[GestureType.HAND_OVER_MOUTH] = self.detect_hand_over_mouth(pose_results.pose_landmarks, hand_landmarks_list)
            current_frame_confidences[GestureType.WAVING_ONE_ARM] = self.detect_waving_one_arm(pose_results.pose_landmarks)
            
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
            left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
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
                         (chest_rect[0] + chest_rect[2], chest_rect[1] + chest_rect[3]), 
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
            
            cv2.circle(frame, (mouth_x, mouth_y), mouth_radius, (255, 0, 255), 1)
        
        # Update gesture histories and process detection
        current_time = time.time()
        
        # Track which gestures to display
        active_gestures = []
        
        for gesture in GestureType:
            # Update history
            self.gestures[gesture]["history"].append(current_frame_confidences[gesture])
            
            # Keep only the last N frames in history
            if len(self.gestures[gesture]["history"]) > self.HISTORY_FRAMES:
                self.gestures[gesture]["history"].pop(0)
            
            # Calculate average confidence over history
            if self.gestures[gesture]["history"]:
                avg_confidence = sum(self.gestures[gesture]["history"]) / len(self.gestures[gesture]["history"])
                self.gestures[gesture]["confidence"] = avg_confidence
            else:
                self.gestures[gesture]["confidence"] = 0.0
            
            # Check if gesture is consistently detected with high confidence
            if self.gestures[gesture]["confidence"] >= self.DETECTION_THRESHOLD:
                # Start timing if we haven't already
                if self.gestures[gesture]["start_time"] is None:
                    self.gestures[gesture]["start_time"] = current_time
                    
                # If the gesture has been held long enough
                elapsed_time = current_time - self.gestures[gesture]["start_time"]
                if elapsed_time >= self.REQUIRED_DURATION and not self.gestures[gesture]["detected"]:
                    self.gestures[gesture]["detected"] = True
                    print(self.gesture_alerts[gesture])
                
                # Add to active gestures for display
                active_gestures.append((gesture, elapsed_time, self.gestures[gesture]["confidence"]))
            else:
                # Reset timing if gesture is not detected consistently
                self.gestures[gesture]["start_time"] = None
                self.gestures[gesture]["detected"] = False
        
        # Display all gestures with significant confidence (not just top 2)
        # Sort by confidence rather than elapsed time
        active_gestures.sort(key=lambda x: x[2], reverse=True)
        
        # Display timestamp on frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
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
                
                cv2.rectangle(frame, (10, y_offset + 10), (10 + bar_width, y_offset + 30), (0, 0, 0), cv2.FILLED)
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