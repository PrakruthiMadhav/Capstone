# detectors/one_knee_down_detector.py
import numpy as np

def detect_one_knee_down(pose_landmarks, mp_pose):
    """
    Detect if person has one knee down (about to collapse/weakness)
    Returns confidence score between 0-1
    """
    if not pose_landmarks:
        return 0.0

    # Get relevant landmarks
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

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
    
    # More lenient knee bend threshold
    left_leg_bent = left_knee_angle < 140  # Increased from 120
    right_leg_bent = right_knee_angle < 140  # Increased from 120
    
    # Check if one leg is bent (knee down) and other is relatively straight
    one_knee_down = (left_leg_bent and not right_leg_bent) or (right_leg_bent and not left_leg_bent)
    
    # Give some confidence even if both knees are slightly bent
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
    
    # More lenient position checks
    # The down knee should be lower in frame than the straight knee
    proper_knee_position = down_knee.y >= straight_knee.y - 0.05  # Added tolerance
    
    # More lenient foot position check
    foot_knee_distance = np.sqrt((down_foot.x - down_knee.x)**2 + (down_foot.y - down_knee.y)**2)
    foot_close_to_knee = foot_knee_distance < 0.25  # Increased from 0.15
    
    # More lenient ground detection
    # Calculate approx ground based on the down foot and knee
    estimated_ground_y = max(down_foot.y, down_knee.y)
    
    # Down knee should be close to the ground
    knee_ground_dist = abs(1.0 - down_knee.y)
    knee_near_ground = knee_ground_dist < 0.4  # Increased from 0.3
    
    # Calculate body position relative to knee
    hip_midpoint_y = (left_hip.y + right_hip.y) / 2
    
    # Body should be higher than down knee
    body_above_knee = hip_midpoint_y < down_knee.y + 0.1  # Added tolerance
    
    # More lenient body distance check
    body_knee_dist = abs(hip_midpoint_y - down_knee.y)
    proper_body_dist = 0.05 < body_knee_dist < 0.6  # Widened range (was 0.1-0.5)
    
    # Calculate confidence components
    angle_conf = 0.0
    if left_leg_bent and (not right_leg_bent or left_knee_angle < right_knee_angle):
        angle_conf = max(0.0, min(1.0, (180 - left_knee_angle) / 90))
    else:
        angle_conf = max(0.0, min(1.0, (180 - right_knee_angle) / 90))
    
    # More generous position confidence
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
    
    # Adjusted weights to prioritize angle and position
    confidence = (
        angle_conf * 0.35 +           # Increased (was 0.30)
        position_conf * 0.25 +        # Increased (was 0.20)
        ground_conf * 0.15 +          # Decreased (was 0.20)
        body_position_conf * 0.15 +   # Unchanged
        foot_position_conf * 0.10     # Decreased (was 0.15)
    )
    
    # Lower threshold for clear knee bends
    if angle_conf > 0.6 and position_conf > 0.4:
        confidence = max(confidence, 0.55)  # Lower threshold for minimum confidence
    
    return confidence