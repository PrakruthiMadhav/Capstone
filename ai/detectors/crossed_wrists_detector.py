# detectors/crossed_wrists_detector.py
import numpy as np

def detect_crossed_wrists(pose_landmarks, mp_pose):
    """
    Detect if person has wrists crossed (captive/abuse/hostage signal)
    Returns confidence score between 0-1
    """
    if not pose_landmarks:
        return 0.0

    # Get relevant landmarks
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # More lenient wrist proximity check
    wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
    wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)

    # Calculate shoulder width for scaling
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    
    # Increased thresholds for better detection
    max_horizontal_dist = shoulder_width * 0.35  # Increased from 0.25
    max_vertical_dist = shoulder_width * 0.35    # Increased from 0.25
    
    wrists_close = (wrist_horizontal_distance < max_horizontal_dist and 
                wrist_vertical_distance < max_vertical_dist)
    
    # Even if wrists aren't perfectly close, give more confidence
    proximity_conf = max(0.0, 1.0 - (
        wrist_horizontal_distance / max_horizontal_dist + 
        wrist_vertical_distance / max_vertical_dist) / 2)
    
    # Only apply strict crossing check if proximity is decent
    if proximity_conf < 0.3:
        return proximity_conf * 0.4  # Still return some confidence
    
    # More lenient arm crossing detection
    left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
    right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
    
    # Allow more flexibility in arm directions
    left_going_right = left_arm_vector[0] > -0.05  # Allow slight leftward movement
    right_going_left = right_arm_vector[0] < 0.05   # Allow slight rightward movement
    
    # Calculate crossing confidence
    crossing_conf = 0.0
    if left_going_right and right_going_left:
        crossing_conf = 1.0  # Perfect crossing
    elif left_going_right or right_going_left:
        crossing_conf = 0.6  # Increased from 0.4 for partial crossing
    else:
        # Still give some confidence even if arms aren't perfectly crossing
        crossing_conf = 0.3
    
    # Position checks for wrists
    chest_center_x = (left_shoulder.x + right_shoulder.x) / 2
    chest_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # More lenient height check
    optimal_wrist_y = chest_center_y + (shoulder_width * 0.1)
    wrist_avg_y = (left_wrist.y + right_wrist.y) / 2
    
    # Allow wider vertical range for detection
    wrist_too_high = wrist_avg_y < (nose.y - 0.05)  # More lenient
    
    # Get hip position to check if wrists are too low
    if hasattr(mp_pose.PoseLandmark, 'LEFT_HIP'):
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_center_y = (left_hip.y + right_hip.y) / 2
        wrist_too_low = wrist_avg_y > hip_center_y + 0.1  # Added buffer
    else:
        # Approximate hip position if not available
        estimated_hip_y = chest_center_y + (shoulder_width * 1.0)
        wrist_too_low = wrist_avg_y > estimated_hip_y + 0.1  # Added buffer
    
    # Less severe height penalties
    height_penalty = 1.0
    if wrist_too_high:
        height_penalty = 0.7  # Less severe (was 0.5)
    elif wrist_too_low:
        height_penalty = 0.8  # Less severe (was 0.7)
    
    # More lenient centering requirement
    wrist_avg_x = (left_wrist.x + right_wrist.x) / 2
    horizontal_offset = abs(wrist_avg_x - chest_center_x)
    center_conf = max(0.0, 1.0 - (horizontal_offset / (shoulder_width * 0.7)))  # More lenient
    
    # Adjusted weights to prioritize proximity and crossing
    confidence = (
        proximity_conf * 0.40 +    # Increased weight (was 0.30)
        crossing_conf * 0.40 +     # Increased weight (was 0.35)
        center_conf * 0.20         # Reduced weight (was combined 0.35)
    ) * height_penalty
    
    # Boost confidence for clear cases with more lenient criteria
    if proximity_conf > 0.6 and crossing_conf > 0.6:
        confidence = max(confidence, 0.65)  # Lower threshold, higher minimum
    
    return confidence