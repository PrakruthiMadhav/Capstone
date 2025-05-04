# detectors/crossed_wrists_detector.py
import numpy as np

def detect_crossed_wrists(pose_landmarks, mp_pose):
    """
    Significantly improved detection of crossed wrists to work in various positions
    including in front of chest or anywhere in frame
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
    
    # Calculate reference measurements for scaling
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    
    # IMPROVED: 1. WRIST PROXIMITY CHECK - Much more lenient
    wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
    wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)
    
    # IMPROVED: Scale thresholds based on shoulder width but much more lenient
    max_horizontal_dist = shoulder_width * 0.6  # Very lenient horizontal distance
    max_vertical_dist = shoulder_width * 0.5    # Very lenient vertical distance
    
    # Calculate if wrists are close enough
    wrists_close = (wrist_horizontal_distance < max_horizontal_dist and 
                  wrist_vertical_distance < max_vertical_dist)
    
    # IMPROVED: Calculate proximity confidence - more generous scoring
    proximity_conf = max(0.3, 1.0 - (
        wrist_horizontal_distance / max_horizontal_dist * 0.6 + 
        wrist_vertical_distance / max_vertical_dist * 0.4))
    
    # IMPROVED: Always give significant baseline confidence if wrists are reasonably close
    if not wrists_close:
        return max(0.0, proximity_conf - 0.2)  # Return low confidence but not zero
    
    # IMPROVED: 2. MIDLINE CROSSING CHECK - The most important factor
    # Check if left wrist is on right side and right wrist is on left side
    body_midline_x = shoulder_center_x
    
    # IMPROVED: Add tolerance for midline detection
    midline_tolerance = shoulder_width * 0.1  # Much more lenient
    left_wrist_on_right = left_wrist.x > (body_midline_x - midline_tolerance)
    right_wrist_on_left = right_wrist.x < (body_midline_x + midline_tolerance)
    
    # IMPROVED: Calculate crossing amount with more leniency
    left_cross_amount = 0
    if left_wrist_on_right:
        left_cross_amount = (left_wrist.x - body_midline_x) / (shoulder_width / 2)
    
    right_cross_amount = 0
    if right_wrist_on_left:
        right_cross_amount = (body_midline_x - right_wrist.x) / (shoulder_width / 2)
    
    # IMPROVED: 3. ARM DIRECTION CHECK - Much more lenient
    left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
    right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
    
    # IMPROVED: Allow far more flexibility in arm directions
    left_going_right = left_arm_vector[0] > -0.1  # Much more lenient
    right_going_left = right_arm_vector[0] < 0.1   # Much more lenient
    
    # IMPROVED: Calculate crossing confidence with stronger values
    crossing_conf = 0.5  # Start with a moderate baseline confidence
    if left_wrist_on_right and right_wrist_on_left:
        # Both wrists crossing midline - high confidence
        crossing_amount = (left_cross_amount + right_cross_amount) / 2
        crossing_conf = min(1.0, 0.7 + crossing_amount * 0.6)  # Base 0.7 + boost
    elif left_wrist_on_right or right_wrist_on_left:
        # One wrist crossing - moderate confidence
        crossing_conf = 0.7  # Increased from typical 0.5
    
    # CRITICALLY IMPORTANT:
    # IMPROVED: 4. COMPLETELY REMOVE HEIGHT POSITION CHECK
    # This allows crossing to be detected anywhere - in front of chest, stomach, etc.
    height_conf = 1.0  # Always perfect height score, no matter where the crossing happens
    
    # IMPROVED: 5. MINIMAL ARM DIRECTION CHECK
    # Arms don't need to be in perfect position, just somewhat crossed
    arm_direction_conf = 0.7  # Give good baseline score regardless
    if left_going_right and right_going_left:
        arm_direction_conf = 1.0  # Perfect arm directions
    elif left_going_right or right_going_left:
        arm_direction_conf = 0.8  # One arm in correct direction
    
    # IMPROVED: 6. Add X-pattern detection (one wrist above the other)
    x_pattern = (left_wrist.y < right_wrist.y and left_wrist.x > right_wrist.x) or \
                (right_wrist.y < left_wrist.y and right_wrist.x > left_wrist.x)
    
    x_pattern_boost = 1.2 if x_pattern else 1.0  # Stronger boost for X pattern
    
    # IMPROVED: 7. COMBINED CONFIDENCE CALCULATION - Adjusted weights
    # Heavily prioritize proximity and crossing as key indicators
    confidence = (
        proximity_conf * 0.3 +     # High weight for wrist proximity
        crossing_conf * 0.5 +      # Very high weight for midline crossing - most important
        arm_direction_conf * 0.2   # Lower weight for arm direction
    ) * x_pattern_boost             # Boost for X pattern
    
    # IMPROVED: 8. Special case for perfect crossing
    # If wrists are definitely crossed (clear X-pattern), ensure very high confidence
    if left_wrist_on_right and right_wrist_on_left and x_pattern:
        confidence = max(confidence, 0.8)  # Ensure at least 0.8 confidence
    
    # IMPROVED: 9. Add explicit boost for chest-area detection 
    # If wrists are crossed in front of chest (common position), boost confidence
    chest_y_min = min(left_shoulder.y, right_shoulder.y)
    
    # Approximate chest bottom
    chest_y_max = chest_y_min + shoulder_width * 0.6
    
    wrists_avg_y = (left_wrist.y + right_wrist.y) / 2
    wrists_in_chest_area = chest_y_min <= wrists_avg_y <= chest_y_max
    
    if wrists_in_chest_area and confidence > 0.6:
        confidence = min(1.0, confidence * 1.15)  # 15% boost for chest position
    
    return min(1.0, confidence)  # Ensure max confidence of 1.0