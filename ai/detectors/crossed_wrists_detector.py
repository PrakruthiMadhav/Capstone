# detectors/crossed_wrists_detector.py - IMPROVED VERSION
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

    # Calculate reference measurements for scaling
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # 1. WRIST PROXIMITY CHECK - More lenient
    wrist_horizontal_distance = abs(left_wrist.x - right_wrist.x)
    wrist_vertical_distance = abs(left_wrist.y - right_wrist.y)
    
    # Scale thresholds based on shoulder width for different body sizes
    max_horizontal_dist = shoulder_width * 0.5  # Very lenient horizontal distance
    max_vertical_dist = shoulder_width * 0.4    # More lenient vertical distance
    
    # Calculate if wrists are close enough
    wrists_close = (wrist_horizontal_distance < max_horizontal_dist and 
                  wrist_vertical_distance < max_vertical_dist)
    
    # Calculate proximity confidence - more generous scoring
    proximity_conf = max(0.0, 1.0 - (
        wrist_horizontal_distance / max_horizontal_dist * 0.7 + 
        wrist_vertical_distance / max_vertical_dist * 0.3)) # Weight horizontal more
    
    # Still give minimal confidence even if wrists aren't close
    if proximity_conf < 0.3:
        return proximity_conf * 0.5  # Increased base confidence
    
    # 2. MIDLINE CROSSING CHECK - Key indicator of crossed wrists
    # Check if left wrist is on right side and right wrist is on left side
    body_midline_x = shoulder_center_x
    
    # Add tolerance for midline detection
    midline_tolerance = shoulder_width * 0.05
    left_wrist_on_right = left_wrist.x > (body_midline_x - midline_tolerance)
    right_wrist_on_left = right_wrist.x < (body_midline_x + midline_tolerance)
    
    # Calculate crossing amount - how far past midline
    left_cross_amount = 0
    if left_wrist_on_right:
        left_cross_amount = (left_wrist.x - body_midline_x) / (shoulder_width / 2)
    
    right_cross_amount = 0
    if right_wrist_on_left:
        right_cross_amount = (body_midline_x - right_wrist.x) / (shoulder_width / 2)
    
    # 3. ARM DIRECTION CHECK - More lenient
    left_arm_vector = [left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y]
    right_arm_vector = [right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y]
    
    # Allow more flexibility in arm directions
    left_going_right = left_arm_vector[0] > -0.05  # Allow slight leftward movement
    right_going_left = right_arm_vector[0] < 0.05   # Allow slight rightward movement
    
    # Calculate crossing confidence - more lenient
    crossing_conf = 0.0
    if left_wrist_on_right and right_wrist_on_left:
        # Both wrists crossing midline - high confidence
        crossing_amount = (left_cross_amount + right_cross_amount) / 2
        crossing_conf = min(1.0, 0.7 + crossing_amount * 0.6)  # Base 0.7 + boost
    elif left_wrist_on_right or right_wrist_on_left:
        # One wrist crossing - moderate confidence
        crossing_conf = 0.65  # Increased from typical 0.5
    else:
        # No crossing but wrists are close - low confidence
        crossing_conf = 0.3   # Still give some confidence
    
    # 4. ARM DIRECTION CONFIDENCE - More lenient
    arm_direction_conf = 0.0
    if left_going_right and right_going_left:
        arm_direction_conf = 1.0  # Perfect arm directions
    elif left_going_right or right_going_left:
        arm_direction_conf = 0.7  # One arm in correct direction
    else:
        arm_direction_conf = 0.4  # Neither perfect but still possible
    
    # 5. HEIGHT POSITION CHECK - More lenient vertical range
    # Calculate optimal height range (between shoulders and hips)
    
    # Try to use hip landmarks if available
    if hasattr(mp_pose.PoseLandmark, 'LEFT_HIP') and hasattr(mp_pose.PoseLandmark, 'RIGHT_HIP'):
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_center_y = (left_hip.y + right_hip.y) / 2
    else:
        # Approximate hip position if not available
        hip_center_y = shoulder_center_y + shoulder_width * 1.1
    
    # Define acceptable height range with more lenient bounds
    min_height = nose.y  # Top bound at nose height
    max_height = hip_center_y + (shoulder_width * 0.2)  # Lower bound below hips
    
    # Calculate average wrist height
    wrist_avg_y = (left_wrist.y + right_wrist.y) / 2
    
    # Check if wrists are in acceptable height range
    height_in_range = min_height <= wrist_avg_y <= max_height
    
    # Calculate height position confidence
    if height_in_range:
        # Ideal position is between shoulders and mid-torso
        optimal_height = shoulder_center_y + (hip_center_y - shoulder_center_y) * 0.4
        height_diff = abs(wrist_avg_y - optimal_height) / (hip_center_y - shoulder_center_y)
        height_conf = max(0.5, 1.0 - height_diff)  # Minimum 0.5 if in range
    else:
        # Outside ideal range but still give some confidence
        if wrist_avg_y < min_height:
            height_diff = (min_height - wrist_avg_y) / shoulder_width
        else:
            height_diff = (wrist_avg_y - max_height) / shoulder_width
        height_conf = max(0.0, 0.5 - height_diff)  # Reduce from 0.5 based on distance
    
    # 6. HORIZONTAL CENTERING - More lenient
    wrist_avg_x = (left_wrist.x + right_wrist.x) / 2
    horizontal_offset = abs(wrist_avg_x - body_midline_x)
    center_conf = max(0.3, 1.0 - (horizontal_offset / (shoulder_width * 0.8)))  # Minimum 0.3
    
    # 7. WRIST RELATIVE POSITION - Check for X pattern
    x_pattern = (left_wrist.y > right_wrist.y and left_wrist.x > body_midline_x) or \
                (right_wrist.y > left_wrist.y and right_wrist.x < body_midline_x)
    x_pattern_boost = 1.1 if x_pattern else 1.0  # Slight boost for X pattern
    
    # 8. COMBINED CONFIDENCE CALCULATION - Adjusted weights
    # Prioritize proximity and crossing as key indicators
    confidence = (
        proximity_conf * 0.35 +     # High weight for wrist proximity
        crossing_conf * 0.35 +      # High weight for midline crossing
        arm_direction_conf * 0.15 + # Lower weight for arm direction
        height_conf * 0.05 +        # Minimal impact from height
        center_conf * 0.10          # Lower weight for centering
    ) * x_pattern_boost             # Boost for X pattern
    
    # 9. SPECIAL BOOSTS FOR CLEAR CASES - More lenient
    # If key indicators are strong, ensure good confidence
    if proximity_conf > 0.6 and crossing_conf > 0.6:
        confidence = max(confidence, 0.7)  # Ensure at least 0.7 confidence for clear cases
    
    # If perfect crossing with good proximity, ensure very high confidence
    if left_wrist_on_right and right_wrist_on_left and proximity_conf > 0.7:
        confidence = max(confidence, 0.8)  # Ensure at least 0.8 confidence for perfect cases
    
    # 10. DETAILED DEBUG OUTPUT
    if proximity_conf > 0.3 or crossing_conf > 0.3:  # Only log potential detections
        print(f"Crossed wrists: prox={proximity_conf:.2f}, cross={crossing_conf:.2f}, " +
              f"arms_dir={arm_direction_conf:.2f}, height={height_conf:.2f}, " +
              f"center={center_conf:.2f}, boost={x_pattern_boost:.1f}, final={confidence:.2f}")
        
        # Log crossing details
        if left_wrist_on_right and right_wrist_on_left:
            print(f"  Both wrists crossing! L->R: {left_cross_amount:.2f}, R->L: {right_cross_amount:.2f}")
        elif left_wrist_on_right:
            print(f"  Only left wrist crossing right: {left_cross_amount:.2f}")
        elif right_wrist_on_left:
            print(f"  Only right wrist crossing left: {right_cross_amount:.2f}")
    
    return confidence