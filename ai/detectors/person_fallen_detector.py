# detectors/person_fallen_detector.py
import numpy as np

def detect_person_fallen(pose_landmarks, mp_pose, DEBUG=False, frame_count=0):
    """
    Detect if person is in a horizontal position (fallen)
    Returns confidence score between 0-1
    """
    if not pose_landmarks:
        return 0.0

    # Get key landmarks
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

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

    # For fallen detection, body should be nearly horizontal
    # This means angle should be close to 90 degrees with vertical
    horizontal_confidence = 1.0 - (abs(angle_degrees - 90) / 90)
    
    # If body is not sufficiently horizontal, return very low confidence
    if horizontal_confidence < 0.7:  # Require at least 70% horizontality
        return max(0.0, horizontal_confidence * 0.1)  # Very low confidence
    
    # Check body length axis
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
    
    # Check for body flatness (all key points at similar height)
    key_y_values = [
        left_shoulder.y, right_shoulder.y,
        left_hip.y, right_hip.y,
        nose.y
    ]
    
    y_variance = np.var(key_y_values)
    flatness_confidence = max(0.0, 1.0 - (y_variance * 50))  # Lower variance = higher confidence
    
    # Check height in frame - fallen person is usually lower in frame
    body_center_y = (shoulder_midpoint[1] + hip_midpoint[1]) / 2
    height_in_frame_confidence = min(1.0, body_center_y * 1.2)  # Higher y = lower in frame
    
    # Check relative foot position for fallen person
    # For lying down, feet should be far from head horizontally
    head_foot_horizontal_dist = abs(foot_midpoint[0] - nose.x)
    relative_body_length = head_foot_horizontal_dist / body_magnitude
    length_confidence = min(1.0, relative_body_length * 2)
    
    # Check if person might be kneeling instead of fallen
    # For kneeling, one or both knees would be bent
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
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
    if DEBUG and frame_count % 15 == 0:
        print(f"Fallen: horz={horizontal_confidence:.2f}, align={length_angle_confidence:.2f}, " +
            f"flat={flatness_confidence:.2f}, height={height_in_frame_confidence:.2f}, " +
            f"length={length_confidence:.2f}, knee_pen={knee_penalty:.1f}, final={confidence:.2f}")

    return confidence