# detectors/l_shape_sign_detector.py
import numpy as np

def detect_l_shape_sign(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Detect if a person is making the L shape sign with both hands (index and thumb extended) 
    with thumbs facing each other and index fingers pointing up.
    Returns confidence score between 0-1
    """
    if not pose_landmarks or len(hand_landmarks_list) < 2:
        # Need both hands for proper L shape detection
        return 0.0

    # Dictionary to store data for left and right hands
    hand_data = {
        "left": {"landmarks": None, "l_shape_quality": 0.0, "thumb_direction": None, "index_direction": None},
        "right": {"landmarks": None, "l_shape_quality": 0.0, "thumb_direction": None, "index_direction": None}
    }
    
    # Get shoulder landmarks for reference
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    
    # First pass: identify left and right hands and analyze basic L shapes
    for hand_landmarks in hand_landmarks_list:
        # Determine if left or right hand based on wrist position
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        is_left_hand = wrist.x < shoulder_midpoint_x
        hand_key = "left" if is_left_hand else "right"
        
        # Store landmarks
        hand_data[hand_key]["landmarks"] = hand_landmarks
        
        # Get relevant landmarks for the L shape
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
        
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        
        # 1. Check if index finger is POINTING UP
        # For index pointing up, tip should be significantly higher than MCP
        index_pointing_up = (index_tip.y < index_mcp.y - 0.05) and (index_tip.y < index_pip.y)
        
        # Measure the vertical component of index finger extension
        index_vertical_extension = index_mcp.y - index_tip.y
        
        # 2. Calculate thumb direction vector (from MCP to tip)
        thumb_vec = np.array([thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y, thumb_tip.z - thumb_mcp.z])
        
        # Normalize the vector
        thumb_vec_norm = np.linalg.norm(thumb_vec)
        if thumb_vec_norm > 0:
            thumb_vec = thumb_vec / thumb_vec_norm
        
        # Store thumb direction vector for later comparison
        hand_data[hand_key]["thumb_direction"] = thumb_vec
        
        # 3. Calculate index direction vector (from MCP to tip)
        index_vec = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y, index_tip.z - index_mcp.z])
        
        # Normalize the vector
        index_vec_norm = np.linalg.norm(index_vec)
        if index_vec_norm > 0:
            index_vec = index_vec / index_vec_norm
        
        # Store index direction vector
        hand_data[hand_key]["index_direction"] = index_vec
        
        # 4. Calculate angle between thumb and index finger
        dot_product = np.dot(thumb_vec, index_vec)
        dot_product = max(-1.0, min(1.0, dot_product))  # Ensure within valid range
        angle_between = np.degrees(np.arccos(dot_product))
        
        # 5. Check if other fingers are curled
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        other_fingers_curled_count = middle_curled + ring_curled + pinky_curled
        
        # 6. Check thumb extension - for L shape, thumb should extend horizontally
        # For left hand, thumb should extend to the right (positive x)
        # For right hand, thumb should extend to the left (negative x)
        left_thumb_correct = is_left_hand and thumb_tip.x > thumb_mcp.x
        right_thumb_correct = (not is_left_hand) and thumb_tip.x < thumb_mcp.x
        
        thumb_direction_correct = left_thumb_correct or right_thumb_correct
        
        # 7. Calculate the L shape quality for this hand
        index_extension_quality = min(1.0, index_vertical_extension * 10)  # Scale up for better differentiation
        
        angle_quality = 0.0
        if 80 <= angle_between <= 100:  # Near perpendicular (ideal for L shape)
            angle_quality = 1.0
        elif 70 <= angle_between < 80 or 100 < angle_between <= 110:
            angle_quality = 0.7
        elif 60 <= angle_between < 70 or 110 < angle_between <= 120:
            angle_quality = 0.4
        else:
            angle_quality = 0.2  # Not a good L shape angle
        
        curl_quality = other_fingers_curled_count / 3.0
        
        thumb_direction_quality = 1.0 if thumb_direction_correct else 0.3
        index_up_quality = 1.0 if index_pointing_up else 0.2
        
        # Combined L shape quality score
        l_shape_quality = (
            index_extension_quality * 0.2 +
            angle_quality * 0.3 +
            curl_quality * 0.1 +
            thumb_direction_quality * 0.2 +
            index_up_quality * 0.2
        )
        
        # Store the L shape quality score
        hand_data[hand_key]["l_shape_quality"] = l_shape_quality
        
        # Debug output
        if DEBUG and frame_count % 10 == 0 and l_shape_quality > 0.4:
            hand_name = "Left" if is_left_hand else "Right"
            print(f"{hand_name} L-shape: index_up={index_pointing_up}, " +
                  f"thumb_dir_correct={thumb_direction_correct}, " +
                  f"angle={angle_between:.1f}°, " +
                  f"others_curled={other_fingers_curled_count}/3, " +
                  f"quality={l_shape_quality:.2f}")
    
    # Second pass: check if both hands are present and making L shapes
    if hand_data["left"]["landmarks"] is None or hand_data["right"]["landmarks"] is None:
        # Need both hands for the correct gesture
        return 0.0
        
    # Check quality of L shapes on both hands
    left_quality = hand_data["left"]["l_shape_quality"]
    right_quality = hand_data["right"]["l_shape_quality"]
    
    # If either hand isn't making a good L shape, return low confidence
    if left_quality < 0.6 or right_quality < 0.6:
        return max(0.0, (left_quality + right_quality) / 4)  # Very low confidence
    
    # Check if thumbs are facing each other
    # Get thumb direction vectors
    left_thumb_dir = hand_data["left"]["thumb_direction"]
    right_thumb_dir = hand_data["right"]["thumb_direction"]
    
    # For thumbs facing each other:
    # - Left thumb should point right (positive x)
    # - Right thumb should point left (negative x)
    thumbs_facing_each_other = (left_thumb_dir[0] > 0.7) and (right_thumb_dir[0] < -0.7)
    
    # Get thumb tip positions
    left_thumb_tip = hand_data["left"]["landmarks"].landmark[mp_hands.HandLandmark.THUMB_TIP]
    right_thumb_tip = hand_data["right"]["landmarks"].landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    # Check if thumbs are aligned horizontally (similar y-values)
    thumbs_aligned = abs(left_thumb_tip.y - right_thumb_tip.y) < 0.1
    
    # Check if index fingers are both pointing up
    left_index_dir = hand_data["left"]["index_direction"]
    right_index_dir = hand_data["right"]["index_direction"]
    
    # Index fingers should have strong negative y component (pointing up)
    indexes_pointing_up = (left_index_dir[1] < -0.8) and (right_index_dir[1] < -0.8)
    
    # Calculate final confidence based on all criteria
    # Base confidence is average of both hand L shapes
    base_confidence = (left_quality + right_quality) / 2.0
    
    # Apply multipliers for thumb orientation and index direction
    thumbs_facing_mult = 1.0 if thumbs_facing_each_other else 0.3
    thumbs_aligned_mult = 1.0 if thumbs_aligned else 0.7
    indexes_up_mult = 1.0 if indexes_pointing_up else 0.4
    
    # Calculate final confidence
    final_confidence = base_confidence * thumbs_facing_mult * thumbs_aligned_mult * indexes_up_mult
    
    # Apply threshold for more consistent detection
    if final_confidence < 0.7:  # High threshold for clean detection
        final_confidence *= 0.6  # Reduce low confidences
    
    # Debug output for final determination
    if DEBUG and frame_count % 10 == 0 and final_confidence > 0.3:
        print(f"L-shape (both hands): thumbs_facing={thumbs_facing_each_other}, " +
              f"thumbs_aligned={thumbs_aligned}, indexes_up={indexes_pointing_up}, " +
              f"left_qual={left_quality:.2f}, right_qual={right_quality:.2f}, " +
              f"final_conf={final_confidence:.2f}")
    
    return final_confidence