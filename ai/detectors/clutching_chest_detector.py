# detectors/clutching_chest_detector.py
import numpy as np

def detect_clutching_chest(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Detect if a hand is clutching the chest (health emergency)
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list:
        return 0.0

    # Get chest area landmarks
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    
    # Check for crossed arms pattern - this is NOT clutching chest
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
            if DEBUG and frame_count % 15 == 0:
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
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        palm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        
        # Determine if this is left or right hand (approximation)
        is_left_hand = wrist.x < (left_shoulder.x + right_shoulder.x) / 2
        
        # Check hand cupping/grabbing posture
        # For clutching chest, fingers should be curled or flexed
        
        # Get fingertip positions
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        
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
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
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
            if DEBUG and frame_count % 15 == 0 and confidence > 0.3:
                hand_str = "left" if is_left_hand else "right"
                print(f"Clutching chest ({hand_str}): on_heart={palm_in_heart_area}, " +
                    f"fingers_curled={fingers_curled}, facing_in={palm_facing_inward}, " +
                    f"both_hands_penalty={both_hands_on_chest_penalty:.1f}, conf={confidence:.2f}")
    
    return max_confidence