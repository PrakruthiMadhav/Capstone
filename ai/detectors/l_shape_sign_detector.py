# detectors/l_shape_sign_detector.py
import numpy as np

def detect_l_shape_sign(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Completely rewritten L-shape sign detector with more reliable detection
    Returns confidence score between 0-1
    """
    if not pose_landmarks:
        return 0.0
    
    # CRITICAL CHANGE: Can work with one or two hands
    # If only one hand is making a clear L-shape, that's good enough
    if len(hand_landmarks_list) == 0:
        return 0.0
        
    # Get shoulder landmarks for reference
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    
    # Track individual hand L-shape qualities
    left_hand_quality = 0.0
    right_hand_quality = 0.0
    
    # Track if we've seen each hand
    left_hand_seen = False
    right_hand_seen = False
    
    # Analyze each hand independently
    for hand_landmarks in hand_landmarks_list:
        # Determine if left or right hand based on wrist position
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        is_left_hand = wrist.x < shoulder_midpoint_x
        
        # Flag that we've seen this hand
        if is_left_hand:
            left_hand_seen = True
        else:
            right_hand_seen = True
        
        # Get key landmarks
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        
        # CRITICAL CHANGE: Simplified L-shape detection focusing on core features
        
        # 1. Check if index finger is extended
        # For vertical extension, tip should be above pip and mcp
        index_extended_up = index_tip.y < index_pip.y < index_mcp.y
        
        # Also measure length of extension
        index_extension = np.sqrt(
            (index_tip.x - index_mcp.x)**2 +
            (index_tip.y - index_mcp.y)**2
        )
        
        # 2. Check if thumb is extended horizontally
        # The key is that thumb should be extended at roughly 90 degrees from index
        
        # For left hand, thumb should extend to the right (positive x)
        # For right hand, thumb should extend to the left (negative x)
        thumb_extended_horizontally = False
        if is_left_hand:
            thumb_extended_horizontally = thumb_tip.x > thumb_mcp.x + 0.03
        else:
            thumb_extended_horizontally = thumb_tip.x < thumb_mcp.x - 0.03
            
        # Measure length of thumb extension
        thumb_extension = np.sqrt(
            (thumb_tip.x - thumb_mcp.x)**2 +
            (thumb_tip.y - thumb_mcp.y)**2
        )
        
        # 3. Check if other fingers are curled
        # This is a strong distinguishing feature from surrender pose
        middle_curled = middle_tip.y > middle_pip.y 
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y
        
        # For a stronger check, verify they're actually curled toward palm
        palm_center = index_mcp  # Approximate palm center
        
        # Calculate distances from fingertips to palm
        middle_to_palm = np.sqrt(
            (middle_tip.x - palm_center.x)**2 +
            (middle_tip.y - palm_center.y)**2
        )
        
        ring_to_palm = np.sqrt(
            (ring_tip.x - palm_center.x)**2 +
            (ring_tip.y - palm_center.y)**2
        )
        
        pinky_to_palm = np.sqrt(
            (pinky_tip.x - palm_center.x)**2 +
            (pinky_tip.y - palm_center.y)**2
        )
        
        # Check for sufficient curling (fingertips close to palm)
        middle_truly_curled = middle_curled and middle_to_palm < 0.1
        ring_truly_curled = ring_curled and ring_to_palm < 0.1
        pinky_truly_curled = pinky_curled and pinky_to_palm < 0.1
        
        # Count how many fingers are truly curled
        curled_count = middle_truly_curled + ring_truly_curled + pinky_truly_curled
        
        # 4. Calculate angle between thumb and index
        thumb_vec = np.array([thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y])
        index_vec = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])
        
        # Normalize vectors
        thumb_norm = np.linalg.norm(thumb_vec)
        index_norm = np.linalg.norm(index_vec)
        
        angle_between = 0
        if thumb_norm > 0 and index_norm > 0:
            thumb_vec = thumb_vec / thumb_norm
            index_vec = index_vec / index_norm
            
            dot_product = np.dot(thumb_vec, index_vec)
            dot_product = max(-1.0, min(1.0, dot_product))
            angle_between = np.degrees(np.arccos(dot_product))
        
        # 5. Calculate L-shape quality score
        l_shape_quality = 0.0
        
        # Only if we have the minimum conditions for an L-shape
        if index_extended_up and thumb_extended_horizontally:
            # Calculate component scores
            
            # Angle quality - 90° is ideal
            if 75 <= angle_between <= 105:
                angle_quality = 1.0
            elif 60 <= angle_between < 75 or 105 < angle_between <= 120:
                angle_quality = 0.7
            else:
                angle_quality = 0.3
            
            # Extension quality - both thumb and index should be well extended
            extension_quality = min(1.0, (index_extension + thumb_extension) * 5)
            
            # Curl quality - how well other fingers are curled
            curl_quality = curled_count / 3.0
            
            # Combined quality score with strong weights on defining features
            l_shape_quality = (
                angle_quality * 0.4 +     # Angle is most important
                extension_quality * 0.3 +  # Good extension is important
                curl_quality * 0.3         # Curled fingers help distinguish from other gestures
            )
            
            # CRITICAL: If other fingers aren't curled at all, apply severe penalty
            # This is crucial to distinguish from surrender pose
            if curled_count == 0:
                l_shape_quality *= 0.3  # 70% reduction if no fingers are curled
        
        # Store quality for this hand
        if is_left_hand:
            left_hand_quality = l_shape_quality
        else:
            right_hand_quality = l_shape_quality
        
        # Debug output
        if DEBUG and frame_count % 10 == 0 and l_shape_quality > 0.3:
            hand_name = "Left" if is_left_hand else "Right"
            print(f"{hand_name} L-shape: index_up={index_extended_up}, " +
                  f"thumb_horiz={thumb_extended_horizontally}, " +
                  f"angle={angle_between:.1f}°, " +
                  f"curled={curled_count}/3, " +
                  f"quality={l_shape_quality:.2f}")
    
    # CRITICAL CHANGE: Calculate final confidence based on what we've seen
    # For two-handed tracking, check if both hands make Ls with thumbs facing each other
    if left_hand_seen and right_hand_seen:
        # Both hands detected - check if they both make good L shapes
        if left_hand_quality > 0.6 and right_hand_quality > 0.6:
            # Strong case: both hands making clear L shapes
            final_confidence = (left_hand_quality + right_hand_quality) / 2.0
            
            # Apply a boost for two clear L shapes
            final_confidence = min(1.0, final_confidence * 1.1)
            
            if DEBUG and frame_count % 10 == 0 and final_confidence > 0.5:
                print(f"TWO-HAND L-shape: left={left_hand_quality:.2f}, " +
                      f"right={right_hand_quality:.2f}, final={final_confidence:.2f}")
                
            return final_confidence
        elif left_hand_quality > 0.7 or right_hand_quality > 0.7:
            # One hand is making a very clear L shape - good enough
            return max(left_hand_quality, right_hand_quality) * 0.9
        else:
            # Neither hand is making a clear enough L shape
            return max(left_hand_quality, right_hand_quality) * 0.7
    
    # Only one hand detected or visible
    elif left_hand_seen:
        # For single-hand detection, require a stronger L shape
        if left_hand_quality > 0.7:
            # Very clear L shape on left hand
            return left_hand_quality * 0.9  # Slight reduction for single hand
        else:
            return left_hand_quality * 0.7
    
    elif right_hand_seen:
        # For single-hand detection, require a stronger L shape
        if right_hand_quality > 0.7:
            # Very clear L shape on right hand
            return right_hand_quality * 0.9  # Slight reduction for single hand
        else:
            return right_hand_quality * 0.7
    
    # No hands detected
    return 0.0