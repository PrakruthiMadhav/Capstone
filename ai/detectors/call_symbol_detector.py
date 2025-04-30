# detectors/call_symbol_detector.py
import numpy as np

def detect_call_symbol(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Detect if a person is making the call symbol (index and pinky extended, others curled)
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list:
        return 0.0

    # Variables to track if either hand is making the call symbol
    left_hand_call_gesture = False
    right_hand_call_gesture = False
    left_conf = 0.0
    right_conf = 0.0
    
    # Get shoulder landmarks for reference
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    
    for hand_landmarks in hand_landmarks_list:
        # Determine if left or right hand based on wrist position relative to shoulder midpoint
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        is_left_hand = wrist.x < shoulder_midpoint_x
        
        # Get all finger landmarks
        # Tips of fingers
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        # Middle joints (PIP joints)
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        
        # Knuckles (MCP joints)
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        
        # Get palm center (approximated by middle MCP)
        palm_center = middle_mcp
        
        # 1. PRECISE CHECK FOR INDEX FINGER EXTENSION
        # Index should be fully extended - measure in 3D space
        index_extension_dist = np.sqrt(
            (index_tip.x - index_mcp.x)**2 + 
            (index_tip.y - index_mcp.y)**2 + 
            (index_tip.z - index_mcp.z)**2
        )
        
        # Check if index is straight (tips above PIP, PIP above MCP)
        index_straight = (index_tip.y < index_pip.y) and (index_pip.y < index_mcp.y)
        
        # 2. PRECISE CHECK FOR PINKY EXTENSION
        # Pinky should be fully extended - measure in 3D space
        pinky_extension_dist = np.sqrt(
            (pinky_tip.x - pinky_mcp.x)**2 + 
            (pinky_tip.y - pinky_mcp.y)**2 + 
            (pinky_tip.z - pinky_mcp.z)**2
        )
        
        # Check if pinky is straight (tips above PIP, PIP above MCP)
        pinky_straight = (pinky_tip.y < pinky_pip.y) and (pinky_pip.y < pinky_mcp.y)
        
        # 3. PRECISE CHECK FOR MIDDLE AND RING FINGER CURLING
        # Middle and ring should be curled toward palm
        # Check if tips are below PIPs (curled)
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        
        # Also check distance from tips to palm - should be small if curled
        middle_tip_to_palm_dist = np.sqrt(
            (middle_tip.x - palm_center.x)**2 + 
            (middle_tip.y - palm_center.y)**2
        )
        
        ring_tip_to_palm_dist = np.sqrt(
            (ring_tip.x - palm_center.x)**2 + 
            (ring_tip.y - palm_center.y)**2
        )
        
        # Middle and ring are properly curled if tips are close to palm
        middle_properly_curled = middle_curled and (middle_tip_to_palm_dist < 0.08)
        ring_properly_curled = ring_curled and (ring_tip_to_palm_dist < 0.08)
        
        # 4. CHECK THUMB POSITION
        # For call sign, thumb is typically tucked in or touching the curled fingers
        thumb_to_middle_dist = np.sqrt(
            (thumb_tip.x - middle_tip.x)**2 + 
            (thumb_tip.y - middle_tip.y)**2
        )
        
        thumb_properly_positioned = thumb_to_middle_dist < 0.1
        
        # 5. CHECK FOR PROPER EXTENSION CONTRAST
        # Call symbol has strong contrast: index and pinky extended, others curled
        # Calculate extension ratios
        if middle_tip_to_palm_dist > 0 and ring_tip_to_palm_dist > 0:
            # How much more extended are index and pinky compared to middle and ring?
            index_middle_ratio = index_extension_dist / middle_tip_to_palm_dist
            pinky_ring_ratio = pinky_extension_dist / ring_tip_to_palm_dist
            
            strong_extension_contrast = (index_middle_ratio > 2.0) and (pinky_ring_ratio > 2.0)
        else:
            strong_extension_contrast = False
        
        # 6. CHECK ORIENTATION OF HAND
        # For call symbol, hand is usually in "phone call" position
        # (palm facing side of face, fingers pointing up)
        
        # Check if index and pinky are pointing up
        index_pinky_upward = (index_tip.y < index_mcp.y - 0.05) and (pinky_tip.y < pinky_mcp.y - 0.05)
        
        # 7. CALCULATE CONFIDENCE SCORES FOR KEY FEATURES
        
        # Index finger extension quality
        if index_straight:
            index_quality = min(1.0, index_extension_dist * 10)
        else:
            index_quality = 0.0
        
        # Pinky extension quality
        if pinky_straight:
            pinky_quality = min(1.0, pinky_extension_dist * 10)
        else:
            pinky_quality = 0.0
        
        # Middle and ring curling quality
        middle_curl_quality = 1.0 if middle_properly_curled else 0.0
        ring_curl_quality = 1.0 if ring_properly_curled else 0.0
        
        # Combined curl quality
        curl_quality = (middle_curl_quality + ring_curl_quality) / 2.0
        
        # Thumb position quality
        thumb_quality = 1.0 if thumb_properly_positioned else 0.5
        
        # Extension contrast quality
        contrast_quality = 1.0 if strong_extension_contrast else 0.5
        
        # Orientation quality
        orientation_quality = 1.0 if index_pinky_upward else 0.3
        
        # Calculate final confidence with weighted components
        call_symbol_conf = (
            index_quality * 0.25 +          # Index must be extended
            pinky_quality * 0.25 +          # Pinky must be extended
            curl_quality * 0.20 +           # Middle and ring must be curled
            thumb_quality * 0.10 +          # Thumb should be properly positioned
            contrast_quality * 0.10 +       # Clear contrast between extended and curled
            orientation_quality * 0.10      # Hand should be in proper orientation
        )
        
        # Apply threshold for more consistent detection - must be very clear
        if call_symbol_conf < 0.7:  # Higher threshold for clean detection
            call_symbol_conf *= 0.5  # Drastically reduce low confidences
        
        # Store result based on which hand
        if is_left_hand:
            left_hand_call_gesture = call_symbol_conf > 0.7  # Higher threshold
            left_conf = call_symbol_conf
        else:
            right_hand_call_gesture = call_symbol_conf > 0.7  # Higher threshold
            right_conf = call_symbol_conf
            
        # Debug output
        if DEBUG and frame_count % 10 == 0 and call_symbol_conf > 0.3:
            hand_name = "Left" if is_left_hand else "Right"
            print(f"{hand_name} Call Symbol: index={index_straight} ({index_quality:.2f}), " +
                  f"pinky={pinky_straight} ({pinky_quality:.2f}), " +
                  f"mid_curled={middle_properly_curled}, ring_curled={ring_properly_curled}, " +
                  f"contrast={strong_extension_contrast}, orientation={index_pinky_upward}, " +
                  f"conf={call_symbol_conf:.2f}")
    
    # Return the maximum confidence from either hand
    # Either hand can make the call symbol
    if left_hand_call_gesture or right_hand_call_gesture:
        max_conf = max(left_conf, right_conf)
        return max_conf
    else:
        return 0.0