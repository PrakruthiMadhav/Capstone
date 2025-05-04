# detectors/call_symbol_detector.py 
import numpy as np

def detect_call_symbol(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Improved call symbol detector that explicitly distinguishes from L-shape sign and surrender pose
    Call symbol: index and pinky extended, others curled (like a phone)
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list:
        return 0.0

    # Variables to track if either hand is making the call symbol
    left_hand_call_gesture = False
    right_hand_call_gesture = False
    left_conf = 0.0
    right_conf = 0.0
    
    # Get face landmarks for proximity checks
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    
    # Estimate mouth position
    eye_y = (left_eye.y + right_eye.y) / 2
    face_center_x = nose.x
    eye_nose_distance = nose.y - eye_y
    estimated_mouth_y = nose.y + eye_nose_distance * 0.8
    estimated_ear_y = eye_y
    
    # Get shoulder landmarks for reference
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    
    for hand_landmarks in hand_landmarks_list:
        # Determine if left or right hand based on wrist position
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        is_left_hand = wrist.x < shoulder_midpoint_x
        
        # Get all finger landmarks
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        # Get all joints
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        
        # Get palm center
        palm_center = middle_mcp
        
        # CRITICAL CHECKS FOR CALL SYMBOL:
        
        # 1. INDEX AND PINKY SHOULD BE EXTENDED
        index_extended = index_tip.y < index_mcp.y - 0.05
        pinky_extended = pinky_tip.y < pinky_mcp.y - 0.05
        
        # 2. MIDDLE AND RING SHOULD BE CURLED
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        
        # 3. THUMB SHOULD BE TUCKED OR CLOSE TO PALM
        thumb_to_palm_dist = np.sqrt(
            (thumb_tip.x - palm_center.x)**2 + 
            (thumb_tip.y - palm_center.y)**2
        )
        thumb_tucked = thumb_to_palm_dist < 0.1
        
        # CRITICAL: DISTINGUISH FROM SURRENDER POSE
        # In surrender pose, all fingers are extended
        # In call symbol, only index and pinky are extended
        
        # Check if all fingers are extended (would be surrender pose)
        all_fingers_extended = (
            index_extended and
            middle_tip.y < middle_mcp.y - 0.05 and
            ring_tip.y < ring_mcp.y - 0.05 and
            pinky_extended
        )
        
        # If all fingers are extended, this is not a call symbol
        if all_fingers_extended:
            if DEBUG and frame_count % 20 == 0:
                hand_name = "Left" if is_left_hand else "Right"
                print(f"{hand_name} hand has ALL FINGERS EXTENDED (likely surrender pose, not call symbol)")
            continue
        
        # CRITICAL: DISTINGUISH FROM L-SHAPE
        # In L-shape, thumb and index are extended
        # In call symbol, index and pinky are extended
        
        # Check if thumb is extended horizontally (L-shape pattern)
        thumb_extended = False
        if is_left_hand:
            thumb_extended = thumb_tip.x > thumb_mcp.x + 0.05
        else:
            thumb_extended = thumb_tip.x < thumb_mcp.x - 0.05
        
        # If thumb is extended horizontally and pinky is not, this might be an L-shape
        if thumb_extended and index_extended and not pinky_extended:
            if DEBUG and frame_count % 20 == 0:
                hand_name = "Left" if is_left_hand else "Right"
                print(f"{hand_name} hand shows L-SHAPE pattern (not call symbol)")
            continue
        
        # CRITICAL: If we've reached this point, we have:
        # 1. Not all fingers extended (not surrender pose)
        # 2. Not thumb+index extended only (not L-shape)
        # 3. Need to check for the specific call symbol pattern
        
        # 4. Core call symbol check: Index and pinky extended, middle and ring curled
        call_symbol_pattern = index_extended and pinky_extended and middle_curled and ring_curled
        
        if not call_symbol_pattern:
            continue
            
        # 5. ADDITIONAL CHECKS FOR CALL SYMBOL:
        
        # 6. POSITION CHECK - typically near ear/face for "phone call" gesture
        # For call symbol, the hand should be positioned near the side of the head
        
        # Calculate distances to ear (side of head) and mouth
        # Use ear position based on eye height but to the side
        estimated_ear_x = face_center_x - 0.15 if is_left_hand else face_center_x + 0.15
        
        palm_to_ear_dist = np.sqrt(
            (palm_center.x - estimated_ear_x)**2 + 
            (palm_center.y - estimated_ear_y)**2
        )
        
        palm_to_mouth_dist = np.sqrt(
            (palm_center.x - face_center_x)**2 + 
            (palm_center.y - estimated_mouth_y)**2
        )
        
        # Hand should be somewhat near face for call symbol
        hand_near_face = palm_to_ear_dist < 0.3 or palm_to_mouth_dist < 0.3
        
        # Calculate final confidence:
        
        # Core feature confidences
        index_quality = 1.0 if index_extended else 0.0
        pinky_quality = 1.0 if pinky_extended else 0.0
        middle_curl_quality = 1.0 if middle_curled else 0.3
        ring_curl_quality = 1.0 if ring_curled else 0.3
        thumb_quality = 1.0 if thumb_tucked else 0.5
        
        # Position confidence
        position_quality = 1.0 if hand_near_face else 0.6
        
        # Combined curl quality
        curl_quality = (middle_curl_quality + ring_curl_quality) / 2.0
        
        # Calculate final confidence with weighted components
        call_symbol_conf = (
            index_quality * 0.3 +      # Index must be extended
            pinky_quality * 0.3 +      # Pinky must be extended
            curl_quality * 0.2 +       # Middle and ring must be curled
            thumb_quality * 0.1 +      # Thumb should be properly positioned
            position_quality * 0.1     # Hand should be near face/ear
        )
        
        # Apply threshold for more consistent detection
        if call_symbol_conf < 0.7:  # Higher threshold for clean detection
            call_symbol_conf *= 0.6  # Reduce low confidences
            
        # Store result based on which hand
        if is_left_hand:
            left_hand_call_gesture = call_symbol_conf > 0.7
            left_conf = call_symbol_conf
        else:
            right_hand_call_gesture = call_symbol_conf > 0.7
            right_conf = call_symbol_conf
            
        # Debug output
        if DEBUG and frame_count % 10 == 0 and call_symbol_conf > 0.3:
            hand_name = "Left" if is_left_hand else "Right"
            print(f"{hand_name} Call Symbol: index={index_extended}, pinky={pinky_extended}, " +
                  f"middle_curled={middle_curled}, ring_curled={ring_curled}, " +
                  f"thumb_tucked={thumb_tucked}, near_face={hand_near_face}, " +
                  f"all_ext={all_fingers_extended}, conf={call_symbol_conf:.2f}")
    
    # Return the maximum confidence from either hand
    if left_hand_call_gesture or right_hand_call_gesture:
        max_conf = max(left_conf, right_conf)
        return max_conf
    else:
        return 0.0