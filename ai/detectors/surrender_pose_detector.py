# detectors/surrender_pose_detector.py
import numpy as np

def detect_surrender_pose(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, gesture_state, DEBUG=False, frame_count=0):
    """
    Significantly improved surrender pose detection that distinguishes from call symbol.
    Surrender pose requires: BOTH hands raised above shoulders with palms flat and facing forward.
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list or len(hand_landmarks_list) < 2:
        # Need both hands visible for surrender pose
        return 0.0
    
    # Get relevant pose landmarks
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # CRITICAL: Determine midpoint for hand identification
    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    
    # Initialize detection variables
    left_hand_data = {
        "detected": False,
        "raised": False,
        "palm_forward": False,
        "fingers_extended": False,
        "wrist": None
    }
    
    right_hand_data = {
        "detected": False,
        "raised": False,
        "palm_forward": False,
        "fingers_extended": False,
        "wrist": None
    }
    
    # Process each hand
    for hand_landmarks in hand_landmarks_list:
        # CRITICAL: First correctly identify left vs right hand
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        is_left_hand = wrist.x < shoulder_midpoint_x
        
        # Get key hand landmarks
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        # Get palm and finger base landmarks
        palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        
        # CRITICAL CHECK #1: HAND MUST BE RAISED SIGNIFICANTLY ABOVE SHOULDER
        # For surrender pose, hands must be well above shoulders
        is_raised = wrist.y < min(left_shoulder.y, right_shoulder.y) - 0.1
        
        # CRITICAL CHECK #2: PALM MUST BE FACING FORWARD
        # Check if palm is facing the camera (fingertips closer to camera than palm)
        # Use multiple points for more reliable detection
        fingertips_z_avg = (
            index_tip.z + 
            middle_tip.z + 
            ring_tip.z + 
            pinky_tip.z
        ) / 4.0
        
        palm_facing_forward = fingertips_z_avg < palm_center.z - 0.02
        
        # CRITICAL CHECK #3: MUST HAVE ALL FINGERS EXTENDED
        # For surrender pose, all fingers should be extended (not curled)
        # This is critical for distinguishing from call symbol and L-shape
        index_extended = index_tip.y < index_mcp.y - 0.03
        middle_extended = middle_tip.y < middle_mcp.y - 0.03
        ring_extended = ring_tip.y < ring_mcp.y - 0.03
        pinky_extended = pinky_tip.y < pinky_mcp.y - 0.03
        
        # Check that ALL fingers are extended (not just index and pinky)
        all_fingers_extended = (
            index_extended and 
            middle_extended and 
            ring_extended and 
            pinky_extended
        )
        
        # CRITICAL: CHECK FOR CALL SYMBOL SPECIFICALLY
        # Call symbol has index and pinky extended but middle and ring curled
        index_pinky_only = (
            index_extended and 
            pinky_extended and 
            not middle_extended and 
            not ring_extended
        )
        
        # If it looks like a call symbol, it's definitely NOT a surrender pose
        if index_pinky_only:
            all_fingers_extended = False
            palm_facing_forward = False
            
            if DEBUG and frame_count % 20 == 0:
                hand_name = "Left" if is_left_hand else "Right"
                print(f"{hand_name} hand shows CALL SYMBOL pattern (not surrender pose)")
        
        # Store results for this hand
        if is_left_hand:
            left_hand_data["detected"] = True
            left_hand_data["raised"] = is_raised
            left_hand_data["palm_forward"] = palm_facing_forward
            left_hand_data["fingers_extended"] = all_fingers_extended
            left_hand_data["wrist"] = wrist
        else:
            right_hand_data["detected"] = True
            right_hand_data["raised"] = is_raised
            right_hand_data["palm_forward"] = palm_facing_forward
            right_hand_data["fingers_extended"] = all_fingers_extended
            right_hand_data["wrist"] = wrist
    
    # CRITICAL: BOTH HANDS MUST BE DETECTED AND MEET ALL CRITERIA
    # For surrender pose, we need both hands in proper position
    
    # Early rejection: If either hand doesn't meet the basic criteria, return 0
    if not (left_hand_data["detected"] and right_hand_data["detected"]):
        return 0.0
    
    if not (left_hand_data["raised"] and right_hand_data["raised"]):
        # Both hands must be raised
        return 0.0
    
    if not (left_hand_data["palm_forward"] and right_hand_data["palm_forward"]):
        # Both palms must be facing forward
        return 0.0
    
    if not (left_hand_data["fingers_extended"] and right_hand_data["fingers_extended"]):
        # Both hands must have all fingers extended
        return 0.0
    
    # Initialize position history if needed (for stability detection)
    if "wrist_positions" not in gesture_state:
        gesture_state["wrist_positions"] = {"left": [], "right": []}
    
    # Update position history
    if left_hand_data["wrist"] is not None:
        gesture_state["wrist_positions"]["left"].append((
            left_hand_data["wrist"].x, 
            left_hand_data["wrist"].y
        ))
    
    if right_hand_data["wrist"] is not None:
        gesture_state["wrist_positions"]["right"].append((
            right_hand_data["wrist"].x, 
            right_hand_data["wrist"].y
        ))
    
    # Keep only recent history
    max_history = 10
    for hand in ["left", "right"]:
        if len(gesture_state["wrist_positions"][hand]) > max_history:
            gesture_state["wrist_positions"][hand] = gesture_state["wrist_positions"][hand][-max_history:]
    
    # Calculate stability (low movement is good for surrender pose)
    stability_score = 1.0
    if len(gesture_state["wrist_positions"]["left"]) > 3:
        left_x = [p[0] for p in gesture_state["wrist_positions"]["left"]]
        left_y = [p[1] for p in gesture_state["wrist_positions"]["left"]]
        left_variance = np.var(left_x) + np.var(left_y)
        # Low variance = higher stability
        left_stability = max(0.0, 1.0 - (left_variance * 2000))
        stability_score = min(stability_score, left_stability)
    
    if len(gesture_state["wrist_positions"]["right"]) > 3:
        right_x = [p[0] for p in gesture_state["wrist_positions"]["right"]]
        right_y = [p[1] for p in gesture_state["wrist_positions"]["right"]]
        right_variance = np.var(right_x) + np.var(right_y)
        # Low variance = higher stability
        right_stability = max(0.0, 1.0 - (right_variance * 2000))
        stability_score = min(stability_score, right_stability)
    
    # Calculate additional factors for confidence
    
    # Measure how high above shoulders the hands are
    left_height = left_shoulder.y - left_hand_data["wrist"].y
    right_height = right_shoulder.y - right_hand_data["wrist"].y
    
    # Normalize to a reasonable range (0-1)
    # Higher hands = higher confidence
    height_confidence = min(1.0, (left_height + right_height) / 0.8)
    
    # How symmetrical the pose is (both hands at similar height)
    symmetry = 1.0 - min(1.0, abs(left_height - right_height) * 5.0)
    
    # Arms should be somewhat apart for surrender pose
    wrist_distance = abs(left_hand_data["wrist"].x - right_hand_data["wrist"].x)
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    
    # Hands should be at least shoulder width apart
    hands_apart = min(1.0, wrist_distance / shoulder_width)
    
    # Combine factors for overall confidence
    confidence = (
        height_confidence * 0.25 +    # How high hands are raised 
        symmetry * 0.25 +             # Symmetry of pose
        hands_apart * 0.25 +          # Hands appropriately apart
        stability_score * 0.25        # Stability (not moving much)
    )
    
    # CRITICAL: Penalize if hands are too close together
    # This helps distinguish from other gestures
    if wrist_distance < shoulder_width * 0.8:
        confidence *= 0.7  # 30% reduction if hands too close
    
    # Debug visualization
    if DEBUG and frame_count % 10 == 0 and confidence > 0.3:
        print(f"Surrender: height={height_confidence:.2f}, sym={symmetry:.2f}, " +
              f"left_raised={left_hand_data['raised']}, right_raised={right_hand_data['raised']}, " +
              f"left_palm_fwd={left_hand_data['palm_forward']}, right_palm_fwd={right_hand_data['palm_forward']}, " +
              f"left_fingers_ext={left_hand_data['fingers_extended']}, right_fingers_ext={right_hand_data['fingers_extended']}, " +
              f"apart={hands_apart:.2f}, stable={stability_score:.2f}, conf={confidence:.2f}")
    
    return confidence