# detectors/surrender_pose_detector.py
import numpy as np

def detect_surrender_pose(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, gesture_state, DEBUG=False, frame_count=0):
    """
    Detect if person has both hands raised with palms forward (surrender pose)
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list or len(hand_landmarks_list) < 2:
        return 0.0
    
    # Get relevant pose landmarks
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Check if both hands are raised above shoulders
    left_hand_raised = False
    right_hand_raised = False
    left_palm_forward = False
    right_palm_forward = False
    
    left_wrist = None
    right_wrist = None
    
    for hand_landmarks in hand_landmarks_list:
        # Determine if it's left or right hand
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # Check if hand is raised above shoulders
        is_raised = wrist.y < min(left_shoulder.y, right_shoulder.y)
        
        # Check if palm is facing forward
        # For palm forward, fingertips should be closer to camera than wrist
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Better palm forward detection using multiple finger points
        fingertips_z = [
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
        ]
        avg_fingertip_z = sum(fingertips_z) / len(fingertips_z)
        is_palm_forward = avg_fingertip_z < palm_center.z
        
        # Determine if it's left or right hand based on position relative to shoulders
        if wrist.x < (left_shoulder.x + right_shoulder.x) / 2:
            left_hand_raised = is_raised
            left_palm_forward = is_palm_forward
            left_wrist = wrist
        else:
            right_hand_raised = is_raised
            right_palm_forward = is_palm_forward
            right_wrist = wrist
    
    # Check for movement to differentiate from waving
    # Initialize position history if needed
    if "wrist_positions" not in gesture_state:
        gesture_state["wrist_positions"] = {"left": [], "right": []}
    
    # Make sure the wrist_positions dictionary has the required keys
    for key in ["left", "right"]:
        if key not in gesture_state["wrist_positions"]:
            gesture_state["wrist_positions"][key] = []
    
    # Add current positions if hands are detected
    if left_wrist is not None:
        gesture_state["wrist_positions"]["left"].append((left_wrist.x, left_wrist.y))
    if right_wrist is not None:
        gesture_state["wrist_positions"]["right"].append((right_wrist.x, right_wrist.y))
    
    # Keep only recent history
    max_history = 10
    for hand in ["left", "right"]:
        if len(gesture_state["wrist_positions"][hand]) > max_history:
            gesture_state["wrist_positions"][hand] = gesture_state["wrist_positions"][hand][-max_history:]
    
    # Calculate stability (low movement is good for surrender pose)
    stability_score = 1.0
    if left_hand_raised and len(gesture_state["wrist_positions"]["left"]) > 3:
        left_x = [p[0] for p in gesture_state["wrist_positions"]["left"]]
        left_y = [p[1] for p in gesture_state["wrist_positions"]["left"]]
        left_variance = np.var(left_x) + np.var(left_y)
        # Low variance = higher stability
        left_stability = max(0.0, 1.0 - (left_variance * 2000)) 
        stability_score = min(stability_score, left_stability)
    
    if right_hand_raised and len(gesture_state["wrist_positions"]["right"]) > 3:
        right_x = [p[0] for p in gesture_state["wrist_positions"]["right"]]
        right_y = [p[1] for p in gesture_state["wrist_positions"]["right"]]
        right_variance = np.var(right_x) + np.var(right_y)
        # Low variance = higher stability
        right_stability = max(0.0, 1.0 - (right_variance * 2000))
        stability_score = min(stability_score, right_stability)
    
    # Calculate confidence based on hand positions
    if left_hand_raised and right_hand_raised:
        # Get height above shoulders
        if left_wrist and right_wrist:
            # How high above shoulders
            left_height = left_shoulder.y - left_wrist.y
            right_height = right_shoulder.y - right_wrist.y
            
            # Normalize to a reasonable range (0-1)
            height_confidence = min(1.0, (left_height + right_height) / 0.8)
            
            # How symmetrical the pose is (both hands at similar height)
            symmetry = 1.0 - min(1.0, abs(left_height - right_height) * 5.0)
            
            # Check palm orientation - important for surrender pose
            palm_orientation = 0.0
            if left_palm_forward and right_palm_forward:
                palm_orientation = 1.0
            elif left_palm_forward or right_palm_forward:
                palm_orientation = 0.5
            
            # Arms should be somewhat apart for surrender pose
            hands_apart = 0.0
            if left_wrist and right_wrist:
                wrist_distance = abs(left_wrist.x - right_wrist.x)
                # Ideal distance is around shoulder width
                shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                if wrist_distance > shoulder_width * 0.7:
                    hands_apart = 1.0
                else:
                    hands_apart = wrist_distance / (shoulder_width * 0.7)
            
            # Combine factors for overall confidence with stability as key factor
            confidence = (
                height_confidence * 0.2 +  # How high hands are raised
                symmetry * 0.2 +           # Symmetry of pose
                palm_orientation * 0.3 +   # Palm orientation
                hands_apart * 0.1 +        # Hands appropriately apart
                stability_score * 0.2      # Stability (not moving much)
            )
            
            # Debug visualization
            if DEBUG and frame_count % 10 == 0 and confidence > 0.3:
                print(f"Surrender: height={height_confidence:.2f}, sym={symmetry:.2f}, " +
                    f"palms={palm_orientation:.1f}, apart={hands_apart:.2f}, stable={stability_score:.2f}, conf={confidence:.2f}")
            
            return confidence
    
    return 0.0