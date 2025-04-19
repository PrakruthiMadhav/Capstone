# detectors/waving_arm_detector.py
import numpy as np

def detect_waving_one_arm(pose_landmarks, mp_pose, gesture_state, DEBUG=False, frame_count=0):
    """
    Detect if person is waving one arm (help needed - outdoor signaling)
    Returns confidence score between 0-1
    """
    if not pose_landmarks:
        return 0.0
    
    # Get current landmarks
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    
    # Check if at least one arm is raised
    left_arm_raised = left_wrist.y < (left_shoulder.y - 0.05)
    right_arm_raised = right_wrist.y < (right_shoulder.y - 0.05)
    
    # Differentiate from surrender pose - typically only ONE arm is waving
    both_arms_raised = left_arm_raised and right_arm_raised
    
    # Key difference: if BOTH arms are raised, require asymmetric movement
    # for wave detection (unlike surrender pose which is symmetric and stable)
    
    # Add wrist and elbow positions to tracking history
    if "positions" not in gesture_state:
        gesture_state["positions"] = {
            "left_wrist": [], "right_wrist": [],
            "left_elbow": [], "right_elbow": []
        }
    
    # Make sure the positions dictionary has all required keys
    for key in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
        if key not in gesture_state["positions"]:
            gesture_state["positions"][key] = []
            
    # Add current positions
    gesture_state["positions"]["left_wrist"].append((left_wrist.x, left_wrist.y))
    gesture_state["positions"]["right_wrist"].append((right_wrist.x, right_wrist.y))
    gesture_state["positions"]["left_elbow"].append((left_elbow.x, left_elbow.y))
    gesture_state["positions"]["right_elbow"].append((right_elbow.x, right_elbow.y))
    
    # Keep only recent positions
    max_history = 15  # Increased for better pattern detection
    for key in gesture_state["positions"]:
        if len(gesture_state["positions"][key]) > max_history:
            gesture_state["positions"][key] = gesture_state["positions"][key][-max_history:]
    
    # Need at least a few frames to detect movement
    if len(gesture_state["positions"]["left_wrist"]) < 4:
        return 0.0
    
    # Calculate movement for each joint
    movement_scores = {}
    for joint in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
        positions = gesture_state["positions"][joint]
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]
        
        if len(x_positions) > 3:
            # Calculate variance in x and y directions
            x_variance = np.var(x_positions) 
            y_variance = np.var(y_positions)
            
            # Calculate mean absolute difference for consecutive points
            x_diffs = [abs(x_positions[i] - x_positions[i-1]) for i in range(1, len(x_positions))]
            y_diffs = [abs(y_positions[i] - y_positions[i-1]) for i in range(1, len(y_positions))]
            
            # Check for oscillation pattern (direction changes)
            x_direction_changes = sum(1 for i in range(2, len(x_positions)) if 
                                   (x_positions[i] - x_positions[i-1]) * 
                                   (x_positions[i-1] - x_positions[i-2]) <= 0)
            
            oscillation_factor = min(3.0, 1.0 + x_direction_changes / 3.0)
            
            # Combined movement score with oscillation boost
            if x_diffs and y_diffs:
                x_mad = np.mean(x_diffs)
                y_mad = np.mean(y_diffs)
                movement_scores[joint] = ((x_variance + y_variance) * 1500 + 
                                       (x_mad + y_mad) * 500) * oscillation_factor
            else:
                movement_scores[joint] = 0
    
    # Determine which arm is moving more
    left_arm_movement = max(movement_scores.get("left_wrist", 0), movement_scores.get("left_elbow", 0))
    right_arm_movement = max(movement_scores.get("right_wrist", 0), movement_scores.get("right_elbow", 0))
    
    # Critical difference from surrender pose: movement asymmetry
    # One arm should move much more than the other
    movement_asymmetry = max(0, abs(left_arm_movement - right_arm_movement))
    
    # Enhanced waving pattern detection - check for side-to-side movement
    side_to_side_pattern = False
    dominant_plane = "none"
    
    # Calculate dominant plane of movement (horizontal for waving)
    if left_arm_raised and "left_wrist" in movement_scores:
        left_x = [p[0] for p in gesture_state["positions"]["left_wrist"]]
        left_y = [p[1] for p in gesture_state["positions"]["left_wrist"]]
        x_range = max(left_x) - min(left_x)
        y_range = max(left_y) - min(left_y)
        
        if x_range > y_range * 1.5:  # Horizontal movement dominates
            side_to_side_pattern = True
            dominant_plane = "horizontal-left"
    
    if right_arm_raised and "right_wrist" in movement_scores:
        right_x = [p[0] for p in gesture_state["positions"]["right_wrist"]]
        right_y = [p[1] for p in gesture_state["positions"]["right_wrist"]]
        x_range = max(right_x) - min(right_x)
        y_range = max(right_y) - min(right_y)
        
        if x_range > y_range * 1.5:  # Horizontal movement dominates
            side_to_side_pattern = True
            dominant_plane = "horizontal-right"
    
    # Calculate confidence based on which arm is moving
    confidence = 0.0
    moving_arm = "none"
    
    if both_arms_raised:
        # If both arms raised, require asymmetric movement (one arm waving, one still)
        movement_ratio = 0
        if left_arm_movement > 0 and right_arm_movement > 0:
            movement_ratio = max(left_arm_movement, right_arm_movement) / max(0.00001, min(left_arm_movement, right_arm_movement))
        
        if movement_ratio > 3.0:  # One arm moving at least 3x more than the other
            if left_arm_movement > right_arm_movement:
                confidence = min(1.0, left_arm_movement)
                moving_arm = "left"
            else:
                confidence = min(1.0, right_arm_movement)
                moving_arm = "right"
        else:
            # Both arms moving similarly - reduce confidence as this is more like surrender
            confidence = max(0.0, min(0.3, (left_arm_movement + right_arm_movement) / 10))
            moving_arm = "both"
    else:
        # One arm raised
        if left_arm_raised and left_arm_movement > 0.1:
            confidence = min(1.0, left_arm_movement)
            moving_arm = "left"
        elif right_arm_raised and right_arm_movement > 0.1:
            confidence = min(1.0, right_arm_movement)
            moving_arm = "right"
        else:
            confidence = 0.0
            moving_arm = "none"
    
    # Boost confidence if side-to-side pattern detected
    if side_to_side_pattern and confidence > 0.1:
        confidence = min(1.0, confidence * 1.5)
    
    # Debug output
    if DEBUG and frame_count % 3 == 0 and confidence > 0.1:
        print(f"Waving arm: left_mov={left_arm_movement:.6f}, right_mov={right_arm_movement:.6f}, " +
            f"both={both_arms_raised}, asym={movement_asymmetry:.2f}")
        print(f"Waving {moving_arm} arm: pattern={side_to_side_pattern}, " +
            f"plane={dominant_plane}, conf={confidence:.2f}")
    
    return min(1.0, confidence)  # Cap at 1.0