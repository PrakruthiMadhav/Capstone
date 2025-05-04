# detectors/hand_over_mouth_detector.py 
import numpy as np

def detect_hand_over_mouth(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Simplified detection of hand covering mouth area
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list:
        return 0.0

    # Estimate mouth position from face landmarks
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

    # Better mouth position estimation
    face_center_x = nose.x
    eye_y = (left_eye.y + right_eye.y) / 2
    eye_nose_distance = nose.y - eye_y
    estimated_mouth_y = nose.y + eye_nose_distance * 0.8

    # SIMPLIFIED: Increased mouth area radius for better detection
    mouth_radius = 0.2  # Significantly larger radius

    max_confidence = 0.0

    # Check if any hand is over the mouth area
    for hand_landmarks in hand_landmarks_list:
        # SIMPLIFIED: Use multiple points on the hand for detection
        # The palm and all finger tips
        key_points = [
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],  # Palm center
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],  # Added middle joints
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ]
        
        # SIMPLIFIED: Calculate distance from each point to mouth
        min_distance = float('inf')
        for point in key_points:
            distance = np.sqrt((point.x - face_center_x)**2 + 
                              (point.y - estimated_mouth_y)**2)
            min_distance = min(min_distance, distance)
        
        # SIMPLIFIED: Hand is near mouth if any part is within the radius
        if min_distance < mouth_radius:
            # SIMPLIFIED: Clearer proximity confidence calculation
            proximity_confidence = 1.0 - (min_distance / mouth_radius)
            
            # SIMPLIFIED: Just apply minimum threshold of 0.5
            confidence = max(0.5, proximity_confidence)
            
            # SIMPLIFIED: Apply Z-axis check if available (palm facing inward is better)
            palm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            if palm.z < wrist.z:  # Palm facing inward
                confidence = min(1.0, confidence * 1.2)  # Boost confidence
            
            # Update the maximum confidence found
            max_confidence = max(max_confidence, confidence)

            # Debug visualization
            if DEBUG and frame_count % 10 == 0:
                print(f"Hand over mouth: dist={min_distance:.3f}, conf={confidence:.2f}")

    return max_confidence