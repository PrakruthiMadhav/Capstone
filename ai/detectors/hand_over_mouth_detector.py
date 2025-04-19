# detectors/hand_over_mouth_detector.py
import numpy as np

def detect_hand_over_mouth(pose_landmarks, hand_landmarks_list, mp_pose, mp_hands, DEBUG=False, frame_count=0):
    """
    Detect if a hand is covering the mouth area
    Returns confidence score between 0-1
    """
    if not pose_landmarks or not hand_landmarks_list:
        return 0.0

    # Estimate mouth position from face landmarks
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

    # Better mouth position estimation using multiple face landmarks
    face_center_x = nose.x
    eye_y = (left_eye.y + right_eye.y) / 2

    # Mouth is below the nose and at a distance proportional to eye-nose distance
    estimated_mouth_x = face_center_x
    eye_nose_distance = nose.y - eye_y
    estimated_mouth_y = nose.y + eye_nose_distance * 0.8

    # Define mouth area (larger to improve detection)
    mouth_radius = 0.15  # Increased radius for better detection

    max_confidence = 0.0

    # Check if any hand is over the mouth area
    for hand_landmarks in hand_landmarks_list:
        # Check multiple points on the hand for better detection
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Calculate distances from different parts of hand to estimated mouth
        palm_distance = np.sqrt((palm_center.x - estimated_mouth_x)**2 +
                                (palm_center.y - estimated_mouth_y)**2)

        index_distance = np.sqrt((index_tip.x - estimated_mouth_x)**2 +
                                 (index_tip.y - estimated_mouth_y)**2)

        middle_distance = np.sqrt((middle_tip.x - estimated_mouth_x)**2 +
                                  (middle_tip.y - estimated_mouth_y)**2)

        # Use the shortest distance from any hand part to mouth
        min_distance = min(palm_distance, index_distance, middle_distance)

        # More lenient check for hand near mouth
        hand_near_mouth = min_distance < mouth_radius

        if hand_near_mouth:
            # Calculate proximity confidence - closer means higher confidence
            proximity_confidence = 1.0 - (min_distance / mouth_radius)

            # Z-axis check is optional - if palm is facing inward, boost confidence
            palm_facing_inward = palm_center.z < wrist.z
            z_boost = 1.2 if palm_facing_inward else 1.0

            # Calculate final confidence with higher minimum
            confidence = max(0.4, proximity_confidence) * z_boost
            max_confidence = max(max_confidence, min(1.0, confidence))

            # Debug visualization
            if DEBUG and frame_count % 10 == 0:
                print(f"Hand over mouth: dist={min_distance:.3f}, prox_conf={proximity_confidence:.2f}, " +
                      f"z_boost={z_boost:.1f}, final_conf={confidence:.2f}")

    return max_confidence