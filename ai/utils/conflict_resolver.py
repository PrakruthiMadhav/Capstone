# utils/conflict_resolver.py
from gesture_types import GestureType

def resolve_gesture_conflicts(confidences):
    """
    Advanced conflict resolution between gestures that shouldn't be detected simultaneously.
    Returns modified confidences dictionary.
    """
    modified_confidences = confidences.copy()
    
    # First, identify the primary gesture candidate (highest confidence)
    primary_gesture = max(confidences.items(), key=lambda x: x[1])
    primary_type, primary_conf = primary_gesture
    
    # Only apply conflict resolution if primary gesture has meaningful confidence
    if primary_conf < 0.3:
        return modified_confidences
    
    # Define incompatible gesture groups (mutually exclusive gestures)
    # These are gestures that physically cannot occur together
    incompatible_groups = [
        # Group 1: Standing/upper body gestures - can't do these while fallen
        {
            GestureType.CROSSED_WRISTS, 
            GestureType.CLUTCHING_CHEST,
            GestureType.SURRENDER_POSE,
            GestureType.HAND_OVER_MOUTH,
            GestureType.WAVING_ONE_ARM
        },
        
        # Group 2: Lower body positions - can't do multiple at once
        {
            GestureType.ONE_KNEE_DOWN,
            GestureType.PERSON_FALLEN
        },
        
        # Group 3: Arm/hand specific conflicts
        {
            GestureType.CROSSED_WRISTS,
            GestureType.CLUTCHING_CHEST,
            GestureType.SURRENDER_POSE, 
            GestureType.WAVING_ONE_ARM
        },
        
        # Group 4: Hand over mouth vs other hand gestures
        {
            GestureType.HAND_OVER_MOUTH,
            GestureType.CLUTCHING_CHEST,
            GestureType.CROSSED_WRISTS
        }
    ]
    
    # Less aggressive suppression for specific problematic pairs
    critical_conflicts = [
        # Crossed arms vs clutching chest
        (GestureType.CROSSED_WRISTS, GestureType.CLUTCHING_CHEST, 0.7),
        
        # Fallen person vs kneeling 
        (GestureType.PERSON_FALLEN, GestureType.ONE_KNEE_DOWN, 0.8),
        
        # Waving vs surrender (both involve raised arms)
        (GestureType.WAVING_ONE_ARM, GestureType.SURRENDER_POSE, 0.75),
        
        # Hand over mouth vs clutching chest (hand can't be both places)
        (GestureType.HAND_OVER_MOUTH, GestureType.CLUTCHING_CHEST, 0.75)
    ]
    
    # Only apply critical conflicts if the primary confidence is high enough
    if primary_conf > 0.45:
        for g1, g2, suppression_factor in critical_conflicts:
            if primary_type in (g1, g2):
                other_type = g2 if primary_type == g1 else g1
                
                # Less aggressive suppression
                if confidences[other_type] > 0:
                    modified_confidences[other_type] *= (1.0 - suppression_factor * primary_conf)
    
    # Only apply group conflicts when primary confidence is higher
    if primary_conf > 0.5:
        for group in incompatible_groups:
            if primary_type in group:
                # Find other gestures in this incompatible group
                competing_gestures = [(g, confidences[g]) for g in group if g != primary_type]
                
                # Less aggressive suppression for competing gestures
                for competing_type, competing_conf in competing_gestures:
                    if competing_conf > 0.2:  # Only bother with more meaningful competitors
                        # The stronger the primary, the more we suppress competitors
                        # The closer the competitor, the less aggressive the suppression
                        confidence_gap = max(0.05, primary_conf - competing_conf)
                        suppression_strength = min(0.8, 0.4 + confidence_gap)
                        
                        modified_confidences[competing_type] *= (1.0 - suppression_strength)
    
    # Less aggressive winner-takes-most approach
    if primary_conf > 0.65:  # Only apply for high confidence
        # Get all gestures with significant confidence
        significant_gestures = [(g, c) for g, c in confidences.items() 
                               if c > 0.4 and g != primary_type]
        
        # Apply winner-take-most approach with less suppression
        for gesture, conf in significant_gestures:
            # Less aggressive suppression
            suppression = 0.5 + (primary_conf * 0.2)
            modified_confidences[gesture] *= (1.0 - suppression)
    
    return modified_confidences