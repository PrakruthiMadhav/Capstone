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
    if primary_conf < 0.45:  # Reasonable threshold to allow gesture detection
        return modified_confidences
    
    # Define incompatible gesture groups (mutually exclusive gestures)
    # Note: CIRCLE_IN_PALM has been removed from the system
    incompatible_groups = [
        # Group 1: Hand gestures that conflict with each other
        {
            GestureType.CROSSED_WRISTS,
            GestureType.CALL_SYMBOL,
            GestureType.L_SHAPE_SIGN
        },
        
        # Group 2: Face and hand gestures that conflict
        {
            GestureType.HAND_OVER_MOUTH,
            GestureType.CALL_SYMBOL,
            GestureType.L_SHAPE_SIGN
        },
        
        # Group 3: Surrender pose conflicts with hand gestures
        {
            GestureType.SURRENDER_POSE,
            GestureType.CROSSED_WRISTS,
            GestureType.CALL_SYMBOL,
            GestureType.L_SHAPE_SIGN,
            GestureType.HAND_OVER_MOUTH
        }
    ]
    
    # Define specific conflict pairs with custom suppression factors
    critical_conflicts = [
        # Call symbol vs L-shape sign
        (GestureType.CALL_SYMBOL, GestureType.L_SHAPE_SIGN, 0.85),
        
        # Hand over mouth vs other hand gestures - increased suppression for call symbol
        (GestureType.HAND_OVER_MOUTH, GestureType.CALL_SYMBOL, 0.95),  # Increased from 0.85 to 0.95
        (GestureType.HAND_OVER_MOUTH, GestureType.L_SHAPE_SIGN, 0.85),
        (GestureType.HAND_OVER_MOUTH, GestureType.CROSSED_WRISTS, 0.85),
        
        # Surrender pose vs other gestures
        (GestureType.SURRENDER_POSE, GestureType.CROSSED_WRISTS, 0.90),
        (GestureType.SURRENDER_POSE, GestureType.CALL_SYMBOL, 0.90),
        (GestureType.SURRENDER_POSE, GestureType.L_SHAPE_SIGN, 0.90),
    ]
    
    # Apply critical conflicts resolution
    if primary_conf > 0.5:
        for g1, g2, suppression_factor in critical_conflicts:
            if primary_type in (g1, g2):
                other_type = g2 if primary_type == g1 else g1
                modified_confidences[other_type] = modified_confidences[other_type] * (1.0 - suppression_factor)
    
    # Apply winner-takes-most approach for strong primary gestures
    if primary_conf > 0.7:
        # Get all gestures with some confidence
        other_gestures = [(g, c) for g, c in modified_confidences.items() 
                         if g != primary_type and c > 0.15]
        
        # Apply suppression to other gestures 
        for gesture, conf in other_gestures:
            # The stronger the primary, the more we suppress others
            suppression_strength = 0.6 + (primary_conf * 0.2)
            suppression_strength = min(0.85, suppression_strength)  # Cap at 85% suppression
            
            # Prioritize crossed wrists detection
            if primary_type == GestureType.CROSSED_WRISTS:
                # Apply stronger suppression to other gestures
                modified_confidences[gesture] *= (1.0 - suppression_strength)
            elif gesture == GestureType.CROSSED_WRISTS:
                # Less suppression on crossed wrists when it's not primary
                modified_confidences[gesture] *= (1.0 - suppression_strength * 0.7)
            else:
                # Normal suppression for other gesture combinations
                modified_confidences[gesture] *= (1.0 - suppression_strength)
    
    # Special handling for Call Symbol detection (reported as not detecting well)
    # MODIFIED: Increased threshold, added hand-over-mouth check, reduced boost
    if (GestureType.CALL_SYMBOL in confidences and 
        confidences[GestureType.CALL_SYMBOL] > 0.45 and  # Increased from 0.35 to 0.45
        primary_type != GestureType.CALL_SYMBOL):
        
        # Only boost if primary isn't too strong and primary isn't hand over mouth
        if primary_conf < 0.7 and primary_type != GestureType.HAND_OVER_MOUTH:
            modified_confidences[GestureType.CALL_SYMBOL] *= 1.1  # Reduced from 1.2 to 1.1
            modified_confidences[GestureType.CALL_SYMBOL] = min(0.95, modified_confidences[GestureType.CALL_SYMBOL])
    
    # Special handling for L-shape when only one hand is detected
    # L-shape should require both hands as specified by user
    if (GestureType.L_SHAPE_SIGN in confidences and 
        confidences[GestureType.L_SHAPE_SIGN] > 0.4):
        
        # The detector should already handle this, but add extra suppression if primary
        # is not L-shape but L-shape has some confidence
        if primary_type != GestureType.L_SHAPE_SIGN:
            modified_confidences[GestureType.L_SHAPE_SIGN] *= 0.6
    
    # Ensure one dominant gesture with reasonable fallback
    viable_gestures = [(g, c) for g, c in modified_confidences.items() if c > 0.45]
    
    if len(viable_gestures) > 1:
        # Sort by confidence (highest first)
        viable_gestures.sort(key=lambda x: x[1], reverse=True)
        
        # Special case: if crossed wrists is among top gestures, prioritize it
        has_crossed_wrists = any(g == GestureType.CROSSED_WRISTS for g, c in viable_gestures[:2])
        
        if has_crossed_wrists:
            # Boost crossed wrists confidence slightly
            modified_confidences[GestureType.CROSSED_WRISTS] *= 1.1
            modified_confidences[GestureType.CROSSED_WRISTS] = min(1.0, modified_confidences[GestureType.CROSSED_WRISTS])
            
            # Reduce other top gestures more
            for gesture, conf in viable_gestures:
                if gesture != GestureType.CROSSED_WRISTS:
                    modified_confidences[gesture] *= 0.6
        else:
            # Keep the top gesture, reduce others
            top_gesture = viable_gestures[0][0]
            for gesture, conf in viable_gestures[1:]:
                modified_confidences[gesture] *= 0.65
    
    return modified_confidences
