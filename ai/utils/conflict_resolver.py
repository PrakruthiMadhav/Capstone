# utils/conflict_resolver.py
from gesture_types import GestureType

def resolve_gesture_conflicts(confidences):
    """
    Advanced conflict resolution between gestures that shouldn't be detected simultaneously.
    With special focus on L-shape vs Call symbol distinction and Surrender pose confusion.
    Returns modified confidences dictionary.
    """
    modified_confidences = confidences.copy()
    
    # First, identify the primary gesture candidate (highest confidence)
    primary_gesture = max(confidences.items(), key=lambda x: x[1])
    primary_type, primary_conf = primary_gesture
    
    # Only apply conflict resolution if primary gesture has meaningful confidence
    if primary_conf < 0.4:  # Reasonable threshold to allow gesture detection
        return modified_confidences
    
    # CRITICAL: SPECIAL HANDLING FOR L-SHAPE SIGN VS CALL SYMBOL
    # These gestures are frequently confused because they both involve extended fingers
    if confidences[GestureType.L_SHAPE_SIGN] > 0.3 and confidences[GestureType.CALL_SYMBOL] > 0.3:
        # Both gestures have significant confidence - determine which is more likely
        
        # If L-shape is stronger, suppress call symbol completely
        if confidences[GestureType.L_SHAPE_SIGN] > confidences[GestureType.CALL_SYMBOL]:
            # Strong suppression - L-shape is dominant
            modified_confidences[GestureType.CALL_SYMBOL] *= 0.2  # 80% reduction
            
        # If call symbol is stronger, suppress L-shape
        else:
            # Strong suppression - call symbol is dominant
            modified_confidences[GestureType.L_SHAPE_SIGN] *= 0.2  # 80% reduction
    
    # CRITICAL: SPECIAL HANDLING FOR SURRENDER POSE VS CALL SYMBOL
    # These gestures can be confused as both involve raised hands
    if confidences[GestureType.SURRENDER_POSE] > 0.3 and confidences[GestureType.CALL_SYMBOL] > 0.3:
        # If surrender pose is very confident, override call symbol
        if confidences[GestureType.SURRENDER_POSE] > 0.7:
            modified_confidences[GestureType.CALL_SYMBOL] *= 0.3  # Strong reduction
        
        # If call symbol is very confident, override surrender pose
        elif confidences[GestureType.CALL_SYMBOL] > 0.7:
            modified_confidences[GestureType.SURRENDER_POSE] *= 0.3  # Strong reduction
        
        # Otherwise, favor the one with higher confidence
        elif confidences[GestureType.SURRENDER_POSE] > confidences[GestureType.CALL_SYMBOL]:
            modified_confidences[GestureType.CALL_SYMBOL] *= 0.5
        else:
            modified_confidences[GestureType.SURRENDER_POSE] *= 0.5
    
    # CRITICAL: SPECIAL HANDLING FOR L-SHAPE SIGN VS SURRENDER POSE
    # These gestures can be confused when arms are raised
    if confidences[GestureType.L_SHAPE_SIGN] > 0.3 and confidences[GestureType.SURRENDER_POSE] > 0.3:
        # If L-shape is very confident, override surrender pose
        if confidences[GestureType.L_SHAPE_SIGN] > 0.7:
            modified_confidences[GestureType.SURRENDER_POSE] *= 0.3  # Strong reduction
        
        # If surrender pose is very confident, override L-shape
        elif confidences[GestureType.SURRENDER_POSE] > 0.7:
            modified_confidences[GestureType.L_SHAPE_SIGN] *= 0.3  # Strong reduction
        
        # Otherwise, favor the one with higher confidence
        elif confidences[GestureType.L_SHAPE_SIGN] > confidences[GestureType.SURRENDER_POSE]:
            modified_confidences[GestureType.SURRENDER_POSE] *= 0.5
        else:
            modified_confidences[GestureType.L_SHAPE_SIGN] *= 0.5
    
    # Define incompatible gesture groups (mutually exclusive gestures)
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
        # Call symbol vs L-shape sign - INCREASED suppression
        (GestureType.CALL_SYMBOL, GestureType.L_SHAPE_SIGN, 0.95),  # Increased to 95%
        
        # Hand over mouth vs other hand gestures
        (GestureType.HAND_OVER_MOUTH, GestureType.CALL_SYMBOL, 0.90),
        (GestureType.HAND_OVER_MOUTH, GestureType.L_SHAPE_SIGN, 0.90),
        (GestureType.HAND_OVER_MOUTH, GestureType.CROSSED_WRISTS, 0.85),
        
        # Surrender pose vs other gestures
        (GestureType.SURRENDER_POSE, GestureType.CROSSED_WRISTS, 0.90),
        (GestureType.SURRENDER_POSE, GestureType.CALL_SYMBOL, 0.90),
        (GestureType.SURRENDER_POSE, GestureType.L_SHAPE_SIGN, 0.95),  # Increased
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
            suppression_strength = 0.7 + (primary_conf * 0.2)  # Increased base to 0.7
            suppression_strength = min(0.9, suppression_strength)  # Cap at 90% suppression
            
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
    
    # Ensure one dominant gesture
    viable_gestures = [(g, c) for g, c in modified_confidences.items() if c > 0.5]
    
    if len(viable_gestures) > 1:
        # Sort by confidence (highest first)
        viable_gestures.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the top gesture, reduce others
        top_gesture = viable_gestures[0][0]
        for gesture, conf in viable_gestures[1:]:
            modified_confidences[gesture] *= 0.5  # 50% reduction for other viable gestures
    
    return modified_confidences