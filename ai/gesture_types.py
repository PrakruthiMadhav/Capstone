# gesture_types.py - Defines the gesture types and alerts (Circle in Palm removed)
from enum import Enum, auto

class GestureType(str,Enum):
    """Enum to represent different emergency gestures"""
    CROSSED_WRISTS = auto()     # Captive/abuse/hostage signaling
    SURRENDER_POSE = auto()     # Hands raised, palms forward
    HAND_OVER_MOUTH = auto()    # Hand covering mouth (can't speak)
    CALL_SYMBOL = auto()        # Phone gesture - index and pinky extended, others curled
    L_SHAPE_SIGN = auto()       # L gesture - index and thumb extended forming L shape


# Define alert messages for each gesture
GESTURE_ALERTS = {
    GestureType.CROSSED_WRISTS: "🚨 ALERT: Crossed wrists detected! Possible captive/hostage situation! 🚨",
    GestureType.SURRENDER_POSE: "🚨 ALERT: Hands raised palms forward! Possible threat situation! 🚨",
    GestureType.HAND_OVER_MOUTH: "🚨 ALERT: Hand over mouth! Unable to speak/possible threat! 🚨",
    GestureType.CALL_SYMBOL: "🚨 ALERT: Call symbol detected! Person needs to call for help! 🚨",
    GestureType.L_SHAPE_SIGN: "🚨 ALERT: L shape sign detected! 🚨"
}

# Colors for visualization
GESTURE_COLORS = {
    GestureType.CROSSED_WRISTS: (0, 255, 0),      # Green
    GestureType.SURRENDER_POSE: (255, 0, 255),    # Magenta
    GestureType.HAND_OVER_MOUTH: (255, 255, 0),   # Cyan
    GestureType.CALL_SYMBOL: (0, 165, 255),       # Orange
    GestureType.L_SHAPE_SIGN: (255, 0, 128)       # Pink
}
