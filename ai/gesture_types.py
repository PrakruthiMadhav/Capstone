# gesture_types.py - Defines the gesture types and alerts
from enum import Enum, auto

class GestureType(Enum):
    """Enum to represent different emergency gestures"""
    CROSSED_WRISTS = auto()    # Captive/abuse/hostage signaling for police
    CLUTCHING_CHEST = auto()   # Hand on chest (health emergency)
    ONE_KNEE_DOWN = auto()     # About to collapse, weakness for medical team
    PERSON_FALLEN = auto()     # Horizontal body position
    SURRENDER_POSE = auto()    # Hands raised, palms forward
    HAND_OVER_MOUTH = auto()   # Hand covering mouth (can't speak)
    WAVING_ONE_ARM = auto()    # Help needed — outdoor signaling for search & rescue


# Define alert messages for each gesture
GESTURE_ALERTS = {
    GestureType.CROSSED_WRISTS: "🚨 ALERT: Crossed wrists detected! Possible captive/hostage situation! 🚨",
    GestureType.CLUTCHING_CHEST: "🚨 ALERT: Hand clutching chest! Possible cardiac emergency! 🚨",
    GestureType.ONE_KNEE_DOWN: "🚨 ALERT: One knee down! Person about to collapse! Medical assistance needed! 🚨",
    GestureType.PERSON_FALLEN: "🚨 ALERT: Person fallen! Unconscious or injured! 🚨",
    GestureType.SURRENDER_POSE: "🚨 ALERT: Hands raised palms forward! Possible threat situation! 🚨",
    GestureType.HAND_OVER_MOUTH: "🚨 ALERT: Hand over mouth! Unable to speak/possible threat! 🚨",
    GestureType.WAVING_ONE_ARM: "🚨 ALERT: Waving one arm! Help needed - outdoor signaling! 🚨"
}

# Colors for visualization
GESTURE_COLORS = {
    GestureType.CROSSED_WRISTS: (0, 255, 0),      # Green
    GestureType.CLUTCHING_CHEST: (0, 0, 255),     # Red
    GestureType.ONE_KNEE_DOWN: (255, 191, 0),     # Deep Sky Blue
    GestureType.PERSON_FALLEN: (255, 0, 0),       # Blue
    GestureType.SURRENDER_POSE: (255, 0, 255),    # Magenta
    GestureType.HAND_OVER_MOUTH: (255, 255, 0),   # Cyan
    GestureType.WAVING_ONE_ARM: (128, 0, 128)     # Purple
}