# main.py - Entry point for Emergency Gesture Detection Server

import uvicorn
import threading
import cv2
import time
import logging
from fastapi import FastAPI, WebSocket
from PIL import Image
import io
import base64
import numpy as np
import asyncio
import os
import sys

# Configure logging to send logs to stdout
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Add the ai folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))
from ai.gesture_detector import EmergencyGestureDetector
from ai.gesture_types import GestureType

# Create FastAPI app
app = FastAPI(title="Emergency Gesture Detection API")

# Global detector instance
detector = EmergencyGestureDetector()

# Global variable for latest processed frame
latest_processed_frame = None
frame_lock = threading.Lock()

@app.websocket("/ws/frame")
async def websocket_endpoint(websocket: WebSocket):
    global latest_processed_frame
    
    await websocket.accept()
    logger.info("✅ Client connected.")
    
    try:
        while True:
            # Receive base64 encoded image from client
            data = await websocket.receive_text()
            decoded = base64.b64decode(data)
            logger.info("📸 Frame received from client.")
            img_bytes = io.BytesIO(decoded)
            img = Image.open(img_bytes)
            
            # Convert to OpenCV BGR format
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Process the frame
            processed_frame = detector.process_frame(frame)
            
            # Save the latest processed frame
            with frame_lock:
                latest_processed_frame = processed_frame.copy()
            
            # Detect gestures (but don't send anything back)
            detected_gestures = [g for g in GestureType if detector.gestures[g]["detected"]]
            if detected_gestures:
                for gesture in detected_gestures:
                    logger.warning(f"[ALERT] {gesture.name.replace('_', ' ').title()} detected!")

    except Exception as e:
        logger.error("❌ Disconnected or error: %s", e)


def print_startup_info():
    logger.info("🚀 Emergency Gesture Detection Server is Starting...")
    logger.info("Detecting the following gestures:\n")
    for gesture in GestureType:
        logger.info(f"• {gesture.name.replace('_', ' ').title()}")
    logger.info("\nHold gestures for ~2.5s to trigger detection.")
    logger.info("Controls (OpenCV window): [q] Quit | [r] Reset | [d] Debug Toggle\n")


if __name__ == "__main__":
    print_startup_info()
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Cleanup
    detector.release()
    cv2.destroyAllWindows()
