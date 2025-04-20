# main.py - Entry point for Emergency Gesture Detection Server

import uvicorn
import threading
import cv2
import time
from fastapi import FastAPI, WebSocket
from PIL import Image
import io
import base64
import numpy as np
import asyncio
import os
import sys

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
    print("✅ Client connected.")
    
    try:
        while True:
            # Receive base64 encoded image from client
            data = await websocket.receive_text()
            decoded = base64.b64decode(data)
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
                    print(f"[ALERT] {gesture.name.replace('_', ' ').title()} detected!")

                    # 👉 Place to trigger custom alert logic
                    # Example:
                    # log_alert_to_db(gesture)
                    # send_notification(gesture)
                    # trigger_webhook(gesture)

    except Exception as e:
        print("❌ Disconnected or error:", e)


# def display_processed_frames():
#     """Displays processed frames in OpenCV window (optional)"""
#     global latest_processed_frame
    
#     while True:
#         with frame_lock:
#             if latest_processed_frame is not None:
#                 cv2.imshow('Emergency Gesture Detection', latest_processed_frame)
        
#         key = cv2.waitKey(5) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('r'):
#             detector.reset()
#             print("Detection state reset")
#         elif key == ord('d'):
#             detector.DEBUG = not detector.DEBUG
#             print(f"Debug mode {'ON' if detector.DEBUG else 'OFF'}")
        
#         time.sleep(0.33)  # ~3 FPS display loop


def print_startup_info():
    print("🚀 Emergency Gesture Detection Server is Starting...")
    print("Detecting the following gestures:\n")
    for gesture in GestureType:
        print(f"• {gesture.name.replace('_', ' ').title()}")
    print("\nHold gestures for ~2.5s to trigger detection.")
    print("Controls (OpenCV window): [q] Quit | [r] Reset | [d] Debug Toggle\n")


if __name__ == "__main__":
    print_startup_info()
    
    # Start display thread
    # display_thread = threading.Thread(target=display_processed_frames, daemon=True)
    # display_thread.start()
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Cleanup after server stops
    detector.release()
    cv2.destroyAllWindows()
