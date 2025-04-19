# main.py - Entry point
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

from gesture_detector import EmergencyGestureDetector
from gesture_types import GestureType

# Create FastAPI app
app = FastAPI(title="Emergency Gesture Detection API")

# Create a global detector instance
detector = EmergencyGestureDetector()

# Global variables for storing processed frames
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
            
            # Process the frame with our detector
            processed_frame = detector.process_frame(frame)
            
            # Convert processed frame to base64 to send back (optional)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Save the latest processed frame for display
            with frame_lock:
                latest_processed_frame = processed_frame.copy()
            
            # Send back processed frame or just a confirmation
            # Option 1: Send just a confirmation
            await websocket.send_text("✅ Frame processed")
            
            # Option 2: Send back the processed frame (uncomment if needed)
            # await websocket.send_text(processed_base64)
            
            # Check for any detected gestures
            detected_gestures = [g.name.replace('_', ' ').title() 
                                 for g in GestureType 
                                 if detector.gestures[g]["detected"]]
            
            if detected_gestures:
                alert_message = "🚨 EMERGENCY DETECTED: " + ", ".join(detected_gestures)
                await websocket.send_text(alert_message)
            
    except Exception as e:
        print("❌ Disconnected or error:", e)


def display_processed_frames():
    """Function to display processed frames in a separate window"""
    global latest_processed_frame
    
    while True:
        with frame_lock:
            if latest_processed_frame is not None:
                cv2.imshow('Emergency Gesture Detection', latest_processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("Detection state reset")
        elif key == ord('d'):
            detector.DEBUG = not detector.DEBUG
            print(f"Debug mode {'ON' if detector.DEBUG else 'OFF'}")
        
        time.sleep(0.33)  # ~30 FPS


def print_startup_info():
    """Print startup information and instructions"""
    print("Starting WebSocket server...")
    print("Emergency Gesture Detection System")
    print("----------------------------------")
    print("Detecting the following emergency gestures:")
    for gesture in GestureType:
        print(f"- {gesture.name.replace('_', ' ').title()}")
    print("\nGesture must be held for 2.5 seconds to trigger an alert")
    print("Debug mode is ON - detection regions will be shown")
    print("\nControls (in visualization window):")
    print("  q - Quit")
    print("  r - Reset detection state")
    print("  d - Toggle debug mode")


if __name__ == "__main__":
    # Print startup information
    print_startup_info()
    
    # Start a thread for displaying processed frames
    display_thread = threading.Thread(target=display_processed_frames, daemon=True)
    display_thread.start()
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Cleanup
    detector.release()
    cv2.destroyAllWindows()