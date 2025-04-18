from fastapi import FastAPI, WebSocket
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = FastAPI()

@app.websocket("/ws/frame")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            decoded = base64.b64decode(data)
            img_bytes = io.BytesIO(decoded)
            img = Image.open(img_bytes)

            # Convert to OpenCV BGR format
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Optional: Save frame for verification
            cv2.imwrite("received_frame.jpg", frame)
            print("📸 Frame received and saved.")

            # Dummy response
            await websocket.send_text("✅ Frame processed")
    except Exception as e:
        print("❌ Disconnected or error:", e)
