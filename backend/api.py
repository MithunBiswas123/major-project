"""
FastAPI Backend for Sign Language Detection
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import mediapipe as mp
import tensorflow as tf
import joblib
import os

app = FastAPI(title="Sign Language Detection API")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessors
print("Loading model...")
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models/saved/lstm_model.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models/scalers/scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models/scalers/label_encoder.pkl'))
SIGNS = list(label_encoder.classes_)
print(f"✅ Model loaded! Signs: {SIGNS}")

# MediaPipe setup
mp_hands = mp.solutions.hands

def extract_landmarks(results):
    """Extract NORMALIZED landmarks - position independent"""
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= len(results.multi_handedness):
                continue
            
            hand_label = results.multi_handedness[idx].classification[0].label
            
            xs, ys, zs = [], [], []
            for lm in hand_landmarks.landmark:
                xs.append(lm.x)
                ys.append(lm.y)
                zs.append(lm.z)
            
            wrist_x, wrist_y, wrist_z = xs[0], ys[0], zs[0]
            hand_size = np.sqrt((xs[12] - wrist_x)**2 + (ys[12] - wrist_y)**2)
            if hand_size < 0.01:
                hand_size = 0.1
            
            landmarks = []
            for i in range(21):
                landmarks.append((xs[i] - wrist_x) / hand_size)
                landmarks.append((ys[i] - wrist_y) / hand_size)
                landmarks.append((zs[i] - wrist_z) / hand_size)
            
            if hand_label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks
    
    return left_hand + right_hand

@app.get("/")
async def root():
    return {"message": "Sign Language Detection API", "signs": SIGNS}

@app.get("/signs")
async def get_signs():
    return {"signs": SIGNS, "count": len(SIGNS)}

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        while True:
            # Receive base64 image from frontend
            data = await websocket.receive_text()
            
            # Decode image
            img_data = base64.b64decode(data.split(',')[1] if ',' in data else data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid image"})
                continue
            
            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            response = {
                "hand_detected": False,
                "prediction": None,
                "confidence": 0,
                "all_predictions": [],
                "landmarks": []
            }
            
            if results.multi_hand_landmarks:
                response["hand_detected"] = True
                
                # Get landmarks for drawing
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = []
                    for lm in hand_landmarks.landmark:
                        hand_points.append({"x": lm.x, "y": lm.y, "z": lm.z})
                    response["landmarks"].append(hand_points)
                
                # Extract features and predict
                features = extract_landmarks(results)
                features = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                probs = model.predict(features_scaled, verbose=0)[0]
                pred_idx = np.argmax(probs)
                
                response["prediction"] = SIGNS[pred_idx]
                response["confidence"] = float(probs[pred_idx])
                
                # Top 5 predictions
                sorted_indices = np.argsort(probs)[::-1][:5]
                response["all_predictions"] = [
                    {"sign": SIGNS[i], "confidence": float(probs[i])}
                    for i in sorted_indices
                ]
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        hands.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
