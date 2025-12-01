"""
Debug Detection - Shows confidence for ALL signs
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
from collections import deque

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
            
            # Normalize to wrist (landmark 0)
            wrist_x, wrist_y, wrist_z = xs[0], ys[0], zs[0]
            
            # Hand size for scale normalization
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

def main():
    # Load model and preprocessors
    print("Loading model...")
    model = tf.keras.models.load_model('models/saved/lstm_model.h5')
    scaler = joblib.load('models/scalers/scaler.pkl')
    le = joblib.load('models/scalers/label_encoder.pkl')
    
    signs = list(le.classes_)
    print(f"Model loaded! Signs: {signs}")
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Smoothing
    pred_buffer = deque(maxlen=5)
    
    print("\n" + "="*50)
    print("DEBUG DETECTION - Shows ALL sign confidences")
    print("="*50)
    print("Press Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Create sidebar for confidences
        sidebar = np.zeros((480, 200, 3), dtype=np.uint8)
        sidebar[:] = (40, 40, 40)
        
        cv2.putText(sidebar, "CONFIDENCES", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            # Which hand detected
            for idx, handedness in enumerate(results.multi_handedness):
                hand_label = handedness.classification[0].label
                cv2.putText(frame, f"{hand_label} Hand", (10, 30 + idx*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Extract and predict
            features = extract_landmarks(results)
            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            probs = model.predict(features_scaled, verbose=0)[0]
            pred_idx = np.argmax(probs)
            
            # Add to buffer
            pred_buffer.append(pred_idx)
            
            # Get smoothed prediction
            from collections import Counter
            if len(pred_buffer) >= 3:
                most_common = Counter(pred_buffer).most_common(1)[0][0]
                final_sign = signs[most_common]
                final_conf = probs[most_common]
            else:
                final_sign = signs[pred_idx]
                final_conf = probs[pred_idx]
            
            # Main prediction display
            color = (0, 255, 0) if final_conf > 0.7 else (0, 255, 255) if final_conf > 0.4 else (0, 165, 255)
            cv2.rectangle(frame, (0, 440), (640, 480), (30, 30, 30), -1)
            cv2.putText(frame, f"{final_sign.upper()}: {final_conf*100:.1f}%", (180, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Show ALL confidences in sidebar (sorted by confidence)
            sorted_indices = np.argsort(probs)[::-1]
            for i, idx in enumerate(sorted_indices):
                y_pos = 60 + i * 35
                sign_name = signs[idx]
                conf = probs[idx]
                
                # Bar
                bar_len = int(conf * 150)
                bar_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (100, 100, 100)
                cv2.rectangle(sidebar, (10, y_pos), (10 + bar_len, y_pos + 20), bar_color, -1)
                cv2.rectangle(sidebar, (10, y_pos), (160, y_pos + 20), (80, 80, 80), 1)
                
                # Label
                cv2.putText(sidebar, f"{sign_name[:8]}", (10, y_pos + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(sidebar, f"{conf*100:.0f}%", (165, y_pos + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            pred_buffer.clear()
            cv2.putText(frame, "No hand detected", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            cv2.putText(sidebar, "No hand", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Combine frame and sidebar
        combined = np.hstack([frame, sidebar])
        
        cv2.imshow('Sign Detection (DEBUG)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
