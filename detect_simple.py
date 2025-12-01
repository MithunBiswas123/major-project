"""
Simple Detection - Matches Training Exactly
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
from collections import deque

def extract_landmarks(results):
    """Extract 126 features - EXACT MATCH with training"""
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= len(results.multi_handedness):
                continue
            
            hand_label = results.multi_handedness[idx].classification[0].label
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
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
    
    print(f"Model loaded! Signs: {list(le.classes_)}")
    
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
    
    # Smoothing buffer
    pred_buffer = deque(maxlen=7)
    
    print("\n" + "="*50)
    print("SIGN LANGUAGE DETECTION")
    print("="*50)
    print("Press Q to quit")
    print("="*50)
    
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
        
        # Header
        cv2.rectangle(frame, (0, 0), (640, 120), (30, 30, 30), -1)
        
        sign_name = ""
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            # Extract features
            features = extract_landmarks(results)
            features = np.array(features).reshape(1, -1)
            
            # Scale
            features_scaled = scaler.transform(features)
            
            # Predict
            probs = model.predict(features_scaled, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            # Add to buffer
            pred_buffer.append(pred_idx)
            
            # Get most common prediction
            from collections import Counter
            if len(pred_buffer) >= 3:
                most_common = Counter(pred_buffer).most_common(1)[0][0]
                sign_name = le.inverse_transform([most_common])[0]
            else:
                sign_name = le.inverse_transform([pred_idx])[0]
            
            # Color based on confidence
            if confidence >= 0.7:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Display prediction
            cv2.putText(frame, sign_name.upper(), (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
            cv2.putText(frame, f"{confidence*100:.1f}%", (500, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Hand status
            cv2.circle(frame, (600, 30), 15, (0, 255, 0), -1)
            
            # Confidence bar
            bar_len = int(confidence * 400)
            cv2.rectangle(frame, (20, 95), (20 + bar_len, 110), color, -1)
            cv2.rectangle(frame, (20, 95), (420, 110), (100, 100, 100), 2)
            
        else:
            # No hand
            cv2.putText(frame, "Show your hand", (150, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            cv2.circle(frame, (600, 30), 15, (0, 0, 255), -1)
            pred_buffer.clear()
        
        # Footer
        cv2.rectangle(frame, (0, 450), (640, 480), (30, 30, 30), -1)
        cv2.putText(frame, "Q = Quit", (270, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow('Sign Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Done!")

if __name__ == "__main__":
    main()
