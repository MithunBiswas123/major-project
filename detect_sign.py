"""
Simple Sign Language Detection - Standalone Script
Uses webcam + MediaPipe + Your trained ML model
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load model and encoders
print("Loading model...")
model_path = os.path.join(os.path.dirname(__file__), 'models', 'saved', 'best_27_signs.h5')
le_path = os.path.join(os.path.dirname(__file__), 'models', 'scalers', 'label_encoder.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scalers', 'scaler.pkl')

model = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(le_path)
scaler = joblib.load(scaler_path)

print(f"✅ Model loaded! Signs: {list(label_encoder.classes_)}")


def extract_features(landmarks):
    """Extract 126 features from hand landmarks"""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]
    
    # Normalize relative to wrist
    wrist_x, wrist_y, wrist_z = xs[0], ys[0], zs[0]
    
    # Calculate hand size
    hand_size = np.sqrt((xs[12] - wrist_x)**2 + (ys[12] - wrist_y)**2)
    if hand_size < 0.01:
        hand_size = 0.1
    
    # Right hand features (63 values)
    right_hand = []
    for i in range(21):
        right_hand.append((xs[i] - wrist_x) / hand_size)
        right_hand.append((ys[i] - wrist_y) / hand_size)
        right_hand.append((zs[i] - wrist_z) / hand_size)
    
    # Left hand is zeros
    left_hand = [0.0] * 63
    
    return np.array(left_hand + right_hand).reshape(1, -1)


def predict_sign(landmarks):
    """Predict sign from landmarks"""
    features = extract_features(landmarks)
    features = scaler.transform(features)
    pred = model.predict(features, verbose=0)
    idx = np.argmax(pred[0])
    confidence = float(pred[0][idx])
    sign = label_encoder.classes_[idx]
    return sign, confidence


def main():
    print("\n" + "="*50)
    print("SIGN LANGUAGE DETECTION")
    print("="*50)
    print("Press 'q' to quit")
    print("="*50 + "\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_sign = ""
    current_conf = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw hand landmarks and predict
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Predict sign
                sign, conf = predict_sign(hand_landmarks.landmark)
                
                if conf > 0.5:
                    current_sign = sign
                    current_conf = conf
        
        # Display prediction
        if current_sign:
            # Background box
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 255, 0), 2)
            
            # Sign text
            cv2.putText(frame, f"Sign: {current_sign}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {current_conf*100:.1f}%", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
            cv2.putText(frame, "Show your hand", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Sign Language Detection', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
