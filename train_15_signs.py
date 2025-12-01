"""
Train 15 Signs - SPACE KEY TO RECORD
Restore the previous high accuracy model
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Original 15 signs that worked well
SIGNS = ['1', '2', '3', '4', '5', 'A', 'B', 'C', 'D', 'E', 'goodbye', 'hi', 'no', 'peace', 'yes']

SAMPLES_PER_SIGN = 150  # More samples for better accuracy

def extract_landmarks(results):
    """Extract 126 features from both hands - MATCHES DETECTION CODE"""
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= len(results.multi_handedness):
                continue
            
            # Get hand label (Left or Right)
            hand_label = results.multi_handedness[idx].classification[0].label
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if hand_label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks
    
    return left_hand + right_hand

def collect_data():
    """Collect training data - PRESS SPACE TO RECORD"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    all_data = []
    all_labels = []
    
    print("\n" + "="*60)
    print("15 SIGN TRAINING - SPACE KEY MODE")
    print("="*60)
    print(f"Signs: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print("\nControls:")
    print("  SPACE = Record one sample")
    print("  N = Skip to next sign")
    print("  Q = Quit and train")
    print("="*60)
    
    for sign_idx, sign in enumerate(SIGNS):
        print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: {sign}")
        
        samples_collected = 0
        sign_data = []
        
        while samples_collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Progress bar
            progress = samples_collected / SAMPLES_PER_SIGN
            bar_width = 300
            cv2.rectangle(frame, (170, 25), (170 + bar_width, 45), (50, 50, 50), -1)
            cv2.rectangle(frame, (170, 25), (170 + int(bar_width * progress), 45), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, f"Sign: {sign}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"[{sign_idx+1}/{len(SIGNS)}]", (500, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{SAMPLES_PER_SIGN}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "PRESS SPACE TO RECORD", (150, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "N=Next  Q=Quit", (230, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Hand status
            if results.multi_hand_landmarks:
                cv2.putText(frame, f"Hands: {len(results.multi_hand_landmarks)}", (530, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hands!", (520, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Training - SPACE to Record', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE
                if results.multi_hand_landmarks:
                    features = extract_landmarks(results)
                    sign_data.append(features)
                    samples_collected += 1
                    
                    # Green flash
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (640, 480), (0, 255, 0), 10)
                    cv2.imshow('Training - SPACE to Record', flash)
                    cv2.waitKey(30)
                else:
                    print("  No hand detected!")
            
            elif key == ord('n'):
                print(f"  Skipping {sign}")
                break
            
            elif key == ord('q'):
                print("\nQuitting...")
                if sign_data:
                    all_data.extend(sign_data)
                    all_labels.extend([sign] * len(sign_data))
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return np.array(all_data) if all_data else None, all_labels
        
        if sign_data:
            all_data.extend(sign_data)
            all_labels.extend([sign] * len(sign_data))
            print(f"  Completed {sign}: {len(sign_data)} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return np.array(all_data), all_labels

def augment_data(X, y, factor=8):
    """Augment with noise"""
    X_aug = [X]
    y_aug = [y]
    
    for i in range(factor - 1):
        noise = np.random.normal(0, 0.008 * (i + 1), X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    
    return np.vstack(X_aug), np.concatenate(y_aug)

def build_model(num_classes):
    """Build neural network"""
    model = keras.Sequential([
        keras.layers.Input(shape=(126,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("\n" + "="*60)
    print("15 SIGN TRAINER - RESTORE HIGH ACCURACY")
    print("="*60)
    print(f"Signs: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print("Target accuracy: 99%+")
    
    input("\nPress ENTER to start...")
    
    X, y = collect_data()
    
    if X is None or len(X) == 0:
        print("No data collected!")
        return
    
    print(f"\nCollected {len(X)} samples for {len(set(y))} signs")
    
    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Augment 8x
    print("Augmenting data 8x...")
    X_aug, y_aug = augment_data(X_scaled, y_encoded, factor=8)
    print(f"After augmentation: {len(X_aug)} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    # Train
    print("\nTraining model...")
    model = build_model(len(le.classes_))
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # Save
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('models/scalers', exist_ok=True)
    
    model.save('models/saved/lstm_model.h5')
    joblib.dump(scaler, 'models/scalers/scaler.pkl')
    joblib.dump(le, 'models/scalers/label_encoder.pkl')
    
    print("\nSaved model and scalers!")
    print(f"Signs: {list(le.classes_)}")
    print("\nRun detection with: python -c \"from src.detect import detect_signs; detect_signs('models/saved/lstm_model.h5')\"")

if __name__ == "__main__":
    main()
