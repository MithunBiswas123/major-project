"""
Ultimate Training - Position Normalized for Better Recognition
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

# 15 signs
SIGNS = ['1', '2', '3', '4', '5', 'A', 'B', 'C', 'D', 'E', 'goodbye', 'hi', 'no', 'peace', 'yes']

SAMPLES_PER_SIGN = 250  # More samples

def extract_normalized_landmarks(results):
    """Extract NORMALIZED landmarks - position independent"""
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= len(results.multi_handedness):
                continue
            
            hand_label = results.multi_handedness[idx].classification[0].label
            
            # Get all landmarks
            landmarks = []
            xs, ys, zs = [], [], []
            for lm in hand_landmarks.landmark:
                xs.append(lm.x)
                ys.append(lm.y)
                zs.append(lm.z)
            
            # Normalize to wrist (landmark 0) - makes position independent
            wrist_x, wrist_y, wrist_z = xs[0], ys[0], zs[0]
            
            # Calculate hand size for scale normalization
            # Distance from wrist to middle finger tip (landmark 12)
            hand_size = np.sqrt((xs[12] - wrist_x)**2 + (ys[12] - wrist_y)**2)
            if hand_size < 0.01:
                hand_size = 0.1  # Prevent division by zero
            
            # Normalize all landmarks relative to wrist and scale by hand size
            for i in range(21):
                landmarks.append((xs[i] - wrist_x) / hand_size)
                landmarks.append((ys[i] - wrist_y) / hand_size)
                landmarks.append((zs[i] - wrist_z) / hand_size)
            
            if hand_label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks
    
    return left_hand + right_hand

def collect_data():
    """Collect training data with continuous recording while holding SPACE"""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    all_data = []
    all_labels = []
    
    print("\n" + "="*60)
    print("FAST TRAINING - HOLD SPACE TO RECORD")
    print("="*60)
    print(f"Signs: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print("\nControls:")
    print("  HOLD SPACE = Record continuously (move hand around!)")
    print("  RELEASE SPACE = Stop recording")
    print("  N = Next sign")
    print("  Q = Quit and train")
    print("="*60)
    
    for sign_idx, sign in enumerate(SIGNS):
        print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: {sign}")
        print("  HOLD SPACE and move hand to different positions!")
        
        samples_collected = 0
        sign_data = []
        recording = False
        
        while samples_collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Progress bar
            progress = samples_collected / SAMPLES_PER_SIGN
            bar_width = 400
            cv2.rectangle(frame, (120, 30), (120 + bar_width, 50), (50, 50, 50), -1)
            cv2.rectangle(frame, (120, 30), (120 + int(bar_width * progress), 50), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, f"{sign}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(frame, f"{samples_collected}/{SAMPLES_PER_SIGN}", (540, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Recording status
            if recording:
                cv2.putText(frame, "RECORDING...", (220, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(frame, (200, 462), 10, (0, 0, 255), -1)
            else:
                cv2.putText(frame, "HOLD SPACE TO RECORD", (150, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Hand status
            if results.multi_hand_landmarks:
                cv2.circle(frame, (600, 30), 12, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (600, 30), 12, (0, 0, 255), -1)
            
            cv2.imshow('Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Check if SPACE is held
            if key == ord(' '):
                recording = True
            else:
                recording = False
            
            # Record while SPACE held
            if recording and results.multi_hand_landmarks:
                features = extract_normalized_landmarks(results)
                sign_data.append(features)
                samples_collected += 1
            
            if key == ord('n'):
                print(f"  Skipping {sign}")
                break
            
            if key == ord('q'):
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

def augment_data(X, y, factor=12):
    """Heavy augmentation"""
    X_aug = [X]
    y_aug = [y]
    
    for i in range(factor - 1):
        noise = np.random.normal(0, 0.02 * (i % 4 + 1), X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    
    return np.vstack(X_aug), np.concatenate(y_aug)

def build_model(num_classes):
    """Bigger model for better accuracy"""
    model = keras.Sequential([
        keras.layers.Input(shape=(126,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
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
    print("POSITION-NORMALIZED TRAINING")
    print("="*60)
    print("This training uses NORMALIZED landmarks")
    print("so gestures work regardless of hand position!")
    print(f"\nSigns: {SIGNS}")
    
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
    
    # Heavy augmentation
    print("Augmenting data 12x...")
    X_aug, y_aug = augment_data(X_scaled, y_encoded, factor=12)
    print(f"After augmentation: {len(X_aug)} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )
    
    # Train
    print("\nTraining model...")
    model = build_model(len(le.classes_))
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_split=0.15,
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
    
    print("\nSaved!")
    print(f"Signs: {list(le.classes_)}")

if __name__ == "__main__":
    main()
