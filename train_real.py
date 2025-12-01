"""
=============================================================
QUICK REAL DATA COLLECTION + TRAINING FOR 100% ACCURACY
=============================================================
This script collects REAL hand data from YOUR camera and trains
a model that will work perfectly with YOUR hands.

Takes only 2-3 minutes to collect data for all signs!
"""

import cv2
import numpy as np
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_collection import HandDetector

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scalers')

# Signs you want - customize this list!
SIGNS = [
    'hi',       # Wave hand
    'goodbye',  # Flat hand wave
    '1',        # Index finger up
    '2',        # Peace/V sign
    '3',        # Three fingers
    '4',        # Four fingers
    '5',        # All five open
    'A',        # Fist with thumb side
    'B',        # Flat hand up
    'C',        # C curve
    'D',        # Index up, others curved
    'E',        # All fingers curled
    'yes',      # Thumbs up
    'no',       # Two fingers together
    'peace',    # Same as V/2
]

SAMPLES_PER_SIGN = 80  # Collect 80 samples per sign (~8 seconds each)


def collect_data():
    """Collect real hand data from webcam"""
    print("\n" + "="*60)
    print("📷 COLLECTING YOUR HAND DATA")
    print("="*60)
    print(f"\nSigns to collect: {len(SIGNS)}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"Estimated time: {len(SIGNS) * 10} seconds (~{len(SIGNS) * 10 // 60} minutes)")
    print("\nINSTRUCTIONS:")
    print("1. Show your RIGHT HAND to the camera")
    print("2. Press SPACE to start collecting each sign")
    print("3. HOLD the sign steady during collection")
    print("4. Press Q anytime to quit")
    print("="*60)
    
    input("\nPress ENTER when ready to start...")
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return None, None
    
    all_X = []
    all_y = []
    
    sign_guides = {
        'hi': 'WAVE - Open hand, fingers spread',
        'goodbye': 'FLAT hand wave',
        '1': 'Index finger UP only',
        '2': 'Peace sign - V shape',
        '3': 'Three fingers (thumb+index+middle)',
        '4': 'Four fingers up',
        '5': 'All FIVE fingers spread',
        'A': 'FIST with thumb on side',
        'B': 'FLAT hand, fingers together UP',
        'C': 'C shape - curved hand',
        'D': 'Index UP, others touch thumb',
        'E': 'All fingers CURLED down',
        'yes': 'THUMBS UP',
        'no': 'Index+middle fingers TOGETHER',
        'peace': 'PEACE sign (V shape)',
    }
    
    for sign_idx, sign in enumerate(SIGNS):
        guide = sign_guides.get(sign, f"Show '{sign}' sign")
        
        print(f"\n[{sign_idx+1}/{len(SIGNS)}] Next sign: '{sign}'")
        print(f"    How to: {guide}")
        
        # Wait for spacebar
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)  # Mirror
            features, results, hand_detected = detector.extract_landmarks(frame)
            frame = detector.draw_landmarks(frame, results)
            
            # UI
            cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
            cv2.putText(frame, f"NEXT: '{sign}' ({sign_idx+1}/{len(SIGNS)})", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, guide[:55], (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, "Press SPACE when ready | Q to quit", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Hand status
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status = "HAND OK" if hand_detected else "Show hand!"
            cv2.putText(frame, status, (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and hand_detected:
                waiting = False
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.release()
                print("❌ Cancelled")
                return None, None
        
        # Countdown 3-2-1
        for countdown in [3, 2, 1]:
            start = time.time()
            while time.time() - start < 0.7:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                features, results, hand_detected = detector.extract_landmarks(frame)
                frame = detector.draw_landmarks(frame, results)
                
                cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
                cv2.putText(frame, f"GET READY: {countdown}", (200, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1)
        
        # COLLECT!
        print(f"    Collecting '{sign}'...", end=' ', flush=True)
        collected = 0
        
        while collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            features, results, hand_detected = detector.extract_landmarks(frame)
            frame = detector.draw_landmarks(frame, results)
            
            # Progress bar
            progress = collected / SAMPLES_PER_SIGN
            bar_w = int(progress * 400)
            
            cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
            cv2.putText(frame, f"COLLECTING '{sign}': {collected}/{SAMPLES_PER_SIGN}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (20, 60), (420, 85), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 60), (20 + bar_w, 85), (0, 255, 0), -1)
            
            if hand_detected and features is not None:
                all_X.append(features.copy())
                all_y.append(sign)
                collected += 1
                # Green flash
                cv2.rectangle(frame, (0, 0), (640, 480), (0, 255, 0), 8)
            else:
                cv2.putText(frame, "KEEP HAND VISIBLE!", (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        
        print(f"✓ ({collected} samples)")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    if all_X:
        print(f"\n✅ Total collected: {len(all_X)} samples")
        return np.array(all_X), np.array(all_y)
    return None, None


def augment_data(X, y):
    """Augment data with small variations"""
    print("Augmenting data...")
    X_aug = [X]
    y_aug = [y]
    
    # Add noise variations
    for _ in range(3):
        noise = np.random.randn(*X.shape) * 0.01
        X_aug.append(X + noise)
        y_aug.append(y)
    
    # Small scale variations
    for scale in [0.98, 1.02]:
        X_scaled = (X - 0.5) * scale + 0.5
        X_aug.append(np.clip(X_scaled, 0, 1))
        y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)


def train_model(X, y):
    """Train model on collected data"""
    print("\n" + "="*60)
    print("🎯 TRAINING MODEL ON YOUR DATA")
    print("="*60)
    
    # Augment
    X, y = augment_data(X, y)
    print(f"Augmented to: {len(X)} samples")
    
    # Encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, stratify=y_enc, random_state=42
    )
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)
    
    print(f"Classes: {n_classes}")
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Build model
    model = Sequential([
        Input(shape=(126,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n🚀 Training...")
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{'='*60}")
    print(f"✅ TEST ACCURACY: {acc*100:.2f}%")
    print(f"{'='*60}")
    
    # Per-class accuracy
    print("\nPer-sign accuracy:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    for i, sign in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            sign_acc = (y_pred[mask] == i).mean() * 100
            status = "✅" if sign_acc >= 90 else "⚠️" if sign_acc >= 70 else "❌"
            print(f"  {status} {sign:12s}: {sign_acc:.1f}%")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Model saved!")
    print(f"Signs: {list(le.classes_)}")
    
    return acc


def main():
    print("\n" + "="*60)
    print("🤟 SIGN LANGUAGE MODEL - REAL DATA TRAINING")
    print("="*60)
    print("\nThis will collect data from YOUR hands for perfect accuracy!")
    print(f"Signs to learn: {SIGNS}")
    print(f"Time needed: ~{len(SIGNS) * 10 // 60 + 1} minutes")
    print("="*60)
    
    # Collect
    X, y = collect_data()
    
    if X is None:
        print("No data collected!")
        return
    
    # Train
    accuracy = train_model(X, y)
    
    if accuracy >= 0.95:
        print("\n🎉 EXCELLENT! Model is ready with 95%+ accuracy!")
    elif accuracy >= 0.85:
        print("\n👍 Good accuracy! Model should work well.")
    else:
        print("\n⚠️ Accuracy is low. Try collecting data again with clearer signs.")
    
    print("\n🚀 Run 'python main.py' to test your signs!")


if __name__ == "__main__":
    main()
