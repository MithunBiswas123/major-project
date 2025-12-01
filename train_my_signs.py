"""
EASY DATA COLLECTION AND TRAINING
Collects real hand data from YOUR webcam and trains a model
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.insert(0, '.')
from src.data_collection import HandDetector

# Configuration
SIGNS_TO_COLLECT = [
    # Start with just a few signs - easier to learn
    'A', 'B', 'C', 'D', 'E',
    '1', '2', '3', '4', '5',
    'hello', 'yes', 'no', 'ok',
]

SAMPLES_PER_SIGN = 100  # Collect 100 samples per sign (takes ~10 seconds each)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'my_hand_data.csv')


def show_sign_guide(sign):
    """Show how to make each sign"""
    guides = {
        'A': "Make a FIST with thumb on the side",
        'B': "FLAT hand, fingers together pointing up",
        'C': "Curved hand like holding a cup (C shape)",
        'D': "Index finger UP, others curled touching thumb",
        'E': "All fingers curled/bent down",
        '1': "Index finger pointing UP only",
        '2': "Peace sign - index and middle finger up (V shape)",
        '3': "Thumb + index + middle finger up",
        '4': "Four fingers up, thumb folded",
        '5': "All 5 fingers spread open",
        'hello': "Open hand, wave (all fingers spread)",
        'yes': "Make a FIST and nod it",
        'no': "Index and middle finger pinching with thumb",
        'ok': "Thumb and index make a circle, others up",
    }
    return guides.get(sign, f"Show the '{sign}' sign")


def collect_data():
    print("=" * 60)
    print("🎥 COLLECT YOUR HAND DATA")
    print("=" * 60)
    print(f"Signs to collect: {len(SIGNS_TO_COLLECT)}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"Total time: ~{len(SIGNS_TO_COLLECT) * 15} seconds")
    print("=" * 60)
    print("\nINSTRUCTIONS:")
    print("1. Press SPACE when ready to start collecting a sign")
    print("2. HOLD your hand sign STEADY during collection")
    print("3. Keep your hand in the green box area")
    print("4. Press Q to quit anytime")
    print("=" * 60)
    
    input("\nPress ENTER to start...")
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    all_data = []
    
    for sign_idx, sign in enumerate(SIGNS_TO_COLLECT):
        guide = show_sign_guide(sign)
        
        print(f"\n📌 Sign {sign_idx + 1}/{len(SIGNS_TO_COLLECT)}: '{sign}'")
        print(f"   How to: {guide}")
        
        # Wait for user to be ready
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            features, results, hand_detected = detector.extract_landmarks(frame)
            frame = detector.draw_landmarks(frame, results)
            
            # Draw guide area
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0) if hand_detected else (0, 0, 255), 3)
            
            # Header
            cv2.rectangle(frame, (0, 0), (w, 120), (40, 40, 40), -1)
            cv2.putText(frame, f"NEXT: '{sign}'", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, guide[:50], (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Press SPACE when ready, Q to quit", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Hand status
            status = "HAND OK - Ready!" if hand_detected else "Show your hand!"
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, status, (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and hand_detected:
                waiting = False
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.release()
                print("❌ Cancelled")
                return None
        
        # Collect samples with countdown
        print(f"   Collecting in 3...")
        for countdown in [3, 2, 1]:
            start = time.time()
            while time.time() - start < 1:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                features, results, hand_detected = detector.extract_landmarks(frame)
                frame = detector.draw_landmarks(frame, results)
                
                cv2.rectangle(frame, (0, 0), (640, 100), (40, 40, 40), -1)
                cv2.putText(frame, f"GET READY: {countdown}", (200, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1)
            if countdown > 1:
                print(f"   {countdown-1}...")
        
        # Now collect!
        print(f"   GO! Hold '{sign}' steady...")
        collected = 0
        no_hand_frames = 0
        
        while collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            features, results, hand_detected = detector.extract_landmarks(frame)
            frame = detector.draw_landmarks(frame, results)
            
            # Progress
            progress = collected / SAMPLES_PER_SIGN
            bar_width = int(progress * 400)
            
            cv2.rectangle(frame, (0, 0), (640, 100), (40, 40, 40), -1)
            cv2.putText(frame, f"Collecting '{sign}': {collected}/{SAMPLES_PER_SIGN}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (20, 60), (420, 85), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 60), (20 + bar_width, 85), (0, 255, 0), -1)
            
            if hand_detected and features is not None:
                # Save sample
                sample = {'sign': sign}
                for i, val in enumerate(features):
                    sample[f'feature_{i}'] = val
                all_data.append(sample)
                collected += 1
                no_hand_frames = 0
                
                # Flash green border
                cv2.rectangle(frame, (0, 0), (640, 480), (0, 255, 0), 10)
            else:
                no_hand_frames += 1
                cv2.putText(frame, "KEEP HAND VISIBLE!", (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            
            # If hand lost for too long, pause
            if no_hand_frames > 30:
                print("   ⚠️ Hand lost! Show your hand again...")
                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    features, results, hand_detected = detector.extract_landmarks(frame)
                    frame = detector.draw_landmarks(frame, results)
                    cv2.putText(frame, "HAND LOST - Show hand to continue", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow('Data Collection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if hand_detected:
                        no_hand_frames = 0
                        break
        
        print(f"   ✅ Collected {collected} samples for '{sign}'")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    # Save data
    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ Data saved: {OUTPUT_PATH}")
        print(f"   Total samples: {len(df)}")
        print(f"   Signs: {df['sign'].nunique()}")
        return df
    
    return None


def train_on_my_data():
    """Train model on collected data"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import joblib
    
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scalers')
    
    print("\n" + "=" * 60)
    print("🎯 TRAINING ON YOUR DATA")
    print("=" * 60)
    
    if not os.path.exists(OUTPUT_PATH):
        print("❌ No data found! Run collection first.")
        return
    
    # Load data
    df = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded {len(df)} samples, {df['sign'].nunique()} signs")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols].values
    y = df['sign'].values
    
    # Encode
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Data augmentation - add noise variations
    print("Augmenting data...")
    X_aug = [X_scaled]
    y_aug = [y_encoded]
    for _ in range(4):  # 5x data
        noise = np.random.randn(*X_scaled.shape) * 0.02
        X_aug.append(X_scaled + noise)
        y_aug.append(y_encoded)
    X_scaled = np.vstack(X_aug)
    y_encoded = np.hstack(y_aug)
    print(f"Augmented to {len(X_scaled)} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # To categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Build model
    print("Building model...")
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("Training...")
    callbacks = [EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)]
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Model saved!")
    print(f"   Signs learned: {list(label_encoder.classes_)}")
    print("\n🎉 Run 'python main.py' to test!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤟 PERSONAL SIGN LANGUAGE TRAINER")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Collect data (recommended first)")
    print("  2. Train model (after collecting)")
    print("  3. Both (collect then train)")
    print()
    
    choice = input("Choose (1/2/3): ").strip()
    
    if choice == '1':
        collect_data()
    elif choice == '2':
        train_on_my_data()
    elif choice == '3':
        data = collect_data()
        if data is not None:
            train_on_my_data()
    else:
        print("Running both...")
        data = collect_data()
        if data is not None:
            train_on_my_data()
