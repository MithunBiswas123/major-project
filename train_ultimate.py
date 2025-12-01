"""
ULTIMATE Sign Language Training - Maximum Accuracy
Collects extensive data with variations for best real-time performance
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import joblib

# Signs to train
SIGNS = ['1', '2', '3', '4', '5', 'A', 'B', 'C', 'D', 'E', 'hi', 'goodbye', 'yes', 'no', 'peace']

# Paths
MODEL_DIR = 'models/saved'
SCALER_DIR = 'models/scalers'

# Ultimate settings
SAMPLES_PER_SIGN = 200  # Even more samples!


def collect_ultimate_data():
    """Collect data in TWO phases - different positions"""
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        return None, None
    
    all_features = []
    all_labels = []
    
    # TWO COLLECTION PHASES
    phases = [
        ("PHASE 1: CENTER", "Keep hand in CENTER of screen", 100),
        ("PHASE 2: MOVE AROUND", "Move hand to different positions!", 100),
    ]
    
    for phase_name, instruction, samples_per_phase in phases:
        print("\n" + "="*60)
        print(f"🎯 {phase_name}")
        print(f"📝 {instruction}")
        print("="*60)
        input(f"\nPress ENTER to start {phase_name}...")
        
        for sign_idx, sign in enumerate(SIGNS):
            print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: {sign}")
            
            # Countdown
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (0, 0), (640, 120), (50, 50, 50), -1)
                    cv2.putText(frame, f"{phase_name}", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    cv2.putText(frame, f"Next: {sign}", (20, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.putText(frame, f"Starting in {i}...", (20, 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, instruction, (100, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow('Ultimate Training', frame)
                    cv2.waitKey(1000)
            
            # Collect samples
            sign_samples = []
            
            while len(sign_samples) < samples_per_phase:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                progress = len(sign_samples) / samples_per_phase
                
                # Draw UI
                cv2.rectangle(frame, (0, 0), (640, 120), (50, 50, 50), -1)
                cv2.putText(frame, f"{phase_name}", (20, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                cv2.putText(frame, f"Sign: {sign}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"{len(sign_samples)}/{samples_per_phase}", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Progress bar
                bar_width = int(progress * 500)
                cv2.rectangle(frame, (120, 85), (620, 110), (100, 100, 100), -1)
                cv2.rectangle(frame, (120, 85), (120 + bar_width, 110), (0, 255, 0), -1)
                
                # Instruction
                cv2.putText(frame, instruction, (100, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    features = extract_features(results)
                    if features is not None:
                        sign_samples.append(features)
                        cv2.circle(frame, (600, 50), 15, (0, 255, 0), -1)
                else:
                    cv2.putText(frame, "SHOW HAND!", (230, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                cv2.imshow('Ultimate Training', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None, None
            
            all_features.extend(sign_samples)
            all_labels.extend([sign] * len(sign_samples))
            print(f"✅ Collected {len(sign_samples)} samples for '{sign}'")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n✅ Total collected: {len(X)} samples")
    
    return X, y


def extract_features(results):
    """Extract 126 features from hand landmarks"""
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label
            
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if hand_type == 'Left':
                left_hand = np.array(landmarks[:63])
            else:
                right_hand = np.array(landmarks[:63])
    
    features = np.concatenate([left_hand, right_hand])
    
    if np.sum(np.abs(features)) < 0.1:
        return None
    
    return features


def augment_ultimate(X, y, factor=12):
    """Extensive augmentation for robustness"""
    print(f"\n🔄 Ultimate augmentation (factor: {factor})...")
    
    augmented_X = [X.copy()]
    augmented_y = [y.copy()]
    
    for i in range(factor - 1):
        aug_type = i % 6
        
        if aug_type == 0:
            # Tiny noise
            noise = np.random.normal(0, 0.005, X.shape)
            X_aug = X + noise
        elif aug_type == 1:
            # Small noise
            noise = np.random.normal(0, 0.015, X.shape)
            X_aug = X + noise
        elif aug_type == 2:
            # Medium noise
            noise = np.random.normal(0, 0.025, X.shape)
            X_aug = X + noise
        elif aug_type == 3:
            # Small scale
            scale = np.random.uniform(0.95, 1.05, X.shape)
            X_aug = X * scale
        elif aug_type == 4:
            # Large scale
            scale = np.random.uniform(0.90, 1.10, X.shape)
            X_aug = X * scale
        else:
            # Combined noise + scale
            noise = np.random.normal(0, 0.01, X.shape)
            scale = np.random.uniform(0.97, 1.03, X.shape)
            X_aug = (X + noise) * scale
        
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
    
    X_final = np.vstack(augmented_X)
    y_final = np.hstack(augmented_y)
    
    # Shuffle
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    print(f"✅ Augmented: {len(X)} → {len(X_final)} samples")
    
    return X_final, y_final


def train_ultimate_model(X, y):
    """Train the ultimate model"""
    print("\n" + "="*60)
    print("🏆 TRAINING ULTIMATE MODEL")
    print("="*60)
    
    # Augment
    X_aug, y_aug = augment_ultimate(X, y, factor=12)
    
    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_aug)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Classes: {num_classes}")
    
    # Ultimate model - deep and wide
    model = Sequential([
        Input(shape=(126,)),
        
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.000001,
        verbose=1
    )
    
    print("\n🏋️ Training ultimate model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=150,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{'='*60}")
    print(f"🏆 FINAL TEST ACCURACY: {acc*100:.2f}%")
    print(f"{'='*60}")
    
    # Per-class accuracy
    print("\nPer-sign accuracy:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    all_good = True
    for i, sign in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            sign_acc = (y_pred[mask] == i).mean() * 100
            status = "✅" if sign_acc >= 95 else "⚠️" if sign_acc >= 85 else "❌"
            if sign_acc < 95:
                all_good = False
            print(f"  {status} {sign:12s}: {sign_acc:.1f}%")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Model saved!")
    
    return model, scaler, le, acc


def test_realtime(model, scaler, le):
    """Real-time test"""
    print("\n" + "="*60)
    print("🎥 REAL-TIME TEST")
    print("="*60)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    from collections import deque
    pred_buffer = deque(maxlen=5)
    
    print("✅ Show your signs! Press Q to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Header
        cv2.rectangle(frame, (0, 0), (640, 130), (30, 30, 30), -1)
        cv2.putText(frame, "ULTIMATE SIGN DETECTOR", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            features = extract_features(results)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                probs = model.predict(features_scaled, verbose=0)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                
                pred_buffer.append(pred_idx)
                
                # Smoothed prediction
                from collections import Counter
                if len(pred_buffer) >= 3:
                    most_common = Counter(pred_buffer).most_common(1)[0][0]
                    sign_name = le.inverse_transform([most_common])[0]
                else:
                    sign_name = le.inverse_transform([pred_idx])[0]
                
                # Color by confidence
                if confidence > 0.8:
                    color = (0, 255, 0)
                elif confidence > 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                # Display
                cv2.rectangle(frame, (10, 40), (630, 125), color, 3)
                cv2.putText(frame, sign_name.upper(), (25, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
                cv2.putText(frame, f"{confidence*100:.0f}%", (520, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                cv2.circle(frame, (610, 20), 10, (0, 255, 0), -1)
        else:
            cv2.putText(frame, "Show your hand", (180, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            cv2.circle(frame, (610, 20), 10, (0, 0, 255), -1)
            pred_buffer.clear()
        
        # Footer
        cv2.rectangle(frame, (0, 450), (640, 480), (30, 30, 30), -1)
        cv2.putText(frame, "Q = Quit", (280, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        cv2.imshow('Ultimate Sign Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main():
    print("\n" + "="*60)
    print("🏆 ULTIMATE SIGN LANGUAGE TRAINING")
    print("="*60)
    print("\nThis is the BEST training method for maximum accuracy!")
    print("\n📋 What you'll do:")
    print("   PHASE 1: Show each sign with hand in CENTER")
    print("   PHASE 2: Show each sign while MOVING hand around")
    print(f"\nSigns: {SIGNS}")
    print(f"Total time: ~8-10 minutes")
    print("="*60)
    
    # Collect
    X, y = collect_ultimate_data()
    
    if X is None:
        print("No data collected!")
        return
    
    # Train
    model, scaler, le, accuracy = train_ultimate_model(X, y)
    
    if accuracy >= 0.98:
        print("\n🏆 PERFECT! Your model is AMAZING!")
    elif accuracy >= 0.95:
        print("\n🎉 EXCELLENT! Very high accuracy!")
    elif accuracy >= 0.90:
        print("\n👍 GREAT! Model should work very well!")
    else:
        print("\n⚠️ Consider retraining with clearer gestures")
    
    # Test
    print("\n" + "="*60)
    input("Press ENTER to test real-time detection...")
    test_realtime(model, scaler, le)
    
    print("\n✅ Training complete! Use 'python main.py' anytime.")


if __name__ == "__main__":
    main()
