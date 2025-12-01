"""
ROBUST Sign Language Training - Collects MORE data with VARIATIONS
This creates a model that works well in real-time detection
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import joblib

# Signs to train
SIGNS = ['1', '2', '3', '4', '5', 'A', 'B', 'C', 'D', 'E', 'hi', 'goodbye', 'yes', 'no', 'peace']

# Paths
MODEL_DIR = 'models/saved'
SCALER_DIR = 'models/scalers'

# More samples = better real-time accuracy
SAMPLES_PER_SIGN = 150  # More samples!
COLLECTION_TIME = 8  # seconds per sign


def collect_robust_data():
    """Collect data with movement and angle variations"""
    
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
    
    print("\n" + "="*60)
    print("ROBUST DATA COLLECTION")
    print("="*60)
    print("\nINSTRUCTIONS FOR BETTER ACCURACY:")
    print("- Move your hand SLIGHTLY during collection")
    print("- Tilt hand at different angles")
    print("- Move closer/farther from camera")
    print("- This creates a more robust model!")
    print("="*60)
    
    input("\nPress ENTER when ready...")
    
    for sign_idx, sign in enumerate(SIGNS):
        print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: {sign}")
        print("Get ready...")
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
                cv2.putText(frame, f"Next: {sign}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Starting in {i}...", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Move hand slightly for variations!", (150, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1000)
        
        # Collect samples with variations
        sign_samples = []
        start_time = time.time()
        frame_count = 0
        
        while len(sign_samples) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            elapsed = time.time() - start_time
            progress = len(sign_samples) / SAMPLES_PER_SIGN
            
            # Draw UI
            cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
            cv2.putText(frame, f"Sign: {sign}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Samples: {len(sign_samples)}/{SAMPLES_PER_SIGN}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Progress bar
            bar_width = int(progress * 400)
            cv2.rectangle(frame, (200, 65), (600, 85), (100, 100, 100), -1)
            cv2.rectangle(frame, (200, 65), (200 + bar_width, 85), (0, 255, 0), -1)
            
            # Instruction
            cv2.putText(frame, "Move hand slightly for better training!", (100, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
            if results.multi_hand_landmarks:
                # Draw landmarks
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Extract features
                features = extract_features(results)
                if features is not None:
                    sign_samples.append(features)
                    
                    # Visual feedback for collection
                    cv2.circle(frame, (600, 40), 15, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "Show your hand!", (250, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None
            
            frame_count += 1
        
        # Add to dataset
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
            
            # Get landmarks
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if hand_type == 'Left':
                left_hand = np.array(landmarks[:63])
            else:
                right_hand = np.array(landmarks[:63])
    
    # Combine
    features = np.concatenate([left_hand, right_hand])
    
    # Check if valid (has actual hand data)
    if np.sum(np.abs(features)) < 0.1:
        return None
    
    return features


def augment_robust(X, y, factor=8):
    """Create variations to handle real-world conditions"""
    print(f"\n🔄 Augmenting data (factor: {factor})...")
    
    augmented_X = [X.copy()]
    augmented_y = [y.copy()]
    
    for i in range(factor - 1):
        # Different types of augmentation
        aug_type = i % 4
        
        if aug_type == 0:
            # Small noise (simulates hand shake)
            noise = np.random.normal(0, 0.01, X.shape)
            X_aug = X + noise
        elif aug_type == 1:
            # Medium noise
            noise = np.random.normal(0, 0.02, X.shape)
            X_aug = X + noise
        elif aug_type == 2:
            # Scale variation (simulates distance change)
            scale = np.random.uniform(0.92, 1.08, X.shape)
            X_aug = X * scale
        else:
            # Position shift (simulates hand position change)
            shift = np.random.uniform(-0.03, 0.03, X.shape)
            X_aug = X + shift
        
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
    
    X_final = np.vstack(augmented_X)
    y_final = np.hstack(augmented_y)
    
    print(f"✅ Augmented: {len(X)} → {len(X_final)} samples")
    
    return X_final, y_final


def train_robust_model(X, y):
    """Train a robust model for real-time detection"""
    print("\n" + "="*60)
    print("TRAINING ROBUST MODEL")
    print("="*60)
    
    # Augment data
    X_aug, y_aug = augment_robust(X, y, factor=8)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_aug)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert to categorical
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {num_classes}")
    
    # Build robust model - wider layers for better generalization
    model = Sequential([
        Input(shape=(126,)),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    print("\n🏋️ Training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Final evaluation
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
    
    return model, scaler, le, acc


def test_realtime(model, scaler, le):
    """Test the model in real-time immediately after training"""
    print("\n" + "="*60)
    print("🎥 REAL-TIME TEST")
    print("="*60)
    print("Testing your trained model now!")
    print("Press Q to quit")
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
    
    # Prediction buffer for smoothing
    from collections import deque
    pred_buffer = deque(maxlen=7)
    
    print("\n✅ Camera ready! Show your signs...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw header
        cv2.rectangle(frame, (0, 0), (640, 120), (40, 40, 40), -1)
        cv2.putText(frame, "SIGN LANGUAGE DETECTOR", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Extract and predict
            features = extract_features(results)
            if features is not None:
                # Normalize
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predict
                probs = model.predict(features_scaled, verbose=0)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                
                # Add to buffer
                pred_buffer.append(pred_idx)
                
                # Get smoothed prediction (most common in buffer)
                if len(pred_buffer) >= 3:
                    from collections import Counter
                    most_common = Counter(pred_buffer).most_common(1)[0][0]
                    sign_name = le.inverse_transform([most_common])[0]
                else:
                    sign_name = le.inverse_transform([pred_idx])[0]
                
                # Color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                # Display prediction
                cv2.rectangle(frame, (15, 45), (625, 115), color, 3)
                cv2.putText(frame, sign_name.upper(), (30, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
                cv2.putText(frame, f"{confidence*100:.0f}%", (500, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Hand status
                cv2.circle(frame, (600, 25), 10, (0, 255, 0), -1)
        else:
            cv2.putText(frame, "Show your hand", (200, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
            cv2.circle(frame, (600, 25), 10, (0, 0, 255), -1)
            pred_buffer.clear()
        
        # Footer
        cv2.rectangle(frame, (0, 445), (640, 480), (40, 40, 40), -1)
        cv2.putText(frame, "Q = Quit", (270, 468),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Sign Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main():
    print("\n" + "="*60)
    print("🤟 ROBUST SIGN LANGUAGE TRAINING")
    print("="*60)
    print("\nThis collects MORE data with VARIATIONS for better accuracy!")
    print(f"Signs: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"Estimated time: ~{len(SIGNS) * 12 // 60 + 2} minutes")
    print("="*60)
    
    # Collect data
    X, y = collect_robust_data()
    
    if X is None:
        print("No data collected!")
        return
    
    # Train
    model, scaler, le, accuracy = train_robust_model(X, y)
    
    if accuracy >= 0.90:
        print("\n🎉 EXCELLENT! Model accuracy is 90%+")
    elif accuracy >= 0.80:
        print("\n👍 Good accuracy! Should work well.")
    else:
        print("\n⚠️ Accuracy is below 80%. Consider retraining.")
    
    # Test immediately
    print("\n" + "="*60)
    input("Press ENTER to test your model in real-time...")
    test_realtime(model, scaler, le)
    
    print("\n✅ Done! Run 'python main.py' to use detection anytime.")


if __name__ == "__main__":
    main()
