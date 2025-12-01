"""
Train ALL 26 Signs - Combined Model
Letters: A, B, C, D, E
Numbers: 1, 2, 3, 4, 5
Original words: peace, goodbye, yes, no, hi
New 11: hello, welcome, please, run, sorry, thankyou, wait, sick, drink, happy, thirsty
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

# ALL 26 SIGNS
SIGNS = [
    # Letters (5)
    'A', 'B', 'C', 'D', 'E',
    # Numbers (5)
    '1', '2', '3', '4', '5',
    # Original words (5)
    'hi', 'goodbye', 'yes', 'no', 'peace',
    # New 11 words
    'hello', 'welcome', 'please', 'run', 'sorry', 
    'thankyou', 'wait', 'sick', 'drink', 'happy', 'thirsty'
]

# Gesture hints for each sign
HINTS = {
    # Letters
    'A': 'Closed fist, thumb on side',
    'B': 'Flat hand, fingers together, thumb tucked',
    'C': 'Curved hand like holding a cup',
    'D': 'Index up, others touch thumb in circle',
    'E': 'Fingers bent down, thumb tucked',
    # Numbers
    '1': 'Index finger up only',
    '2': 'Index and middle up (peace sign)',
    '3': 'Thumb, index, middle up',
    '4': 'Four fingers up, thumb folded',
    '5': 'All five fingers spread open',
    # Original words
    'hi': 'Open hand wave',
    'goodbye': 'Wave hand away from body',
    'yes': 'Fist nodding up and down',
    'no': 'Index and middle finger snap to thumb',
    'peace': 'Peace sign - V shape',
    # New 11
    'hello': 'Open palm, wave side to side',
    'welcome': 'Open arms gesture, palms up',
    'please': 'Flat hand circles on chest',
    'run': 'Hook index fingers, pump alternating',
    'sorry': 'Fist circles on chest',
    'thankyou': 'Flat hand from chin forward',
    'wait': 'Hold up palm, fingers spread',
    'sick': 'Middle finger touches forehead',
    'drink': 'Thumb to mouth, tilting motion',
    'happy': 'Brush chest upward with flat hands',
    'thirsty': 'Index finger traces down throat',
}

SAMPLES_PER_SIGN = 200  # 200 samples each for good accuracy

def extract_normalized_landmarks(results):
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
            
            # Normalize to wrist
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

def collect_data():
    """Collect training data - HOLD SPACE to record"""
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
    print("ALL 26 SIGNS TRAINING")
    print("="*60)
    print(f"Total signs: {len(SIGNS)}")
    print(f"Signs: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print("\nControls:")
    print("  HOLD SPACE = Record continuously (move hand around!)")
    print("  N = Next sign (skip current)")
    print("  Q = Quit and train with collected data")
    print("="*60)
    
    input("\nPress ENTER to start...")
    
    for sign_idx, sign in enumerate(SIGNS):
        samples = []
        hint = HINTS.get(sign, 'Make the gesture')
        
        print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: {sign.upper()}")
        print(f"  Hint: {hint}")
        print(f"  HOLD SPACE and move hand to different positions!")
        
        while len(samples) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            
            # UI
            cv2.rectangle(frame, (0, 0), (640, 100), (40, 40, 40), -1)
            cv2.putText(frame, f"Sign: {sign.upper()}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"[{sign_idx+1}/{len(SIGNS)}] Samples: {len(samples)}/{SAMPLES_PER_SIGN}",
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(frame, hint[:50], (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Progress bar
            progress = len(samples) / SAMPLES_PER_SIGN
            cv2.rectangle(frame, (20, 440), (620, 460), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 440), (20 + int(600 * progress), 460), (0, 255, 0), -1)
            
            # Instructions
            cv2.putText(frame, "HOLD SPACE=Record | N=Next | Q=Quit", (140, 475),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            cv2.imshow('Training - 26 Signs', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE held = record
            if key == ord(' '):
                if results.multi_hand_landmarks:
                    features = extract_normalized_landmarks(results)
                    samples.append(features)
            elif key == ord('n'):
                print(f"  Skipped {sign} with {len(samples)} samples")
                break
            elif key == ord('q'):
                print("\nQuitting early...")
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                
                if len(all_data) > 0:
                    return np.array(all_data), np.array(all_labels)
                return None, None
        
        # Add collected samples
        if len(samples) > 0:
            all_data.extend(samples)
            all_labels.extend([sign] * len(samples))
            print(f"  Completed {sign}: {len(samples)} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return np.array(all_data), np.array(all_labels)

def augment_data(X, y, multiplier=10):
    """Augment data with noise, scaling, rotation"""
    augmented_X = [X]
    augmented_y = [y]
    
    for i in range(multiplier - 1):
        # Random noise
        noise = np.random.normal(0, 0.02, X.shape)
        X_noisy = X + noise
        augmented_X.append(X_noisy)
        augmented_y.append(y)
        
        # Random scaling
        if i % 3 == 0:
            scale = np.random.uniform(0.9, 1.1, (X.shape[0], 1))
            X_scaled = X * scale
            augmented_X.append(X_scaled)
            augmented_y.append(y)
        
        # Random rotation (2D)
        if i % 4 == 0:
            angle = np.random.uniform(-0.15, 0.15, X.shape[0])
            X_rot = X.copy()
            for j in range(X.shape[0]):
                cos_a, sin_a = np.cos(angle[j]), np.sin(angle[j])
                for k in range(0, X.shape[1], 3):
                    x, y_coord = X[j, k], X[j, k+1]
                    X_rot[j, k] = x * cos_a - y_coord * sin_a
                    X_rot[j, k+1] = x * sin_a + y_coord * cos_a
            augmented_X.append(X_rot)
            augmented_y.append(y)
    
    return np.vstack(augmented_X), np.hstack(augmented_y)

def train_model(X, y):
    """Train neural network model"""
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"\nTraining on {len(X)} samples for {num_classes} classes")
    print(f"Classes: {list(le.classes_)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # Convert to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Build model - larger for 26 classes
    model = keras.Sequential([
        keras.layers.Input(shape=(126,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    # Train
    print("\nTraining model...")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n" + "="*60)
    print(f"TEST ACCURACY: {accuracy*100:.2f}%")
    print("="*60)
    
    return model, scaler, le

def save_model(model, scaler, le):
    """Save model and preprocessors"""
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('models/scalers', exist_ok=True)
    
    model.save('models/saved/lstm_model.h5')
    joblib.dump(scaler, 'models/scalers/scaler.pkl')
    joblib.dump(le, 'models/scalers/label_encoder.pkl')
    
    print("Saved!")
    print(f"Signs: {list(le.classes_)}")

def main():
    print("="*60)
    print("TRAIN ALL 26 SIGNS")
    print("="*60)
    print(f"\nSigns to train ({len(SIGNS)} total):")
    print("  Letters: A, B, C, D, E")
    print("  Numbers: 1, 2, 3, 4, 5")
    print("  Words: hi, goodbye, yes, no, peace")
    print("  New 11: hello, welcome, please, run, sorry,")
    print("          thankyou, wait, sick, drink, happy, thirsty")
    print(f"\nSamples per sign: {SAMPLES_PER_SIGN}")
    print(f"Total raw samples: {len(SIGNS) * SAMPLES_PER_SIGN}")
    
    # Collect data
    X, y = collect_data()
    
    if X is None or len(X) == 0:
        print("No data collected!")
        return
    
    print(f"\nCollected {len(X)} samples for {len(np.unique(y))} signs")
    
    # Augment
    print("Augmenting data 10x...")
    X_aug, y_aug = augment_data(X, y, multiplier=10)
    print(f"After augmentation: {len(X_aug)} samples")
    
    # Train
    model, scaler, le = train_model(X_aug, y_aug)
    
    # Save
    save_model(model, scaler, le)
    
    print("\nRun: python detect_simple.py")

if __name__ == "__main__":
    main()
