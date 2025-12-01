"""
Train with Practical Signs - SPACE KEY TO RECORD
45 Signs: Greetings, Responses, Actions, Emotions
Press SPACE when ready to start recording each gesture
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

# Signs to train (45 total)
SIGNS = [
    # Greetings (10)
    'hello', 'hi', 'bye', 'goodbye', 'welcome',
    'good_morning', 'good_night', 'nice_to_meet', 'see_you', 'take_care',
    # Responses (10)
    'yes', 'no', 'ok', 'maybe', 'please',
    'sorry', 'thank_you', 'excuse_me', 'no_problem', 'you_are_welcome',
    # Actions (15)
    'stop', 'go', 'come', 'wait', 'help',
    'eat', 'drink', 'sleep', 'walk', 'run',
    'sit', 'stand', 'open', 'close', 'give',
    # Emotions (10)
    'happy', 'sad', 'angry', 'scared', 'surprised',
    'tired', 'hungry', 'thirsty', 'sick', 'better'
]

# Gesture hints for each sign
HINTS = {
    # Greetings
    'hello': 'Open palm, wave side to side',
    'hi': 'Raise hand, fingers spread',
    'bye': 'Wave goodbye motion',
    'goodbye': 'Wave with both hands or single wave',
    'welcome': 'Open arms gesture, palms up',
    'good_morning': 'Flat hand rises like sun',
    'good_night': 'Hands together, tilt head/hands',
    'nice_to_meet': 'Shake hands gesture',
    'see_you': 'Point to eyes, then point forward',
    'take_care': 'Cross arms over chest gently',
    # Responses
    'yes': 'Nod fist up and down',
    'no': 'Shake finger/hand side to side',
    'ok': 'Make O with thumb and index finger',
    'maybe': 'Flat hands, palms up, tilt alternating',
    'please': 'Flat hand circles on chest',
    'sorry': 'Fist circles on chest',
    'thank_you': 'Flat hand from chin forward',
    'excuse_me': 'Brush fingertips across palm',
    'no_problem': 'Wave hand dismissively',
    'you_are_welcome': 'Open palm moves toward person',
    # Actions
    'stop': 'Flat palm facing forward, firm',
    'go': 'Point forward with index finger',
    'come': 'Beckon with hand/finger toward self',
    'wait': 'Hold up palm, fingers spread',
    'help': 'Thumbs up on flat palm, lift up',
    'eat': 'Fingers to mouth, eating motion',
    'drink': 'Thumb to mouth, tilting motion',
    'sleep': 'Hand on cheek, head tilted',
    'walk': 'Two fingers walk on flat palm',
    'run': 'Hook index fingers, pump alternating',
    'sit': 'Two fingers sit on other two fingers',
    'stand': 'Two fingers stand on flat palm',
    'open': 'Flat hands together, then apart',
    'close': 'Flat hands apart, then together',
    'give': 'Hands move from self toward other',
    # Emotions
    'happy': 'Brush chest upward with flat hands',
    'sad': 'Drag fingers down face',
    'angry': 'Claw hands in front of face',
    'scared': 'Hands up, shaking, startled look',
    'surprised': 'Open hands near face, wide eyes',
    'tired': 'Hands droop down from chest',
    'hungry': 'Hand moves down from throat to stomach',
    'thirsty': 'Index finger traces down throat',
    'sick': 'Middle finger touches forehead and stomach',
    'better': 'Flat hand on mouth, then thumbs up',
}

SAMPLES_PER_SIGN = 80
RECORD_FRAMES = 1  # Frames to record per sample

def extract_landmarks(results):
    """Extract 126 features from both hands"""
    features = []
    
    # Left hand (21 landmarks * 3 coords = 63)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
        hand = results.multi_hand_landmarks[0]
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Right hand or second hand
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
        hand = results.multi_hand_landmarks[1]
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    return features

def collect_data():
    """Collect training data - PRESS SPACE TO RECORD EACH SAMPLE"""
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
    print("SPACE KEY TRAINING MODE")
    print("="*60)
    print(f"Signs to learn: {len(SIGNS)}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print("\nControls:")
    print("  SPACE = Record one sample")
    print("  N = Skip to next sign")
    print("  Q = Quit and train with collected data")
    print("="*60)
    
    for sign_idx, sign in enumerate(SIGNS):
        print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: {sign.upper()}")
        print(f"Hint: {HINTS.get(sign, 'Make the gesture')}")
        
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
            
            # UI Display
            # Progress bar
            progress = samples_collected / SAMPLES_PER_SIGN
            bar_width = 300
            cv2.rectangle(frame, (170, 25), (170 + bar_width, 45), (50, 50, 50), -1)
            cv2.rectangle(frame, (170, 25), (170 + int(bar_width * progress), 45), (0, 255, 0), -1)
            
            # Text info
            cv2.putText(frame, f"Sign: {sign.upper()}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"[{sign_idx+1}/{len(SIGNS)}]", (500, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{SAMPLES_PER_SIGN}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hint
            hint = HINTS.get(sign, 'Make the gesture')
            # Split long hints
            if len(hint) > 45:
                cv2.putText(frame, hint[:45], (10, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, hint[45:], (10, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            else:
                cv2.putText(frame, hint, (10, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Instructions
            cv2.putText(frame, "PRESS SPACE TO RECORD", (150, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "N=Next Sign  Q=Quit", (200, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Hand detection status
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                cv2.putText(frame, f"Hands: {num_hands}", (530, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hands!", (520, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Training - SPACE to Record', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - Record sample
                if results.multi_hand_landmarks:
                    features = extract_landmarks(results)
                    sign_data.append(features)
                    samples_collected += 1
                    print(f"  Recorded sample {samples_collected}/{SAMPLES_PER_SIGN}")
                    
                    # Visual feedback - green flash
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (640, 480), (0, 255, 0), 10)
                    cv2.imshow('Training - SPACE to Record', flash)
                    cv2.waitKey(50)
                else:
                    print("  No hand detected! Show your hand and try again.")
            
            elif key == ord('n'):  # Skip to next sign
                print(f"  Skipping {sign} (collected {samples_collected} samples)")
                break
            
            elif key == ord('q'):  # Quit
                print("\nQuitting early...")
                if sign_data:
                    all_data.extend(sign_data)
                    all_labels.extend([sign] * len(sign_data))
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return np.array(all_data) if all_data else None, all_labels
        
        # Add collected data for this sign
        if sign_data:
            all_data.extend(sign_data)
            all_labels.extend([sign] * len(sign_data))
            print(f"  Completed {sign}: {len(sign_data)} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return np.array(all_data), all_labels

def augment_data(X, y, factor=5):
    """Augment data with noise variations"""
    X_aug = [X]
    y_aug = [y]
    
    for i in range(factor - 1):
        noise = np.random.normal(0, 0.01 * (i + 1), X.shape)
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
    print("PRACTICAL SIGN LANGUAGE TRAINER")
    print("SPACE KEY RECORDING MODE")
    print("="*60)
    print(f"\nTotal signs to learn: {len(SIGNS)}")
    print("\nCategories:")
    print("  - Greetings (10): hello, hi, bye, welcome, etc.")
    print("  - Responses (10): yes, no, ok, please, thank_you, etc.")
    print("  - Actions (15): stop, go, eat, drink, walk, etc.")
    print("  - Emotions (10): happy, sad, angry, surprised, etc.")
    
    input("\nPress ENTER to start training...")
    
    # Collect data
    X, y = collect_data()
    
    if X is None or len(X) == 0:
        print("No data collected!")
        return
    
    print(f"\nCollected {len(X)} samples for {len(set(y))} signs")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Signs learned: {list(le.classes_)}")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Augment
    print("\nAugmenting data...")
    X_aug, y_aug = augment_data(X_scaled, y_encoded, factor=5)
    print(f"After augmentation: {len(X_aug)} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    # Build and train
    print("\nBuilding model...")
    model = build_model(len(le.classes_))
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    print("Training...")
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
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Save everything
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('models/scalers', exist_ok=True)
    
    model.save('models/saved/lstm_model.h5')
    joblib.dump(scaler, 'models/scalers/scaler.pkl')
    joblib.dump(le, 'models/scalers/label_encoder.pkl')
    
    print("\nSaved:")
    print("  - models/saved/lstm_model.h5")
    print("  - models/scalers/scaler.pkl")
    print("  - models/scalers/label_encoder.pkl")
    
    print(f"\nTrained signs: {list(le.classes_)}")
    print("\nRun 'python -c \"from src.detect import run_detection; run_detection()\"' to test!")

if __name__ == "__main__":
    main()
