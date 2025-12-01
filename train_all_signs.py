"""
FULL Sign Language Training - ALL 111 SIGNS
Trains on complete alphabet, numbers, greetings, responses, actions, etc.
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
from tensorflow.keras.regularizers import l2
import joblib

# Paths
MODEL_DIR = 'models/saved'
SCALER_DIR = 'models/scalers'

# ============================================================================
# ALL 111 SIGNS ORGANIZED BY CATEGORY
# ============================================================================
SIGN_CATEGORIES = {
    'Alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
    
    'Numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    
    'Greetings': ['hello', 'hi', 'bye', 'goodbye', 'welcome', 'good_morning',
                  'good_night', 'nice_to_meet', 'see_you', 'take_care'],
    
    'Responses': ['yes', 'no', 'ok', 'maybe', 'please', 'sorry', 'thank_you',
                  'excuse_me', 'no_problem', 'you_are_welcome'],
    
    'Actions': ['stop', 'go', 'come', 'wait', 'help', 'eat', 'drink', 'sleep',
                'walk', 'run', 'sit', 'stand', 'open', 'close', 'give'],
    
    'Emotions': ['happy', 'sad', 'angry', 'scared', 'surprised', 'tired',
                 'hungry', 'thirsty', 'sick', 'better'],
    
    'Questions': ['what', 'where', 'when', 'why', 'how', 'who', 'which',
                  'whose', 'how_much', 'how_many'],
    
    'Time': ['today', 'tomorrow', 'yesterday', 'now', 'later', 'morning',
             'afternoon', 'evening', 'night', 'week'],
    
    'People': ['me', 'you', 'he', 'she', 'we', 'they', 'friend', 'family',
               'mother', 'father']
}

# Sign descriptions for guidance
SIGN_HINTS = {
    # Alphabet
    'A': 'Closed fist, thumb on side',
    'B': 'Flat hand, fingers together',
    'C': 'Curved hand like cup',
    'D': 'Index up, others touch thumb',
    'E': 'Fingers bent, thumb tucked',
    'F': 'Index+thumb circle, others up',
    'G': 'Index+thumb pointing sideways',
    'H': 'Index+middle sideways',
    'I': 'Pinky up only',
    'J': 'Pinky up, trace J',
    'K': 'Index+middle V, thumb between',
    'L': 'L shape thumb+index',
    'M': '3 fingers over thumb',
    'N': '2 fingers over thumb',
    'O': 'Fingers touch thumb (O shape)',
    'P': 'Like K pointing down',
    'Q': 'Like G pointing down',
    'R': 'Index+middle crossed',
    'S': 'Fist, thumb over fingers',
    'T': 'Thumb between index+middle',
    'U': 'Index+middle together up',
    'V': 'Peace sign',
    'W': '3 fingers spread',
    'X': 'Index hooked',
    'Y': 'Thumb+pinky out',
    'Z': 'Index traces Z',
    # Numbers
    '0': 'O shape',
    '1': 'Index up',
    '2': 'Index+middle up',
    '3': 'Thumb+index+middle up',
    '4': '4 fingers up',
    '5': 'All fingers spread',
    '6': 'Thumb touches pinky',
    '7': 'Thumb touches ring',
    '8': 'Thumb touches middle',
    '9': 'Thumb touches index',
    # Greetings
    'hello': 'Wave near forehead',
    'hi': 'Quick wave',
    'bye': 'Open/close hand wave',
    'goodbye': 'Wave away',
    'welcome': 'Hand toward body',
    'good_morning': 'Chin up + sun rising',
    'good_night': 'Hands on cheek (sleep)',
    'nice_to_meet': 'Index fingers meet',
    'see_you': 'Point eyes then out',
    'take_care': 'Cross hands on chest',
    # Responses
    'yes': 'Fist nodding',
    'no': 'Fingers snap to thumb',
    'ok': 'OK sign',
    'maybe': 'Hands alternating',
    'please': 'Circle on chest',
    'sorry': 'Fist circle on chest',
    'thank_you': 'Chin to outward',
    'excuse_me': 'Brush across palm',
    'no_problem': 'Brush off',
    'you_are_welcome': 'Chest outward',
    # Actions
    'stop': 'Palm out',
    'go': 'Both pointing forward',
    'come': 'Beckoning finger',
    'wait': 'Palms patting',
    'help': 'Thumbs up lifting',
    'eat': 'Fingers to mouth',
    'drink': 'Tilting cup motion',
    'sleep': 'Hand on cheek',
    'walk': 'Fingers walking',
    'run': 'Fast alternating',
    'sit': 'Fingers sitting',
    'stand': 'Fingers standing',
    'open': 'Hands spreading',
    'close': 'Hands together',
    'give': 'Hands outward',
    # Emotions
    'happy': 'Hands up chest',
    'sad': 'Fingers down face',
    'angry': 'Claw face outward',
    'scared': 'Hands up shaking',
    'surprised': 'Hands up opening',
    'tired': 'Hands drop down',
    'hungry': 'Hand throat to stomach',
    'thirsty': 'Finger down throat',
    'sick': 'Forehead + stomach',
    'better': 'Chin to thumbs up',
    # Questions
    'what': 'Palms up shrug',
    'where': 'Finger wagging',
    'when': 'Finger circle+point',
    'why': 'Forehead wiggle',
    'how': 'Fists rolling open',
    'who': 'Finger near mouth circle',
    'which': 'Thumbs alternating',
    'whose': 'Finger mouth circle',
    'how_much': 'Claws opening up',
    'how_many': 'Fist to spread',
    # Time
    'today': 'Hands down then up',
    'tomorrow': 'Thumb forward',
    'yesterday': 'Thumb backward',
    'now': 'Hands drop sharp',
    'later': 'L tilting forward',
    'morning': 'Hand rising up',
    'afternoon': 'Forearm down',
    'evening': 'Hand drooping',
    'night': 'Hand over wrist',
    'week': 'Index across palm',
    # People
    'me': 'Point to self',
    'you': 'Point outward',
    'he': 'Point to side',
    'she': 'Point other side',
    'we': 'Touch both shoulders',
    'they': 'Sweep across',
    'friend': 'Fingers hooked',
    'family': 'F hands circling',
    'mother': 'Thumb on chin',
    'father': 'Thumb on forehead',
}

# Flatten all signs
ALL_SIGNS = []
for category, signs in SIGN_CATEGORIES.items():
    ALL_SIGNS.extend(signs)

print(f"Total signs to train: {len(ALL_SIGNS)}")

# Training settings
SAMPLES_PER_SIGN = 60  # Reasonable for 111 signs


def collect_category_data(category_name, signs, hands, cap, mp_draw, mp_hands):
    """Collect data for one category"""
    features = []
    labels = []
    
    print(f"\n{'='*60}")
    print(f"📁 CATEGORY: {category_name.upper()} ({len(signs)} signs)")
    print(f"{'='*60}")
    
    for sign_idx, sign in enumerate(signs):
        hint = SIGN_HINTS.get(sign, 'Make the sign gesture')
        print(f"\n[{sign_idx+1}/{len(signs)}] Sign: {sign}")
        print(f"   Hint: {hint}")
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0, 0), (640, 140), (40, 40, 40), -1)
                cv2.putText(frame, f"{category_name}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
                cv2.putText(frame, f"Sign: {sign}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
                cv2.putText(frame, f"Starting in {i}...", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, hint, (20, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Sign Language Training', frame)
                cv2.waitKey(1000)
        
        # Collect
        sign_samples = []
        while len(sign_samples) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            progress = len(sign_samples) / SAMPLES_PER_SIGN
            
            # UI
            cv2.rectangle(frame, (0, 0), (640, 120), (40, 40, 40), -1)
            cv2.putText(frame, f"{category_name}: {sign}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"{len(sign_samples)}/{SAMPLES_PER_SIGN}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Progress bar
            bar_w = int(progress * 450)
            cv2.rectangle(frame, (150, 70), (600, 95), (80, 80, 80), -1)
            cv2.rectangle(frame, (150, 70), (150 + bar_w, 95), (0, 255, 0), -1)
            
            cv2.putText(frame, hint, (20, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                feat = extract_features(results)
                if feat is not None:
                    sign_samples.append(feat)
                    cv2.circle(frame, (610, 50), 12, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "SHOW HAND", (220, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.imshow('Sign Language Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, None
            elif key == ord('s'):  # Skip sign
                print(f"   ⏭️ Skipped {sign}")
                break
        
        if len(sign_samples) > 0:
            features.extend(sign_samples)
            labels.extend([sign] * len(sign_samples))
            print(f"   ✅ Collected {len(sign_samples)} samples")
    
    return features, labels


def extract_features(results):
    """Extract 126 features"""
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


def collect_all_data():
    """Collect data for all categories"""
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
    print("🤟 FULL SIGN LANGUAGE TRAINING - 111 SIGNS")
    print("="*60)
    print(f"\nCategories: {list(SIGN_CATEGORIES.keys())}")
    print(f"Total signs: {len(ALL_SIGNS)}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"\nEstimated time: ~{len(ALL_SIGNS) * 5 // 60} minutes")
    print("\nControls:")
    print("  Q = Quit (saves progress)")
    print("  S = Skip current sign")
    print("="*60)
    
    for cat_name, cat_signs in SIGN_CATEGORIES.items():
        input(f"\n➡️ Press ENTER to start '{cat_name}' ({len(cat_signs)} signs)...")
        
        features, labels = collect_category_data(
            cat_name, cat_signs, hands, cap, mp_draw, mp_hands
        )
        
        if features is None:
            print("\n⚠️ Collection interrupted. Saving what we have...")
            break
        
        all_features.extend(features)
        all_labels.extend(labels)
        
        print(f"\n✅ {cat_name} complete! Total so far: {len(all_features)} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    if len(all_features) == 0:
        return None, None
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*60}")
    print(f"✅ COLLECTION COMPLETE!")
    print(f"   Total samples: {len(X)}")
    print(f"   Unique signs: {len(np.unique(y))}")
    print(f"{'='*60}")
    
    return X, y


def augment_data(X, y, factor=8):
    """Augment for robustness"""
    print(f"\n🔄 Augmenting data (x{factor})...")
    
    aug_X = [X.copy()]
    aug_y = [y.copy()]
    
    for i in range(factor - 1):
        t = i % 4
        if t == 0:
            noise = np.random.normal(0, 0.01, X.shape)
            aug_X.append(X + noise)
        elif t == 1:
            noise = np.random.normal(0, 0.02, X.shape)
            aug_X.append(X + noise)
        elif t == 2:
            scale = np.random.uniform(0.95, 1.05, X.shape)
            aug_X.append(X * scale)
        else:
            noise = np.random.normal(0, 0.015, X.shape)
            scale = np.random.uniform(0.97, 1.03, X.shape)
            aug_X.append((X + noise) * scale)
        aug_y.append(y.copy())
    
    X_aug = np.vstack(aug_X)
    y_aug = np.hstack(aug_y)
    
    # Shuffle
    idx = np.random.permutation(len(X_aug))
    print(f"✅ Augmented: {len(X)} → {len(X_aug)}")
    
    return X_aug[idx], y_aug[idx]


def train_model(X, y):
    """Train on all signs"""
    print("\n" + "="*60)
    print("🏋️ TRAINING MODEL")
    print("="*60)
    
    # Augment
    X_aug, y_aug = augment_data(X, y, factor=8)
    
    # Encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y_aug)
    num_classes = len(le.classes_)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Classes: {num_classes}")
    
    # Build model - larger for more classes
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
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=20,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=1
    )
    
    print("\n🏋️ Training...")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{'='*60}")
    print(f"🏆 TEST ACCURACY: {acc*100:.2f}%")
    print(f"{'='*60}")
    
    # Per-category accuracy
    print("\nPer-category accuracy:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    for cat_name, cat_signs in SIGN_CATEGORIES.items():
        cat_correct = 0
        cat_total = 0
        for sign in cat_signs:
            if sign in le.classes_:
                idx = list(le.classes_).index(sign)
                mask = y_test == idx
                if mask.sum() > 0:
                    cat_correct += (y_pred[mask] == idx).sum()
                    cat_total += mask.sum()
        
        if cat_total > 0:
            cat_acc = cat_correct / cat_total * 100
            status = "✅" if cat_acc >= 90 else "⚠️" if cat_acc >= 75 else "❌"
            print(f"  {status} {cat_name:15s}: {cat_acc:.1f}%")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Model saved with {num_classes} signs!")
    
    return model, scaler, le, acc


def test_realtime(model, scaler, le):
    """Test detection"""
    print("\n" + "="*60)
    print("🎥 TESTING")
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
    
    from collections import deque
    pred_buffer = deque(maxlen=5)
    
    print("✅ Show signs! Q to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        cv2.rectangle(frame, (0, 0), (640, 120), (30, 30, 30), -1)
        cv2.putText(frame, f"SIGN DETECTOR ({len(le.classes_)} signs)", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            features = extract_features(results)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                probs = model.predict(features_scaled, verbose=0)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]
                
                pred_buffer.append(pred_idx)
                
                from collections import Counter
                if len(pred_buffer) >= 3:
                    most_common = Counter(pred_buffer).most_common(1)[0][0]
                    sign = le.inverse_transform([most_common])[0]
                else:
                    sign = le.inverse_transform([pred_idx])[0]
                
                color = (0,255,0) if conf > 0.7 else (0,255,255) if conf > 0.4 else (0,165,255)
                
                cv2.rectangle(frame, (10, 40), (630, 115), color, 3)
                cv2.putText(frame, sign.upper(), (25, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
                cv2.putText(frame, f"{conf*100:.0f}%", (520, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
        else:
            cv2.putText(frame, "Show hand", (200, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
            pred_buffer.clear()
        
        cv2.rectangle(frame, (0, 450), (640, 480), (30, 30, 30), -1)
        cv2.putText(frame, "Q = Quit", (280, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
        
        cv2.imshow('Full Sign Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main():
    print("\n" + "="*60)
    print("🤟 COMPLETE SIGN LANGUAGE TRAINING")
    print("="*60)
    print(f"\nThis will train ALL {len(ALL_SIGNS)} signs!")
    print("\nCategories:")
    for cat, signs in SIGN_CATEGORIES.items():
        print(f"  • {cat}: {len(signs)} signs")
    print(f"\nTotal time: ~{len(ALL_SIGNS) * 5 // 60} minutes")
    print("="*60)
    
    # Collect
    X, y = collect_all_data()
    
    if X is None or len(X) == 0:
        print("No data!")
        return
    
    # Train
    model, scaler, le, acc = train_model(X, y)
    
    print(f"\n{'='*60}")
    if acc >= 0.90:
        print("🏆 EXCELLENT! Very high accuracy!")
    elif acc >= 0.80:
        print("👍 Good! Model should work well.")
    else:
        print("⚠️ Consider retraining with clearer gestures.")
    
    # Test
    input("\nPress ENTER to test...")
    test_realtime(model, scaler, le)
    
    print("\n✅ Done! Run 'python main.py' anytime.")


if __name__ == "__main__":
    main()
