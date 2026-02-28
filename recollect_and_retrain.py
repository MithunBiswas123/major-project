"""
=================================================================
  RECOLLECT OLD SIGNS VIA WEBCAM + RETRAIN
=================================================================
The old 27 signs have synthetic data (wrist=[0,0,0], range [-31,16])
that doesn't match what your webcam produces (wrist=[0.5,0.7,~0], 
range [0,1]). That's why the model can't detect them live.

This script:
  1. Lets you recollect all 27 old signs from your webcam
  2. Replaces the old synthetic data with real webcam data
  3. Keeps the 14 new signs data intact
  4. Retrains the model on all-webcam data

Run:  python recollect_and_retrain.py
=================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import sys
import shutil
from datetime import datetime

# ── Settings ─────────────────────────────────────────────────────────────────
SAMPLES_PER_SIGN = 100      # How many samples per sign (match new signs)
CAPTURE_INTERVAL = 0.12     # Seconds between captures (fast)
COUNTDOWN_SECONDS = 3       # Countdown before capture starts

# The 27 old signs that need recollection
OLD_SIGNS = [
    '1', '2', '3', '4', '5',
    'A', 'B', 'C', 'D', 'E',
    'drink', 'eat', 'happy', 'hello', 'help',
    'no', 'peace', 'run', 'sick', 'sorry',
    'stop', 'thankyou', 'thirsty', 'tired',
    'wait', 'welcome', 'yes'
]

# Gesture tips for each sign (to help you remember the gesture)
GESTURE_TIPS = {
    'A': 'Closed fist, thumb on side',
    'B': 'Flat hand, fingers together, thumb across palm',
    'C': 'Curved hand like holding a cup',
    'D': 'Index up, others touch thumb in circle',
    'E': 'Fingers bent down, thumb tucked',
    '1': 'Index finger pointing up',
    '2': 'Index + middle fingers up (peace)',
    '3': 'Thumb + index + middle up',
    '4': 'Four fingers up, thumb folded',
    '5': 'All five fingers spread open',
    'hello': 'Open hand wave near forehead',
    'yes': 'Fist nodding up and down (or S-hand nod)',
    'no': 'Index + middle finger snap to thumb',
    'stop': 'Flat hand facing outward, palm out',
    'eat': 'Fingers to mouth repeatedly',
    'drink': 'Thumb to mouth, tilting hand like cup',
    'happy': 'Flat hands brushing up chest repeatedly',
    'help': 'Thumbs up on flat palm, lifting up',
    'sorry': 'Fist circular motion on chest',
    'thankyou': 'Flat hand from chin moving outward',
    'welcome': 'Open hand gesture toward body',
    'wait': 'Open hands, palms down, patting motion',
    'run': 'Index fingers moving fast alternately',
    'sick': 'Middle finger on forehead, one on stomach',
    'thirsty': 'Index finger tracing down throat',
    'tired': 'Both hands on chest, dropping down',
    'peace': 'Peace sign - index and middle spread (V shape)',
}

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

CSV_PATH = 'data/raw/sign_dataset.csv'
BACKUP_PATH = 'data/raw/sign_dataset_backup_synthetic.csv'


def extract_features(results):
    """Extract 126 raw features from MediaPipe results (same as detect_sign.py)."""
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= len(results.multi_handedness):
                continue
            hand_label = results.multi_handedness[idx].classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)
            if hand_label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks

    return np.concatenate([left_hand, right_hand])


def collect_sign(sign_label, cap):
    """Collect samples for one sign using webcam."""
    collected = []
    collecting = False
    countdown_start = None
    last_capture = 0
    samples_collected = 0

    print(f"\n  >> Press SPACE to start countdown, Q to skip this sign")

    while samples_collected < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_detected = results.multi_hand_landmarks is not None

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

        # UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 140), (40, 40, 40), -1)

        tip = GESTURE_TIPS.get(sign_label, '')
        cv2.putText(frame, f"Sign: {sign_label.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, tip[:60], (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Progress bar
        progress = int((samples_collected / SAMPLES_PER_SIGN) * (w - 40))
        cv2.rectangle(frame, (20, 100), (w - 20, 125), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 100), (20 + progress, 125), (0, 255, 0), -1)
        cv2.putText(frame, f"{samples_collected}/{SAMPLES_PER_SIGN}", (w - 120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Hand status
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status = "HAND OK" if hand_detected else "NO HAND"
        cv2.putText(frame, status, (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Countdown
        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
            else:
                collecting = True
                countdown_start = None

        # Collect
        if collecting and hand_detected:
            now = time.time()
            if now - last_capture >= CAPTURE_INTERVAL:
                features = extract_features(results)
                collected.append(features)
                samples_collected += 1
                last_capture = now
                # Flash green border
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)

        if not collecting and countdown_start is None:
            cv2.putText(frame, "SPACE = Start | Q = Skip", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        elif collecting:
            cv2.putText(frame, "COLLECTING... hold your sign steady!", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Recollect Signs', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not collecting and countdown_start is None:
            countdown_start = time.time()
        elif key == ord('q'):
            print(f"  >> Skipped '{sign_label}' ({samples_collected} samples collected)")
            break

    return collected


def main():
    print("\n" + "=" * 65)
    print("   RECOLLECT OLD SIGNS + RETRAIN")
    print("=" * 65)
    print(f"   Signs to recollect: {len(OLD_SIGNS)}")
    print(f"   Samples per sign:   {SAMPLES_PER_SIGN}")
    print(f"   Total to collect:   {len(OLD_SIGNS) * SAMPLES_PER_SIGN}")
    print("=" * 65)

    # Check what already exists
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH, low_memory=False)
        feat_cols = [c for c in df_existing.columns if c.startswith('feature_')]
        print(f"\n   Existing dataset: {len(df_existing)} samples, "
              f"{df_existing['sign'].nunique()} signs")

        # Show which old signs have synthetic vs webcam data
        for sign in OLD_SIGNS:
            subset = df_existing[df_existing['sign'] == sign]
            if len(subset) > 0:
                feats = subset[feat_cols].values[:1].astype(float)
                wrist = feats[0][:63].reshape(21, 3)[0]
                is_synthetic = np.allclose(wrist, 0, atol=0.01)
                tag = "SYNTHETIC (needs recollection)" if is_synthetic else "WEBCAM (ok)"
                print(f"      {sign:12s}: {len(subset):4d} samples - {tag}")
            else:
                print(f"      {sign:12s}: NO DATA")
    else:
        df_existing = None
        feat_cols = [f'feature_{i}' for i in range(126)]

    print(f"\n   The old synthetic data will be REPLACED with real webcam data.")
    print(f"   New sign data (hi, is, name, etc.) will be kept intact.")
    input("\n   Press ENTER to start webcam collection (or Ctrl+C to cancel)...")

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("   ERROR: Cannot open webcam!")
        return

    all_new_data = []
    collected_signs = []

    for i, sign in enumerate(OLD_SIGNS):
        tip = GESTURE_TIPS.get(sign, '')
        print(f"\n{'=' * 65}")
        print(f"   [{i + 1}/{len(OLD_SIGNS)}] Sign: {sign.upper()}")
        print(f"   Gesture: {tip}")
        print(f"{'=' * 65}")

        samples = collect_sign(sign, cap)
        
        if samples:
            for feat_vec in samples:
                row = {'sign': sign, 'timestamp': datetime.now().isoformat()}
                for j, val in enumerate(feat_vec):
                    row[f'feature_{j}'] = val
                all_new_data.append(row)
            collected_signs.append(sign)
            print(f"   >> Collected {len(samples)} samples for '{sign}'")
        else:
            print(f"   >> No samples for '{sign}'")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if not all_new_data:
        print("\n   No data collected. Exiting.")
        return

    # Build the new dataset
    print(f"\n{'=' * 65}")
    print(f"   BUILDING NEW DATASET")
    print(f"{'=' * 65}")

    new_df = pd.DataFrame(all_new_data)
    print(f"   Newly collected: {len(new_df)} samples for {len(collected_signs)} signs")

    if df_existing is not None:
        # Backup the original
        if not os.path.exists(BACKUP_PATH):
            shutil.copy2(CSV_PATH, BACKUP_PATH)
            print(f"   Backed up original to: {BACKUP_PATH}")

        # Remove OLD synthetic data for signs we recollected
        keep_mask = ~df_existing['sign'].isin(collected_signs)
        df_kept = df_existing[keep_mask].copy()
        print(f"   Kept {len(df_kept)} samples of non-recollected signs")

        # Combine
        combined = pd.concat([df_kept, new_df], ignore_index=True)
    else:
        combined = new_df

    # Save
    combined.to_csv(CSV_PATH, index=False)
    print(f"   Saved: {CSV_PATH} ({len(combined)} total samples)")
    
    # Show summary
    print(f"\n   Dataset summary:")
    for sign in sorted(combined['sign'].unique()):
        cnt = len(combined[combined['sign'] == sign])
        print(f"      {sign:12s}: {cnt} samples")

    # Ask about retraining
    print(f"\n{'=' * 65}")
    retrain = input("   Retrain model now? (y/n): ").strip().lower()
    if retrain == 'y':
        retrain_model()


def retrain_model():
    """Retrain the hybrid model on the updated dataset."""
    print(f"\n{'=' * 65}")
    print(f"   RETRAINING MODEL")
    print(f"{'=' * 65}")

    # Import training pipeline
    sys.path.insert(0, os.path.dirname(__file__))
    from src.preprocessing import DataPreprocessor, normalize_hand_features
    from src.models import get_model, get_callbacks, get_model_path
    from src.config import EPOCHS, BATCH_SIZE, MODELS_DIR

    import tensorflow as tf

    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(augment=True)
    
    if data is None:
        print("   ERROR: Preprocessing failed!")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = data
    num_classes = preprocessor.get_num_classes()
    class_names = preprocessor.get_class_names()

    print(f"\n   Classes: {num_classes}")
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Build model
    model = get_model('hybrid', num_features=X_train.shape[1], num_classes=num_classes)

    # Callbacks
    model_path = get_model_path('hybrid')
    callbacks = get_callbacks(model_path)

    # Class weights
    from collections import Counter
    class_counts = Counter(y_train)
    total = len(y_train)
    n_classes = len(class_counts)
    class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}

    # Train
    print(f"\n   Training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   Test accuracy: {test_acc * 100:.1f}%")

    # Save final model
    final_path = os.path.join(MODELS_DIR, 'hybrid_model.h5')
    model.save(final_path)
    print(f"   Saved: {final_path}")

    # Per-sign accuracy check
    import numpy as np
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n   Per-sign test accuracy:")
    for idx in sorted(set(y_test)):
        mask = y_test == idx
        sign_acc = (y_pred[mask] == y_test[mask]).mean()
        sign_name = class_names[idx]
        print(f"      {sign_name:12s}: {sign_acc * 100:.0f}%")

    print(f"\n{'=' * 65}")
    print(f"   DONE! Model retrained with all-webcam data.")
    print(f"   Run: python detect_sign.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
