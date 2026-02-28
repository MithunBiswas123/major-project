"""
Collect ONLY the missing signs (yes, wait, welcome) via webcam,
replace their old synthetic data, then retrain the model.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import sys
from datetime import datetime

# Signs to recollect (still have old synthetic data at 500 samples)
MISSING_SIGNS = ['wait', 'welcome', 'yes']

SAMPLES_PER_SIGN = 100
CAPTURE_INTERVAL = 0.12
COUNTDOWN_SECONDS = 3

GESTURE_TIPS = {
    'yes': 'Fist nodding up and down (S-hand nod)',
    'wait': 'Open hands, palms down, patting motion',
    'welcome': 'Open hand gesture toward body',
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

CSV_PATH = 'data/raw/sign_dataset.csv'


def extract_features(results):
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
    collected = []
    collecting = False
    countdown_start = None
    last_capture = 0
    samples_collected = 0

    print(f"\n  >> Press SPACE to start countdown, Q to skip")

    while samples_collected < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_detected = results.multi_hand_landmarks is not None

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 140), (40, 40, 40), -1)

        tip = GESTURE_TIPS.get(sign_label, '')
        cv2.putText(frame, f"Sign: {sign_label.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, tip[:60], (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        progress = int((samples_collected / SAMPLES_PER_SIGN) * (w - 40))
        cv2.rectangle(frame, (20, 100), (w - 20, 125), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 100), (20 + progress, 125), (0, 255, 0), -1)
        cv2.putText(frame, f"{samples_collected}/{SAMPLES_PER_SIGN}", (w - 120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status = "HAND OK" if hand_detected else "NO HAND"
        cv2.putText(frame, status, (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
            else:
                collecting = True
                countdown_start = None

        if collecting and hand_detected:
            now = time.time()
            if now - last_capture >= CAPTURE_INTERVAL:
                features = extract_features(results)
                collected.append(features)
                samples_collected += 1
                last_capture = now
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)

        if not collecting and countdown_start is None:
            cv2.putText(frame, "SPACE = Start | Q = Skip", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        elif collecting:
            cv2.putText(frame, "COLLECTING... hold your sign steady!", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Collect Missing Signs', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not collecting and countdown_start is None:
            countdown_start = time.time()
        elif key == ord('q'):
            print(f"  >> Skipped '{sign_label}' ({samples_collected} collected)")
            break

    return collected


def main():
    print("\n" + "=" * 60)
    print("   COLLECT MISSING SIGNS: yes, wait, welcome")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"   Current dataset: {len(df)} samples, {df['sign'].nunique()} signs")
    for s in MISSING_SIGNS:
        cnt = len(df[df['sign'] == s])
        print(f"   {s:12s}: {cnt} samples (old synthetic - will be replaced)")

    input("\n   Press ENTER to open webcam...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("   ERROR: Cannot open webcam!")
        return

    all_new = []
    collected_signs = []

    for i, sign in enumerate(MISSING_SIGNS):
        tip = GESTURE_TIPS.get(sign, '')
        print(f"\n{'=' * 60}")
        print(f"   [{i+1}/{len(MISSING_SIGNS)}] Sign: {sign.upper()}")
        print(f"   Gesture: {tip}")
        print(f"{'=' * 60}")

        samples = collect_sign(sign, cap)
        if samples:
            for feat_vec in samples:
                row = {'sign': sign, 'timestamp': datetime.now().isoformat()}
                for j, val in enumerate(feat_vec):
                    row[f'feature_{j}'] = val
                all_new.append(row)
            collected_signs.append(sign)
            print(f"   >> Collected {len(samples)} samples for '{sign}'")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if not all_new:
        print("\n   No data collected. Exiting.")
        return

    # Replace old synthetic data with new webcam data
    new_df = pd.DataFrame(all_new)
    print(f"\n   Newly collected: {len(new_df)} samples")

    # Remove old synthetic data for collected signs
    keep_mask = ~df['sign'].isin(collected_signs)
    df_kept = df[keep_mask].copy()
    combined = pd.concat([df_kept, new_df], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"   Saved: {CSV_PATH} ({len(combined)} total samples)")

    # Show summary
    print(f"\n   Updated dataset:")
    for sign in sorted(combined['sign'].unique()):
        cnt = len(combined[combined['sign'] == sign])
        print(f"      {sign:12s}: {cnt} samples")

    # Retrain
    print(f"\n{'=' * 60}")
    retrain = input("   Retrain model now? (y/n): ").strip().lower()
    if retrain == 'y':
        retrain_model()


def retrain_model():
    print(f"\n{'=' * 60}")
    print(f"   RETRAINING MODEL")
    print(f"{'=' * 60}")

    sys.path.insert(0, os.path.dirname(__file__))
    from src.preprocessing import DataPreprocessor
    from src.models import get_model, get_callbacks, get_model_path
    from src.config import EPOCHS, BATCH_SIZE, MODELS_DIR
    import tensorflow as tf

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

    model = get_model('hybrid', num_features=X_train.shape[1], num_classes=num_classes)
    model_path = get_model_path('hybrid')
    callbacks = get_callbacks(model_path)

    from collections import Counter
    class_counts = Counter(y_train)
    total = len(y_train)
    n_classes = len(class_counts)
    class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}

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

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   Test accuracy: {test_acc * 100:.1f}%")

    final_path = os.path.join(MODELS_DIR, 'hybrid_model.h5')
    model.save(final_path)
    print(f"   Saved: {final_path}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n   Per-sign test accuracy:")
    for idx in sorted(set(y_test)):
        mask = y_test == idx
        sign_acc = (y_pred[mask] == y_test[mask]).mean()
        print(f"      {class_names[idx]:12s}: {sign_acc * 100:.0f}%")

    print(f"\n{'=' * 60}")
    print(f"   DONE! Run: python detect_sign.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
