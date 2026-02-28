"""
Simple Sign Language Detection - Standalone Script
Uses webcam + MediaPipe + Your trained ML model

IMPORTANT: Feature extraction here MUST match the training pipeline
(HandDetector.extract_landmarks in src/data_collection.py) exactly:
  - Raw x, y, z coordinates (NO wrist normalization, NO hand-size scaling)
  - Left hand = features[0:63], Right hand = features[63:126]
  - Handedness from MediaPipe determines which slot gets filled
  - StandardScaler applied after extraction (same scaler from training)
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
from collections import deque, Counter

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7     # match training config
)

# ── Load model + preprocessors ───────────────────────────────────────────────
print("Loading model...")
base_dir = os.path.dirname(__file__)
model_path  = os.path.join(base_dir, 'models', 'saved', 'hybrid_model.h5')
le_path     = os.path.join(base_dir, 'models', 'saved', 'label_encoder.pkl')
scaler_path = os.path.join(base_dir, 'models', 'saved', 'scaler.pkl')

model         = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(le_path)
scaler        = joblib.load(scaler_path)

print(f"✅ Model loaded!  {len(label_encoder.classes_)} signs: {list(label_encoder.classes_)}")

# ── Prediction smoothing buffer ──────────────────────────────────────────────
BUFFER_SIZE = 7
pred_buffer = deque(maxlen=BUFFER_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extraction  –  must replicate HandDetector.extract_landmarks exactly
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(results):
    """
    Extract 126 raw features from MediaPipe hand results.
    Format: [left_hand_63 + right_hand_63]
    Each hand = 21 landmarks × 3 coords (x, y, z) in raw MediaPipe scale.
    If a hand is not detected its slot stays all-zeros.
    """
    left_hand  = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= len(results.multi_handedness):
                continue

            # MediaPipe gives handedness label
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


def normalize_hand_features(features_126):
    """
    Per-sample hand normalization — MUST match preprocessing.normalize_hand_features.
    Subtract wrist, normalize x,y together and z separately.
    """
    result = features_126.copy().astype(np.float64)

    for start in [0, 63]:
        hand = result[start:start+63].copy()
        if np.any(hand != 0):
            landmarks = hand.reshape(21, 3)
            wrist = landmarks[0].copy()
            landmarks = landmarks - wrist

            non_wrist = landmarks[1:]  # 20 landmarks

            # L2-normalize x,y together (2D hand shape)
            xy = non_wrist[:, :2].flatten()
            xy_norm = np.linalg.norm(xy)
            if xy_norm > 1e-6:
                non_wrist[:, :2] = non_wrist[:, :2] / xy_norm

            # L2-normalize z separately (depth pattern)
            z = non_wrist[:, 2]
            z_norm = np.linalg.norm(z)
            if z_norm > 1e-6:
                non_wrist[:, 2] = z / z_norm
            else:
                non_wrist[:, 2] = 0.0

            landmarks[1:] = non_wrist
            result[start:start+63] = landmarks.flatten()

    return result


def predict_sign(features):
    """Normalize, scale, then predict."""
    # Step 1: Per-sample hand normalization (wrist-relative + L2)
    features = normalize_hand_features(features)

    # Step 2: StandardScaler from training
    features_2d = features.reshape(1, -1)
    features_2d = scaler.transform(features_2d)

    pred = model.predict(features_2d, verbose=0)
    idx  = np.argmax(pred[0])
    conf = float(pred[0][idx])
    sign = label_encoder.classes_[idx]
    return sign, conf, pred[0]


def get_smoothed_prediction():
    """Return the most-voted sign + its average confidence from the buffer."""
    if not pred_buffer:
        return None, 0.0

    signs = [s for s, _ in pred_buffer]
    counter = Counter(signs)
    best_sign, count = counter.most_common(1)[0]
    avg_conf = np.mean([c for s, c in pred_buffer if s == best_sign])
    return best_sign, avg_conf


# ═══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 50)
    print("  SIGN LANGUAGE DETECTION")
    print("=" * 50)
    print("  Press 'q' to quit")
    print("=" * 50 + "\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_sign = ""
    current_conf = 0.0
    frame_count  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = results.multi_hand_landmarks is not None

        # ── Draw hand landmarks ──────────────────────────────────────────
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

        # ── Predict ──────────────────────────────────────────────────────
        if hand_detected:
            features = extract_features(results)
            sign, conf, probs = predict_sign(features)

            # Push raw prediction into smoothing buffer
            pred_buffer.append((sign, conf))

            # Use smoothed result
            smoothed_sign, smoothed_conf = get_smoothed_prediction()
            if smoothed_sign and smoothed_conf > 0.15:
                current_sign = smoothed_sign
                current_conf = smoothed_conf

            # Debug: print top-3 every ~1 second
            if frame_count % 30 == 0:
                top3 = np.argsort(probs)[-3:][::-1]
                info = "  ".join(
                    f"{label_encoder.classes_[i]}:{probs[i]*100:.1f}%"
                    for i in top3
                )
                print(f"[Top 3] {info}")
        else:
            # Clear buffer when no hand is visible so stale results don't linger
            pred_buffer.clear()
            current_sign = ""
            current_conf = 0.0

        frame_count += 1

        # ── Display prediction ───────────────────────────────────────────
        if current_sign:
            # Colour by confidence
            if current_conf >= 0.6:
                box_color = (0, 255, 0)    # green
            elif current_conf >= 0.35:
                box_color = (0, 255, 255)  # yellow
            else:
                box_color = (0, 165, 255)  # orange

            cv2.rectangle(frame, (10, 10), (450, 110), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (450, 110), box_color, 2)

            cv2.putText(frame, f"Sign: {current_sign.upper()}", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, box_color, 3)
            cv2.putText(frame, f"Confidence: {current_conf*100:.1f}%", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
            cv2.putText(frame, "Show your hand", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
