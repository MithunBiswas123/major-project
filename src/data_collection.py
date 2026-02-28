"""
Data Collection Module for Sign Language Detection
Collects hand landmark data from webcam using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

from .config import (
    SIGNS, SIGN_LABELS, SIGN_CATEGORIES, NUM_SIGNS,
    SAMPLES_PER_SIGN, CAPTURE_INTERVAL, COUNTDOWN_SECONDS,
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    NUM_LANDMARKS, FEATURES_PER_HAND,
    RAW_DATA_DIR, DATASET_CSV, DISPLAY_COLORS,
    CAMERA_WIDTH, CAMERA_HEIGHT
)


class HandDetector:
    """MediaPipe hand detection wrapper"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF
        )
    
    def detect(self, frame):
        """Detect hands in frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results
    
    def extract_landmarks(self, frame):
        """Extract landmarks from both hands"""
        results = self.detect(frame)
        
        left_hand = np.zeros(FEATURES_PER_HAND)
        right_hand = np.zeros(FEATURES_PER_HAND)
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            
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
        
        features = np.concatenate([left_hand, right_hand])
        return features, results, hand_detected
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        return frame
    
    def release(self):
        """Release resources"""
        self.hands.close()


class DataCollector:
    """Collect training data for sign language detection"""
    
    def __init__(self, samples_per_sign=SAMPLES_PER_SIGN):
        self.samples_per_sign = samples_per_sign
        self.detector = HandDetector()
        self.data = []
        
        # Ensure directories exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    def collect_sign(self, sign_label, description):
        """Collect samples for a single sign"""
        
        # Open camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        samples_collected = 0
        collecting = False
        countdown_start = None
        last_capture = 0
        
        print(f"\n📹 Collecting: {sign_label}")
        print(f"   Description: {description}")
        print("   Press SPACE to start, Q to quit")
        
        while samples_collected < self.samples_per_sign:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            features, results, hand_detected = self.detector.extract_landmarks(frame)
            
            # Draw landmarks
            display = self.detector.draw_landmarks(frame.copy(), results)
            
            # UI Header
            cv2.rectangle(display, (0, 0), (CAMERA_WIDTH, 130), 
                         DISPLAY_COLORS['background'], -1)
            
            # Sign info
            cv2.putText(display, f"Sign: {sign_label}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, DISPLAY_COLORS['primary'], 3)
            cv2.putText(display, f"Samples: {samples_collected}/{self.samples_per_sign}",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DISPLAY_COLORS['text'], 2)
            
            # Progress bar
            progress = int((samples_collected / self.samples_per_sign) * 400)
            cv2.rectangle(display, (20, 100), (420, 120), (100, 100, 100), -1)
            cv2.rectangle(display, (20, 100), (20 + progress, 120), 
                         DISPLAY_COLORS['success'], -1)
            
            # Hand status
            status = "Hand Detected ✓" if hand_detected else "Show Hand!"
            color = DISPLAY_COLORS['success'] if hand_detected else DISPLAY_COLORS['error']
            cv2.putText(display, status, (500, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Countdown
            if countdown_start is not None:
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECONDS - int(elapsed)
                
                if remaining > 0:
                    cv2.putText(display, str(remaining), (600, 350),
                               cv2.FONT_HERSHEY_SIMPLEX, 5, DISPLAY_COLORS['primary'], 10)
                else:
                    collecting = True
                    countdown_start = None
            
            # Collect samples
            if collecting and hand_detected:
                current_time = time.time()
                if current_time - last_capture >= CAPTURE_INTERVAL:
                    # Save sample
                    self.data.append({
                        'sign': sign_label,
                        'features': features.tolist(),
                        'timestamp': datetime.now().isoformat()
                    })
                    samples_collected += 1
                    last_capture = current_time
                    
                    # Visual feedback
                    cv2.rectangle(display, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT),
                                 DISPLAY_COLORS['success'], 15)
            
            # Instructions
            if not collecting and countdown_start is None:
                cv2.putText(display, "Press SPACE to start", (500, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, DISPLAY_COLORS['text'], 1)
            elif collecting:
                cv2.putText(display, "Collecting...", (500, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, DISPLAY_COLORS['success'], 1)
            
            cv2.imshow('Data Collection', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not collecting and countdown_start is None:
                countdown_start = time.time()
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"   ✅ Collected {samples_collected} samples")
        return samples_collected >= self.samples_per_sign
    
    def collect_multiple_signs(self, signs_to_collect=None):
        """Collect data for multiple signs"""
        
        if signs_to_collect is None:
            signs_to_collect = SIGN_LABELS
        
        print("\n" + "=" * 60)
        print("🎬 SIGN LANGUAGE DATA COLLECTION")
        print("=" * 60)
        print(f"Signs to collect: {len(signs_to_collect)}")
        print(f"Samples per sign: {self.samples_per_sign}")
        print(f"Total samples: {len(signs_to_collect) * self.samples_per_sign}")
        print("=" * 60)
        
        for i, sign in enumerate(signs_to_collect):
            desc = SIGNS.get(sign, 'Custom sign')
            print(f"\n[{i+1}/{len(signs_to_collect)}] Next: {sign}")
            
            input(f"Press ENTER when ready to collect '{sign}'...")
            
            success = self.collect_sign(sign, desc)
            
            if not success:
                cont = input(f"Incomplete. Continue? (y/n): ")
                if cont.lower() != 'y':
                    break
        
        # Save to CSV
        self.save_to_csv()
        
        return self.data
    
    def collect_by_category(self, category):
        """Collect signs from a specific category"""
        signs = SIGN_CATEGORIES.get(category, [])
        if not signs:
            print(f"❌ Category '{category}' not found")
            return None
        
        print(f"\n📁 Collecting {category} signs ({len(signs)} signs)")
        return self.collect_multiple_signs(signs)
    
    def save_to_csv(self):
        """Save collected data to CSV"""
        if not self.data:
            print("❌ No data to save")
            return
        
        # Prepare DataFrame
        rows = []
        for item in self.data:
            row = {'sign': item['sign'], 'timestamp': item['timestamp']}
            for i, val in enumerate(item['features']):
                row[f'feature_{i}'] = val
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Append or create
        if os.path.exists(DATASET_CSV):
            existing = pd.read_csv(DATASET_CSV)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(DATASET_CSV, index=False)
        
        print(f"\n💾 Saved {len(self.data)} samples to {DATASET_CSV}")
        print(f"📊 Total dataset size: {len(df)} samples")
        
        # Print distribution
        print("\n📊 Sign distribution:")
        print(df['sign'].value_counts())
    
    def release(self):
        """Release resources"""
        self.detector.release()


def quick_collect(signs, samples=30):
    """Quick collection utility"""
    collector = DataCollector(samples_per_sign=samples)
    collector.collect_multiple_signs(signs)
    collector.release()
    return collector.data


# ============================================================================
# CONVERSATION SIGN COLLECTION
# ============================================================================

# All 14 conversation signs you want to collect
CONVERSATION_SIGNS = [
    'hi', 'what', 'is', 'your', 'name',
    'my', 'to', 'who', 'where', 'why',
    'when', 'which', 'me', 'how'
]

# How-to tips for each sign (shown during collection)
GESTURE_TIPS = {
    'hi':    'Wave your open hand left-right near your shoulder',
    'what':  'Open both palms up, wiggle fingers downward (shrug)',
    'is':    'Hold pinky up (I-hand), move it slightly forward from chin',
    'your':  'Push flat open palm forward toward the camera/person',
    'name':  'Both hands in H-shape, tap middle fingers on index fingers twice',
    'my':    'Place your flat dominant hand on your chest',
    'to':    'Point both index fingers at each other, tap tips together',
    'who':   'Circle your index finger around your chin/mouth area',
    'where': 'Hold up index finger and waggle it side to side',
    'why':   'Touch forehead with fingers, move hand outward and down',
    'when':  'Circle dominant index finger around non-dominant index tip',
    'which': 'Both thumbs up, alternate hands up and down',
    'me':    'Point your index finger toward your own chest',
    'how':   'Both fists knuckles together, then roll/open forward',
}


def get_already_collected_signs():
    """Check which signs already have data in the CSV (so we don't erase them)"""
    if not os.path.exists(DATASET_CSV):
        return set()
    df = pd.read_csv(DATASET_CSV)
    return set(df['sign'].unique())


def collect_conversation_signs(samples=100):
    """
    Collect the 14 conversation signs.
    - Existing CSV data is NEVER deleted (append-only).
    - Signs that already have enough data are shown but can be skipped.
    """
    already = get_already_collected_signs()

    print("\n" + "=" * 60)
    print("  CONVERSATION SIGN LANGUAGE COLLECTION")
    print("=" * 60)
    print(f"  Signs to collect : {', '.join(CONVERSATION_SIGNS)}")
    print(f"  Samples per sign : {samples}")
    print(f"  Total new samples: {len(CONVERSATION_SIGNS) * samples}")
    print("=" * 60)

    if already:
        print(f"\n  Signs already in dataset: {', '.join(sorted(already))}")
        print("  (Old data will NOT be erased. New samples are appended.)")

    collector = DataCollector(samples_per_sign=samples)

    for i, sign in enumerate(CONVERSATION_SIGNS):
        desc = SIGNS.get(sign, '')
        tip  = GESTURE_TIPS.get(sign, '')
        has_data = sign in already

        print(f"\n[{i+1}/{len(CONVERSATION_SIGNS)}] Sign: {sign.upper()}")
        print(f"   Description : {desc}")
        print(f"   How to do it: {tip}")
        if has_data:
            print(f"   ** Already has data in CSV (will ADD more, not replace) **")
        print("-" * 60)

        choice = input(f"   Press ENTER to collect '{sign}', or 's' to skip: ").strip().lower()
        if choice == 's':
            print(f"   Skipped {sign}")
            continue

        success = collector.collect_sign(sign, desc)

        if not success:
            retry = input("   Incomplete. Retry? (y/n): ").strip().lower()
            if retry == 'y':
                collector.collect_sign(sign, desc)

    # save_to_csv APPENDS to existing CSV — old data stays safe
    collector.save_to_csv()
    collector.release()

    print("\n" + "=" * 60)
    print("  ALL DONE! New data appended to existing dataset.")
    print("  Old sign data is intact.")
    print("=" * 60)
    print("\n  Next step: retrain the model with:")
    print("    python -m src.train")

    return collector.data


if __name__ == "__main__":
    print("\nChoose what to collect:")
    print("  1) Conversation signs (hi, what, is, your, name, ...)")
    print("  2) Collect by category")
    print("  3) Collect ALL signs")
    print("  4) Quick test (single sign)")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    if choice == '1':
        n = input("Samples per sign (default 100): ").strip()
        n = int(n) if n.isdigit() else 100
        collect_conversation_signs(samples=n)
    elif choice == '2':
        print(f"Categories: {list(SIGN_CATEGORIES.keys())}")
        cat = input("Enter category name: ").strip()
        collector = DataCollector(samples_per_sign=100)
        collector.collect_by_category(cat)
        collector.release()
    elif choice == '3':
        collector = DataCollector(samples_per_sign=100)
        collector.collect_multiple_signs()
        collector.release()
    else:
        sign = input("Enter sign label: ").strip()
        desc = SIGNS.get(sign, 'Custom sign')
        collector = DataCollector(samples_per_sign=30)
        collector.collect_sign(sign, desc)
        collector.save_to_csv()
        collector.release()
