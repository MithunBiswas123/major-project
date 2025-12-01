"""
Collect REAL hand sign data from your webcam
This creates actual training data from YOUR hand gestures
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.insert(0, '.')
from src.data_collection import HandDetector

# Signs to collect - start with fewer, easier signs
SIGNS_TO_COLLECT = [
    'A', 'B', 'C', 'D', 'E',  # Letters
    '1', '2', '3', '4', '5',  # Numbers
    'hello', 'yes', 'no', 'ok', 'peace'  # Common gestures
]

SAMPLES_PER_SIGN = 50  # Collect 50 samples of each sign

def collect_data():
    print("=" * 60)
    print("🎥 REAL DATA COLLECTION")
    print("=" * 60)
    print(f"Signs to collect: {len(SIGNS_TO_COLLECT)}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print("=" * 60)
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    all_data = []
    
    for sign_idx, sign in enumerate(SIGNS_TO_COLLECT):
        print(f"\n📌 Sign {sign_idx + 1}/{len(SIGNS_TO_COLLECT)}: '{sign}'")
        print("   Press SPACE when ready to start collecting")
        print("   Hold your hand sign steady during collection")
        
        # Wait for user to be ready
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            features, results, hand_detected = detector.extract_landmarks(frame)
            frame = detector.draw_landmarks(frame, results)
            
            # Show instructions
            cv2.rectangle(frame, (0, 0), (640, 100), (50, 50, 50), -1)
            cv2.putText(frame, f"NEXT SIGN: {sign}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, "Press SPACE when ready, Q to quit", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Hand status
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status = "HAND OK" if hand_detected else "NO HAND"
            cv2.putText(frame, status, (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                waiting = False
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.release()
                print("❌ Collection cancelled")
                return
        
        # Collect samples
        collected = 0
        print(f"   Collecting... hold the '{sign}' sign!")
        
        while collected < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            features, results, hand_detected = detector.extract_landmarks(frame)
            frame = detector.draw_landmarks(frame, results)
            
            # Progress bar
            progress = collected / SAMPLES_PER_SIGN
            bar_width = int(progress * 400)
            cv2.rectangle(frame, (0, 0), (640, 80), (50, 50, 50), -1)
            cv2.putText(frame, f"Collecting '{sign}': {collected}/{SAMPLES_PER_SIGN}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (20, 50), (420, 70), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 50), (20 + bar_width, 70), (0, 255, 0), -1)
            
            if hand_detected and features is not None:
                # Save this sample
                sample = {'sign': sign}
                for i, val in enumerate(features):
                    sample[f'feature_{i}'] = val
                all_data.append(sample)
                collected += 1
                
                # Visual feedback
                cv2.circle(frame, (550, 60), 20, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (550, 60), 20, (0, 0, 255), -1)
                cv2.putText(frame, "Show hand!", (480, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(30) & 0xFF  # Small delay between samples
            if key == ord('q'):
                break
        
        print(f"   ✅ Collected {collected} samples for '{sign}'")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    # Save data
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'real_sign_dataset.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Data saved to: {output_path}")
        print(f"   Total samples: {len(df)}")
        print(f"   Signs collected: {df['sign'].nunique()}")
    else:
        print("❌ No data collected")

if __name__ == "__main__":
    collect_data()
