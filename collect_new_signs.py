"""
Direct webcam data collection for 14 conversation signs.
Just run: python collect_new_signs.py
Webcam opens immediately - no menus!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_collection import DataCollector, CONVERSATION_SIGNS, GESTURE_TIPS
from src.config import SIGNS, DATASET_CSV
import pandas as pd

# How many samples per sign
SAMPLES = 100

def main():
    # Show what signs already have data
    if os.path.exists(DATASET_CSV):
        df = pd.read_csv(DATASET_CSV)
        existing = set(df['sign'].unique())
        print(f"\nExisting data: {len(df)} samples for {len(existing)} signs")
        print(f"Signs: {sorted(existing)}")
        print("\nOLD DATA WILL NOT BE ERASED!")
    else:
        existing = set()
    
    print("\n" + "=" * 60)
    print("  COLLECTING 14 CONVERSATION SIGNS")
    print("=" * 60)
    print(f"  Samples per sign: {SAMPLES}")
    print(f"  Signs: {', '.join(CONVERSATION_SIGNS)}")
    print("=" * 60)
    
    collector = DataCollector(samples_per_sign=SAMPLES)
    
    for i, sign in enumerate(CONVERSATION_SIGNS):
        desc = SIGNS.get(sign, '')
        tip = GESTURE_TIPS.get(sign, '')
        has_data = sign in existing
        
        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/14] Sign: {sign.upper()}")
        print(f"  Description : {desc}")
        print(f"  How to do it: {tip}")
        if has_data:
            print(f"  ** Already has data - will ADD more, not replace **")
        print(f"{'=' * 60}")
        print(f"\n  Press ENTER to open webcam for '{sign}'...")
        print(f"  (or type 's' + ENTER to skip this sign)")
        
        choice = input("  > ").strip().lower()
        if choice == 's':
            print(f"  Skipped '{sign}'")
            continue
        
        print(f"\n  WEBCAM OPENING for '{sign}'...")
        print(f"  >> Press SPACE to start 3-second countdown")
        print(f"  >> Hold your '{sign}' hand sign steady")
        print(f"  >> Press Q to quit early\n")
        
        success = collector.collect_sign(sign, desc)
        
        if not success:
            retry = input("  Incomplete. Retry? (y/n): ").strip().lower()
            if retry == 'y':
                collector.collect_sign(sign, desc)
    
    # Save - this APPENDS to existing CSV
    collector.save_to_csv()
    collector.release()
    
    print("\n" + "=" * 60)
    print("  DONE! New data appended to existing dataset.")
    print("  Old sign data is safe and intact.")
    print("=" * 60)
    print("\n  Next step - retrain the model:")
    print("  python -m src.train")


if __name__ == "__main__":
    main()
