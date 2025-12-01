"""
Generate Synthetic Dataset for Sign Language Detection
Creates sample CSV data for testing and demonstration
Run this to create a larger training dataset
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SIGN_LABELS, DATASET_CSV, TOTAL_FEATURES

# Configuration - use signs from config
SIGNS_TO_GENERATE = SIGN_LABELS[:50]  # Use first 50 signs for training data
SAMPLES_PER_SIGN = 50
OUTPUT_PATH = DATASET_CSV


def generate_sign_pattern(sign):
    """Generate unique base pattern for each sign"""
    # Use sign name to create reproducible unique pattern
    np.random.seed(hash(sign) % (2**32))
    base = np.random.randn(TOTAL_FEATURES) * 0.3
    
    # Add structure based on hand shape
    # First 63 features = left hand, next 63 = right hand
    base[:63] += np.random.uniform(0.3, 0.7, 63)  # Left hand position
    
    return base


def generate_sample(base_pattern, variation=0.05):
    """Generate single sample with variation from base"""
    noise = np.random.randn(TOTAL_FEATURES) * variation
    sample = base_pattern + noise
    
    # Keep values in reasonable range
    sample = np.clip(sample, -1, 1)
    
    return sample


def create_dataset():
    """Create full synthetic dataset"""
    print("=" * 50)
    print("🔄 GENERATING SYNTHETIC DATASET")
    print("=" * 50)
    
    data = []
    
    for sign in SIGNS_TO_GENERATE:
        print(f"Generating {SAMPLES_PER_SIGN} samples for: {sign}")
        
        base_pattern = generate_sign_pattern(sign)
        
        for i in range(SAMPLES_PER_SIGN):
            sample = generate_sample(base_pattern)
            
            row = {
                'sign': sign,
                'timestamp': datetime.now().isoformat()
            }
            
            for j, val in enumerate(sample):
                row[f'feature_{j}'] = round(val, 6)
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "=" * 50)
    print("✅ DATASET GENERATED")
    print("=" * 50)
    print(f"📁 Location: {OUTPUT_PATH}")
    print(f"📊 Total samples: {len(df)}")
    print(f"🤟 Total signs: {df['sign'].nunique()}")
    print(f"📈 Features: {TOTAL_FEATURES}")
    print("\n📋 Sample distribution:")
    print(df['sign'].value_counts())
    
    return df


if __name__ == "__main__":
    df = create_dataset()
