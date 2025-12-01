"""
Real-Time Sign Language Detection
Directly opens camera for sign recognition
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    LSTM_MODEL_PATH, HYBRID_MODEL_PATH, CNN_MODEL_PATH, TRANSFORMER_MODEL_PATH
)
from src.detect import SignLanguageDetector


def main():
    """Start real-time sign language detection"""
    print("\n" + "=" * 60)
    print("🤟 REAL-TIME SIGN LANGUAGE DETECTION")
    print("=" * 60)
    
    # Find the best available model (priority: LSTM > Hybrid > CNN > Transformer)
    models = [
        ('LSTM', LSTM_MODEL_PATH),
        ('Hybrid', HYBRID_MODEL_PATH),
        ('CNN', CNN_MODEL_PATH),
        ('Transformer', TRANSFORMER_MODEL_PATH)
    ]
    
    model_path = None
    model_name = None
    
    for name, path in models:
        if os.path.exists(path):
            model_path = path
            model_name = name
            break
    
    if model_path is None:
        print("\n❌ No trained model found!")
        print("Please train a model first by running:")
        print("  python -c \"from src.train import SignLanguageTrainer; t = SignLanguageTrainer('lstm'); t.run_full_pipeline()\"")
        return
    
    print(f"\n✅ Using {model_name} model")
    print("\n📹 Starting camera...")
    print("\nControls:")
    print("  Q - Quit")
    print("  R - Reset prediction buffer")
    print("-" * 60)
    
    # Start detection
    detector = SignLanguageDetector(model_path)
    detector.run()
    detector.release()
    
    print("\n👋 Detection ended. Goodbye!")


if __name__ == "__main__":
    main()
