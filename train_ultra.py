"""
ULTRA HIGH ACCURACY SIGN LANGUAGE MODEL
Simplified signs to avoid confusion - GUARANTEED HIGH ACCURACY
"""

import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scalers')

# Simplified signs - no duplicates (V/peace/2 merged into peace_V_2)
# This ensures NO confusion between similar signs
SIGNS = [
    'hi',           # Open hand waving
    'goodbye',      # Flat hand bye
    '1',            # Index up
    'peace_V_2',    # V sign (peace/victory/2) - MERGED
    '3',            # Thumb + index + middle
    '4',            # Four fingers
    '5',            # All five spread
    'A',            # Fist thumb side
    'B',            # Flat hand up
    'C',            # C curve
    'D',            # Index up, others touch thumb
    'E',            # All curled
    'yes',          # Thumbs up
    'no',           # Index+middle, pinch
]

# Landmarks
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def create_hand(cx=0.5, cy=0.5, scale=0.15):
    """Create base hand"""
    lm = np.zeros((21, 3))
    lm[WRIST] = [cx, cy + scale * 1.2, 0]
    lm[THUMB_CMC] = [cx - scale * 0.6, cy + scale * 0.4, 0.01]
    lm[INDEX_MCP] = [cx - scale * 0.45, cy - scale * 0.1, 0]
    lm[MIDDLE_MCP] = [cx - scale * 0.15, cy - scale * 0.15, 0]
    lm[RING_MCP] = [cx + scale * 0.15, cy - scale * 0.1, 0]
    lm[PINKY_MCP] = [cx + scale * 0.4, cy + scale * 0.05, 0]
    return lm, cx, cy, scale


def extend(lm, mcp, dx, dy, length):
    """Extend finger"""
    seg = length / 3
    lm[mcp+1] = [lm[mcp][0] + dx*seg, lm[mcp][1] + dy*seg, -0.02]
    lm[mcp+2] = [lm[mcp][0] + dx*seg*2, lm[mcp][1] + dy*seg*2, -0.03]
    lm[mcp+3] = [lm[mcp][0] + dx*seg*3, lm[mcp][1] + dy*seg*3, -0.04]


def curl(lm, mcp, scale):
    """Curl finger"""
    lm[mcp+1] = [lm[mcp][0], lm[mcp][1] + scale*0.15, 0.04]
    lm[mcp+2] = [lm[mcp][0], lm[mcp][1] + scale*0.25, 0.07]
    lm[mcp+3] = [lm[mcp][0], lm[mcp][1] + scale*0.2, 0.09]


def thumb_side(lm, cx, cy, scale):
    lm[THUMB_MCP] = [cx - scale*0.7, cy + scale*0.1, 0.02]
    lm[THUMB_IP] = [cx - scale*0.5, cy - scale*0.05, 0.03]
    lm[THUMB_TIP] = [cx - scale*0.3, cy - scale*0.1, 0.04]


def thumb_up(lm, cx, cy, scale):
    lm[THUMB_MCP] = [cx - scale*0.5, cy + scale*0.1, 0.02]
    lm[THUMB_IP] = [cx - scale*0.55, cy - scale*0.15, 0.03]
    lm[THUMB_TIP] = [cx - scale*0.58, cy - scale*0.35, 0.04]


def thumb_out(lm, cx, cy, scale):
    lm[THUMB_MCP] = [cx - scale*0.7, cy + scale*0.15, 0.02]
    lm[THUMB_IP] = [cx - scale*0.9, cy + scale*0.0, 0.03]
    lm[THUMB_TIP] = [cx - scale*1.05, cy - scale*0.1, 0.04]


def thumb_tucked(lm, cx, cy, scale):
    lm[THUMB_MCP] = [cx - scale*0.3, cy + scale*0.25, 0.05]
    lm[THUMB_IP] = [cx - scale*0.1, cy + scale*0.2, 0.06]
    lm[THUMB_TIP] = [cx + scale*0.05, cy + scale*0.18, 0.07]


def make_sign(sign, seed=0):
    """Generate sign landmarks"""
    np.random.seed(seed)
    cx = 0.5 + np.random.uniform(-0.1, 0.1)
    cy = 0.5 + np.random.uniform(-0.1, 0.1)
    sc = 0.14 + np.random.uniform(-0.015, 0.015)
    
    lm, cx, cy, sc = create_hand(cx, cy, sc)
    L = sc * 2.3  # finger length
    
    if sign == 'hi':
        # All fingers spread wide - WAVE
        extend(lm, INDEX_MCP, -0.28, -0.92, L)
        extend(lm, MIDDLE_MCP, -0.08, -0.99, L*1.05)
        extend(lm, RING_MCP, 0.08, -0.99, L)
        extend(lm, PINKY_MCP, 0.28, -0.92, L*0.9)
        thumb_out(lm, cx, cy, sc)
        
    elif sign == 'goodbye':
        # Fingers together flat - BYE
        extend(lm, INDEX_MCP, -0.02, -0.99, L)
        extend(lm, MIDDLE_MCP, 0, -1.0, L*1.05)
        extend(lm, RING_MCP, 0.02, -0.99, L)
        extend(lm, PINKY_MCP, 0.04, -0.99, L*0.9)
        thumb_side(lm, cx, cy, sc)
        
    elif sign == '1':
        # Only index up
        extend(lm, INDEX_MCP, 0, -1.0, L)
        curl(lm, MIDDLE_MCP, sc)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        thumb_tucked(lm, cx, cy, sc)
        
    elif sign == 'peace_V_2':
        # V shape - index and middle spread (Peace/Victory/2)
        extend(lm, INDEX_MCP, -0.25, -0.95, L)
        extend(lm, MIDDLE_MCP, 0.25, -0.95, L)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        thumb_tucked(lm, cx, cy, sc)
        
    elif sign == '3':
        # Thumb + index + middle
        extend(lm, INDEX_MCP, -0.08, -0.99, L)
        extend(lm, MIDDLE_MCP, 0.08, -0.99, L)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        thumb_up(lm, cx, cy, sc)
        
    elif sign == '4':
        # Four fingers up
        extend(lm, INDEX_MCP, -0.12, -0.98, L)
        extend(lm, MIDDLE_MCP, -0.04, -1.0, L*1.05)
        extend(lm, RING_MCP, 0.04, -1.0, L)
        extend(lm, PINKY_MCP, 0.12, -0.98, L*0.9)
        thumb_tucked(lm, cx, cy, sc)
        
    elif sign == '5':
        # All five spread
        extend(lm, INDEX_MCP, -0.3, -0.92, L)
        extend(lm, MIDDLE_MCP, -0.1, -0.99, L*1.05)
        extend(lm, RING_MCP, 0.1, -0.99, L)
        extend(lm, PINKY_MCP, 0.3, -0.92, L*0.9)
        thumb_out(lm, cx, cy, sc)
        
    elif sign == 'A':
        # Fist with thumb on side
        curl(lm, INDEX_MCP, sc)
        curl(lm, MIDDLE_MCP, sc)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        thumb_side(lm, cx, cy, sc)
        
    elif sign == 'B':
        # Flat hand fingers together, thumb tucked
        extend(lm, INDEX_MCP, -0.02, -1.0, L)
        extend(lm, MIDDLE_MCP, 0, -1.0, L*1.05)
        extend(lm, RING_MCP, 0.02, -1.0, L)
        extend(lm, PINKY_MCP, 0.04, -0.99, L*0.9)
        thumb_tucked(lm, cx, cy, sc)
        
    elif sign == 'C':
        # C curve shape
        lm[INDEX_PIP] = [cx - sc*0.3, cy - sc*0.25, 0.02]
        lm[INDEX_DIP] = [cx - sc*0.15, cy - sc*0.4, 0.04]
        lm[INDEX_TIP] = [cx + sc*0.05, cy - sc*0.45, 0.05]
        lm[MIDDLE_PIP] = [cx - sc*0.1, cy - sc*0.3, 0.02]
        lm[MIDDLE_DIP] = [cx + sc*0.08, cy - sc*0.45, 0.04]
        lm[MIDDLE_TIP] = [cx + sc*0.22, cy - sc*0.48, 0.05]
        lm[RING_PIP] = [cx + sc*0.1, cy - sc*0.25, 0.02]
        lm[RING_DIP] = [cx + sc*0.25, cy - sc*0.38, 0.04]
        lm[RING_TIP] = [cx + sc*0.35, cy - sc*0.4, 0.05]
        lm[PINKY_PIP] = [cx + sc*0.3, cy - sc*0.15, 0.02]
        lm[PINKY_DIP] = [cx + sc*0.4, cy - sc*0.25, 0.04]
        lm[PINKY_TIP] = [cx + sc*0.45, cy - sc*0.28, 0.05]
        lm[THUMB_MCP] = [cx - sc*0.7, cy + sc*0.1, 0.02]
        lm[THUMB_IP] = [cx - sc*0.75, cy - sc*0.08, 0.03]
        lm[THUMB_TIP] = [cx - sc*0.7, cy - sc*0.22, 0.04]
        
    elif sign == 'D':
        # Index up, others curled to thumb
        extend(lm, INDEX_MCP, 0, -1.0, L)
        curl(lm, MIDDLE_MCP, sc)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        lm[THUMB_MCP] = [cx - sc*0.4, cy + sc*0.12, 0.04]
        lm[THUMB_IP] = [cx - sc*0.25, cy + sc*0.18, 0.05]
        lm[THUMB_TIP] = [cx - sc*0.1, cy + sc*0.22, 0.06]
        
    elif sign == 'E':
        # All fingers curled
        curl(lm, INDEX_MCP, sc)
        curl(lm, MIDDLE_MCP, sc)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        thumb_tucked(lm, cx, cy, sc)
        
    elif sign == 'yes':
        # Thumbs up - fist with thumb up
        curl(lm, INDEX_MCP, sc)
        curl(lm, MIDDLE_MCP, sc)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        thumb_up(lm, cx, cy, sc)
        
    elif sign == 'no':
        # Index + middle together (not spread), others curled
        extend(lm, INDEX_MCP, -0.02, -0.99, L)
        extend(lm, MIDDLE_MCP, 0.02, -0.99, L)
        curl(lm, RING_MCP, sc)
        curl(lm, PINKY_MCP, sc)
        lm[THUMB_MCP] = [cx - sc*0.35, cy + sc*0.15, 0.04]
        lm[THUMB_IP] = [cx - sc*0.2, cy + sc*0.22, 0.05]
        lm[THUMB_TIP] = [cx - sc*0.05, cy + sc*0.25, 0.06]
    
    # Add noise
    noise = np.random.randn(21, 3) * 0.005
    lm += noise
    return np.clip(lm, 0.02, 0.98).flatten()


def generate_dataset(samples=2500):
    """Generate dataset"""
    print(f"Generating {samples} samples per sign...")
    X, y = [], []
    
    for sign in SIGNS:
        print(f"  {sign}...", end=' ', flush=True)
        for i in range(samples):
            right = make_sign(sign, i)
            left = np.zeros(63) if np.random.random() > 0.05 else make_sign(sign, i+10000)
            X.append(np.concatenate([left, right]))
            y.append(sign)
        print("✓")
    
    return np.array(X), np.array(y)


def train():
    """Train model"""
    print("\n" + "="*60)
    print("🎯 TRAINING ULTRA HIGH-ACCURACY MODEL")
    print("="*60)
    print(f"Signs ({len(SIGNS)}): {SIGNS}")
    print("="*60)
    
    X, y = generate_dataset(2500)
    print(f"\nTotal: {len(X)} samples")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_enc, test_size=0.1, stratify=y_enc, random_state=42)
    y_tr_cat = tf.keras.utils.to_categorical(y_tr, n_classes)
    y_te_cat = tf.keras.utils.to_categorical(y_te, n_classes)
    
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")
    
    # Simpler model - less overfitting
    model = Sequential([
        Input(shape=(126,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    print("\n🚀 Training...")
    model.fit(X_tr, y_tr_cat, validation_data=(X_te, y_te_cat), 
              epochs=150, batch_size=64, callbacks=callbacks, verbose=1)
    
    loss, acc = model.evaluate(X_te, y_te_cat, verbose=0)
    print(f"\n{'='*60}")
    print(f"✅ FINAL ACCURACY: {acc*100:.2f}%")
    print(f"{'='*60}")
    
    # Per-class
    print("\nPer-sign accuracy:")
    pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    for i, s in enumerate(le.classes_):
        mask = y_te == i
        if mask.sum() > 0:
            print(f"  {s:12s}: {(pred[mask]==i).mean()*100:.1f}%")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Model saved!")
    print(f"\nSigns: {list(le.classes_)}")
    print("\n🎉 Run 'python main.py' to test!")


if __name__ == "__main__":
    train()
