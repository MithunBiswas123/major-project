"""Test per-sample normalization to unify old and new data formats."""
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

df = pd.read_csv('data/raw/sign_dataset.csv', low_memory=False)
feat_cols = [c for c in df.columns if c.startswith('feature_')]

def normalize_hand_features(features_126):
    """Normalize each hand: subtract wrist, L2-normalize.
    This makes features invariant to position and scale."""
    result = features_126.copy()
    
    for start in [0, 63]:  # left hand, right hand
        hand = result[start:start+63].copy()
        
        if np.any(hand != 0):
            # Reshape to (21, 3)
            landmarks = hand.reshape(21, 3)
            
            # Subtract wrist (landmark 0)
            wrist = landmarks[0].copy()
            landmarks = landmarks - wrist
            
            # Compute L2 norm of the hand vector (excluding wrist which is now 0)
            norm = np.linalg.norm(landmarks[1:])  # exclude wrist (always 0 after subtraction)
            if norm > 1e-6:
                landmarks[1:] = landmarks[1:] / norm
            
            result[start:start+63] = landmarks.flatten()
    
    return result

print("=== BEFORE NORMALIZATION ===")
for sign in ['A', 'hello', '1', 'hi', 'is', 'name']:
    subset = df[df['sign'] == sign][feat_cols].values.astype(float)
    subset = np.nan_to_num(subset, nan=0.0)
    nz = subset[subset != 0]
    print(f"  {sign:10s}: range=[{nz.min():.4f}, {nz.max():.4f}], mean={nz.mean():.4f}, std={nz.std():.4f}")

print("\n=== AFTER PER-SAMPLE NORMALIZATION ===")
for sign in ['A', 'hello', '1', 'hi', 'is', 'name']:
    subset = df[df['sign'] == sign][feat_cols].values.astype(float)
    subset = np.nan_to_num(subset, nan=0.0)
    
    normalized = np.array([normalize_hand_features(row) for row in subset])
    nz = normalized[normalized != 0]
    print(f"  {sign:10s}: range=[{nz.min():.4f}, {nz.max():.4f}], mean={nz.mean():.4f}, std={nz.std():.4f}")

# Now test: train a quick model with normalized data and see accuracy
print("\n=== QUICK MODEL TEST WITH NORMALIZED DATA ===")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

X_all = df[feat_cols].values.astype(float)
X_all = np.nan_to_num(X_all, nan=0.0)
y_all = df['sign'].values

# Normalize all samples
X_norm = np.array([normalize_hand_features(row) for row in X_all])

le = LabelEncoder()
y_enc = le.fit_transform(y_all)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Simple dense model for quick test
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

model = Sequential([
    Input(shape=(126,)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_s, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=0)

loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"  Test accuracy with normalized features: {acc*100:.1f}%")

# Per-sign accuracy
y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
from collections import Counter
for sign_idx in sorted(set(y_test)):
    mask = y_test == sign_idx
    sign_acc = (y_pred[mask] == y_test[mask]).mean()
    sign_name = le.classes_[sign_idx]
    count = mask.sum()
    print(f"    {sign_name:10s}: {sign_acc*100:.0f}% ({count} test samples)")
