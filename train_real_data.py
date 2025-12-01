"""
Train model on REAL collected data
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'real_sign_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scalers')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def train():
    print("=" * 60)
    print("🎯 TRAINING ON REAL DATA")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print("❌ No real data found!")
        print("   Run 'python collect_real_data.py' first to collect data")
        return
    
    # Load data
    print("\n📊 Loading real dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Samples: {len(df)}")
    print(f"   Signs: {df['sign'].nunique()}")
    print(f"   Signs: {df['sign'].unique().tolist()}")
    
    # Prepare features and labels
    X = df.drop('sign', axis=1).values
    y = df['sign'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"   Classes: {num_classes}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Build simple Dense model (works better with real data)
    print("\n🏗️ Building model...")
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train
    print("\n🎯 Training...")
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save model (as Dense model, not LSTM)
    model.save(os.path.join(MODEL_DIR, 'real_model.h5'))
    print(f"   Model saved: real_model.h5")
    
    # Save scaler and encoder
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    print(f"   Scaler & encoder saved")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNow update main.py to use 'real_model.h5'")

if __name__ == "__main__":
    train()
