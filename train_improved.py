"""
Improved LSTM Training for Higher Accuracy
Features:
- More LSTM layers
- Better regularization
- Learning rate scheduling
- Data augmentation
- Early stopping with patience
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'sign_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scalers')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

print("=" * 60)
print("🚀 IMPROVED LSTM TRAINING FOR HIGH ACCURACY")
print("=" * 60)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Samples: {len(df)}")
print(f"   Signs: {df['sign'].nunique()}")

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

# Data augmentation function
def augment_data(X, y, augment_factor=3):
    """Augment data with noise and slight variations"""
    augmented_X = [X]
    augmented_y = [y]
    
    for i in range(augment_factor):
        # Add Gaussian noise
        noise_level = 0.02 + (i * 0.01)  # Increasing noise
        noisy_X = X + np.random.normal(0, noise_level, X.shape)
        augmented_X.append(noisy_X)
        augmented_y.append(y)
        
        # Add slight scaling variation
        scale = np.random.uniform(0.95, 1.05, X.shape)
        scaled_X = X * scale
        augmented_X.append(scaled_X)
        augmented_y.append(y)
    
    return np.vstack(augmented_X), np.hstack(augmented_y)

# Augment training data
print("\n🔄 Augmenting data...")
X_augmented, y_augmented = augment_data(X_scaled, y_encoded, augment_factor=2)
print(f"   Augmented samples: {len(X_augmented)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Reshape for LSTM [samples, timesteps, features]
# Use 6 timesteps (simulating sequence of hand positions)
n_timesteps = 6
n_features = X_train.shape[1] // n_timesteps * n_timesteps // n_timesteps

# Pad or reshape
if X_train.shape[1] % n_timesteps != 0:
    # Pad to make divisible
    pad_size = n_timesteps - (X_train.shape[1] % n_timesteps)
    X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
    X_test = np.pad(X_test, ((0, 0), (0, pad_size)), mode='constant')

n_features = X_train.shape[1] // n_timesteps
X_train = X_train.reshape(-1, n_timesteps, n_features)
X_test = X_test.reshape(-1, n_timesteps, n_features)

print(f"\n   Input shape: {X_train.shape}")

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Build improved model
print("\n🏗️ Building IMPROVED model...")

model = Sequential([
    # First Bidirectional LSTM layer
    Bidirectional(LSTM(256, return_sequences=True, 
                       kernel_regularizer=l2(0.001)),
                  input_shape=(n_timesteps, n_features)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second Bidirectional LSTM layer
    Bidirectional(LSTM(128, return_sequences=True,
                       kernel_regularizer=l2(0.001))),
    BatchNormalization(),
    Dropout(0.3),
    
    # Third LSTM layer
    LSTM(64, return_sequences=False,
         kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Dense layers
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Output layer
    Dense(num_classes, activation='softmax')
])

# Compile with Adam optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_lstm_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n🎯 Training model...")
print("   This may take a few minutes...\n")

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("📊 EVALUATION RESULTS")
print("=" * 60)

loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n   ✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"   📉 Test Loss: {loss:.4f}")

# Save final model
model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
print(f"\n   💾 Model saved: lstm_model.h5")

# Save scaler and encoder
joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
print(f"   💾 Scaler saved: scaler.pkl")
print(f"   💾 Label encoder saved: label_encoder.pkl")

# Save training history
best_val_acc = max(history.history['val_accuracy'])
print(f"\n   🏆 Best Validation Accuracy: {best_val_acc * 100:.2f}%")

# Per-class accuracy
print("\n📈 Per-class Performance (sample):")
predictions = model.predict(X_test, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

from collections import Counter
correct_per_class = Counter()
total_per_class = Counter()

for true, pred in zip(y_test, pred_classes):
    total_per_class[true] += 1
    if true == pred:
        correct_per_class[true] += 1

# Show some classes
print("\n   Sign       | Accuracy")
print("   " + "-" * 25)
for i in range(min(10, num_classes)):
    class_name = label_encoder.inverse_transform([i])[0]
    acc = correct_per_class[i] / total_per_class[i] * 100 if total_per_class[i] > 0 else 0
    print(f"   {class_name:<10} | {acc:.1f}%")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\n🎉 Run 'python main.py' to test detection!")
