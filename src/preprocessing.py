"""
Data Preprocessing Module for Sign Language Detection
Handles data loading, augmentation, normalization, and splitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

from .config import (
    DATASET_CSV, PROCESSED_X, PROCESSED_Y, ENCODER_PATH, SCALER_PATH,
    PROCESSED_DATA_DIR, TOTAL_FEATURES, TEST_SPLIT, VALIDATION_SPLIT
)


def normalize_hand_features(features_126):
    """
    Per-sample hand normalization that unifies different data formats.
    
    For EACH detected hand:
      1. Subtract wrist position (landmark 0) from all landmarks
      2. L2-normalize x,y components together (2D hand shape)
      3. L2-normalize z components separately (depth pattern)
    
    Why separate xy/z normalization?
      Old data was collected with z having ~50% weight in the feature vector,
      while webcam/new data has z at only ~5%. A single L2-norm makes the
      same gesture look completely different depending on data source.
      Separate normalization ensures both formats produce identical features.
    
    This makes features invariant to:
      - Hand position in frame (translation invariance)
      - Distance from camera / hand size (scale invariance)
      - Different x,y vs z scale ratios (format invariance)
    """
    result = features_126.copy().astype(np.float64)
    
    for start in [0, 63]:  # left hand slot, right hand slot
        hand = result[start:start+63].copy()
        
        if np.any(hand != 0):  # hand detected
            landmarks = hand.reshape(21, 3)
            
            # Subtract wrist (landmark 0) — makes landmark 0 = [0,0,0]
            wrist = landmarks[0].copy()
            landmarks = landmarks - wrist
            
            non_wrist = landmarks[1:]  # 20 landmarks
            
            # L2-normalize x,y together (preserves 2D hand shape)
            xy = non_wrist[:, :2].flatten()  # 40 values
            xy_norm = np.linalg.norm(xy)
            if xy_norm > 1e-6:
                non_wrist[:, :2] = non_wrist[:, :2] / xy_norm
            
            # L2-normalize z separately (preserves depth pattern)
            z = non_wrist[:, 2]  # 20 values
            z_norm = np.linalg.norm(z)
            if z_norm > 1e-6:
                non_wrist[:, 2] = z / z_norm
            else:
                non_wrist[:, 2] = 0.0  # no depth info
            
            landmarks[1:] = non_wrist
            result[start:start+63] = landmarks.flatten()
    
    return result


def normalize_batch(X):
    """Apply normalize_hand_features to an entire batch."""
    return np.array([normalize_hand_features(row) for row in X])


class DataPreprocessor:
    """Preprocess data for training"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def load_dataset(self, csv_path=DATASET_CSV):
        """Load dataset from CSV"""
        if not os.path.exists(csv_path):
            print(f"❌ Dataset not found: {csv_path}")
            return None, None
        
        df = pd.read_csv(csv_path)
        
        print(f"✅ Loaded dataset: {len(df)} samples")
        print(f"📊 Signs: {df['sign'].nunique()}")
        print(f"📊 Features: {len(df.columns) - 2}")  # Excluding sign and timestamp
        
        return df
    
    def extract_features_labels(self, df):
        """Extract features and labels from DataFrame"""
        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        X = df[feature_cols].values
        y = df['sign'].values
        
        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Labels shape: {y.shape}")
        
        return X, y
    
    def encode_labels(self, y):
        """Encode string labels to integers"""
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Encoded {len(self.label_encoder.classes_)} classes")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        return y_encoded
    
    def normalize_features(self, X, fit=True):
        """Normalize features using StandardScaler"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        print(f"✅ Features normalized")
        print(f"   Range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
        
        return X_scaled
    
    def augment_data(self, X, y, augmentation_factor=2):
        """Augment data with noise and transformations"""
        print(f"🔄 Augmenting data (factor: {augmentation_factor})...")
        
        augmented_X = [X]
        augmented_y = [y]
        
        for i in range(augmentation_factor - 1):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.02, X.shape)
            X_noisy = X + noise
            
            # Add scaling variation
            scale = np.random.uniform(0.95, 1.05, X.shape)
            X_scaled = X * scale
            
            augmented_X.extend([X_noisy, X_scaled])
            augmented_y.extend([y, y])
        
        X_aug = np.vstack(augmented_X)
        y_aug = np.hstack(augmented_y)
        
        print(f"✅ Augmented: {len(X)} → {len(X_aug)} samples")
        
        return X_aug, y_aug
    
    def balance_classes(self, X, y):
        """Balance classes by upsampling minority classes to match majority"""
        from collections import Counter
        
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        if max_count == min_count:
            print("✅ Classes already balanced")
            return X, y
        
        print(f"⚖️  Balancing classes: min={min_count}, max={max_count}")
        
        balanced_X = []
        balanced_y = []
        
        for cls in np.unique(y):
            cls_mask = (y == cls)
            cls_X = X[cls_mask]
            cls_y = y[cls_mask]
            cls_count = len(cls_X)
            
            # Always include original samples
            balanced_X.append(cls_X)
            balanced_y.append(cls_y)
            
            # Upsample if needed (with small noise to avoid exact duplicates)
            if cls_count < max_count:
                deficit = max_count - cls_count
                indices = np.random.choice(cls_count, size=deficit, replace=True)
                extra_X = cls_X[indices] + np.random.normal(0, 0.01, (deficit, cls_X.shape[1]))
                extra_y = np.full(deficit, cls)
                balanced_X.append(extra_X)
                balanced_y.append(extra_y)
        
        X_bal = np.vstack(balanced_X)
        y_bal = np.hstack(balanced_y)
        
        # Shuffle
        perm = np.random.permutation(len(X_bal))
        X_bal = X_bal[perm]
        y_bal = y_bal[perm]
        
        print(f"✅ Balanced: {len(X)} → {len(X_bal)} samples ({len(np.unique(y_bal))} classes, {max_count} each)")
        
        return X_bal, y_bal
    
    def split_data(self, X, y, test_size=TEST_SPLIT, val_size=VALIDATION_SPLIT):
        """Split data into train, validation, and test sets"""
        
        # First split: train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
        )
        
        print(f"✅ Data split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self):
        """Save label encoder and scaler"""
        os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
        
        joblib.dump(self.label_encoder, ENCODER_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        
        print(f"✅ Saved: {ENCODER_PATH}")
        print(f"✅ Saved: {SCALER_PATH}")
    
    def load_preprocessors(self):
        """Load label encoder and scaler"""
        if os.path.exists(ENCODER_PATH):
            self.label_encoder = joblib.load(ENCODER_PATH)
        
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            self.is_fitted = True
        
        print("✅ Loaded preprocessors")
    
    def save_processed_data(self, X, y):
        """Save processed numpy arrays"""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        np.save(PROCESSED_X, X)
        np.save(PROCESSED_Y, y)
        
        print(f"✅ Saved processed data")
    
    def load_processed_data(self):
        """Load processed numpy arrays"""
        if os.path.exists(PROCESSED_X) and os.path.exists(PROCESSED_Y):
            X = np.load(PROCESSED_X)
            y = np.load(PROCESSED_Y)
            return X, y
        return None, None
    
    def preprocess_pipeline(self, augment=True, augmentation_factor=2):
        """Complete preprocessing pipeline
        
        IMPORTANT: Split FIRST, then augment only the training set.
        This prevents data leakage (augmented copies of the same
        sample appearing in both train and test sets).
        """
        print("\n" + "=" * 60)
        print("⚙️  DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_dataset()
        if df is None:
            return None
        
        # Extract features and labels
        X, y = self.extract_features_labels(df)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # ── Hand normalization (unifies old + new data formats) ──
        print("🔧 Applying per-sample hand normalization...")
        X = normalize_batch(X)
        print("✅ Hand features normalized (wrist-relative, L2-scaled)")
        
        # Encode labels
        y_encoded = self.encode_labels(y)
        
        # ── Step 1: Balance classes BEFORE splitting ──
        X, y_encoded = self.balance_classes(X, y_encoded)
        
        # ── Step 2: Split FIRST (on clean, un-augmented data) ──
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y_encoded)
        
        # ── Step 3: Augment ONLY the training set ──
        if augment:
            X_train, y_train = self.augment_data(X_train, y_train, augmentation_factor)
        
        # ── Step 4: Normalize (fit on train, transform all) ──
        X_train = self.normalize_features(X_train, fit=True)
        X_val   = self.normalize_features(X_val,   fit=False)
        X_test  = self.normalize_features(X_test,  fit=False)
        
        # Save preprocessors
        self.save_preprocessors()
        
        # Save processed data
        self.save_processed_data(X_train, y_train)
        
        print(f"\n📊 Final dataset sizes:")
        print(f"   Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_num_classes(self):
        """Get number of classes"""
        return len(self.label_encoder.classes_)
    
    def decode_labels(self, y_encoded):
        """Decode integer labels to strings"""
        return self.label_encoder.inverse_transform(y_encoded)
    
    def get_class_names(self):
        """Get list of class names"""
        return list(self.label_encoder.classes_)


def preprocess_data(augment=True):
    """Utility function to preprocess data"""
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_pipeline(augment=augment)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(augment=True)
    
    if data:
        X_train, X_val, X_test, y_train, y_val, y_test = data
        print(f"\nFinal shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
