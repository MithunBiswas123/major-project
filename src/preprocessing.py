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
        """Complete preprocessing pipeline"""
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
        
        # Encode labels
        y_encoded = self.encode_labels(y)
        
        # Augment data
        if augment:
            X, y_encoded = self.augment_data(X, y_encoded, augmentation_factor)
        
        # Normalize features
        X_scaled = self.normalize_features(X)
        
        # Split data
        data_splits = self.split_data(X_scaled, y_encoded)
        
        # Save preprocessors
        self.save_preprocessors()
        
        # Save processed data
        self.save_processed_data(X_scaled, y_encoded)
        
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return data_splits
    
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
