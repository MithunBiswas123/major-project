"""
Utility Functions for Sign Language Detection
Helper functions for various tasks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle

from .config import (
    SIGNS, SIGN_LABELS, SIGN_CATEGORIES, NUM_SIGNS,
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, DATASET_CSV,
    PLOTS_DIR, LOGS_DIR
)


def ensure_directories():
    """Create all necessary directories"""
    dirs = [
        DATA_DIR, os.path.join(DATA_DIR, 'raw'), os.path.join(DATA_DIR, 'processed'),
        MODELS_DIR, os.path.join(MODELS_DIR, 'saved'),
        OUTPUTS_DIR, PLOTS_DIR, LOGS_DIR
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Directories created")


def print_project_info():
    """Print project information"""
    print("\n" + "=" * 60)
    print("🤟 SIGN LANGUAGE DETECTION PROJECT")
    print("=" * 60)
    print(f"📊 Total signs: {NUM_SIGNS}")
    print(f"📁 Categories: {len(SIGN_CATEGORIES)}")
    print("\n📋 Categories:")
    for cat, signs in SIGN_CATEGORIES.items():
        print(f"   • {cat}: {len(signs)} signs")
    print("=" * 60)


def print_all_signs():
    """Print all signs with descriptions"""
    print("\n" + "=" * 70)
    print("🤟 ALL SIGN LANGUAGE GESTURES")
    print("=" * 70)
    
    for category, signs in SIGN_CATEGORIES.items():
        print(f"\n📁 {category} ({len(signs)} signs)")
        print("-" * 40)
        for sign in signs:
            desc = SIGNS.get(sign, "")
            if desc != sign:
                print(f"   {sign:12} → {desc}")
            else:
                print(f"   {sign}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {NUM_SIGNS} signs")
    print("=" * 70)


def analyze_dataset(csv_path=DATASET_CSV):
    """Analyze dataset statistics"""
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "=" * 60)
    print("📊 DATASET ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Total signs: {df['sign'].nunique()}")
    print(f"Features: {len([c for c in df.columns if c.startswith('feature_')])}")
    
    # Distribution
    print("\n📈 Sample distribution:")
    print(df['sign'].value_counts())
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"\n⚠️  Missing values: {missing}")
    
    return df


def plot_dataset_distribution(csv_path=DATASET_CSV, save=True):
    """Plot dataset distribution"""
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    
    if df is None:
        print("❌ Dataset not found")
        return
    
    # Sign distribution
    plt.figure(figsize=(15, 8))
    
    counts = df['sign'].value_counts()
    
    if len(counts) > 30:
        # For many signs, use horizontal bar chart
        plt.figure(figsize=(12, max(8, len(counts) * 0.25)))
        counts.sort_values().plot(kind='barh', color='steelblue')
        plt.xlabel('Number of Samples')
        plt.ylabel('Sign')
    else:
        counts.plot(kind='bar', color='steelblue')
        plt.xlabel('Sign')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
    
    plt.title('Dataset Distribution by Sign', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = os.path.join(PLOTS_DIR, 'dataset_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {plot_path}")
    
    plt.show()


def plot_feature_heatmap(csv_path=DATASET_CSV, save=True):
    """Plot feature correlation heatmap"""
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    
    if df is None:
        print("❌ Dataset not found")
        return
    
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    
    # Sample features for visualization
    n_features = min(20, len(feature_cols))
    sample_cols = feature_cols[:n_features]
    
    plt.figure(figsize=(12, 10))
    corr = df[sample_cols].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation (First 20 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plot_path = os.path.join(PLOTS_DIR, 'feature_correlation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {plot_path}")
    
    plt.show()


def compare_models():
    """Compare different trained models"""
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('_training_log.json')]
    
    if not log_files:
        print("❌ No training logs found")
        return None
    
    results = []
    for log_file in log_files:
        with open(os.path.join(LOGS_DIR, log_file), 'r') as f:
            data = json.load(f)
            results.append(data)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)
    print(df[['model_type', 'test_accuracy', 'epochs_trained']].to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['model_type'], df['test_accuracy'] * 100, color='steelblue')
    plt.xlabel('Model Type')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    for i, (model, acc) in enumerate(zip(df['model_type'], df['test_accuracy'])):
        plt.text(i, acc * 100 + 1, f'{acc*100:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return df


def create_sample_dataset(num_samples=100, num_signs=10):
    """Create a synthetic sample dataset for testing"""
    print(f"Creating sample dataset: {num_samples} samples, {num_signs} signs")
    
    # Select signs
    selected_signs = SIGN_LABELS[:num_signs]
    
    # Generate random features
    data = []
    for _ in range(num_samples):
        sign = np.random.choice(selected_signs)
        features = np.random.randn(126)  # 126 features for 2 hands
        
        row = {
            'sign': sign,
            'timestamp': datetime.now().isoformat()
        }
        for i, f in enumerate(features):
            row[f'feature_{i}'] = f
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(DATASET_CSV, index=False)
    
    print(f"✅ Sample dataset saved: {DATASET_CSV}")
    print(f"   Samples: {len(df)}")
    print(f"   Signs: {df['sign'].nunique()}")
    
    return df


def validate_model(model_path):
    """Validate a trained model"""
    import tensorflow as tf
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model valid: {model_path}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"❌ Model invalid: {e}")
        return False


def export_model_info(model_path, output_path=None):
    """Export model information to JSON"""
    import tensorflow as tf
    
    try:
        model = tf.keras.models.load_model(model_path)
        
        info = {
            'model_path': model_path,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params(),
            'layers': len(model.layers),
            'layer_info': []
        }
        
        for layer in model.layers:
            info['layer_info'].append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'params': layer.count_params()
            })
        
        if output_path is None:
            output_path = model_path.replace('.h5', '_info.json')
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Model info exported: {output_path}")
        return info
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def clean_old_files(keep_latest=3):
    """Clean old model and log files"""
    import glob
    
    # Clean old models
    for pattern in ['*.h5', '*.keras']:
        files = sorted(glob.glob(os.path.join(MODELS_DIR, 'saved', pattern)), 
                      key=os.path.getmtime, reverse=True)
        for f in files[keep_latest:]:
            os.remove(f)
            print(f"🗑️  Removed: {f}")
    
    print("✅ Cleanup complete")


if __name__ == "__main__":
    # Test utilities
    print_project_info()
    print_all_signs()
    ensure_directories()
