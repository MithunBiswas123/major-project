"""
Training Module for Sign Language Detection
Complete training pipeline with evaluation and visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

import tensorflow as tf

from .config import (
    EPOCHS, BATCH_SIZE, MODELS_DIR, PLOTS_DIR, LOGS_DIR,
    CNN_MODEL_PATH, LSTM_MODEL_PATH, HYBRID_MODEL_PATH, TRANSFORMER_MODEL_PATH
)
from .preprocessing import DataPreprocessor
from .models import (
    get_model, get_callbacks, get_model_path, print_model_summary
)


class SignLanguageTrainer:
    """Complete training pipeline for sign language detection"""
    
    def __init__(self, model_type='hybrid'):
        self.model_type = model_type
        self.model = None
        self.history = None
        self.preprocessor = DataPreprocessor()
        
        # Create directories
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
    
    def prepare_data(self, augment=True):
        """Prepare and preprocess data"""
        print("\n" + "=" * 60)
        print("📊 PREPARING DATA")
        print("=" * 60)
        
        data = self.preprocessor.preprocess_pipeline(augment=augment)
        
        if data is None:
            raise ValueError("Failed to preprocess data. Check if dataset exists.")
        
        self.X_train, self.X_val, self.X_test, \
        self.y_train, self.y_val, self.y_test = data
        
        self.num_classes = self.preprocessor.get_num_classes()
        self.class_names = self.preprocessor.get_class_names()
        
        print(f"\n✅ Data prepared:")
        print(f"   Classes: {self.num_classes}")
        print(f"   Features: {self.X_train.shape[1]}")
        
        return data
    
    def build_model(self):
        """Build the model"""
        print("\n" + "=" * 60)
        print(f"🏗️  BUILDING {self.model_type.upper()} MODEL")
        print("=" * 60)
        
        self.model = get_model(
            self.model_type,
            num_features=self.X_train.shape[1],
            num_classes=self.num_classes
        )
        
        print_model_summary(self.model)
        
        return self.model
    
    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the model"""
        print("\n" + "=" * 60)
        print("🚀 STARTING TRAINING")
        print("=" * 60)
        print(f"   Model: {self.model_type.upper()}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print("=" * 60)
        
        model_path = get_model_path(self.model_type)
        callbacks = get_callbacks(model_path)
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        return self.history
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        
        # Test evaluation
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"\n📈 Test Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Predictions
        y_pred = np.argmax(self.model.predict(self.X_test, verbose=0), axis=1)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=self.class_names[:len(np.unique(self.y_test))]
        ))
        
        return test_loss, test_acc, y_pred
    
    def plot_history(self, save=True):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(PLOTS_DIR, f'{self.model_type}_history.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {plot_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_pred, save=True):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        # If too many classes, show top N
        n_classes = len(self.class_names)
        
        plt.figure(figsize=(min(20, n_classes), min(16, n_classes * 0.8)))
        
        if n_classes > 30:
            # For many classes, use simpler visualization
            sns.heatmap(cm, cmap='Blues', cbar=True)
            plt.title(f'Confusion Matrix ({n_classes} classes)', fontsize=14, fontweight='bold')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names[:n_classes],
                       yticklabels=self.class_names[:n_classes])
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(PLOTS_DIR, f'{self.model_type}_confusion.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {plot_path}")
        
        plt.show()
    
    def save_training_log(self, test_acc, test_loss):
        """Save training log"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'epochs_trained': len(self.history.history['loss']),
            'final_train_acc': float(self.history.history['accuracy'][-1]),
            'final_val_acc': float(self.history.history['val_accuracy'][-1]),
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss)
        }
        
        log_path = os.path.join(LOGS_DIR, f'{self.model_type}_training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"📝 Saved: {log_path}")
    
    def run_full_pipeline(self, epochs=EPOCHS, batch_size=BATCH_SIZE, augment=True):
        """Run complete training pipeline"""
        print("\n" + "=" * 70)
        print("🎯 SIGN LANGUAGE DETECTION - FULL TRAINING PIPELINE")
        print("=" * 70)
        
        # Step 1: Prepare data
        self.prepare_data(augment=augment)
        
        # Step 2: Build model
        self.build_model()
        
        # Step 3: Train
        self.train(epochs=epochs, batch_size=batch_size)
        
        # Step 4: Evaluate
        test_loss, test_acc, y_pred = self.evaluate()
        
        # Step 5: Visualize
        self.plot_history()
        self.plot_confusion_matrix(y_pred)
        
        # Step 6: Save log
        self.save_training_log(test_acc, test_loss)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING PIPELINE COMPLETE")
        print(f"   Model saved: {get_model_path(self.model_type)}")
        print(f"   Test Accuracy: {test_acc*100:.2f}%")
        print("=" * 70)
        
        return test_acc


def train_model(model_type='hybrid', epochs=EPOCHS, augment=True):
    """Utility function to train a model"""
    trainer = SignLanguageTrainer(model_type=model_type)
    return trainer.run_full_pipeline(epochs=epochs, augment=augment)


def train_all_models(epochs=EPOCHS):
    """Train all model architectures for comparison"""
    results = {}
    
    for model_type in ['dnn', 'cnn', 'lstm', 'hybrid', 'transformer']:
        print(f"\n{'='*70}")
        print(f"TRAINING {model_type.upper()} MODEL")
        print(f"{'='*70}")
        
        try:
            trainer = SignLanguageTrainer(model_type=model_type)
            acc = trainer.run_full_pipeline(epochs=epochs)
            results[model_type] = acc
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            results[model_type] = None
    
    # Print comparison
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)
    for model_type, acc in results.items():
        status = f"{acc*100:.2f}%" if acc else "Failed"
        print(f"   {model_type.upper():12} : {status}")
    
    return results


if __name__ == "__main__":
    # Train hybrid model
    trainer = SignLanguageTrainer(model_type='hybrid')
    trainer.run_full_pipeline(epochs=50)
