"""
Real-time Sign Language Detection Module
Uses trained model with webcam for live prediction
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

from .config import (
    SIGNS, SIGN_LABELS, SIGN_CATEGORIES, NUM_SIGNS,
    CONFIDENCE_THRESHOLD, PREDICTION_BUFFER_SIZE,
    CAMERA_WIDTH, CAMERA_HEIGHT, DISPLAY_COLORS,
    TOTAL_FEATURES
)
from .data_collection import HandDetector
from .preprocessing import DataPreprocessor


class SignLanguageDetector:
    """Real-time sign language detection using trained model"""
    
    def __init__(self, model_path, preprocessor_path=None):
        self.model_path = model_path
        self.model = None
        self.detector = HandDetector()
        self.preprocessor = DataPreprocessor()
        
        # Prediction smoothing - smaller buffer for faster response
        self.prediction_buffer = deque(maxlen=5)
        self.last_prediction = None
        self.confidence_history = deque(maxlen=10)
        
        # Very low threshold - always show prediction when hand detected
        self.confidence_threshold = 0.05  # 5% - basically always show
        
        # Load model and preprocessors
        self.load_model()
        self.preprocessor.load_preprocessors()
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output classes: {self.model.output_shape[-1]}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_features(self, features):
        """Preprocess features for prediction"""
        # Reshape to 2D
        features = features.reshape(1, -1)
        
        # Normalize using loaded scaler
        if self.preprocessor.is_fitted:
            features = self.preprocessor.scaler.transform(features)
        
        # Check if model expects 3D input (LSTM)
        if len(self.model.input_shape) == 3:
            # Reshape for LSTM: (samples, timesteps, features)
            n_timesteps = self.model.input_shape[1]  # e.g., 6
            expected_features = n_timesteps * self.model.input_shape[2]  # e.g., 6 * 21 = 126
            
            # Pad if needed
            if features.shape[1] < expected_features:
                pad_size = expected_features - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_size)), mode='constant')
            elif features.shape[1] > expected_features:
                features = features[:, :expected_features]
            
            # Reshape to 3D
            features = features.reshape(-1, n_timesteps, self.model.input_shape[2])
        
        return features
    
    def predict(self, features):
        """Make prediction on features"""
        # Preprocess
        features = self.preprocess_features(features)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        
        # Get top prediction
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Add to buffer
        self.prediction_buffer.append((class_idx, confidence))
        
        return class_idx, confidence, predictions[0]
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from buffer"""
        if not self.prediction_buffer:
            return None, 0.0
        
        # Count predictions
        predictions = [p[0] for p in self.prediction_buffer]
        confidences = [p[1] for p in self.prediction_buffer]
        
        # Most common prediction
        from collections import Counter
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]
        
        # Average confidence for most common
        avg_confidence = np.mean([c for p, c in self.prediction_buffer 
                                  if p == most_common[0]])
        
        return most_common[0], avg_confidence
    
    def get_sign_name(self, class_idx):
        """Get sign name from class index"""
        try:
            if self.preprocessor.label_encoder.classes_ is not None:
                return self.preprocessor.label_encoder.inverse_transform([class_idx])[0]
        except:
            pass
        
        if class_idx < len(SIGN_LABELS):
            return SIGN_LABELS[class_idx]
        return f"Class_{class_idx}"
    
    def draw_ui(self, frame, sign_name, confidence, hand_detected, fps):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Header background - larger
        cv2.rectangle(frame, (0, 0), (w, 160), (40, 40, 40), -1)
        
        # Title
        cv2.putText(frame, "SIGN LANGUAGE DETECTOR", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Hand status indicator
        if hand_detected:
            cv2.circle(frame, (w - 100, 25), 12, (0, 255, 0), -1)
            cv2.putText(frame, "HAND OK", (w - 85, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.circle(frame, (w - 100, 25), 12, (0, 0, 255), -1)
            cv2.putText(frame, "NO HAND", (w - 85, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if hand_detected and sign_name:
            # ALWAYS show prediction when hand detected
            # Color based on confidence
            if confidence >= 0.5:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence >= 0.25:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            # Prediction box
            cv2.rectangle(frame, (15, 50), (w - 15, 150), color, 3)
            
            # Sign name - BIG and centered
            display_name = sign_name.upper() if len(sign_name) <= 2 else sign_name.upper()
            text_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, display_name, (text_x, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
            
            # Confidence percentage
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Confidence bar
            bar_width = int(confidence * (w - 250))
            cv2.rectangle(frame, (180, 135), (180 + bar_width, 148), color, -1)
            cv2.rectangle(frame, (180, 135), (w - 50, 148), (100, 100, 100), 1)
        
        else:
            # No hand detected
            cv2.putText(frame, "SHOW YOUR HAND", (w//2 - 150, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
            cv2.putText(frame, "Make a sign gesture", (w//2 - 100, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Footer
        cv2.rectangle(frame, (0, h - 35), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, "Q=Quit | SPACE=Toggle Smoothing | S=Screenshot",
                   (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        return frame
    
    def run(self):
        """Run real-time detection"""
        print("\n" + "=" * 60)
        print("🎥 STARTING REAL-TIME DETECTION")
        print("=" * 60)
        print("Controls:")
        print("  Q - Quit")
        print("  SPACE - Toggle smoothing")
        print("  S - Save screenshot")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        use_smoothing = True
        prev_time = time.time()
        fps = 0
        frame_count = 0
        
        print("✅ Camera opened. Starting detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time + 1e-6)
            prev_time = current_time
            
            # Extract landmarks
            features, results, hand_detected = self.detector.extract_landmarks(frame)
            
            # Draw landmarks
            frame = self.detector.draw_landmarks(frame, results)
            
            sign_name = None
            confidence = 0.0
            
            if hand_detected and features is not None:
                # Make prediction
                class_idx, conf, probs = self.predict(features)
                
                if use_smoothing:
                    class_idx, confidence = self.get_smoothed_prediction()
                else:
                    confidence = conf
                
                # Always get sign name if hand detected (lowered threshold in init)
                sign_name = self.get_sign_name(class_idx)
                
                # Debug: print top predictions occasionally
                if frame_count % 30 == 0:  # Every ~1 second
                    top_3 = np.argsort(probs)[-3:][::-1]
                    print(f"Top 3: ", end="")
                    for idx in top_3:
                        name = self.get_sign_name(idx)
                        print(f"{name}:{probs[idx]*100:.1f}% ", end="")
                    print()
            
            frame_count += 1
            
            # Draw UI
            frame = self.draw_ui(frame, sign_name, confidence, hand_detected, fps)
            
            cv2.imshow('Sign Language Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                use_smoothing = not use_smoothing
                print(f"Smoothing: {'ON' if use_smoothing else 'OFF'}")
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f'screenshot_{timestamp}.png', frame)
                print(f"📸 Screenshot saved")
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
        
        print("✅ Detection stopped")
    
    def release(self):
        """Release resources"""
        self.detector.release()


def detect_signs(model_path):
    """Utility function to run detection"""
    detector = SignLanguageDetector(model_path)
    detector.run()
    detector.release()


if __name__ == "__main__":
    # Test detection with default model
    from .config import HYBRID_MODEL_PATH
    
    detect_signs(HYBRID_MODEL_PATH)
