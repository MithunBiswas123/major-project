"""
Generate Realistic ASL Hand Landmark Dataset
Based on actual finger positions for each ASL sign
"""

import numpy as np
import pandas as pd
import os

# MediaPipe hand landmarks indices
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

class ASLLandmarkGenerator:
    """Generate realistic hand landmarks for ASL signs"""
    
    def __init__(self):
        # Base hand structure (normalized coordinates)
        self.base_hand = self._create_base_hand()
        
    def _create_base_hand(self):
        """Create base hand landmark positions (open hand, palm facing camera)"""
        landmarks = np.zeros((21, 3))
        
        # Wrist at center bottom
        landmarks[0] = [0.5, 0.9, 0.0]
        
        # Thumb (angled to side)
        landmarks[1] = [0.35, 0.75, 0.02]  # CMC
        landmarks[2] = [0.25, 0.65, 0.04]  # MCP
        landmarks[3] = [0.18, 0.55, 0.03]  # IP
        landmarks[4] = [0.12, 0.45, 0.02]  # TIP
        
        # Index finger
        landmarks[5] = [0.38, 0.55, 0.0]   # MCP
        landmarks[6] = [0.38, 0.40, 0.0]   # PIP
        landmarks[7] = [0.38, 0.28, 0.0]   # DIP
        landmarks[8] = [0.38, 0.18, 0.0]   # TIP
        
        # Middle finger
        landmarks[9] = [0.50, 0.52, 0.0]   # MCP
        landmarks[10] = [0.50, 0.35, 0.0]  # PIP
        landmarks[11] = [0.50, 0.22, 0.0]  # DIP
        landmarks[12] = [0.50, 0.12, 0.0]  # TIP
        
        # Ring finger
        landmarks[13] = [0.62, 0.55, 0.0]  # MCP
        landmarks[14] = [0.62, 0.40, 0.0]  # PIP
        landmarks[15] = [0.62, 0.28, 0.0]  # DIP
        landmarks[16] = [0.62, 0.18, 0.0]  # TIP
        
        # Pinky finger
        landmarks[17] = [0.74, 0.60, 0.0]  # MCP
        landmarks[18] = [0.74, 0.48, 0.0]  # PIP
        landmarks[19] = [0.74, 0.38, 0.0]  # DIP
        landmarks[20] = [0.74, 0.30, 0.0]  # TIP
        
        return landmarks
    
    def _curl_finger(self, landmarks, finger_indices, curl_amount=0.8):
        """Curl a finger (bend it into palm)"""
        mcp, pip, dip, tip = finger_indices
        base_y = landmarks[mcp][1]
        
        # Curl the finger joints
        landmarks[pip][1] = base_y + 0.05 * curl_amount
        landmarks[pip][2] = -0.05 * curl_amount
        landmarks[dip][1] = base_y + 0.10 * curl_amount
        landmarks[dip][2] = -0.08 * curl_amount
        landmarks[tip][1] = base_y + 0.12 * curl_amount
        landmarks[tip][2] = -0.06 * curl_amount
        
        return landmarks
    
    def _extend_finger(self, landmarks, finger_indices):
        """Extend finger straight up"""
        mcp, pip, dip, tip = finger_indices
        base_x = landmarks[mcp][0]
        base_y = landmarks[mcp][1]
        
        landmarks[pip] = [base_x, base_y - 0.15, 0.0]
        landmarks[dip] = [base_x, base_y - 0.27, 0.0]
        landmarks[tip] = [base_x, base_y - 0.37, 0.0]
        
        return landmarks
    
    def _thumb_across_palm(self, landmarks):
        """Move thumb across palm"""
        landmarks[2] = [0.35, 0.70, 0.05]
        landmarks[3] = [0.42, 0.65, 0.06]
        landmarks[4] = [0.48, 0.62, 0.05]
        return landmarks
    
    def _thumb_up(self, landmarks):
        """Thumb pointing up"""
        landmarks[1] = [0.30, 0.75, 0.02]
        landmarks[2] = [0.25, 0.60, 0.03]
        landmarks[3] = [0.22, 0.45, 0.02]
        landmarks[4] = [0.20, 0.32, 0.01]
        return landmarks
    
    def _thumb_out(self, landmarks):
        """Thumb pointing outward"""
        landmarks[1] = [0.35, 0.75, 0.02]
        landmarks[2] = [0.22, 0.72, 0.03]
        landmarks[3] = [0.12, 0.70, 0.02]
        landmarks[4] = [0.05, 0.68, 0.01]
        return landmarks
    
    def generate_sign(self, sign_name):
        """Generate landmarks for a specific ASL sign"""
        landmarks = self.base_hand.copy()
        
        # Finger indices: [MCP, PIP, DIP, TIP]
        THUMB = [1, 2, 3, 4]
        INDEX = [5, 6, 7, 8]
        MIDDLE = [9, 10, 11, 12]
        RING = [13, 14, 15, 16]
        PINKY = [17, 18, 19, 20]
        
        sign = sign_name.upper()
        
        # ============ ALPHABET SIGNS ============
        if sign == 'A':
            # Fist with thumb on side
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_up(landmarks)
            
        elif sign == 'B':
            # Flat hand, fingers together, thumb tucked
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'C':
            # Curved hand like C shape
            landmarks[8][0] = 0.35  # Index curves
            landmarks[12][0] = 0.45
            landmarks[16][0] = 0.55
            landmarks[20][0] = 0.65
            landmarks[4][0] = 0.20
            # All fingers slightly curved
            for tip in [8, 12, 16, 20]:
                landmarks[tip][1] += 0.1
                landmarks[tip][2] = -0.03
            
        elif sign == 'D':
            # Index up, others touch thumb
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._curl_finger(landmarks, MIDDLE, 0.9)
            landmarks = self._curl_finger(landmarks, RING, 0.9)
            landmarks = self._curl_finger(landmarks, PINKY, 0.9)
            landmarks[4] = [0.45, 0.60, 0.05]  # Thumb touches middle
            
        elif sign == 'E':
            # All fingers bent, thumb tucked
            landmarks = self._curl_finger(landmarks, INDEX, 0.7)
            landmarks = self._curl_finger(landmarks, MIDDLE, 0.7)
            landmarks = self._curl_finger(landmarks, RING, 0.7)
            landmarks = self._curl_finger(landmarks, PINKY, 0.7)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'F':
            # Index and thumb form circle, others up
            landmarks[4] = [0.38, 0.45, 0.05]  # Thumb to index
            landmarks[8] = [0.35, 0.42, 0.04]  # Index bent to thumb
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            
        elif sign == 'G':
            # Index and thumb pointing sideways
            landmarks[4] = [0.15, 0.60, 0.0]  # Thumb out
            landmarks[8] = [0.15, 0.55, 0.0]  # Index out
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == 'H':
            # Index and middle pointing sideways
            landmarks[8] = [0.20, 0.50, 0.0]
            landmarks[12] = [0.25, 0.48, 0.0]
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'I':
            # Pinky up only
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'J':
            # Like I but traced J shape
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks[20][0] = 0.68  # Pinky curves for J
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'K':
            # Index and middle up, thumb between
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks[4] = [0.42, 0.45, 0.05]  # Thumb between
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == 'L':
            # L shape with thumb and index
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._thumb_out(landmarks)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == 'M':
            # Three fingers over thumb
            landmarks = self._curl_finger(landmarks, INDEX, 0.8)
            landmarks = self._curl_finger(landmarks, MIDDLE, 0.8)
            landmarks = self._curl_finger(landmarks, RING, 0.8)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks[4] = [0.55, 0.70, 0.08]
            
        elif sign == 'N':
            # Two fingers over thumb
            landmarks = self._curl_finger(landmarks, INDEX, 0.8)
            landmarks = self._curl_finger(landmarks, MIDDLE, 0.8)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks[4] = [0.48, 0.70, 0.08]
            
        elif sign == 'O':
            # All fingers form O with thumb
            for tip in [8, 12, 16, 20]:
                landmarks[tip][0] = 0.40
                landmarks[tip][1] = 0.50
                landmarks[tip][2] = 0.02
            landmarks[4] = [0.40, 0.52, 0.04]
            
        elif sign == 'P':
            # Like K but pointing down
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks[8][1] += 0.3  # Point down
            landmarks[12][1] += 0.3
            landmarks[4] = [0.42, 0.50, 0.05]
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == 'Q':
            # Like G but pointing down
            landmarks[4] = [0.30, 0.75, 0.0]
            landmarks[8] = [0.32, 0.78, 0.0]
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == 'R':
            # Index and middle crossed
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks[8][0] = 0.45  # Cross index over
            landmarks[12][0] = 0.42
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'S':
            # Fist with thumb over fingers
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'T':
            # Thumb between index and middle
            landmarks = self._curl_finger(landmarks, INDEX, 0.9)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks[4] = [0.40, 0.55, 0.06]
            
        elif sign == 'U':
            # Index and middle up together
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks[8][0] = 0.42  # Close together
            landmarks[12][0] = 0.48
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'V':
            # Peace sign / V shape
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks[8][0] = 0.32  # Spread apart
            landmarks[12][0] = 0.55
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'W':
            # Three fingers up spread
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks[8][0] = 0.32
            landmarks[12][0] = 0.50
            landmarks[16][0] = 0.68
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'X':
            # Index hooked
            landmarks = self._curl_finger(landmarks, INDEX, 0.5)
            landmarks[8][2] = -0.08  # Hook shape
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'Y':
            # Thumb and pinky out (hang loose)
            landmarks = self._thumb_out(landmarks)
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._extend_finger(landmarks, PINKY)
            
        elif sign == 'Z':
            # Index traces Z shape
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks[8][0] = 0.45  # Angled
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        # ============ NUMBER SIGNS ============
        elif sign == '0':
            # All fingers form O
            for tip in [8, 12, 16, 20]:
                landmarks[tip] = [0.42, 0.52, 0.02]
            landmarks[4] = [0.42, 0.54, 0.04]
            
        elif sign == '1':
            # Index up
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == '2':
            # Index and middle up (V)
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == '3':
            # Thumb, index, middle up
            landmarks = self._thumb_out(landmarks)
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == '4':
            # Four fingers up, thumb tucked
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == '5':
            # All five fingers spread
            landmarks = self._thumb_out(landmarks)
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            # Spread fingers
            landmarks[8][0] = 0.30
            landmarks[12][0] = 0.45
            landmarks[16][0] = 0.60
            landmarks[20][0] = 0.75
            
        elif sign == '6':
            # Pinky and thumb touch, others up
            landmarks[4] = [0.70, 0.55, 0.03]  # Thumb to pinky
            landmarks[20] = [0.72, 0.52, 0.02]  # Pinky to thumb
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            
        elif sign == '7':
            # Ring and thumb touch, others up
            landmarks[4] = [0.60, 0.52, 0.03]
            landmarks[16] = [0.62, 0.50, 0.02]
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, PINKY)
            
        elif sign == '8':
            # Middle and thumb touch, others up
            landmarks[4] = [0.48, 0.48, 0.03]
            landmarks[12] = [0.50, 0.46, 0.02]
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            
        elif sign == '9':
            # Index and thumb touch, others up
            landmarks[4] = [0.36, 0.45, 0.03]
            landmarks[8] = [0.38, 0.43, 0.02]
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            
        # ============ COMMON WORDS ============
        elif sign in ['HELLO', 'HI']:
            # Open hand wave
            landmarks = self._thumb_out(landmarks)
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            # Slight tilt
            for i in range(21):
                landmarks[i][0] += 0.05
                
        elif sign in ['GOODBYE', 'BYE']:
            # Open hand
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_out(landmarks)
            
        elif sign == 'YES':
            # Fist nodding (S hand)
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'NO':
            # Index and middle closing to thumb
            landmarks[4] = [0.38, 0.50, 0.04]
            landmarks[8] = [0.36, 0.48, 0.03]
            landmarks[12] = [0.42, 0.48, 0.03]
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            
        elif sign == 'THANKS':
            # Flat hand from chin outward (B hand)
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_across_palm(landmarks)
            # Angled outward
            for i in range(5, 21):
                landmarks[i][0] -= 0.05
                
        elif sign == 'PLEASE':
            # Flat hand on chest circular (B hand)
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_across_palm(landmarks)
            
        elif sign == 'OK':
            # OK gesture
            landmarks[4] = [0.38, 0.48, 0.05]
            landmarks[8] = [0.36, 0.46, 0.04]
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            
        elif sign == 'SORRY':
            # A hand circular on chest
            landmarks = self._curl_finger(landmarks, INDEX, 1.0)
            landmarks = self._curl_finger(landmarks, MIDDLE, 1.0)
            landmarks = self._curl_finger(landmarks, RING, 1.0)
            landmarks = self._curl_finger(landmarks, PINKY, 1.0)
            landmarks = self._thumb_up(landmarks)
            
        elif sign in ['WELCOME', 'NICE_TO_MEET', 'GOOD_MORNING', 'GOOD_NIGHT', 
                      'SEE_YOU', 'TAKE_CARE', 'MAYBE']:
            # Open hand variations
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_out(landmarks)
            
        else:
            # Default: open hand
            landmarks = self._extend_finger(landmarks, INDEX)
            landmarks = self._extend_finger(landmarks, MIDDLE)
            landmarks = self._extend_finger(landmarks, RING)
            landmarks = self._extend_finger(landmarks, PINKY)
            landmarks = self._thumb_out(landmarks)
        
        return landmarks
    
    def add_noise(self, landmarks, noise_level=0.02):
        """Add realistic noise to landmarks"""
        noisy = landmarks.copy()
        noise = np.random.normal(0, noise_level, landmarks.shape)
        noisy += noise
        # Keep in valid range
        noisy = np.clip(noisy, 0, 1)
        return noisy
    
    def add_variation(self, landmarks, var_level=0.03):
        """Add natural hand variation"""
        varied = landmarks.copy()
        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        varied *= scale
        # Random rotation (small)
        angle = np.random.uniform(-0.1, 0.1)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(len(varied)):
            x, y = varied[i][0] - 0.5, varied[i][1] - 0.5
            varied[i][0] = x * cos_a - y * sin_a + 0.5
            varied[i][1] = x * sin_a + y * cos_a + 0.5
        # Random translation
        varied[:, 0] += np.random.uniform(-0.1, 0.1)
        varied[:, 1] += np.random.uniform(-0.1, 0.1)
        return np.clip(varied, 0, 1)


def generate_dataset(signs, samples_per_sign=100, output_path=None):
    """Generate complete dataset"""
    generator = ASLLandmarkGenerator()
    
    all_data = []
    
    print(f"\n🔄 Generating dataset for {len(signs)} signs...")
    print(f"   Samples per sign: {samples_per_sign}")
    
    for sign in signs:
        base_landmarks = generator.generate_sign(sign)
        
        for i in range(samples_per_sign):
            # Add variation and noise
            varied = generator.add_variation(base_landmarks)
            noisy = generator.add_noise(varied)
            
            # Flatten for both hands (duplicate for 2 hands = 126 features)
            left_hand = noisy.flatten()
            right_hand = generator.add_noise(generator.add_variation(base_landmarks)).flatten()
            features = np.concatenate([left_hand, right_hand])
            
            # Create row
            row = {'sign': sign}
            for j, val in enumerate(features):
                row[f'feature_{j}'] = val
            all_data.append(row)
        
        print(f"   ✅ Generated {samples_per_sign} samples for '{sign}'")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'raw', 'sign_dataset.csv'
        )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(df.columns) - 1}")
    
    return df


if __name__ == "__main__":
    # Signs to generate (matching the model's training)
    SIGNS = [
        # Alphabet
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        # Numbers
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # Common words
        'hello', 'hi', 'goodbye', 'bye', 'yes', 'no',
        'thanks', 'please', 'sorry', 'ok', 'welcome',
        'good_morning', 'good_night', 'see_you', 'take_care',
        'nice_to_meet', 'maybe'
    ]
    
    # Generate
    generate_dataset(SIGNS, samples_per_sign=100)
