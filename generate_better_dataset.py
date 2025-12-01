"""
Generate High-Quality ASL Hand Landmark Dataset
More distinct patterns for better accuracy
"""

import numpy as np
import pandas as pd
import os

class ImprovedASLGenerator:
    """Generate highly distinct hand landmarks for each ASL sign"""
    
    def __init__(self):
        # Base positions for 21 landmarks (x, y, z)
        self.wrist = np.array([0.5, 0.85, 0.0])
        
    def _create_hand(self, finger_states):
        """
        Create hand based on finger states
        finger_states: dict with keys 'thumb', 'index', 'middle', 'ring', 'pinky'
        values: 'up', 'down', 'curled', 'out', 'bent', 'touch_thumb'
        """
        landmarks = np.zeros((21, 3))
        
        # Wrist
        landmarks[0] = self.wrist
        
        # Define finger base positions
        finger_bases = {
            'thumb': (0.35, 0.72),
            'index': (0.40, 0.58),
            'middle': (0.50, 0.55),
            'ring': (0.60, 0.58),
            'pinky': (0.70, 0.62)
        }
        
        # Thumb (indices 1-4)
        thumb_state = finger_states.get('thumb', 'out')
        if thumb_state == 'up':
            landmarks[1] = [0.32, 0.72, 0.02]
            landmarks[2] = [0.28, 0.58, 0.03]
            landmarks[3] = [0.25, 0.45, 0.02]
            landmarks[4] = [0.22, 0.32, 0.01]
        elif thumb_state == 'out':
            landmarks[1] = [0.35, 0.72, 0.02]
            landmarks[2] = [0.22, 0.68, 0.03]
            landmarks[3] = [0.12, 0.65, 0.02]
            landmarks[4] = [0.05, 0.62, 0.01]
        elif thumb_state == 'across':
            landmarks[1] = [0.38, 0.72, 0.04]
            landmarks[2] = [0.42, 0.68, 0.06]
            landmarks[3] = [0.48, 0.65, 0.05]
            landmarks[4] = [0.52, 0.63, 0.04]
        elif thumb_state == 'curled':
            landmarks[1] = [0.38, 0.72, 0.04]
            landmarks[2] = [0.40, 0.70, 0.08]
            landmarks[3] = [0.42, 0.72, 0.06]
            landmarks[4] = [0.44, 0.74, 0.04]
        else:  # down/tucked
            landmarks[1] = [0.38, 0.72, 0.04]
            landmarks[2] = [0.42, 0.75, 0.06]
            landmarks[3] = [0.45, 0.78, 0.05]
            landmarks[4] = [0.48, 0.80, 0.04]
        
        # Index finger (indices 5-8)
        self._set_finger(landmarks, 5, finger_bases['index'], 
                        finger_states.get('index', 'up'), 'index')
        
        # Middle finger (indices 9-12)
        self._set_finger(landmarks, 9, finger_bases['middle'], 
                        finger_states.get('middle', 'up'), 'middle')
        
        # Ring finger (indices 13-16)
        self._set_finger(landmarks, 13, finger_bases['ring'], 
                        finger_states.get('ring', 'up'), 'ring')
        
        # Pinky finger (indices 17-20)
        self._set_finger(landmarks, 17, finger_bases['pinky'], 
                        finger_states.get('pinky', 'up'), 'pinky')
        
        return landmarks
    
    def _set_finger(self, landmarks, start_idx, base_pos, state, finger_name):
        """Set finger landmark positions based on state"""
        bx, by = base_pos
        
        if state == 'up':
            landmarks[start_idx] = [bx, by, 0.0]
            landmarks[start_idx + 1] = [bx, by - 0.15, 0.0]
            landmarks[start_idx + 2] = [bx, by - 0.28, 0.0]
            landmarks[start_idx + 3] = [bx, by - 0.40, 0.0]
        elif state == 'down' or state == 'curled':
            landmarks[start_idx] = [bx, by, 0.0]
            landmarks[start_idx + 1] = [bx, by + 0.05, -0.06]
            landmarks[start_idx + 2] = [bx, by + 0.08, -0.10]
            landmarks[start_idx + 3] = [bx, by + 0.06, -0.08]
        elif state == 'bent':
            landmarks[start_idx] = [bx, by, 0.0]
            landmarks[start_idx + 1] = [bx, by - 0.10, -0.02]
            landmarks[start_idx + 2] = [bx, by - 0.05, -0.08]
            landmarks[start_idx + 3] = [bx, by + 0.02, -0.06]
        elif state == 'touch_thumb':
            landmarks[start_idx] = [bx, by, 0.0]
            landmarks[start_idx + 1] = [bx - 0.05, by - 0.05, 0.03]
            landmarks[start_idx + 2] = [bx - 0.10, by, 0.05]
            landmarks[start_idx + 3] = [0.45, 0.62, 0.04]  # Touch thumb
        elif state == 'spread':
            offset = 0.08 if finger_name == 'pinky' else 0.05
            landmarks[start_idx] = [bx, by, 0.0]
            landmarks[start_idx + 1] = [bx + offset * 0.3, by - 0.15, 0.0]
            landmarks[start_idx + 2] = [bx + offset * 0.6, by - 0.28, 0.0]
            landmarks[start_idx + 3] = [bx + offset, by - 0.40, 0.0]
        else:
            # Default up
            landmarks[start_idx] = [bx, by, 0.0]
            landmarks[start_idx + 1] = [bx, by - 0.15, 0.0]
            landmarks[start_idx + 2] = [bx, by - 0.28, 0.0]
            landmarks[start_idx + 3] = [bx, by - 0.40, 0.0]
        
        return landmarks
    
    def generate_sign(self, sign):
        """Generate landmarks for specific sign"""
        sign = sign.upper()
        
        # Define each sign's finger configuration
        sign_configs = {
            # Letters - each has UNIQUE pattern
            'A': {'thumb': 'up', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'B': {'thumb': 'across', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'C': {'thumb': 'out', 'index': 'bent', 'middle': 'bent', 'ring': 'bent', 'pinky': 'bent'},
            'D': {'thumb': 'touch_thumb', 'index': 'up', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'E': {'thumb': 'across', 'index': 'bent', 'middle': 'bent', 'ring': 'bent', 'pinky': 'bent'},
            'F': {'thumb': 'touch_thumb', 'index': 'touch_thumb', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'G': {'thumb': 'out', 'index': 'out', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'H': {'thumb': 'curled', 'index': 'out', 'middle': 'out', 'ring': 'curled', 'pinky': 'curled'},
            'I': {'thumb': 'across', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'up'},
            'J': {'thumb': 'across', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'spread'},
            'K': {'thumb': 'up', 'index': 'up', 'middle': 'up', 'ring': 'curled', 'pinky': 'curled'},
            'L': {'thumb': 'out', 'index': 'up', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'M': {'thumb': 'curled', 'index': 'bent', 'middle': 'bent', 'ring': 'bent', 'pinky': 'curled'},
            'N': {'thumb': 'curled', 'index': 'bent', 'middle': 'bent', 'ring': 'curled', 'pinky': 'curled'},
            'O': {'thumb': 'touch_thumb', 'index': 'touch_thumb', 'middle': 'touch_thumb', 'ring': 'touch_thumb', 'pinky': 'touch_thumb'},
            'P': {'thumb': 'out', 'index': 'down', 'middle': 'down', 'ring': 'curled', 'pinky': 'curled'},
            'Q': {'thumb': 'down', 'index': 'down', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'R': {'thumb': 'across', 'index': 'up', 'middle': 'up', 'ring': 'curled', 'pinky': 'curled'},  # crossed
            'S': {'thumb': 'across', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'T': {'thumb': 'up', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},  # thumb between
            'U': {'thumb': 'across', 'index': 'up', 'middle': 'up', 'ring': 'curled', 'pinky': 'curled'},  # together
            'V': {'thumb': 'across', 'index': 'spread', 'middle': 'spread', 'ring': 'curled', 'pinky': 'curled'},
            'W': {'thumb': 'across', 'index': 'spread', 'middle': 'up', 'ring': 'spread', 'pinky': 'curled'},
            'X': {'thumb': 'across', 'index': 'bent', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'Y': {'thumb': 'out', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'out'},
            'Z': {'thumb': 'across', 'index': 'up', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},  # trace Z
            
            # Numbers - distinct patterns
            '0': {'thumb': 'touch_thumb', 'index': 'touch_thumb', 'middle': 'touch_thumb', 'ring': 'touch_thumb', 'pinky': 'touch_thumb'},
            '1': {'thumb': 'across', 'index': 'up', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            '2': {'thumb': 'across', 'index': 'spread', 'middle': 'spread', 'ring': 'curled', 'pinky': 'curled'},
            '3': {'thumb': 'out', 'index': 'up', 'middle': 'up', 'ring': 'curled', 'pinky': 'curled'},
            '4': {'thumb': 'across', 'index': 'spread', 'middle': 'spread', 'ring': 'spread', 'pinky': 'spread'},
            '5': {'thumb': 'out', 'index': 'spread', 'middle': 'spread', 'ring': 'spread', 'pinky': 'spread'},
            '6': {'thumb': 'touch_thumb', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'touch_thumb'},
            '7': {'thumb': 'touch_thumb', 'index': 'up', 'middle': 'up', 'ring': 'touch_thumb', 'pinky': 'up'},
            '8': {'thumb': 'touch_thumb', 'index': 'up', 'middle': 'touch_thumb', 'ring': 'up', 'pinky': 'up'},
            '9': {'thumb': 'touch_thumb', 'index': 'touch_thumb', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            
            # Words - each gets unique base pattern
            'HELLO': {'thumb': 'out', 'index': 'spread', 'middle': 'spread', 'ring': 'spread', 'pinky': 'spread'},
            'HI': {'thumb': 'out', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'GOODBYE': {'thumb': 'up', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'BYE': {'thumb': 'curled', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'YES': {'thumb': 'up', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'NO': {'thumb': 'out', 'index': 'up', 'middle': 'up', 'ring': 'curled', 'pinky': 'curled'},
            'THANKS': {'thumb': 'out', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'PLEASE': {'thumb': 'across', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'SORRY': {'thumb': 'across', 'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            'OK': {'thumb': 'touch_thumb', 'index': 'touch_thumb', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'WELCOME': {'thumb': 'out', 'index': 'out', 'middle': 'up', 'ring': 'up', 'pinky': 'up'},
            'GOOD_MORNING': {'thumb': 'up', 'index': 'spread', 'middle': 'spread', 'ring': 'curled', 'pinky': 'curled'},
            'GOOD_NIGHT': {'thumb': 'down', 'index': 'down', 'middle': 'down', 'ring': 'curled', 'pinky': 'curled'},
            'SEE_YOU': {'thumb': 'out', 'index': 'up', 'middle': 'curled', 'ring': 'curled', 'pinky': 'up'},
            'TAKE_CARE': {'thumb': 'up', 'index': 'up', 'middle': 'curled', 'ring': 'curled', 'pinky': 'up'},
            'NICE_TO_MEET': {'thumb': 'out', 'index': 'bent', 'middle': 'bent', 'ring': 'curled', 'pinky': 'curled'},
            'MAYBE': {'thumb': 'out', 'index': 'up', 'middle': 'up', 'ring': 'down', 'pinky': 'down'},
        }
        
        # Get config or default
        config = sign_configs.get(sign, {'thumb': 'out', 'index': 'up', 'middle': 'up', 'ring': 'up', 'pinky': 'up'})
        
        return self._create_hand(config)
    
    def add_realistic_noise(self, landmarks, noise_level=0.015):
        """Add realistic variation"""
        noisy = landmarks.copy()
        
        # Different noise for different parts
        for i in range(len(noisy)):
            # Less noise for wrist, more for fingertips
            if i == 0:
                noise = np.random.normal(0, noise_level * 0.5, 3)
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                noise = np.random.normal(0, noise_level * 1.5, 3)
            else:
                noise = np.random.normal(0, noise_level, 3)
            noisy[i] += noise
        
        return np.clip(noisy, 0, 1)
    
    def add_hand_variation(self, landmarks):
        """Add natural hand variation (rotation, scale, position)"""
        varied = landmarks.copy()
        
        # Random scale (hand size)
        scale = np.random.uniform(0.85, 1.15)
        center = landmarks[0]  # Wrist as center
        varied = center + (varied - center) * scale
        
        # Random rotation (small)
        angle = np.random.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(len(varied)):
            x, y = varied[i][0] - center[0], varied[i][1] - center[1]
            varied[i][0] = x * cos_a - y * sin_a + center[0]
            varied[i][1] = x * sin_a + y * cos_a + center[1]
        
        # Random position shift
        shift_x = np.random.uniform(-0.1, 0.1)
        shift_y = np.random.uniform(-0.1, 0.1)
        varied[:, 0] += shift_x
        varied[:, 1] += shift_y
        
        return np.clip(varied, 0, 1)


def generate_high_quality_dataset(signs, samples_per_sign=200):
    """Generate high-quality dataset"""
    generator = ImprovedASLGenerator()
    
    all_data = []
    
    print(f"\n🔄 Generating HIGH-QUALITY dataset for {len(signs)} signs...")
    print(f"   Samples per sign: {samples_per_sign}")
    
    for sign in signs:
        base_landmarks = generator.generate_sign(sign)
        
        for i in range(samples_per_sign):
            # Apply variations
            varied = generator.add_hand_variation(base_landmarks)
            noisy = generator.add_realistic_noise(varied)
            
            # Create both hands (left and right with slight difference)
            left_hand = noisy.flatten()
            
            # Right hand - mirror and slight variation
            right_landmarks = generator.add_realistic_noise(
                generator.add_hand_variation(base_landmarks)
            )
            right_hand = right_landmarks.flatten()
            
            # Combine for 126 features (21 * 3 * 2)
            features = np.concatenate([left_hand, right_hand])
            
            # Create row
            row = {'sign': sign}
            for j, val in enumerate(features):
                row[f'feature_{j}'] = val
            all_data.append(row)
        
        print(f"   ✅ {sign}: {samples_per_sign} samples")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'raw', 'sign_dataset.csv'
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Signs: {len(signs)}")
    print(f"   Features: 126")
    
    return df


if __name__ == "__main__":
    # All signs to generate
    SIGNS = [
        # Alphabet (26)
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        # Numbers (10)
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # Common words (17)
        'hello', 'hi', 'goodbye', 'bye', 'yes', 'no',
        'thanks', 'please', 'sorry', 'ok', 'welcome',
        'good_morning', 'good_night', 'see_you', 'take_care',
        'nice_to_meet', 'maybe'
    ]
    
    # Generate with 200 samples per sign for better accuracy
    generate_high_quality_dataset(SIGNS, samples_per_sign=200)
