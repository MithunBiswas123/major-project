"""
Generate Realistic Sign Language Dataset
Creates hand landmark patterns based on actual ASL sign formations
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Base hand landmark structure (21 landmarks x 3 coords = 63 per hand)
# Landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky

def normalize_landmarks(landmarks):
    """Normalize landmarks to 0-1 range"""
    landmarks = np.array(landmarks)
    min_val = landmarks.min()
    max_val = landmarks.max()
    if max_val - min_val > 0:
        landmarks = (landmarks - min_val) / (max_val - min_val)
    return landmarks

def create_base_hand():
    """Create a base hand position (open palm facing camera)"""
    # Wrist at center bottom
    wrist = [0.5, 0.8, 0.0]
    
    # Thumb (pointing right-ish)
    thumb = [
        [0.35, 0.75, 0.02],  # CMC
        [0.28, 0.68, 0.03],  # MCP
        [0.22, 0.60, 0.02],  # IP
        [0.18, 0.52, 0.01],  # TIP
    ]
    
    # Index finger (pointing up)
    index = [
        [0.42, 0.65, 0.0],   # MCP
        [0.40, 0.50, 0.0],   # PIP
        [0.39, 0.38, 0.0],   # DIP
        [0.38, 0.28, 0.0],   # TIP
    ]
    
    # Middle finger (pointing up, longest)
    middle = [
        [0.50, 0.63, 0.0],   # MCP
        [0.50, 0.46, 0.0],   # PIP
        [0.50, 0.32, 0.0],   # DIP
        [0.50, 0.20, 0.0],   # TIP
    ]
    
    # Ring finger
    ring = [
        [0.58, 0.65, 0.0],   # MCP
        [0.60, 0.50, 0.0],   # PIP
        [0.61, 0.38, 0.0],   # DIP
        [0.62, 0.28, 0.0],   # TIP
    ]
    
    # Pinky finger
    pinky = [
        [0.66, 0.68, 0.0],   # MCP
        [0.70, 0.56, 0.0],   # PIP
        [0.72, 0.46, 0.0],   # DIP
        [0.74, 0.38, 0.0],   # TIP
    ]
    
    landmarks = [wrist] + thumb + index + middle + ring + pinky
    return np.array(landmarks).flatten()

def curl_finger(landmarks, finger_idx, curl_amount=0.8):
    """Curl a specific finger (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)"""
    landmarks = landmarks.copy().reshape(21, 3)
    
    # Finger base indices
    bases = {0: 1, 1: 5, 2: 9, 3: 13, 4: 17}
    base = bases[finger_idx]
    
    # Move fingertip towards palm
    for i in range(4):
        idx = base + i
        if idx < 21:
            # Curl by moving y up and z forward
            landmarks[idx, 1] += curl_amount * 0.1 * (i + 1)
            landmarks[idx, 2] += curl_amount * 0.05 * (i + 1)
    
    return landmarks.flatten()

def extend_finger(landmarks, finger_idx):
    """Extend a specific finger straight"""
    landmarks = landmarks.copy().reshape(21, 3)
    
    bases = {0: 1, 1: 5, 2: 9, 3: 13, 4: 17}
    base = bases[finger_idx]
    
    # Straighten finger
    base_y = landmarks[base, 1]
    for i in range(4):
        idx = base + i
        if idx < 21:
            landmarks[idx, 1] = base_y - 0.12 * (i + 1)
            landmarks[idx, 2] = 0.0
    
    return landmarks.flatten()

def create_fist():
    """Create a closed fist"""
    landmarks = create_base_hand()
    for finger in range(5):
        landmarks = curl_finger(landmarks, finger, 1.0)
    return landmarks

def create_sign_pattern(sign):
    """Create landmark pattern for a specific sign"""
    base = create_base_hand()
    
    # ASL Alphabet patterns
    patterns = {
        # Letters
        'A': lambda: curl_all_except_thumb(base),  # Fist with thumb on side
        'B': lambda: flat_hand_thumb_tucked(base),  # Flat hand
        'C': lambda: curved_hand(base),  # C shape
        'D': lambda: d_shape(base),  # Index up, others touch thumb
        'E': lambda: curled_fingers_thumb_tucked(base),  # All fingers curled
        'F': lambda: f_shape(base),  # Index+thumb circle, others up
        'G': lambda: pointing_sideways(base),  # Index+thumb sideways
        'H': lambda: h_shape(base),  # Index+middle sideways
        'I': lambda: pinky_up(base),  # Only pinky up
        'J': lambda: pinky_up(base),  # Pinky up (J has motion)
        'K': lambda: k_shape(base),  # Index+middle up, thumb between
        'L': lambda: l_shape(base),  # L shape
        'M': lambda: m_shape(base),  # Three fingers over thumb
        'N': lambda: n_shape(base),  # Two fingers over thumb
        'O': lambda: o_shape(base),  # All fingers touch thumb
        'P': lambda: p_shape(base),  # Like K but pointing down
        'Q': lambda: q_shape(base),  # Like G but pointing down
        'R': lambda: r_shape(base),  # Index+middle crossed
        'S': lambda: create_fist(),  # Fist with thumb over fingers
        'T': lambda: t_shape(base),  # Thumb between index+middle
        'U': lambda: u_shape(base),  # Index+middle up together
        'V': lambda: v_shape(base),  # Index+middle spread (peace)
        'W': lambda: w_shape(base),  # Three fingers up spread
        'X': lambda: x_shape(base),  # Index bent like hook
        'Y': lambda: y_shape(base),  # Thumb+pinky out
        'Z': lambda: index_up(base),  # Index up (Z has motion)
        
        # Numbers
        '0': lambda: o_shape(base),  # Same as O
        '1': lambda: index_up(base),  # Index up
        '2': lambda: v_shape(base),  # Two fingers
        '3': lambda: three_shape(base),  # Thumb+index+middle
        '4': lambda: four_shape(base),  # Four fingers up
        '5': lambda: create_base_hand(),  # Open hand
        '6': lambda: six_shape(base),  # Thumb+pinky touch
        '7': lambda: seven_shape(base),  # Thumb+ring touch
        '8': lambda: eight_shape(base),  # Thumb+middle touch
        '9': lambda: nine_shape(base),  # Thumb+index touch, others up
        
        # Common words
        'hello': lambda: create_base_hand(),  # Open palm wave
        'goodbye': lambda: create_base_hand(),  # Wave
        'yes': lambda: create_fist(),  # Fist nod
        'no': lambda: n_shape(base),  # Index+middle+thumb pinch
        'thanks': lambda: flat_hand_from_chin(base),  # Flat hand
        'please': lambda: flat_on_chest(base),  # Flat hand circular
        'sorry': lambda: create_fist(),  # Fist on chest
        'help': lambda: thumbs_up_on_palm(base),  # Fist on palm
        'ok': lambda: f_shape(base),  # OK sign
        'good': lambda: flat_hand_from_chin(base),  # Flat from chin
        'bad': lambda: flat_hand_from_chin(base),  # Flat from chin down
        'hi': lambda: create_base_hand(),  # Wave
        'bye': lambda: create_base_hand(),  # Wave
        'welcome': lambda: create_base_hand(),  # Open gesture
        'maybe': lambda: create_base_hand(),  # Flat hands tilt
        'see_you': lambda: v_shape(base),  # V from eyes
        'take_care': lambda: create_fist(),  # Both fists
        'good_morning': lambda: flat_hand_from_chin(base),
        'good_night': lambda: flat_hand_from_chin(base),
        'nice_to_meet': lambda: create_base_hand(),
    }
    
    if sign in patterns:
        return patterns[sign]()
    else:
        # Default to base hand with slight variation
        return add_noise(base, 0.05)

# Helper functions for sign patterns
def curl_all_except_thumb(base):
    landmarks = base.copy()
    for finger in [1, 2, 3, 4]:  # All except thumb
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def flat_hand_thumb_tucked(base):
    landmarks = base.copy().reshape(21, 3)
    # Tuck thumb across palm
    landmarks[1:5, 0] += 0.15
    landmarks[1:5, 2] += 0.05
    return landmarks.flatten()

def curved_hand(base):
    landmarks = base.copy().reshape(21, 3)
    # Curve all fingers
    for i in range(5, 21):
        landmarks[i, 2] += 0.08
        landmarks[i, 0] -= 0.03
    return landmarks.flatten()

def d_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)  # Index up
    for finger in [2, 3, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    # Thumb touches curled fingers
    landmarks = landmarks.reshape(21, 3)
    landmarks[4, 0] = 0.35
    landmarks[4, 1] = 0.55
    return landmarks.flatten()

def f_shape(base):
    landmarks = base.copy()
    # Index touches thumb, others up
    landmarks = landmarks.reshape(21, 3)
    landmarks[8, 0] = landmarks[4, 0]  # Index tip to thumb tip
    landmarks[8, 1] = landmarks[4, 1]
    return landmarks.flatten()

def pointing_sideways(base):
    landmarks = base.copy().reshape(21, 3)
    # Rotate hand sideways
    landmarks[:, 0] -= 0.1
    landmarks[5:9, 1] = landmarks[5:9, 1].mean()  # Index horizontal
    return landmarks.flatten()

def curled_fingers_thumb_tucked(base):
    """E sign - all fingers curled, thumb tucked"""
    landmarks = create_fist()
    landmarks = landmarks.reshape(21, 3)
    landmarks[4, 0] = 0.40  # Thumb tucked under
    landmarks[4, 1] = 0.70
    return landmarks.flatten()

def h_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)  # Index
    landmarks = extend_finger(landmarks, 2)  # Middle
    for finger in [0, 3, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def pinky_up(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 4)  # Pinky up
    for finger in [0, 1, 2, 3]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def k_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)  # Index
    landmarks = extend_finger(landmarks, 2)  # Middle
    # Spread them
    landmarks = landmarks.reshape(21, 3)
    landmarks[5:9, 0] -= 0.05
    landmarks[9:13, 0] += 0.05
    return landmarks.flatten()

def l_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 0)  # Thumb out
    landmarks = extend_finger(landmarks, 1)  # Index up
    for finger in [2, 3, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def m_shape(base):
    landmarks = create_fist()
    landmarks = landmarks.reshape(21, 3)
    # Thumb under three fingers
    landmarks[4, 1] = 0.72
    return landmarks.flatten()

def n_shape(base):
    landmarks = create_fist()
    landmarks = landmarks.reshape(21, 3)
    # Thumb under two fingers
    landmarks[4, 1] = 0.70
    landmarks[4, 0] = 0.40
    return landmarks.flatten()

def o_shape(base):
    landmarks = base.copy().reshape(21, 3)
    # All fingertips touch thumb tip
    thumb_tip = landmarks[4].copy()
    for tip in [8, 12, 16, 20]:
        landmarks[tip] = thumb_tip + np.random.randn(3) * 0.02
    return landmarks.flatten()

def p_shape(base):
    landmarks = k_shape(base).reshape(21, 3)
    # Point downward
    landmarks[:, 1] = 1 - landmarks[:, 1]
    return landmarks.flatten()

def q_shape(base):
    landmarks = pointing_sideways(base).reshape(21, 3)
    landmarks[:, 1] += 0.2
    return landmarks.flatten()

def r_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)
    landmarks = extend_finger(landmarks, 2)
    # Cross index over middle
    landmarks = landmarks.reshape(21, 3)
    landmarks[5:9, 0] += 0.03
    landmarks[9:13, 0] -= 0.03
    return landmarks.flatten()

def t_shape(base):
    landmarks = create_fist()
    landmarks = landmarks.reshape(21, 3)
    # Thumb between index and middle
    landmarks[4, 0] = 0.45
    landmarks[4, 1] = 0.62
    return landmarks.flatten()

def u_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)
    landmarks = extend_finger(landmarks, 2)
    for finger in [0, 3, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    # Keep together
    landmarks = landmarks.reshape(21, 3)
    landmarks[5:9, 0] = (landmarks[5:9, 0] + landmarks[9:13, 0]) / 2 - 0.02
    landmarks[9:13, 0] = (landmarks[5:9, 0] + landmarks[9:13, 0]) / 2 + 0.02
    return landmarks.flatten()

def v_shape(base):
    landmarks = u_shape(base).reshape(21, 3)
    # Spread index and middle
    landmarks[5:9, 0] -= 0.08
    landmarks[9:13, 0] += 0.08
    return landmarks.flatten()

def w_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)
    landmarks = extend_finger(landmarks, 2)
    landmarks = extend_finger(landmarks, 3)
    for finger in [0, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    # Spread them
    landmarks = landmarks.reshape(21, 3)
    landmarks[5:9, 0] -= 0.06
    landmarks[13:17, 0] += 0.06
    return landmarks.flatten()

def x_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)
    # Bend index like hook
    landmarks = landmarks.reshape(21, 3)
    landmarks[7, 1] += 0.1
    landmarks[8, 1] += 0.15
    for finger in [0, 2, 3, 4]:
        landmarks = curl_finger(landmarks.flatten(), finger, 0.9)
        landmarks = landmarks.reshape(21, 3)
    return landmarks.flatten()

def y_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 0)  # Thumb out
    landmarks = extend_finger(landmarks, 4)  # Pinky out
    for finger in [1, 2, 3]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def index_up(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 1)
    for finger in [0, 2, 3, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def three_shape(base):
    landmarks = base.copy()
    landmarks = extend_finger(landmarks, 0)  # Thumb
    landmarks = extend_finger(landmarks, 1)  # Index
    landmarks = extend_finger(landmarks, 2)  # Middle
    for finger in [3, 4]:
        landmarks = curl_finger(landmarks, finger, 0.9)
    return landmarks

def four_shape(base):
    landmarks = base.copy()
    for finger in [1, 2, 3, 4]:
        landmarks = extend_finger(landmarks, finger)
    landmarks = curl_finger(landmarks, 0, 0.9)  # Thumb in
    return landmarks

def six_shape(base):
    landmarks = base.copy().reshape(21, 3)
    # Thumb touches pinky
    landmarks[4] = landmarks[20] + np.array([0.02, 0, 0])
    return landmarks.flatten()

def seven_shape(base):
    landmarks = base.copy().reshape(21, 3)
    # Thumb touches ring
    landmarks[4] = landmarks[16] + np.array([0.02, 0, 0])
    return landmarks.flatten()

def eight_shape(base):
    landmarks = base.copy().reshape(21, 3)
    # Thumb touches middle
    landmarks[4] = landmarks[12] + np.array([0.02, 0, 0])
    return landmarks.flatten()

def nine_shape(base):
    landmarks = base.copy().reshape(21, 3)
    # Thumb touches index, others up
    landmarks[4] = landmarks[8] + np.array([0.02, 0, 0])
    return landmarks.flatten()

def flat_hand_from_chin(base):
    return base.copy()

def flat_on_chest(base):
    return base.copy()

def thumbs_up_on_palm(base):
    landmarks = create_fist()
    landmarks = extend_finger(landmarks, 0)  # Thumb up
    return landmarks

def add_noise(landmarks, noise_level=0.03):
    """Add random noise to landmarks"""
    noise = np.random.randn(len(landmarks)) * noise_level
    return landmarks + noise

def generate_dataset(signs, samples_per_sign=100):
    """Generate complete dataset"""
    data = []
    
    for sign in signs:
        print(f"Generating {samples_per_sign} samples for '{sign}'...")
        
        for i in range(samples_per_sign):
            # Get base pattern for sign
            pattern = create_sign_pattern(sign)
            
            # Add realistic variation
            noise_level = np.random.uniform(0.02, 0.06)
            landmarks = add_noise(pattern, noise_level)
            
            # Random small rotation/scale
            landmarks = landmarks.reshape(21, 3)
            
            # Small rotation
            angle = np.random.uniform(-0.1, 0.1)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_new = landmarks[:, 0] * cos_a - landmarks[:, 1] * sin_a
            y_new = landmarks[:, 0] * sin_a + landmarks[:, 1] * cos_a
            landmarks[:, 0] = x_new
            landmarks[:, 1] = y_new
            
            # Small scale variation
            scale = np.random.uniform(0.9, 1.1)
            landmarks *= scale
            
            # Small translation
            landmarks[:, 0] += np.random.uniform(-0.05, 0.05)
            landmarks[:, 1] += np.random.uniform(-0.05, 0.05)
            
            landmarks = landmarks.flatten()
            
            # Create two-hand feature vector (126 features)
            # For single hand signs, second hand is zeros or mirrored
            if np.random.random() > 0.3:
                # Just one hand
                full_features = np.concatenate([landmarks, np.zeros(63)])
            else:
                # Mirror for second hand
                landmarks2 = landmarks.copy().reshape(21, 3)
                landmarks2[:, 0] = 1 - landmarks2[:, 0]  # Mirror x
                full_features = np.concatenate([landmarks, landmarks2.flatten()])
            
            # Add small noise to full features
            full_features = add_noise(full_features, 0.01)
            
            # Normalize to 0-1
            full_features = np.clip(full_features, 0, 1)
            
            data.append({
                'sign': sign,
                **{f'feature_{j}': full_features[j] for j in range(126)},
                'timestamp': datetime.now().isoformat()
            })
    
    return pd.DataFrame(data)

def main():
    # Signs to generate
    signs = [
        # Alphabet
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        # Numbers
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # Common words
        'hello', 'goodbye', 'yes', 'no', 'thanks', 'please',
        'sorry', 'help', 'ok', 'good', 'bad', 'hi', 'bye',
        'welcome'
    ]
    
    print(f"Generating realistic dataset for {len(signs)} signs...")
    
    # Generate more samples for better training
    df = generate_dataset(signs, samples_per_sign=150)
    
    # Save
    output_path = os.path.join(
        os.path.dirname(__file__), 
        'data', 'raw', 'sign_dataset.csv'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Signs: {len(signs)}")
    print(f"   Features: 126")
    
    # Show distribution
    print("\n📊 Samples per sign:")
    print(df['sign'].value_counts().head(10))

if __name__ == "__main__":
    main()
