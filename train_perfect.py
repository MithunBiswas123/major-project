"""
PERFECT PRE-TRAINED SIGN LANGUAGE MODEL
Generates anatomically accurate hand landmark data that matches MediaPipe output
"""

import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
SCALER_DIR = os.path.join(BASE_DIR, 'models', 'scalers')

# All signs
SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'hello', 'yes', 'no', 'ok', 'thank_you'
]

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def create_base_hand(center_x=0.5, center_y=0.5, scale=0.15):
    """Create a base open hand with anatomically correct proportions"""
    landmarks = np.zeros((21, 3))
    
    # Wrist at center
    landmarks[WRIST] = [center_x, center_y + scale * 1.5, 0]
    
    # Palm base points (MCP joints) - spread horizontally
    palm_y = center_y + scale * 0.5
    landmarks[THUMB_CMC] = [center_x - scale * 0.8, palm_y + scale * 0.3, 0]
    landmarks[INDEX_MCP] = [center_x - scale * 0.5, palm_y - scale * 0.2, 0]
    landmarks[MIDDLE_MCP] = [center_x - scale * 0.15, palm_y - scale * 0.3, 0]
    landmarks[RING_MCP] = [center_x + scale * 0.2, palm_y - scale * 0.2, 0]
    landmarks[PINKY_MCP] = [center_x + scale * 0.5, palm_y, 0]
    
    return landmarks, center_x, center_y, scale


def set_finger_extended(landmarks, finger_mcp, direction_x, direction_y, length=0.3, scale=0.15):
    """Set a finger to extended position"""
    mcp = landmarks[finger_mcp]
    pip_idx = finger_mcp + 1
    dip_idx = finger_mcp + 2
    tip_idx = finger_mcp + 3
    
    segment = length * scale / 3
    landmarks[pip_idx] = [mcp[0] + direction_x * segment, mcp[1] + direction_y * segment, mcp[2] - 0.02]
    landmarks[dip_idx] = [mcp[0] + direction_x * segment * 2, mcp[1] + direction_y * segment * 2, mcp[2] - 0.03]
    landmarks[tip_idx] = [mcp[0] + direction_x * segment * 3, mcp[1] + direction_y * segment * 3, mcp[2] - 0.04]


def set_finger_curled(landmarks, finger_mcp, scale=0.15):
    """Set a finger to curled/closed position"""
    mcp = landmarks[finger_mcp]
    pip_idx = finger_mcp + 1
    dip_idx = finger_mcp + 2
    tip_idx = finger_mcp + 3
    
    # Curl towards palm
    landmarks[pip_idx] = [mcp[0], mcp[1] + scale * 0.1, mcp[2] + 0.03]
    landmarks[dip_idx] = [mcp[0], mcp[1] + scale * 0.15, mcp[2] + 0.05]
    landmarks[tip_idx] = [mcp[0], mcp[1] + scale * 0.12, mcp[2] + 0.06]


def set_thumb_out(landmarks, center_x, scale):
    """Thumb extended outward"""
    landmarks[THUMB_MCP] = [center_x - scale * 1.0, landmarks[THUMB_CMC][1] - scale * 0.2, 0.01]
    landmarks[THUMB_IP] = [center_x - scale * 1.2, landmarks[THUMB_CMC][1] - scale * 0.4, 0.02]
    landmarks[THUMB_TIP] = [center_x - scale * 1.3, landmarks[THUMB_CMC][1] - scale * 0.5, 0.03]


def set_thumb_across(landmarks, center_x, center_y, scale):
    """Thumb across palm (like in A, S, etc.)"""
    landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y + scale * 0.3, 0.04]
    landmarks[THUMB_IP] = [center_x - scale * 0.2, center_y + scale * 0.2, 0.05]
    landmarks[THUMB_TIP] = [center_x, center_y + scale * 0.15, 0.06]


def set_thumb_up(landmarks, center_x, scale):
    """Thumb pointing up"""
    base = landmarks[THUMB_CMC]
    landmarks[THUMB_MCP] = [base[0] - scale * 0.3, base[1] - scale * 0.3, 0.01]
    landmarks[THUMB_IP] = [base[0] - scale * 0.4, base[1] - scale * 0.5, 0.02]
    landmarks[THUMB_TIP] = [base[0] - scale * 0.45, base[1] - scale * 0.7, 0.03]


def set_thumb_tucked(landmarks, center_x, center_y, scale):
    """Thumb tucked in (like in fist)"""
    landmarks[THUMB_MCP] = [center_x - scale * 0.4, center_y + scale * 0.4, 0.05]
    landmarks[THUMB_IP] = [center_x - scale * 0.2, center_y + scale * 0.35, 0.06]
    landmarks[THUMB_TIP] = [center_x - scale * 0.05, center_y + scale * 0.3, 0.07]


def generate_sign_landmarks(sign, variation=0):
    """Generate landmarks for a specific sign with variations"""
    # Random variations for realism
    cx = 0.5 + np.random.uniform(-0.15, 0.15)
    cy = 0.5 + np.random.uniform(-0.15, 0.15)
    sc = 0.15 + np.random.uniform(-0.03, 0.03)
    
    landmarks, center_x, center_y, scale = create_base_hand(cx, cy, sc)
    
    # Define each sign's hand shape
    if sign == 'A':
        # Fist with thumb on side
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_across(landmarks, center_x, center_y, scale)
        
    elif sign == 'B':
        # Flat hand, fingers up, thumb across
        set_finger_extended(landmarks, INDEX_MCP, -0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.05, -1, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.1, -1, 0.32, scale)
        set_thumb_across(landmarks, center_x, center_y, scale)
        
    elif sign == 'C':
        # Curved hand like C shape
        set_finger_extended(landmarks, INDEX_MCP, 0.3, -0.8, 0.25, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.35, -0.7, 0.28, scale)
        set_finger_extended(landmarks, RING_MCP, 0.4, -0.6, 0.25, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.45, -0.5, 0.22, scale)
        set_thumb_out(landmarks, center_x, scale)
        
    elif sign == 'D':
        # Index up, others curled touching thumb
        set_finger_extended(landmarks, INDEX_MCP, 0, -1, 0.35, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'E':
        # All fingers curled
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'F':
        # Thumb and index make circle, others up
        # Index curled to meet thumb
        landmarks[INDEX_PIP] = [center_x - scale * 0.3, center_y, 0.03]
        landmarks[INDEX_DIP] = [center_x - scale * 0.4, center_y + scale * 0.1, 0.04]
        landmarks[INDEX_TIP] = [center_x - scale * 0.45, center_y + scale * 0.2, 0.05]
        set_finger_extended(landmarks, MIDDLE_MCP, 0, -1, 0.35, scale)
        set_finger_extended(landmarks, RING_MCP, 0.05, -1, 0.33, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.1, -1, 0.30, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y + scale * 0.1, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.45, center_y + scale * 0.15, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.4, center_y + scale * 0.2, 0.05]
        
    elif sign == 'G':
        # Index and thumb pointing sideways
        set_finger_extended(landmarks, INDEX_MCP, -0.9, -0.2, 0.35, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.7, center_y - scale * 0.1, 0.02]
        landmarks[THUMB_IP] = [center_x - scale * 0.9, center_y - scale * 0.15, 0.03]
        landmarks[THUMB_TIP] = [center_x - scale * 1.1, center_y - scale * 0.2, 0.04]
        
    elif sign == 'H':
        # Index and middle extended horizontally
        set_finger_extended(landmarks, INDEX_MCP, -0.9, -0.2, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, -0.85, -0.25, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'I':
        # Pinky up only
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.1, -1, 0.30, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'J':
        # Like I but with motion (pinky up, curved path)
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.2, -0.9, 0.30, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'K':
        # Index and middle up in V, thumb between
        set_finger_extended(landmarks, INDEX_MCP, -0.2, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.2, -1, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.3, center_y - scale * 0.1, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.2, center_y - scale * 0.3, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.1, center_y - scale * 0.4, 0.05]
        
    elif sign == 'L':
        # L shape - index up, thumb out
        set_finger_extended(landmarks, INDEX_MCP, 0, -1, 0.35, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_out(landmarks, center_x, scale)
        
    elif sign == 'M':
        # Three fingers over thumb
        landmarks[INDEX_PIP] = [center_x - scale * 0.4, center_y + scale * 0.2, 0.04]
        landmarks[INDEX_DIP] = [center_x - scale * 0.35, center_y + scale * 0.35, 0.05]
        landmarks[INDEX_TIP] = [center_x - scale * 0.3, center_y + scale * 0.4, 0.06]
        landmarks[MIDDLE_PIP] = [center_x - scale * 0.15, center_y + scale * 0.2, 0.04]
        landmarks[MIDDLE_DIP] = [center_x - scale * 0.1, center_y + scale * 0.35, 0.05]
        landmarks[MIDDLE_TIP] = [center_x - scale * 0.05, center_y + scale * 0.4, 0.06]
        landmarks[RING_PIP] = [center_x + scale * 0.1, center_y + scale * 0.2, 0.04]
        landmarks[RING_DIP] = [center_x + scale * 0.15, center_y + scale * 0.35, 0.05]
        landmarks[RING_TIP] = [center_x + scale * 0.2, center_y + scale * 0.4, 0.06]
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'N':
        # Two fingers over thumb
        landmarks[INDEX_PIP] = [center_x - scale * 0.35, center_y + scale * 0.2, 0.04]
        landmarks[INDEX_DIP] = [center_x - scale * 0.3, center_y + scale * 0.35, 0.05]
        landmarks[INDEX_TIP] = [center_x - scale * 0.25, center_y + scale * 0.4, 0.06]
        landmarks[MIDDLE_PIP] = [center_x - scale * 0.1, center_y + scale * 0.2, 0.04]
        landmarks[MIDDLE_DIP] = [center_x - scale * 0.05, center_y + scale * 0.35, 0.05]
        landmarks[MIDDLE_TIP] = [center_x, center_y + scale * 0.4, 0.06]
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'O':
        # All fingers curved to meet thumb (O shape)
        for finger in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]:
            offset = (finger - 5) * 0.05
            landmarks[finger + 1] = [center_x + offset, center_y - scale * 0.1, 0.03]
            landmarks[finger + 2] = [center_x + offset * 0.5, center_y + scale * 0.1, 0.05]
            landmarks[finger + 3] = [center_x - scale * 0.2, center_y + scale * 0.15, 0.06]
        landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.35, center_y + scale * 0.1, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.2, center_y + scale * 0.15, 0.05]
        
    elif sign == 'P':
        # Like K but pointing down
        set_finger_extended(landmarks, INDEX_MCP, -0.3, 0.9, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.1, 0.95, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y + scale * 0.2, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.6, center_y + scale * 0.4, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.65, center_y + scale * 0.55, 0.05]
        
    elif sign == 'Q':
        # Like G but pointing down
        set_finger_extended(landmarks, INDEX_MCP, -0.2, 0.95, 0.35, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y + scale * 0.2, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.55, center_y + scale * 0.4, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.6, center_y + scale * 0.55, 0.05]
        
    elif sign == 'R':
        # Index and middle crossed
        set_finger_extended(landmarks, INDEX_MCP, 0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, -0.1, -1, 0.35, scale)
        # Cross them
        landmarks[INDEX_TIP][0] += scale * 0.15
        landmarks[MIDDLE_TIP][0] -= scale * 0.15
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'S':
        # Fist with thumb over fingers
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.3, center_y + scale * 0.15, 0.06]
        landmarks[THUMB_IP] = [center_x - scale * 0.1, center_y + scale * 0.1, 0.07]
        landmarks[THUMB_TIP] = [center_x + scale * 0.05, center_y + scale * 0.08, 0.08]
        
    elif sign == 'T':
        # Thumb between index and middle
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.35, center_y + scale * 0.05, 0.04]
        landmarks[THUMB_IP] = [center_x - scale * 0.25, center_y - scale * 0.05, 0.05]
        landmarks[THUMB_TIP] = [center_x - scale * 0.15, center_y - scale * 0.1, 0.06]
        
    elif sign == 'U':
        # Index and middle up together
        set_finger_extended(landmarks, INDEX_MCP, -0.05, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.05, -1, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'V':
        # Peace sign - index and middle spread
        set_finger_extended(landmarks, INDEX_MCP, -0.25, -0.95, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.25, -0.95, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'W':
        # Three fingers up spread
        set_finger_extended(landmarks, INDEX_MCP, -0.25, -0.95, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.25, -0.95, 0.35, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'X':
        # Index bent like hook
        landmarks[INDEX_PIP] = [center_x - scale * 0.3, center_y - scale * 0.2, 0.02]
        landmarks[INDEX_DIP] = [center_x - scale * 0.25, center_y - scale * 0.05, 0.04]
        landmarks[INDEX_TIP] = [center_x - scale * 0.3, center_y + scale * 0.05, 0.05]
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == 'Y':
        # Thumb and pinky out (hang loose)
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.3, -0.9, 0.30, scale)
        set_thumb_out(landmarks, center_x, scale)
        
    elif sign == 'Z':
        # Index traces Z (pointing position)
        set_finger_extended(landmarks, INDEX_MCP, -0.1, -0.95, 0.35, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    # NUMBERS
    elif sign == '0':
        # Like O - all fingers curved to thumb
        for finger in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]:
            offset = (finger - 5) * 0.04
            landmarks[finger + 1] = [center_x + offset, center_y - scale * 0.05, 0.03]
            landmarks[finger + 2] = [center_x + offset * 0.3, center_y + scale * 0.12, 0.05]
            landmarks[finger + 3] = [center_x - scale * 0.15, center_y + scale * 0.18, 0.06]
        landmarks[THUMB_MCP] = [center_x - scale * 0.45, center_y + scale * 0.05, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.3, center_y + scale * 0.12, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.15, center_y + scale * 0.18, 0.05]
        
    elif sign == '1':
        # Index up only
        set_finger_extended(landmarks, INDEX_MCP, 0, -1, 0.35, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == '2':
        # Index and middle up spread (like V)
        set_finger_extended(landmarks, INDEX_MCP, -0.2, -0.95, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.2, -0.95, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == '3':
        # Thumb, index, middle up
        set_finger_extended(landmarks, INDEX_MCP, -0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.1, -1, 0.35, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_up(landmarks, center_x, scale)
        
    elif sign == '4':
        # Four fingers up, thumb tucked
        set_finger_extended(landmarks, INDEX_MCP, -0.15, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, -0.05, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.05, -1, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.15, -1, 0.32, scale)
        set_thumb_tucked(landmarks, center_x, center_y, scale)
        
    elif sign == '5':
        # All five fingers spread
        set_finger_extended(landmarks, INDEX_MCP, -0.25, -0.95, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, -0.08, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.08, -0.98, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.25, -0.92, 0.32, scale)
        set_thumb_out(landmarks, center_x, scale)
        
    elif sign == '6':
        # Thumb and pinky touching, others up - OR - thumb touches pinky
        set_finger_extended(landmarks, INDEX_MCP, -0.15, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, -0.05, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.05, -1, 0.35, scale)
        # Pinky curled to meet thumb
        landmarks[PINKY_PIP] = [center_x + scale * 0.35, center_y + scale * 0.1, 0.03]
        landmarks[PINKY_DIP] = [center_x + scale * 0.2, center_y + scale * 0.2, 0.05]
        landmarks[PINKY_TIP] = [center_x + scale * 0.05, center_y + scale * 0.25, 0.06]
        landmarks[THUMB_MCP] = [center_x - scale * 0.4, center_y + scale * 0.15, 0.04]
        landmarks[THUMB_IP] = [center_x - scale * 0.2, center_y + scale * 0.2, 0.05]
        landmarks[THUMB_TIP] = [center_x + scale * 0.05, center_y + scale * 0.25, 0.06]
        
    elif sign == '7':
        # Index, middle, thumb up, ring and pinky down
        set_finger_extended(landmarks, INDEX_MCP, -0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.1, -1, 0.35, scale)
        # Ring curled to meet thumb
        landmarks[RING_PIP] = [center_x + scale * 0.15, center_y + scale * 0.1, 0.03]
        landmarks[RING_DIP] = [center_x + scale * 0.05, center_y + scale * 0.2, 0.05]
        landmarks[RING_TIP] = [center_x - scale * 0.05, center_y + scale * 0.25, 0.06]
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.35, center_y + scale * 0.15, 0.04]
        landmarks[THUMB_IP] = [center_x - scale * 0.15, center_y + scale * 0.2, 0.05]
        landmarks[THUMB_TIP] = [center_x - scale * 0.05, center_y + scale * 0.25, 0.06]
        
    elif sign == '8':
        # Index, middle, ring up, thumb and pinky meet
        set_finger_extended(landmarks, INDEX_MCP, -0.15, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0, -1, 0.38, scale)
        # Middle curled to meet thumb
        landmarks[MIDDLE_PIP] = [center_x, center_y + scale * 0.05, 0.03]
        landmarks[MIDDLE_DIP] = [center_x - scale * 0.1, center_y + scale * 0.15, 0.05]
        landmarks[MIDDLE_TIP] = [center_x - scale * 0.15, center_y + scale * 0.22, 0.06]
        set_finger_extended(landmarks, RING_MCP, 0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.2, -0.95, 0.32, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.4, center_y + scale * 0.1, 0.04]
        landmarks[THUMB_IP] = [center_x - scale * 0.25, center_y + scale * 0.18, 0.05]
        landmarks[THUMB_TIP] = [center_x - scale * 0.15, center_y + scale * 0.22, 0.06]
        
    elif sign == '9':
        # Like F - thumb and index make circle, others up
        landmarks[INDEX_PIP] = [center_x - scale * 0.25, center_y - scale * 0.1, 0.03]
        landmarks[INDEX_DIP] = [center_x - scale * 0.35, center_y + scale * 0.05, 0.04]
        landmarks[INDEX_TIP] = [center_x - scale * 0.4, center_y + scale * 0.15, 0.05]
        set_finger_extended(landmarks, MIDDLE_MCP, 0, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.1, -0.98, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.2, -0.95, 0.32, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y + scale * 0.05, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.45, center_y + scale * 0.12, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.4, center_y + scale * 0.15, 0.05]
        
    # WORDS
    elif sign == 'hello':
        # Open hand waving - all fingers spread
        set_finger_extended(landmarks, INDEX_MCP, -0.3, -0.9, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, -0.1, -0.98, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.1, -0.98, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.3, -0.9, 0.32, scale)
        set_thumb_out(landmarks, center_x, scale)
        
    elif sign == 'yes':
        # Fist nodding (like A/S)
        set_finger_curled(landmarks, INDEX_MCP, scale)
        set_finger_curled(landmarks, MIDDLE_MCP, scale)
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        set_thumb_up(landmarks, center_x, scale)
        
    elif sign == 'no':
        # Index and middle pinching with thumb
        landmarks[INDEX_PIP] = [center_x - scale * 0.3, center_y - scale * 0.1, 0.02]
        landmarks[INDEX_DIP] = [center_x - scale * 0.35, center_y + scale * 0.05, 0.03]
        landmarks[INDEX_TIP] = [center_x - scale * 0.3, center_y + scale * 0.15, 0.04]
        landmarks[MIDDLE_PIP] = [center_x - scale * 0.1, center_y - scale * 0.1, 0.02]
        landmarks[MIDDLE_DIP] = [center_x - scale * 0.15, center_y + scale * 0.05, 0.03]
        landmarks[MIDDLE_TIP] = [center_x - scale * 0.2, center_y + scale * 0.15, 0.04]
        set_finger_curled(landmarks, RING_MCP, scale)
        set_finger_curled(landmarks, PINKY_MCP, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.45, center_y + scale * 0.05, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.35, center_y + scale * 0.12, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.25, center_y + scale * 0.15, 0.05]
        
    elif sign == 'ok':
        # Thumb and index circle, others extended
        landmarks[INDEX_PIP] = [center_x - scale * 0.25, center_y - scale * 0.05, 0.03]
        landmarks[INDEX_DIP] = [center_x - scale * 0.35, center_y + scale * 0.1, 0.04]
        landmarks[INDEX_TIP] = [center_x - scale * 0.38, center_y + scale * 0.2, 0.05]
        set_finger_extended(landmarks, MIDDLE_MCP, 0.05, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.15, -0.98, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.25, -0.95, 0.32, scale)
        landmarks[THUMB_MCP] = [center_x - scale * 0.5, center_y + scale * 0.1, 0.03]
        landmarks[THUMB_IP] = [center_x - scale * 0.42, center_y + scale * 0.18, 0.04]
        landmarks[THUMB_TIP] = [center_x - scale * 0.38, center_y + scale * 0.2, 0.05]
        
    elif sign == 'thank_you':
        # Flat hand from chin outward - similar to B
        set_finger_extended(landmarks, INDEX_MCP, 0, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0.05, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.1, -0.98, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.15, -0.95, 0.32, scale)
        set_thumb_across(landmarks, center_x, center_y, scale)
    
    else:
        # Default open hand
        set_finger_extended(landmarks, INDEX_MCP, -0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, MIDDLE_MCP, 0, -1, 0.38, scale)
        set_finger_extended(landmarks, RING_MCP, 0.1, -1, 0.35, scale)
        set_finger_extended(landmarks, PINKY_MCP, 0.2, -1, 0.32, scale)
        set_thumb_out(landmarks, center_x, scale)
    
    # Add small random noise for variation
    noise = np.random.randn(21, 3) * 0.008
    landmarks += noise
    
    # Clip to valid range
    landmarks = np.clip(landmarks, 0, 1)
    
    return landmarks.flatten()


def generate_dataset(samples_per_sign=500):
    """Generate complete dataset"""
    print(f"Generating {samples_per_sign} samples per sign...")
    
    X = []
    y = []
    
    for sign in SIGNS:
        print(f"  Generating '{sign}'...", end=' ')
        for i in range(samples_per_sign):
            # Generate right hand landmarks (main hand for signing)
            right_hand = generate_sign_landmarks(sign, i)
            
            # Left hand is usually empty (zeros) - but sometimes present
            if np.random.random() < 0.1:  # 10% chance both hands visible
                left_hand = generate_sign_landmarks(sign, i + 1000)
            else:
                left_hand = np.zeros(63)
            
            # Combine: [left_hand(63), right_hand(63)] = 126 features
            features = np.concatenate([left_hand, right_hand])
            
            X.append(features)
            y.append(sign)
        print(f"✓")
    
    return np.array(X), np.array(y)


def train_model():
    """Train and save the model"""
    print("\n" + "=" * 60)
    print("🎯 TRAINING PERFECT SIGN LANGUAGE MODEL")
    print("=" * 60)
    
    # Generate data
    X, y = generate_dataset(samples_per_sign=600)
    print(f"\nDataset: {len(X)} samples, {len(SIGNS)} signs")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # To categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Build model - deeper for better accuracy
    print("\nBuilding model...")
    model = Sequential([
        Input(shape=(126,)),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    ]
    
    # Train
    print("\n🚀 Training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Final Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save everything
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(SCALER_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(SCALER_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Model saved to: {MODEL_DIR}")
    print(f"✅ Scaler saved to: {SCALER_DIR}")
    print(f"\nSigns learned ({num_classes}): {list(label_encoder.classes_)}")
    print("\n🎉 Run 'python main.py' to test detection!")
    
    return model, scaler, label_encoder


if __name__ == "__main__":
    train_model()
