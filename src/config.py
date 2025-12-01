"""
Configuration file for Sign Language Detection ML System
Contains all 111 signs and system settings
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')

# Dataset paths
DATASET_CSV = os.path.join(RAW_DATA_DIR, 'sign_dataset.csv')
PROCESSED_X = os.path.join(PROCESSED_DATA_DIR, 'X_data.npy')
PROCESSED_Y = os.path.join(PROCESSED_DATA_DIR, 'y_data.npy')

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, 'sign_model.h5')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.h5')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, 'hybrid_model.h5')
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, 'transformer_model.h5')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# ============================================================================
# 111 SIGN DEFINITIONS
# ============================================================================
SIGNS = {
    # ==================== ALPHABET (A-Z) ====================
    'A': 'Closed fist, thumb on side',
    'B': 'Flat hand, fingers together, thumb across palm',
    'C': 'Curved hand like holding a cup',
    'D': 'Index up, others touch thumb in circle',
    'E': 'Fingers bent down, thumb tucked',
    'F': 'Index and thumb form circle, other fingers up',
    'G': 'Index and thumb pointing sideways',
    'H': 'Index and middle pointing sideways together',
    'I': 'Pinky finger up only',
    'J': 'Pinky up, trace J shape in air',
    'K': 'Index and middle up in V, thumb between them',
    'L': 'L shape with thumb and index finger',
    'M': 'Three fingers over thumb',
    'N': 'Two fingers over thumb',
    'O': 'All fingers curved to touch thumb tip',
    'P': 'Like K but pointing downward',
    'Q': 'Like G but pointing downward',
    'R': 'Index and middle crossed',
    'S': 'Closed fist, thumb over fingers',
    'T': 'Thumb between index and middle finger',
    'U': 'Index and middle together pointing up',
    'V': 'Peace sign - index and middle spread',
    'W': 'Three fingers up and spread (index, middle, ring)',
    'X': 'Index finger hooked/bent',
    'Y': 'Thumb and pinky extended outward',
    'Z': 'Index finger traces Z shape in air',
    
    # ==================== NUMBERS (0-9) ====================
    '0': 'O shape with fingers',
    '1': 'Index finger pointing up',
    '2': 'Index and middle fingers up',
    '3': 'Thumb, index, and middle up',
    '4': 'Four fingers up, thumb folded',
    '5': 'All five fingers spread open',
    '6': 'Thumb touches pinky, others up',
    '7': 'Thumb touches ring finger, others up',
    '8': 'Thumb touches middle finger, others up',
    '9': 'Thumb touches index finger, others up',
    
    # ==================== GREETINGS (10) ====================
    'hello': 'Open hand wave near forehead',
    'hi': 'Quick wave with open hand',
    'bye': 'Open and close hand repeatedly (wave goodbye)',
    'goodbye': 'Wave hand away from body',
    'welcome': 'Open hand gesture toward body',
    'good_morning': 'Flat hand from chin moving up + sun rising gesture',
    'good_night': 'Hands together near cheek (sleeping)',
    'nice_to_meet': 'Both index fingers coming together',
    'see_you': 'Point to eyes then point outward',
    'take_care': 'Cross hands over chest',
    
    # ==================== RESPONSES (10) ====================
    'yes': 'Fist nodding up and down',
    'no': 'Index and middle finger snap to thumb',
    'ok': 'Thumb and index form circle, others up',
    'maybe': 'Flat hands alternating up and down',
    'please': 'Flat hand circular motion on chest',
    'sorry': 'Fist circular motion on chest',
    'thank_you': 'Flat hand from chin moving outward',
    'excuse_me': 'Fingers brush across opposite palm',
    'no_problem': 'Brush off gesture with one hand',
    'you_are_welcome': 'Hand from chest moving outward',
    
    # ==================== ACTIONS (15) ====================
    'stop': 'Flat hand facing outward, palm out',
    'go': 'Both index fingers pointing and moving forward',
    'come': 'Index finger beckoning toward self',
    'wait': 'Open hands, palms down, patting motion',
    'help': 'Thumbs up on flat palm, lifting up',
    'eat': 'Fingers to mouth repeatedly',
    'drink': 'Thumb to mouth, tilting hand like cup',
    'sleep': 'Hand on cheek, head tilting',
    'walk': 'Two fingers walking motion',
    'run': 'Index fingers moving fast alternately',
    'sit': 'Two fingers sitting on other two fingers',
    'stand': 'Two fingers standing on palm',
    'open': 'Both hands together then spreading apart',
    'close': 'Both hands apart then coming together',
    'give': 'Flat hands moving from self outward',
    
    # ==================== EMOTIONS (10) ====================
    'happy': 'Flat hands brushing up chest repeatedly',
    'sad': 'Fingers down face like tears',
    'angry': 'Claw hand at face, pulling outward',
    'scared': 'Both hands up, fingers spread, shaking',
    'surprised': 'Eyes wide, hands up near face opening',
    'tired': 'Both hands on chest, dropping down',
    'hungry': 'Hand moving down from throat to stomach',
    'thirsty': 'Index finger tracing down throat',
    'sick': 'Middle finger on forehead, one on stomach',
    'better': 'Flat hand on chin moving to thumbs up',
    
    # ==================== QUESTIONS (10) ====================
    'what': 'Open hands, palms up, shrugging motion',
    'where': 'Index finger wagging side to side',
    'when': 'Index finger circling then pointing',
    'why': 'Fingers on forehead wiggling outward',
    'how': 'Fists together, knuckles up, rolling open',
    'who': 'Index finger circling near mouth',
    'which': 'Thumbs up alternating up and down',
    'whose': 'Index finger near mouth circling',
    'how_much': 'Claw hands opening upward',
    'how_many': 'Fist opening to spread fingers upward',
    
    # ==================== TIME (10) ====================
    'today': 'Both hands down then up in present motion',
    'tomorrow': 'Thumb on cheek moving forward',
    'yesterday': 'Thumb touching cheek moving backward',
    'now': 'Both bent hands dropping down sharply',
    'later': 'L hand, thumb on palm, tilting forward',
    'morning': 'Flat hand in elbow crook rising up',
    'afternoon': 'Forearm at angle moving downward',
    'evening': 'Forearm horizontal, hand drooping',
    'night': 'Bent hand over wrist (sun setting)',
    'week': 'Index sliding across palm',
    
    # ==================== PEOPLE (10) ====================
    'me': 'Index finger pointing to self/chest',
    'you': 'Index finger pointing outward',
    'he': 'Index finger pointing to side',
    'she': 'Index finger pointing to other side',
    'we': 'Index finger touching shoulder then other shoulder',
    'they': 'Index finger sweeping across',
    'friend': 'Index fingers hooked together',
    'family': 'F hands circling together',
    'mother': 'Open hand, thumb on chin',
    'father': 'Open hand, thumb on forehead',
}

# Sign categories for organized display
SIGN_CATEGORIES = {
    'Alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
    'Numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'Greetings': ['hello', 'hi', 'bye', 'goodbye', 'welcome', 'good_morning',
                  'good_night', 'nice_to_meet', 'see_you', 'take_care'],
    'Responses': ['yes', 'no', 'ok', 'maybe', 'please', 'sorry', 'thank_you',
                  'excuse_me', 'no_problem', 'you_are_welcome'],
    'Actions': ['stop', 'go', 'come', 'wait', 'help', 'eat', 'drink', 'sleep',
                'walk', 'run', 'sit', 'stand', 'open', 'close', 'give'],
    'Emotions': ['happy', 'sad', 'angry', 'scared', 'surprised', 'tired',
                 'hungry', 'thirsty', 'sick', 'better'],
    'Questions': ['what', 'where', 'when', 'why', 'how', 'who', 'which',
                  'whose', 'how_much', 'how_many'],
    'Time': ['today', 'tomorrow', 'yesterday', 'now', 'later', 'morning',
             'afternoon', 'evening', 'night', 'week'],
    'People': ['me', 'you', 'he', 'she', 'we', 'they', 'friend', 'family',
               ]
}

# Get list of all sign labels
SIGN_LABELS = list(SIGNS.keys())
NUM_SIGNS = len(SIGN_LABELS)

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================
SAMPLES_PER_SIGN = 30          # Number of samples to collect per sign
CAPTURE_INTERVAL = 0.15        # Seconds between captures
COUNTDOWN_SECONDS = 3          # Countdown before collecting each sign

# ============================================================================
# MEDIAPIPE SETTINGS
# ============================================================================
MP_MAX_HANDS = 2               # Maximum hands to detect
MP_MIN_DETECTION_CONF = 0.7    # Minimum detection confidence
MP_MIN_TRACKING_CONF = 0.7     # Minimum tracking confidence
MP_MODEL_COMPLEXITY = 1        # Model complexity (0, 1, or 2)

# Hand landmark features
NUM_LANDMARKS = 21             # 21 landmarks per hand
NUM_COORDS = 3                 # x, y, z coordinates
FEATURES_PER_HAND = NUM_LANDMARKS * NUM_COORDS  # 63 features
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126 features (both hands)

# ============================================================================
# MODEL SETTINGS
# ============================================================================
# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Model types available
MODEL_TYPES = ['cnn', 'lstm', 'hybrid', 'transformer']
DEFAULT_MODEL = 'hybrid'

# ============================================================================
# DETECTION SETTINGS
# ============================================================================
CONFIDENCE_THRESHOLD = 0.6     # Minimum confidence for prediction
PREDICTION_BUFFER_SIZE = 5     # Number of frames for smoothing
STABLE_PREDICTIONS_NEEDED = 3  # Consistent predictions needed

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
DISPLAY_COLORS = {
    'primary': (0, 255, 255),    # Yellow
    'success': (0, 255, 0),      # Green
    'warning': (0, 165, 255),    # Orange
    'error': (0, 0, 255),        # Red
    'info': (255, 255, 0),       # Cyan
    'text': (255, 255, 255),     # White
    'background': (50, 50, 50),  # Dark gray
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def ensure_dirs():
    """Create all necessary directories"""
    dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
            MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, LOGS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_sign_description(sign):
    """Get description for a sign"""
    return SIGNS.get(sign, 'Unknown sign')

def get_signs_by_category(category):
    """Get all signs in a category"""
    return SIGN_CATEGORIES.get(category, [])

def print_all_signs():
    """Print all signs organized by category"""
    print("\n" + "=" * 70)
    print(f"{'SIGN LANGUAGE ML - ALL ' + str(NUM_SIGNS) + ' SIGNS':^70}")
    print("=" * 70)
    
    for category, signs in SIGN_CATEGORIES.items():
        print(f"\n📁 {category.upper()} ({len(signs)} signs)")
        print("-" * 50)
        for i, sign in enumerate(signs, 1):
            desc = SIGNS[sign][:40] + '...' if len(SIGNS[sign]) > 40 else SIGNS[sign]
            print(f"   {i:2d}. {sign:15s} - {desc}")
    
    print("\n" + "=" * 70)
    print(f"Total: {NUM_SIGNS} signs")
    print("=" * 70)

# Initialize directories on import
ensure_dirs()
