# 🤟 Sign Language Detection - Machine Learning Project

A comprehensive real-time sign language detection system using deep learning and computer vision. This project uses MediaPipe for hand landmark detection and TensorFlow/Keras for sign classification.

## 📊 Features

- **111 Sign Language Gestures**
  - A-Z (26 letters)
  - 0-9 (10 numbers)
  - 75 Custom signs (greetings, actions, emotions, etc.)

- **Multiple Neural Network Architectures**
  - CNN (Convolutional Neural Network)
  - LSTM (Long Short-Term Memory)
  - Hybrid (CNN + LSTM + GRU( Gated Recurrent Unit))
  - Transformer-based model

- **Real-time Detection**
  - Live webcam detection
  - Prediction smoothing
  - Confidence thresholding
  - FPS display

## 📁 Project Structure

```
SignLanguageML/
├── data/
│   ├── raw/              # Raw collected data
│   └── processed/        # Preprocessed numpy arrays
├── models/
│   └── saved/            # Trained model files
├── notebooks/            # Jupyter notebooks
├── outputs/
│   ├── plots/            # Training visualizations
│   └── logs/             # Training logs
├── src/
│   ├── __init__.py       # Package init
│   ├── config.py         # Configuration (111 signs)
│   ├── data_collection.py    # Webcam data collector
│   ├── preprocessing.py      # Data preprocessing
│   ├── models.py             # Neural network architectures
│   ├── train.py              # Training pipeline
│   ├── detect.py             # Real-time detection
│   └── utils.py              # Utility functions
├── main.py               # Main application
├── requirements.txt      # Dependencies
├── run.bat               # Windows launcher
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

**Windows:**
```bash
run.bat
```

**Python:**
```bash
python main.py
```

### 3. Workflow

1. **Collect Data** - Use webcam to record hand signs
2. **Preprocess** - Augment and normalize data
3. **Train** - Train your chosen model architecture
4. **Detect** - Run real-time detection

## 📹 Data Collection

The system captures 126 features per frame:
- 21 landmarks × 3 coordinates (x, y, z) × 2 hands

To collect data:
1. Run `python main.py`
2. Select option 3 (Collect Training Data)
3. Follow on-screen instructions
4. Press SPACE to start recording
5. Hold the sign gesture steady

## 🧠 Model Architectures

### CNN Model
- 3 Conv1D blocks with MaxPooling
- Global Average Pooling
- Dense classification head

### LSTM Model
- 2 Bidirectional LSTM layers
- Dense classification head

### Hybrid Model (Recommended)
- CNN branch for spatial features
- LSTM branch for sequential patterns
- GRU branch for additional context
- Concatenated dense layers

### Transformer Model
- Multi-head self-attention
- Feed-forward networks
- Positional information

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Training settings
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Detection settings
CONFIDENCE_THRESHOLD = 0.6
PREDICTION_BUFFER_SIZE = 5

# Data collection
SAMPLES_PER_SIGN = 50
```

## 📊 111 Signs Reference

### Alphabet (A-Z)
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

### Numbers (0-9)
0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### Greetings
Hello, Goodbye, Thank You, Please, Sorry, Welcome, Good Morning, Good Night, Nice to Meet You

### Responses
Yes, No, Maybe, OK, I Don't Know, I Understand, I Don't Understand, Agree, Disagree

### Actions
Eat, Drink, Sleep, Wake Up, Walk, Run, Stop, Go, Come, Wait, Sit, Stand, Help, Work, Play, Learn, Teach, Read, Write, Listen, Speak, Watch, Open, Close, Give, Take

### Emotions
Happy, Sad, Angry, Scared, Surprised, Tired, Excited, Bored, Confused, Proud

### Questions
What, Where, When, Who, Why, How, How Much, How Many

### Time
Now, Later, Before, After, Today, Tomorrow, Yesterday

### People
I/Me, You, He/She, We, They, Friend, Family, Mother, Father, Child

## 🎮 Controls

### Data Collection
- **SPACE** - Start countdown/recording
- **Q** - Quit

### Real-time Detection
- **Q** - Quit
- **SPACE** - Toggle smoothing
- **S** - Save screenshot

## 📈 Training Tips

1. **Collect enough data**: At least 50 samples per sign
2. **Use data augmentation**: Enabled by default
3. **Start with Hybrid model**: Best balance of accuracy and speed
4. **Monitor validation loss**: Stop training if it plateaus

## 🔧 Troubleshooting

### Camera not opening
- Check camera permissions
- Try different camera index (0, 1, 2)
- Install camera drivers

### Low accuracy
- Collect more training data
- Ensure consistent lighting
- Hold signs steady during collection
- Try different model architectures

### Import errors
- Run `pip install -r requirements.txt`
- Check Python version (3.10+ recommended)

## 📄 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for improvements.
