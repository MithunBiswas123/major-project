"""
Neural Network Models for Sign Language Detection
Contains CNN, LSTM, Hybrid, and Transformer architectures
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LSTM, GRU, Bidirectional,
    Attention, MultiHeadAttention, LayerNormalization,
    Reshape, Flatten, concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from .config import (
    TOTAL_FEATURES, NUM_SIGNS, LEARNING_RATE,
    CNN_MODEL_PATH, LSTM_MODEL_PATH, HYBRID_MODEL_PATH, TRANSFORMER_MODEL_PATH
)


def create_cnn_model(num_features=TOTAL_FEATURES, num_classes=NUM_SIGNS, 
                     units=[256, 128, 64], dropout_rate=0.3):
    """
    Create a 1D CNN model for sign classification
    
    Architecture:
    - 3 Conv1D blocks with MaxPooling
    - Global Average Pooling
    - Dense layers with dropout
    """
    
    inputs = Input(shape=(num_features,), name='input_layer')
    
    # Reshape for Conv1D: (batch, timesteps, features)
    x = Reshape((num_features, 1))(inputs)
    
    # Conv Block 1
    x = Conv1D(units[0], kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Conv1D(units[0], kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)
    
    # Conv Block 2
    x = Conv1D(units[1], kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Conv1D(units[1], kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)
    
    # Conv Block 3
    x = Conv1D(units[2], kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs, outputs, name='CNN_SignLanguage')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(num_features=TOTAL_FEATURES, num_classes=NUM_SIGNS,
                      lstm_units=[128, 64], dropout_rate=0.4):
    """
    Create a Bidirectional LSTM model
    
    Architecture:
    - 2 Bidirectional LSTM layers
    - Dense layers with dropout
    """
    
    inputs = Input(shape=(num_features,), name='input_layer')
    
    # Reshape for LSTM: (batch, timesteps, features)
    # Treat each set of 3 coordinates as a timestep
    x = Reshape((42, 3))(inputs)  # 42 landmarks * 3 coords
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True,
                           kernel_regularizer=l2(0.001)))(x)
    x = Dropout(dropout_rate)(x)
    
    x = Bidirectional(LSTM(lstm_units[1], return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs, outputs, name='LSTM_SignLanguage')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_hybrid_model(num_features=TOTAL_FEATURES, num_classes=NUM_SIGNS,
                        cnn_units=[64, 128], lstm_units=64, dropout_rate=0.3):
    """
    Create a hybrid CNN-LSTM model
    
    Architecture:
    - CNN for local feature extraction
    - LSTM for sequential patterns
    - Combined dense layers
    """
    
    inputs = Input(shape=(num_features,), name='input_layer')
    
    # CNN Branch
    x_cnn = Reshape((num_features, 1))(inputs)
    x_cnn = Conv1D(cnn_units[0], kernel_size=3, activation='relu', padding='same')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Conv1D(cnn_units[1], kernel_size=3, activation='relu', padding='same')(x_cnn)
    x_cnn = GlobalAveragePooling1D()(x_cnn)
    
    # LSTM Branch
    x_lstm = Reshape((42, 3))(inputs)  # 42 landmarks * 3 coords
    x_lstm = LSTM(lstm_units, return_sequences=True)(x_lstm)
    x_lstm = LSTM(lstm_units, return_sequences=False)(x_lstm)
    
    # GRU Branch
    x_gru = Reshape((42, 3))(inputs)
    x_gru = GRU(lstm_units, return_sequences=False)(x_gru)
    
    # Concatenate branches
    x = concatenate([x_cnn, x_lstm, x_gru])
    
    # Dense layers
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs, outputs, name='Hybrid_CNN_LSTM_GRU')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_transformer_model(num_features=TOTAL_FEATURES, num_classes=NUM_SIGNS,
                             num_heads=4, ff_dim=128, dropout_rate=0.3):
    """
    Create a Transformer-based model
    
    Architecture:
    - Positional encoding
    - Multi-head self-attention
    - Feed-forward layers
    """
    
    inputs = Input(shape=(num_features,), name='input_layer')
    
    # Reshape to sequence
    x = Reshape((42, 3))(inputs)  # 42 landmarks * 3 coords
    
    # Project to higher dimension
    x = Dense(64)(x)
    
    # Transformer Block 1
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=64//num_heads
    )(x, x)
    x1 = LayerNormalization()(x + attention_output)
    x1 = Dropout(dropout_rate)(x1)
    
    # Feed Forward
    ff = Dense(ff_dim, activation='relu')(x1)
    ff = Dense(64)(ff)
    x2 = LayerNormalization()(x1 + ff)
    
    # Transformer Block 2
    attention_output2 = MultiHeadAttention(
        num_heads=num_heads, key_dim=64//num_heads
    )(x2, x2)
    x3 = LayerNormalization()(x2 + attention_output2)
    x3 = Dropout(dropout_rate)(x3)
    
    ff2 = Dense(ff_dim, activation='relu')(x3)
    ff2 = Dense(64)(ff2)
    x4 = LayerNormalization()(x3 + ff2)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x4)
    
    # Classification head
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs, outputs, name='Transformer_SignLanguage')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_simple_dnn(num_features=TOTAL_FEATURES, num_classes=NUM_SIGNS,
                      units=[512, 256, 128, 64], dropout_rate=0.3):
    """
    Create a simple Deep Neural Network
    
    Good baseline model for comparison
    """
    
    model = Sequential([
        Input(shape=(num_features,), name='input_layer'),
        
        Dense(units[0], activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(units[1], activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(units[2], activation='relu'),
        Dropout(dropout_rate),
        
        Dense(units[3], activation='relu'),
        Dropout(dropout_rate),
        
        Dense(num_classes, activation='softmax', name='output_layer')
    ], name='Simple_DNN')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_path, patience=15):
    """Get training callbacks"""
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def get_model(model_type='hybrid', num_features=TOTAL_FEATURES, num_classes=NUM_SIGNS):
    """Factory function to get model by type"""
    
    models = {
        'cnn': create_cnn_model,
        'lstm': create_lstm_model,
        'hybrid': create_hybrid_model,
        'transformer': create_transformer_model,
        'dnn': create_simple_dnn
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type.lower()](num_features=num_features, num_classes=num_classes)


def get_model_path(model_type):
    """Get model save path by type"""
    paths = {
        'cnn': CNN_MODEL_PATH,
        'lstm': LSTM_MODEL_PATH,
        'hybrid': HYBRID_MODEL_PATH,
        'transformer': TRANSFORMER_MODEL_PATH,
        'dnn': HYBRID_MODEL_PATH.replace('hybrid', 'dnn')
    }
    return paths.get(model_type.lower(), HYBRID_MODEL_PATH)


def print_model_summary(model):
    """Print model summary with parameter count"""
    print("\n" + "=" * 60)
    print(f"📊 MODEL: {model.name}")
    print("=" * 60)
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = total_params - trainable_params
    
    print(f"\n📈 Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Non-trainable: {non_trainable:,}")
    print("=" * 60)


if __name__ == "__main__":
    # Test all models
    print("Testing model architectures...")
    
    for model_type in ['dnn', 'cnn', 'lstm', 'hybrid', 'transformer']:
        print(f"\n{'=' * 40}")
        print(f"Creating {model_type.upper()} model...")
        model = get_model(model_type)
        print(f"✅ {model.name} created successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
