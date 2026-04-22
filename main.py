import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from historical_data import load_and_preprocess_data
from image_generation import generate_all_images
from feature_extractor_hog import extract_candlestick_features
from feature_extractor_vit import extract_vit_features

def build_multi_horizon_targets(df, window_size, horizons=[1, 3, 7, 10]):
    """
    Constructs Buy/Sell/Hold targets for multiple future horizons.
    For simplicity: 1 if future price > current close, else 0 (Up/Down).
    """
    targets = []
    # Limit length so we have room for the max horizon in the future
    max_h = max(horizons)
    valid_len = len(df) - window_size - max_h
    
    for i in range(valid_len):
        current_close = df.iloc[i + window_size - 1]['Close']
        target_row = []
        for h in horizons:
            future_close = df.iloc[i + window_size - 1 + h]['Close']
            # Simple binary movement: 1 (Up) vs 0 (Down)
            target_row.append(1 if future_close > current_close else 0)
        targets.append(target_row)
        
    return np.array(targets), valid_len

def create_multi_horizon_model(input_dim, num_horizons=4):
    """
    Creates a simple Dense network for multi-horizon Stock Classification.
    Outputs independent predictions for 1/3/7/10 day intervals.
    """
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    
    # We output a sigmoid probability for each horizon (multi-label classification)
    outputs = Dense(num_horizons, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # ------------------------------------------------------------------
    # Step 1: Load and Preprocess Historical Data
    # ------------------------------------------------------------------
    # Default target set to 000001.SS.csv
    # For reference, a suitable starting point is the NYSE US 100 index:
    # https://www.investing.com/indices/nyse-us-100
    csv_path = '000001.SS.csv' 
    
    try:
        df = load_and_preprocess_data(csv_path)
    except FileNotFoundError:
        print(f"Dataset {csv_path} not found. Please place it in the directory.")
        return

    # ------------------------------------------------------------------
    # Step 2: Extract Targets (Multi-Horizon: 1, 3, 7, 10 days)
    # ------------------------------------------------------------------
    window_size = 5
    horizons = [1, 3, 7, 10]
    Y_targets, valid_len = build_multi_horizon_targets(df, window_size, horizons)
    
    # ------------------------------------------------------------------
    # Step 3: Generate Candlestick Images from Time-Series data
    # ------------------------------------------------------------------
    # We only generate images up to valid_len to match our target dimensions
    df_valid = df.iloc[:valid_len + window_size]
    image_paths = generate_all_images(df_valid, window_size=window_size, output_dir='./candlestick_images')

    # ------------------------------------------------------------------
    # Step 4: Extract Visual & Engineered Features
    # ------------------------------------------------------------------
    if len(image_paths) == 0:
        print("No images generated. Adjust window size or dataset.")
        return

    print("Extracting explicit Candlestick features...")
    candlestick_features = extract_candlestick_features(image_paths)
    
    print("Extracting deep ViT features...")
    visual_tokens = extract_vit_features(image_paths)

    # ------------------------------------------------------------------
    # Step 5: Extract Temporal Sequence Features (for the TFT model)
    # ------------------------------------------------------------------
    print("Structuring Historical Sequence Features for TFT...")
    historical_features = []
    
    for i in range(valid_len):
        subset = df.iloc[i : i + window_size].values.flatten()
        historical_features.append(subset)
        
    historical_features = np.array(historical_features)

    # ------------------------------------------------------------------
    # Step 6: Multi-Modal Fusion and Training
    # ------------------------------------------------------------------
    print("Combining Multi-Modal Representations...")
    
    X_combined = np.concatenate([
        historical_features,   # Processed via TFT context
        visual_tokens,         # Processed via ViT
        candlestick_features   # Extracted explicitly
    ], axis=1)
    
    print(f"Final Multi-Modal Input Shape: {X_combined.shape}")
    print(f"Target Output Shape: {Y_targets.shape}")
    
    print("Initializing Multi-Horizon Predictor Model...")
    model = create_multi_horizon_model(input_dim=X_combined.shape[1], num_horizons=len(horizons))
    
    print("Commencing TimeSeriesSplit Model Training Phase...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_combined)):
        print(f"\n--- Training Fold {fold + 1} ---")
        X_train, X_val = X_combined[train_index], X_combined[val_index]
        Y_train, Y_val = Y_targets[train_index], Y_targets[val_index]
        
        # Reset model weights here if treating folds independently
        # For demonstration context, we continue training
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=5,
            batch_size=32,
            verbose=1
        )
    print("Multi-horizon training walkthrough completed!")

if __name__ == "__main__":
    main()
