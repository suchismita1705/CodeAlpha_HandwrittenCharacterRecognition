import os
import numpy as np
from sklearn.model_selection import train_test_split

def execute_preprocessing_pipeline():
    print("📦 Step 1: Extracting raw data cache maps from disk storage...")
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load raw NumPy arrays
    X_train_raw = np.load(os.path.join(raw_dir, "X_train.npy"))
    y_train_raw = np.load(os.path.join(raw_dir, "y_train.npy"))
    X_test_raw = np.load(os.path.join(raw_dir, "X_test.npy"))
    y_test_raw = np.load(os.path.join(raw_dir, "y_test.npy"))
    
    print("🔄 Step 2: Correcting structural transposition across all array spaces...")
    # The raw array dimensions are (Samples, Height, Width, Channels) -> (N, 28, 28, 1)
    # We swap Axis 1 (Height) and Axis 2 (Width) using np.transpose to flip images right-side up permanently
    X_train_upright = np.transpose(X_train_raw, axes=(0, 2, 1, 3))
    X_test_upright = np.transpose(X_test_raw, axes=(0, 2, 1, 3))
    
    print("⚖️ Step 3: Normalizing pixel values to a continuous [0.0, 1.0] float spectrum...")
    # Cast matrices to float32 precision and divide by max intensity value (255.0)
    X_train_norm = X_train_upright.astype('float32') / 255.0
    X_test_norm = X_test_upright.astype('float32') / 255.0
    
    print("✂️ Step 4: Carving out an isolated 10% Validation slice from Training space...")
    # Stratify using labels array to ensure class distribution ratios stay perfectly uniform
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_norm, 
        y_train_raw, 
        test_size=0.10, 
        random_state=42, 
        stratify=y_train_raw
    )
    
    print("💾 Step 5: Serializing clean processed array maps to 'data/processed/'...")
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_norm)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test_raw)
    
    print("\n✅ PREPROCESSING PIPELINE SUCCESSFUL!")
    print(f"📊 Processed Training Matrices   : {X_train.shape} | Labels: {y_train.shape}")
    print(f"📊 Processed Validation Matrices : {X_val.shape}  | Labels: {y_val.shape}")
    print(f"📊 Processed Testing Matrices    : {X_test_norm.shape}  | Labels: {y_test_raw.shape}")
    print(f"📉 Min/Max Pixel Bounds Uniformity: {X_train.min()} -> {X_train.max()}")

if __name__ == "__main__":
    execute_preprocessing_pipeline()