import os
import sys
import numpy as np
import tensorflow as tf

# Ensure the root project directory is added to your Python paths environment lookup context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.cnn_model import build_handwritten_cnn

def execute_model_training():
    print("📦 Step 1: Extracting processed tensor splits from local matrix cache...")
    processed_dir = os.path.join("data", "processed")
    models_out_dir = "saved_models"
    os.makedirs(models_out_dir, exist_ok=True)
    
    # Load standardized data arrays from disk maps
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    print(f"📊 Training Array Stream   : {X_train.shape} | Targets: {y_train.shape}")
    print(f"📊 Validation Array Stream : {X_val.shape} | Targets: {y_val.shape}")
    
    print("\n🏗️ Step 2: Instantiating CNN architectural blueprint...")
    # Initialize our custom neural network structure 
    model = build_handwritten_cnn(input_shape=(28, 28, 1), num_classes=47)
    
    print("⚙️ Step 3: Compiling model execution configurations...")
    # Compile model using adaptive momentum optimization and categorical crossentropy variants
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🎯 Step 4: Structuring production performance monitoring callbacks...")
    # Setup automated model serialization tracks to prevent over-memorization leaks
    checkpoint_path = os.path.join(models_out_dir, "best_handwritten_model.keras")
    
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\n🚀 Step 5: Commencing neural network training optimization parameters...")
    # Configure production hyperparameters for local resource consumption
    BATCH_SIZE = 256
    EPOCHS = 10  # 10 full epochs is perfect for a balanced, solid B.Tech training cycle
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save the secondary training evolution history logs arrays for analysis profiles later
    history_path = os.path.join(models_out_dir, "training_history.npy")
    np.save(history_path, history.history)
    
    print("\n✅ TRAINING COMPLETION SUCCESSFUL!")
    print(f"💾 Absolute production model file written safely to: {os.path.abspath(checkpoint_path)}")
    print(f"📈 Final Achieved Training Accuracy   : {history.history['accuracy'][-1]*100:.2f}%")
    print(f"📉 Final Achieved Validation Accuracy : {history.history['val_accuracy'][-1]*100:.2f}%")

if __name__ == "__main__":
    execute_model_training()