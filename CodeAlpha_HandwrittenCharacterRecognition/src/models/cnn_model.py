import tensorflow as tf
from tensorflow.keras import layers, models

def build_handwritten_cnn(input_shape=(28, 28, 1), num_classes=47):
    """
    Builds an industry-standard custom Convolutional Neural Network (CNN)
    from scratch for EMNIST handwritten character recognition.
    """
    model = models.Sequential(name="Handwritten_Character_CNN")
    
    # 1. FIRST FEATURE EXTRACTION BLOCK
    # 32 filters of size 3x3 to capture basic edges and structural borders
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="conv_1"))
    model.add(layers.BatchNormalization(name="batch_norm_1"))
    model.add(layers.MaxPooling2D((2, 2), name="max_pool_1"))
    
    # 2. SECOND FEATURE EXTRACTION BLOCK
    # 64 filters of size 3x3 to capture complex strokes, intersections, and loops
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="conv_2"))
    model.add(layers.BatchNormalization(name="batch_norm_2"))
    model.add(layers.MaxPooling2D((2, 2), name="max_pool_2"))
    
    # 3. THIRD FEATURE EXTRACTION BLOCK
    # 128 filters to capture deep abstract features unique to alphanumeric classes
    model.add(layers.Conv2D(128, (3, 3), activation='relu', name="conv_3"))
    model.add(layers.BatchNormalization(name="batch_norm_3"))
    
    # 4. FLATTEN AND CLASSIFICATION HEAD
    # Flatten transforms the 2D feature maps into a 1D vector for the dense layers
    model.add(layers.Flatten(name="flatten"))
    
    # Fully connected dense layer for deep abstract feature mapping
    model.add(layers.Dense(256, activation='relu', name="dense_dense_1"))
    
    # Dropout prevents overfitting by turning off 50% of paths randomly during training
    model.add(layers.Dropout(0.5, name="dropout_regularization"))
    
    # Final Output Layer: Softmax converts raw network outputs into probability percentages
    model.add(layers.Dense(num_classes, activation='softmax', name="output_layer"))
    
    return model

if __name__ == "__main__":
    # Standalone validation block to inspect architecture dimensions
    print("🏗️ Initializing CNN architectural compilation...")
    sample_model = build_handwritten_cnn()
    
    print("\n📋 --- CNN MODEL SUMMARY BLUEPRINT ---")
    sample_model.summary()
    print("\n✅ CNN structure created successfully with no dimensional structural leaks!")