import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_layer_explainability():
    print("🧠 Step 1: Ingesting preprocessed arrays and loading the model...")
    processed_dir = os.path.join("data", "processed")
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file missing at {model_path}. Please run training first!")
        return
        
    # Load model and processed test dataset split
    model = tf.keras.models.load_model(model_path)
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    
    print("🖼️ Step 2: Extracting a character sample image and converting to tensor...")
    # Select a stable sample from your testing split matrix
    sample_idx = 0
    input_image = X_test[sample_idx]  # Shape: (28, 28, 1)
    
    # Expand to 4D batch format tensor for input processing
    batch_tensor = np.expand_dims(input_image, axis=0)  # Shape: (1, 28, 28, 1)
    current_tensor = tf.convert_to_tensor(batch_tensor, dtype=tf.float32)
    
    print("🎯 Step 3: Computing layer activations via independent layer iteration loop...")
    # BULLEPROOF INTERMEDIATE EXTRACTION: Trace tensor activations layer-by-layer
    target_layer_names = ["conv_1", "conv_2", "conv_3"]
    activations = {}
    
    for layer in model.layers:
        # Feed tensor forward into the next structural operation block
        current_tensor = layer(current_tensor)
        
        # If the current layer matches our target, intercept its output map matrix values
        if layer.name in target_layer_names:
            activations[layer.name] = current_tensor.numpy()
    
    print("🎨 Step 4: Rendering feature maps extraction chart canvas layout...")
    # Plot configuration: Visualize 8 distinct channel filters across our 3 target convolutional layers
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    fig.suptitle("CNN Hidden Layer Feature Map Visualizations (What Our Model Sees)", fontsize=16, fontweight='bold')
    
    for layer_idx, layer_name in enumerate(target_layer_names):
        layer_activation = activations[layer_name]  # Shape: (1, Height, Width, Filters)
        num_filters = layer_activation.shape[-1]
        
        for filter_idx in range(8):
            ax = axes[layer_idx, filter_idx]
            if filter_idx < num_filters:
                # Extract the 2D feature activation grid from the targeted filter channel slot
                feature_map = layer_activation[0, :, :, filter_idx]
                ax.imshow(feature_map, cmap='viridis')
            
            ax.axis('off')
            if filter_idx == 0:
                ax.text(-5, layer_activation.shape[1] // 2, layer_name, fontsize=12, fontweight='bold', va='center', ha='right')
                
    plt.tight_layout()
    
    # Save the output visualization figure plot to the reports directory
    output_path = os.path.join(reports_dir, "feature_maps.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\n✅ EXPLAINABILITY REPORT GENERATED SUCCESSFULLY!")
    print(f"🖼️ Layer feature map diagram exported safely to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_layer_explainability()