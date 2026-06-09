import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def execute_model_evaluation():
    print("📦 Step 1: Loading test dataset splits and trained model cache...")
    processed_dir = os.path.join("data", "processed")
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load preprocessed test arrays
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    # 2. Load the trained CNN model
    if not os.path.exists(model_path):
        print(f"❌ Error: Compiled model file not found at {model_path}. Please run training first!")
        return
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully from disk.")
    
    # Define EMNIST Balanced 47-class sequence layout map
    class_mapping = [
        '0','1','2','3','4','5','6','7','8','9',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','d','e','f','g','h','n','q','r','t'
    ]
    
    print("\n🎯 Step 2: Generating model predictions across unseen image streams...")
    # Generate raw probability arrays for each test sample image
    raw_predictions = model.predict(X_test, batch_size=256, verbose=1)
    
    # Extract the class index with the highest probability score for each image
    y_pred = np.argmax(raw_predictions, axis=1)
    
    print("\n📊 Step 3: Compiling textual Classification Metrics Report...")
    # Compute Precision, Recall, and F1-Scores across all 47 classes
    metrics_report = classification_report(
        y_test, 
        y_pred, 
        target_names=class_mapping,
        digits=4
    )
    print("\n=== COMPLETE TEST DATA REPORT ===")
    print(metrics_report)
    
    # Save text report to file
    with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
        f.write(metrics_report)
        
    print("\n🎨 Step 4: Generating and rendering Confusion Matrix Heatmap graph...")
    # Compute the 47x47 matrix values
    cm = confusion_matrix(y_test, y_pred)
    
    # Create an extended canvas grid for crisp scannability
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_mapping, 
        yticklabels=class_mapping,
        cbar=True,
        annot_kws={"size": 8}
    )
    
    plt.title("EMNIST Balanced 47-Class Confusion Matrix Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Predicted Label Target Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual True Handwriting Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    # Export graphic plot representation to reports directory
    heatmap_out_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(heatmap_out_path, dpi=300)
    plt.close()
    
    print(f"\n✅ EVALUATION COMPLETION SUCCESSFUL!")
    print(f"📄 Text metrics exported to: {os.path.abspath(os.path.join(reports_dir, 'classification_report.txt'))}")
    print(f"🖼️ Heatmap diagram exported to: {os.path.abspath(heatmap_out_path)}")

if __name__ == "__main__":
    execute_model_evaluation()