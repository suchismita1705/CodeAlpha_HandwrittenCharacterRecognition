import os
import numpy as np
import matplotlib.pyplot as plt

def run_exploratory_analysis():
    print("🔍 Step 1: Loading raw array cache from disk map...")
    raw_dir = os.path.join("data", "raw")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load numpy matrices
    X_train = np.load(os.path.join(raw_dir, "X_train.npy"))
    y_train = np.load(os.path.join(raw_dir, "y_train.npy"))
    
    print("\n📊 --- DATA INTEGRITY PROFILE ---")
    print(f"Training Images Matrix Shape : {X_train.shape} -> (Samples, Height, Width, Channels)")
    print(f"Training Labels Array Shape  : {y_train.shape} -> (Samples,)")
    print(f"Image Data Pixel Array Type  : {X_train.dtype}")
    print(f"Label Data Array Type        : {y_train.dtype}")
    print(f"Minimum Pixel Value Intensity: {X_train.min()}")
    print(f"Maximum Pixel Value Intensity: {X_train.max()}")
    
    # Define the official EMNIST Balanced 47-class alphanumeric sequence list
    # Indices 0-9 = Digits, 10-35 = Uppercase, 36-46 = Distinct Lowercase letters
    class_mapping = [
        '0','1','2','3','4','5','6','7','8','9',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','d','e','f','g','h','n','q','r','t'
    ]
    
    print(f"Total Configured Target Classes: {len(class_mapping)}")
    
    print("\n🎨 Step 2: Extracting random samples and correcting spatial transpositions...")
    
    # Set up a clean matplotlib plot grid (3 rows x 5 columns)
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    fig.suptitle("EMNIST Balanced Dataset Character Visualization (Corrected Orientation)", fontsize=16, fontweight='bold')
    
    # Seed generator for replication consistency
    np.random.seed(42)
    random_indices = np.random.choice(X_train.shape[0], size=15, replace=False)
    
    for idx, sample_idx in enumerate(random_indices):
        ax = axes[idx // 5, idx % 5]
        
        # Pull raw image and corresponding label integer
        raw_img = X_train[sample_idx]
        label_idx = y_train[sample_idx]
        character_label = class_mapping[label_idx]
        
        # FIX THE ORIENTATION BUG:
        # Squeeze out the single color channel (28, 28, 1) -> (28, 28)
        img_2d = np.squeeze(raw_img)
        # Transpose the rows and columns to flip and rotate right side up
        corrected_img = img_2d.T
        
        # Render image to canvas using grayscale spectrum maps
        ax.imshow(corrected_img, cmap='gray')
        ax.set_title(f"Label ID: {label_idx} ({character_label})", fontsize=12, color='darkblue', fontweight='semibold')
        ax.axis('off')
        
    plt.tight_layout()
    
    # Export the visual visualization figure plot out to reports directory
    output_path = os.path.join(reports_dir, "dataset_samples.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\n✅ EDA PIPELINE SUCCESSFUL!")
    print(f"🖼️ Sample grid visualization plot successfully generated and exported to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_exploratory_analysis()