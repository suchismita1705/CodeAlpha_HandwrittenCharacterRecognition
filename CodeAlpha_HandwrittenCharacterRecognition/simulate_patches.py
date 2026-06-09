import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def simulate_transformer_patching():
    print("🚀 Step 1: Locating an existing handwritten sample image asset...")
    reports_dir = "reports"
    output_path = os.path.join(reports_dir, "transformer_patches_visualized.png")
    
    # We will use your existing confusion matrix image as a high-contrast structural text sample to slice up!
    sample_source = os.path.join(reports_dir, "confusion_matrix.png")
    
    if not os.path.exists(sample_source):
        # Fallback safeguard: Create a dummy word matrix if the report image isn't found
        print("⚠️ Sample image report not found, generating a synthetic text string block...")
        raw_img = np.zeros((112, 112), dtype=np.uint8)
        cv2.putText(raw_img, "AI", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
    else:
        # Load the real image asset and convert straight to grayscale
        raw_img = cv2.imread(sample_source, cv2.IMREAD_GRAYSCALE)
        # Crop a clean square block from the upper corner where structural labels sit
        raw_img = cv2.resize(raw_img[:400, :400], (112, 112))

    print("✂️ Step 2: Slicing the image space into a 1D sequence of Transformer patches...")
    # Define our grid dimensions: We will slice the 112x112 image into a 4x4 grid of 28x28 patches
    image_height, image_width = raw_img.shape
    patch_size = 28
    
    patches = []
    # Loop over rows and columns sequentially to capture spatial patch grids
    for i in range(0, image_height, patch_size):
        for j in range(0, image_width, patch_size):
            patch = raw_img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            
    print("🎨 Step 3: Compiling layout to show original vs. linear token sequence...")
    # Setup a canvas layout: Top row holds the full image, bottom row holds the linear sequence of tokens
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("How a Vision Transformer (ViT) Sees an Image as a Sentence", fontsize=16, fontweight='bold')
    
    # Plot 1: The full input image
    ax_orig = plt.subplot2grid((2, 16), (0, 4), colspan=8)
    ax_orig.imshow(raw_img, cmap='gray')
    ax_orig.set_title("Original 2D Input Image Layer (112 x 112)", fontsize=11, fontweight='bold', pad=8)
    ax_orig.axis('off')
    
    # Plot 2-17: The unrolled linear vector patch tokens (1D sequence array)
    for idx, patch in enumerate(patches):
        ax_patch = plt.subplot2grid((2, 16), (1, idx))
        ax_patch.imshow(patch, cmap='viridis') # Using viridis color maps to isolate fine patch line features
        ax_patch.axis('off')
        ax_patch.set_title(f"Tkn {idx+1}", fontsize=9, color='#4B5563')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\n✅ SIMULATION EXPERIMENT COMPLETE!")
    print(f"🖼️ Transformer patch map diagram exported to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    simulate_transformer_patching()