import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def run_transformer_ocr():
    print("🧠 Step 1: Initializing pre-trained TrOCR Transformer architecture from Hugging Face...")
    # Load the specialized visual feature processor and text token sequence decoder
    # This model is pre-trained explicitly on thousands of unstructured handwritten documents
    model_name = "microsoft/trocr-base-handwritten"
    
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Force execution on local CPU space safely
    device = torch.device("cpu")
    model.to(device)
    
    print("🖼️ Step 2: Preparing a sample handwritten word image asset...")
    test_image_path = "test_word.png"
    
    # If you don't have a test image ready, this block automatically creates a clean, high-contrast 
    # handwritten-style image file right in your directory so the script can run instantly!
    if not os.path.exists(test_image_path):
        print(f"📝 '{test_image_path}' not detected. Generating a synthetic handwritten word block...")
        canvas = np.zeros((150, 450, 3), dtype=np.uint8)
        # Generate clean white text on a black background
        cv2.putText(canvas, "CODEALPHA", (25, 95), cv2.FONT_HERSHEY_TRIPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(test_image_path, canvas)
        
    # Load the target handwriting image using Pillow for Hugging Face compatibility
    raw_image = Image.open(test_image_path).convert("RGB")
    
    print("✂️ Step 3: Extracting patch embeddings and processing image states...")
    # The processor automatically resizes, normalizes, and segments the input into standard patches
    pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values.to(device)
    
    print("🔮 Step 4: Launching autoregressive text token generation pass...")
    # The transformer views all patches via Self-Attention and predicts the letters step-by-step
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        
    print("📝 Step 5: Decoding linguistic output sequence parameters...")
    # Map the generated numerical ID arrays back into human-readable text characters
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n==================================================")
    print("🎯 TRANSFORMER INFERENCE EXECUTION SUCCESSFUL!")
    print(f"🖼️ Target Source File Analyzed: {os.path.abspath(test_image_path)}")
    print(f"✨ Reconstructed Text Output: '{predicted_text}'")
    print("==================================================")

if __name__ == "__main__":
    run_transformer_ocr()