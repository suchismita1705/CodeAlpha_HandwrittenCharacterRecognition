import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# 1. WEB APP INTERFACE GRAPHICAL STYLE CONFIGURATIONS
st.set_page_config(
    page_title="AI Handwritten Recognition", 
    page_icon="✍️", 
    layout="centered"
)

# Custom CSS styling injection for clean card rendering layout blocks
st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 18px; color: #4B5563; margin-bottom: 30px; }
    .prediction-box { background-color: #EFF6FF; padding: 25px; border-radius: 12px; border: 2px solid #BFDBFE; text-align: center; margin-top: 20px; }
    .metric-text { font-size: 65px; font-weight: bold; color: #2563EB; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Handwritten Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Engineering B.Tech Portfolio Project</div>', unsafe_allow_html=True)

# 2. CACHE TRAINED BRAIN GRAPH IN RUNTIME MEMORY
@st.cache_resource
def load_trained_brain():
    """Loads the serialized Keras CNN model asset graph from local disk."""
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_trained_brain()

# Official EMNIST Balanced 47-class sequence mapping dictionary list
CLASS_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

if model is None:
    st.error("❌ Error: 'best_handwritten_model.keras' was not found inside your saved_models folder. Please run src/training/train.py first!")
else:
    # 3. INTERACTIVE FILE DROPZONE WIDGET
    uploaded_file = st.file_uploader(
        "Upload an image file containing a single handwritten letter or digit...", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # Load and display original file canvas matrix layout using PIL
        source_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(source_image, caption="Uploaded Original Handwriting", use_container_width=True)
            
        # 4. DIGITAL SIGNAL PROCESSING COMPUTER VISION PIPELINE
        # Convert PIL storage instance directly into a clean single-channel grayscale NumPy array
        img_gray = np.array(source_image.convert('L'))
        
        # SMART CANVAS BACKGROUND CORRECTION:
        # If average intensity > 127, it means dark ink on a light background.
        # We invert it using bitwise_not to match the white ink on a black background that EMNIST expects.
        if np.mean(img_gray) > 127:
            img_gray = cv2.bitwise_not(img_gray)
            
        # Downscale image using area-relation interpolation to fit model footprint dimensions (28x28)
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Standardize matrix floating limits from [0, 255] to clean [0.0, 1.0] continuous space
        img_norm = img_resized.astype('float32') / 255.0
        
        with col2:
            st.image(img_norm, caption="Model's Eye View (Processed 28x28)", width=170)
            
        # Reshape array structure container to fulfill 4D Tensor Batch expectations: (1, 28, 28, 1)
        processed_tensor = np.reshape(img_norm, (1, 28, 28, 1))
        
        # 5. LIVE DEEP LEARNING MODEL INFERENCE PIPELINE
        st.markdown("---")
        if st.button("🚀 Run AI Character Prediction", use_container_width=True):
            with st.spinner("Analyzing stroke geometry and feature configurations..."):
                # Run the forward pass prediction mapping
                raw_probabilities = model.predict(processed_tensor, verbose=0)
                
                # Fetch highest index probability coordinates
                pred_class_idx = np.argmax(raw_probabilities, axis=1)[0]
                confidence_score = raw_probabilities[0][pred_class_idx] * 100
                predicted_character = CLASS_MAPPING[pred_class_idx]
                
                # Render polished user interface dashboard cards
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.write("### Predicted Alphanumeric Character")
                st.markdown(f'<div class="metric-text">{predicted_character}</div>', unsafe_allow_html=True)
                st.write(f"**Network Classification Confidence Rating:** {confidence_score:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Provide system context tips if predictions fall under baseline safety bounds
                if confidence_score < 75.0:
                    st.warning("💡 Pro-Tip: For optimal accuracy, make sure your character is written clearly and fills the majority of the image frame boundaries!")