import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. APPLICATION GRAPHICAL LAYOUT CONFIGURATIONS
st.set_page_config(page_title="AI Handwritten Recognition", page_icon="✍️", layout="centered")

st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 18px; color: #4B5563; margin-bottom: 25px; }
    .prediction-box { background-color: #EFF6FF; padding: 25px; border-radius: 12px; border: 2px solid #BFDBFE; text-align: center; margin-top: 20px; }
    .metric-text { font-size: 65px; font-weight: bold; color: #2563EB; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Handwritten Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive Deep Learning UI Pad</div>', unsafe_allow_html=True)

# 2. RUNTIME MEMORY MODEL LOADER 
@st.cache_resource
def load_trained_brain():
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_trained_brain()

CLASS_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

if model is None:
    st.error("❌ Error: 'best_handwritten_model.keras' missing from saved_models folder. Please run your training script first!")
else:
    # 3. CHOOSE MODE INTERACTIVE NAVIGATION TABS
    app_mode = st.radio("Select Input Workspace Method:", ("✏️ Draw Live Character", "📁 Upload Image File"), horizontal=True)
    
    img_gray = None
    
    # --- MODE A: DRAWING BOARD CANVAS ---
    if app_mode == "✏️ Draw Live Character":
        st.write("Draw a single bold letter or digit inside the frame box below:")
        
        # Instantiate a black-background drawing pad layout
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=16, # Bold stroke lines resemble EMNIST thickness patterns
            stroke_color="#FFFFFF", # White ink
            background_color="#000000", # Black backdrop matching EMNIST
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True
        )
        
        # Check if user has initialized a stroke path on the matrix pad
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
            # Extract raw 4-channel RGBA frame and pull out grayscale layer
            rgba_array = canvas_result.image_data
            img_gray = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2GRAY)

    # --- MODE B: FILE UPLOADER STREAM ---
    else:
        uploaded_file = st.file_uploader("Upload an image file containing a single handwritten character...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            source_image = Image.open(uploaded_file)
            st.image(source_image, caption="Uploaded Original Handwriting", width=200)
            
            # Convert user image directly into a grayscale map matrix array
            img_gray = np.array(source_image.convert('L'))
            
            # Auto-invert if user uploaded dark ink on a light background canvas
            if np.mean(img_gray) > 127:
                img_gray = cv2.bitwise_not(img_gray)

    # 4. COMPUTER VISION PREPROCESSING AND MODEL INFERENCE PIPELINE
    if img_gray is not None:
        # Resize dimensions down using area interpolation to protect stroke resolution mapping configurations (28x28)
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Standardize matrix floating limits to continuous values [0.0, 1.0]
        img_norm = img_resized.astype('float32') / 255.0
        
        # Display model insight thumbnail panel sidebar tracking window layout
        st.image(img_norm, caption="Model's Eye View (Processed 28x28)", width=120)
        
        # Reshape array structure to complete 4D batch tensor expectations: (1, 28, 28, 1)
        processed_tensor = np.reshape(img_norm, (1, 28, 28, 1))
        
        st.markdown("---")
        if st.button("🚀 Run AI Character Prediction", use_container_width=True):
            with st.spinner("Analyzing structural drawing stroke geometry parameters..."):
                raw_probabilities = model.predict(processed_tensor, verbose=0)
                pred_class_idx = np.argmax(raw_probabilities, axis=1)[0]
                confidence_score = raw_probabilities[0][pred_class_idx] * 100
                predicted_character = CLASS_MAPPING[pred_class_idx]
                
                # Render interface dashboard notification cards
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.write("### Predicted Alphanumeric Character")
                st.markdown(f'<div class="metric-text">{predicted_character}</div>', unsafe_allow_html=True)
                st.write(f"**Network Classification Confidence Rating:** {confidence_score:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)