import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. APPLICATION GRAPHICAL LAYOUT CONFIGURATIONS
st.set_page_config(page_title="AI Word Recognition", page_icon="✍️", layout="centered")

st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 18px; color: #4B5563; margin-bottom: 25px; }
    .prediction-box { background-color: #F0FDF4; padding: 25px; border-radius: 12px; border: 2px solid #BBF7D0; text-align: center; margin-top: 20px; }
    .word-text { font-size: 70px; font-weight: bold; color: #16A34A; letter-spacing: 5px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Handwritten Word Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Multi-Character Segmentation Pipeline</div>', unsafe_allow_html=True)

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
    st.error("❌ Error: 'best_handwritten_model.keras' missing. Please run your training script first!")
else:
    app_mode = st.radio("Select Input Workspace Method:", ("✏️ Draw Live Word/Sequence", "📁 Upload Image File"), horizontal=True)
    
    img_gray = None
    
    # --- MODE A: EXPANDED DRAWING CANVAS ---
    if app_mode == "✏️ Draw Live Word/Sequence":
        st.write("Draw multiple letters or numbers across the wide canvas area below (leave clean horizontal spaces between them):")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=200,
            width=600,  # Expanded horizontally to allow multi-character text lines
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True
        )
        
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
            rgba_array = canvas_result.image_data
            img_gray = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2GRAY)

    # --- MODE B: FILE UPLOADER STREAM ---
    else:
        uploaded_file = st.file_uploader("Upload an image file containing handwritten text...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            source_image = Image.open(uploaded_file)
            st.image(source_image, caption="Uploaded Original Handwriting", use_container_width=True)
            img_gray = np.array(source_image.convert('L'))
            if np.mean(img_gray) > 127:
                img_gray = cv2.bitwise_not(img_gray)

    # 4. ADVANCED OPENCV IMAGE SEGMENTATION AND BATCH INFERENCE
    if img_gray is not None:
        # Find external contours bounding individual characters
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            # Filter out tiny pixel noise blocks
            if w > 3 and h > 3:
                bounding_boxes.append((x, y, w, h))
                
        # INDUSTRY INSIGHT: Sort bounding boxes left-to-right so the word reads correctly!
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        
        if len(bounding_boxes) == 0:
            st.warning("⚠️ Write or upload a clearer character line to begin analysis.")
        else:
            st.markdown(f"### 🎯 Segmentation Diagnostics: Detected **{len(bounding_boxes)}** Distinct Characters")
            
            # Render visual debug layout highlighting segmentation blocks
            debug_canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            processed_segments = []
            
            for idx, (x, y, w, h) in enumerate(bounding_boxes):
                # Draw visual green bounding box tracking lines on debug image canvas
                cv2.rectangle(debug_canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_canvas, str(idx+1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Slicing out segment region of interest
                crop = img_gray[y:y+h, x:x+w]
                
                # Advanced Padding Fix: Make segment square and inject uniform border bounds 
                # This perfectly mimics the balanced spatial profile of the EMNIST training samples
                max_dim = max(w, h)
                padded_square = np.zeros((max_dim, max_dim), dtype=np.uint8)
                pad_y = (max_dim - h) // 2
                pad_x = (max_dim - w) // 2
                padded_square[pad_y:pad_y+h, pad_x:pad_x+w] = crop
                
                # Inject extra perimeter cushion margin to keep stroke boundaries away from the outer edges
                final_padded = cv2.copyMakeBorder(padded_square, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
                
                # Scale straight down to 28x28 matrix inputs
                seg_resized = cv2.resize(final_padded, (28, 28), interpolation=cv2.INTER_AREA)
                seg_norm = seg_resized.astype('float32') / 255.0
                processed_segments.append(seg_norm)
                
            # Render our custom green segmentation bounding boxes live to layout panel screen
            st.image(debug_canvas, caption="OpenCV Real-Time Word Segmentation Trace Mapping", use_container_width=True)
            
            st.markdown("---")
            if st.button("🚀 Run Advanced Word Prediction Sequence", use_container_width=True):
                predicted_word_chars = []
                
                # Show individual thumbnail segment classification frames dynamically inside columns
                cols = st.columns(min(len(processed_segments), 8))
                
                for idx, seg_tensor in enumerate(processed_segments):
                    # Format single slice tensor container shape to match 4D expectations: (1, 28, 28, 1)
                    input_tensor = np.reshape(seg_tensor, (1, 28, 28, 1))
                    
                    # Fetch raw probabilities distribution maps
                    probs = model.predict(input_tensor, verbose=0)
                    class_id = np.argmax(probs, axis=1)[0]
                    char_out = CLASS_MAPPING[class_id]
                    predicted_word_chars.append(char_out)
                    
                    # Output individual segment visualization columns
                    with cols[idx % 8]:
                        st.image(seg_tensor, caption=f"#{idx+1}: '{char_out}'", width=65)
                        
                # Reconstruct full matched string
                final_word = "".join(predicted_word_chars)
                
                # Render comprehensive portfolio result dashboard cards
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.write("### AI Reconstructed Word Output Sequence")
                st.markdown(f'<div class="word-text">{final_word}</div>', unsafe_allow_html=True)
                st.write(f"Processed batch sequence mapping execution successful.")
                st.markdown('</div>', unsafe_allow_html=True)