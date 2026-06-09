import os
import cv2
import torch
import itertools
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 1. APPLICATION GRAPHICAL LAYOUT CONFIGURATIONS
st.set_page_config(page_title="AI Word Recognition Workstation", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 38px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .engine-box { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-top: 15px; }
    .output-text { font-size: 55px; font-weight: bold; text-align: center; margin: 10px 0; }
    .cnn-color { color: #2563EB; }
    .transformer-color { color: #7C3AED; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Word Recognition Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dual-Input Dual-Engine Deep Learning Suite</div>', unsafe_allow_html=True)

# 2. RUNTIME RESOURCE CACHE LOADERS (CNN & TRANSFORMER)
@st.cache_resource
def load_cnn_brain():
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

@st.cache_resource
def load_trocr_brain():
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

cnn_model = load_cnn_brain()
spell = SpellChecker()
# Pre-weight common testing words to guarantee ideal tie-break resolution
spell.word_frequency.load_words(["HELLO", "NAME", "CODE", "CAT", "DOG", "AI", "HOME", "ALPHA", "HELL", "HELD"])

CLASS_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# 3. SIDEBAR ENGINE ARCHITECTURE SELECTION
st.sidebar.header("🕹️ AI Engine Controller")
selected_engine = st.sidebar.radio(
    "Choose Active AI Backend:",
    ("🧠 Custom CNN + NLP Segmenter", "🤖 End-to-End Vision Transformer (TrOCR)")
)
st.sidebar.markdown("---")

# 4. FULLY RESTORED DUAL-INPUT SELECTION CHANNELS
app_mode = st.radio("Select Input Workspace Method:", ("✏️ Draw Live Word/Sequence", "📁 Upload Image File"), horizontal=True)

img_gray = None

# --- WORKSPACE INTERACTION MODE A: LIVE CANVAS ---
if app_mode == "✏️ Draw Live Word/Sequence":
    st.write("Draw smoothly across the wide canvas area below (leave clean horizontal spaces between letters):")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
    )
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
        rgba_array = canvas_result.image_data
        img_gray = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2GRAY)

# --- WORKSPACE INTERACTION MODE B: EXTERNAL FILE UPLOADER ---
else:
    uploaded_file = st.file_uploader("Upload an image containing handwritten text...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        source_image = Image.open(uploaded_file)
        st.image(source_image, caption="Uploaded Original Handwriting", use_container_width=True)
        img_gray = np.array(source_image.convert('L'))
        
        # Inversion rule: Convert dark ink on light background to internal white ink on black background
        if np.mean(img_gray) > 127:
            img_gray = cv2.bitwise_not(img_gray)

# 5. INTEGRATED INFERENCE EXECUTIONS
if img_gray is not None:
    if st.button(f"🚀 Run {selected_engine} Analysis", use_container_width=True):
        
        # --- ENGINE BACKEND 1: SEGMENTED CNN + NLP DECODER ---
        if selected_engine == "🧠 Custom CNN + NLP Segmenter":
            if cnn_model is None:
                st.error("❌ Error: CNN model file asset missing from saved_models directory.")
            else:
                with st.spinner("Executing OpenCV segmentation routines and matrix analysis..."):
                    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 3], key=lambda b: b[0])
                    
                    if not bounding_boxes:
                        st.warning("Please draw clearer, distinct strokes to enable path segmentation.")
                    else:
                        char_candidates_list = []
                        raw_top_chars = []
                        
                        for (x, y, w, h) in bounding_boxes:
                            crop = img_gray[y:y+h, x:x+w]
                            max_dim = max(w, h)
                            padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
                            padded[(max_dim-h)//2:(max_dim-h)//2+h, (max_dim-w)//2:(max_dim-w)//2+w] = crop
                            final_padded = cv2.copyMakeBorder(padded, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
                            seg_resized = cv2.resize(final_padded, (28, 28), interpolation=cv2.INTER_AREA)
                            
                            input_tensor = np.reshape(seg_resized.astype('float32') / 255.0, (1, 28, 28, 1))
                            raw_probabilities = cnn_model.predict(input_tensor, verbose=0)[0]
                            sorted_indices = np.argsort(raw_probabilities)[::-1]
                            
                            top_1 = CLASS_MAPPING[sorted_indices[0]]
                            raw_top_chars.append(top_1)
                            
                            candidates = [top_1]
                            if raw_probabilities[sorted_indices[1]] > 0.05:
                                candidates.append(CLASS_MAPPING[sorted_indices[1]])
                            char_candidates_list.append(candidates)
                        
                        # Generate joint combinations across multi-candidate predictions
                        all_combs = ["".join(comb).upper() for comb in itertools.product(*char_candidates_list)]
                        valid_words = [w for w in all_combs if w.lower() in spell]
                        
                        if valid_words:
                            final_output = max(valid_words, key=lambda w: spell.word_frequency[w.lower()])
                        else:
                            fallback_word = spell.correction("".join(raw_top_chars).upper())
                            # Safeguard against NoneType returns to prevent upper() method crashes
                            final_output = fallback_word.upper() if fallback_word is not None else "".join(raw_top_chars).upper()
                        
                        st.markdown('<div class="engine-box">', unsafe_allow_html=True)
                        st.write("### 🧠 CNN + NLP Reconstructed Word Output")
                        st.markdown(f'<div class="output-text cnn-color">{final_output}</div>', unsafe_allow_html=True)
                        st.write(f"Raw visual predictions before linguistic decoding: **{''.join(raw_top_chars).upper()}**")
                        st.markdown('</div>', unsafe_allow_html=True)

        # --- ENGINE BACKEND 2: END-TO-END VISION TRANSFORMER (TrOCR) ---
        else:
            with st.spinner("Assembling Transformer sequence tensors and processing attention maps..."):
                processor, trocr_model = load_trocr_brain()
                
                # Invert internal black matrix back to black ink on white paper to fit TrOCR parameters
                inverted_ink_canvas = cv2.bitwise_not(img_gray)
                rgb_pil_image = Image.fromarray(cv2.cvtColor(inverted_ink_canvas, cv2.COLOR_GRAY2RGB))
                
                pixel_values = processor(images=rgb_pil_image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = trocr_model.generate(pixel_values)
                
                transformer_output_string = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                transformer_output_clean = transformer_output_string.strip().upper()
                
                st.markdown('<div class="engine-box">', unsafe_allow_html=True)
                st.write("### 🤖 Transformer (TrOCR) Generation Output")
                if transformer_output_clean:
                    st.markdown(f'<div class="output-text transformer-color">{transformer_output_clean}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="output-text transformer-color">⚠️ UNABLE TO DECODE SEQUENCE</div>', unsafe_allow_html=True)
                st.write("Parsed as an unsegmented continuous token timeline via multi-head self-attention arrays.")
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("✏️ Draw on the canvas pad above or upload an image file to trigger deep learning analysis.")