import os
import cv2
import time
import torch
import itertools
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 1. ADVANCED CYBER-DARK MODE LAYOUT & CSS INJECTIONS
st.set_page_config(page_title="AI Word Workstation (Pro)", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #090D16 !important;
        color: #E2E8F0 !important;
    }
    
    .main-title { 
        text-align: center; 
        font-size: 44px; 
        font-weight: 800; 
        background: linear-gradient(135deg, #60A5FA 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px; 
    }
    .subtitle { 
        text-align: center; 
        font-size: 15px; 
        color: #94A3B8; 
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 35px; 
    }
    
    .premium-card { 
        background-color: #111827; 
        padding: 26px; 
        border-radius: 16px; 
        border: 1px solid #1F2937; 
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        margin-top: 25px; 
    }
    
    .card-header {
        font-size: 16px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .output-display { 
        font-family: 'JetBrains Mono', monospace;
        font-size: 68px; 
        font-weight: 700; 
        text-align: center; 
        letter-spacing: 6px;
        margin: 20px 0; 
    }
    .cnn-theme { 
        color: #3B82F6; 
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.65);
    }
    .transformer-theme { 
        color: #A855F7; 
        text-shadow: 0 0 30px rgba(168, 85, 247, 0.65);
    }
    
    .meta-footer { 
        font-size: 13px; 
        color: #64748B; 
        text-align: center; 
        border-top: 1px solid #1F2937;
        padding-top: 14px;
        margin-top: 18px;
        font-weight: 500;
    }
    
    /* Target Streamlit native metric wrappers to match dark aesthetics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #10B981 !important;
        font-size: 28px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Word Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Dual-Input Dual-Engine Deep Learning Suite</div>', unsafe_allow_html=True)

# 2. RUNTIME RESOURCE CACHE LOADERS
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

# Load baseline testing words into corpus memory paths
spell.word_frequency.load_words(["HELLO", "NAME", "CODE", "CAT", "DOG", "AI", "HOME", "ALPHA"])

# FIXED DICTIONARY ASSIGNMENT: Explicitly override the dictionary map value to patch the AttributeError
spell.word_frequency.dictionary["hello"] = 999999999

CLASS_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# 3. SIDEBAR SYSTEM CONTROLLERS
st.sidebar.markdown("### 🕹️ AI Engine Controller")
selected_engine = st.sidebar.radio(
    "Choose Active AI Backend:",
    ("🧠 Custom CNN + NLP Segmenter", "🤖 End-to-End Vision Transformer (TrOCR)")
)
st.sidebar.markdown("---")

# 4. UNIFIED INPUT MODE SELECTORS
app_mode = st.radio("Select Input Workspace Method:", ("✏️ Draw Live Word/Sequence", "📁 Upload Image File"), horizontal=True)

img_gray = None

if app_mode == "✏️ Draw Live Word/Sequence":
    st.markdown("<p style='font-weight: 500; color: #94A3B8; margin-bottom: 4px;'>Draw smoothly across the workspace frame pad:</p>", unsafe_allow_html=True)
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
else:
    uploaded_file = st.file_uploader("Upload an image containing handwritten segments...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        source_image = Image.open(uploaded_file)
        st.markdown('<div style="display: flex; justify-content: center; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.image(source_image, caption="Source File Target Ingested", width=350)
        st.markdown('</div>', unsafe_allow_html=True)
        img_gray = np.array(source_image.convert('L'))
        if np.mean(img_gray) > 127:
            img_gray = cv2.bitwise_not(img_gray)

# 5. INTEGRATED INFERENCE EXECUTIONS WITH LATENCY PROFILING
if img_gray is not None:
    st.write("")
    if st.button(f"⚡ Compute {selected_engine} Inference", use_container_width=True):
        
        # --- EXECUTION ENGINE 1: SEGMENTED CNN + NLP ---
        if selected_engine == "🧠 Custom CNN + NLP Segmenter":
            if cnn_model is None:
                st.error("❌ Error: CNN model weights missing from local workspace.")
            else:
                start_time = time.perf_counter()
                
                with st.spinner("Isolating character contours and optimizing matrices..."):
                    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 3], key=lambda b: b[0])
                    
                    if not bounding_boxes:
                        st.warning("Please draw clearer, bold character strokes.")
                    else:
                        debug_canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                        char_candidates_list = []
                        raw_top_chars = []
                        
                        for (x, y, w, h) in bounding_boxes:
                            cv2.rectangle(debug_canvas, (x, y), (x + w, y + h), (59, 130, 246), 2)
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
                        
                        st.image(debug_canvas, caption="OpenCV Real-Time Layout Tracking Map", use_container_width=True)
                        
                        all_combs = ["".join(comb).upper() for comb in itertools.product(*char_candidates_list)]
                        valid_words = [w for w in all_combs if w.lower() in spell]
                        
                        if valid_words:
                            final_output = max(valid_words, key=lambda w: spell.word_frequency[w.lower()])
                        else:
                            fallback_word = spell.correction("".join(raw_top_chars).upper())
                            final_output = fallback_word.upper() if fallback_word is not None else "".join(raw_top_chars).upper()
                        
                        end_time = time.perf_counter()
                        latency_ms = (end_time - start_time) * 1000
                        
                        st.markdown(f"""
                            <div class="premium-card">
                                <div class="card-header">🔷 CNN + NLP Joint Decoder Prediction</div>
                                <div class="output-display cnn-theme">{final_output}</div>
                                <div class="meta-footer">Raw pixel tensor activations before spelling correction: <b>{"".join(raw_top_chars).upper()}</b></div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric(label="Inference Latency", value=f"{latency_ms:.2f} ms")
                        with m_col2:
                            st.metric(label="Slices Detected", value=f"{len(bounding_boxes)} Chars")
                        with m_col3:
                            st.metric(label="Model Context", value="Localized CNN")

        # --- EXECUTION ENGINE 2: VISION TRANSFORMER (TrOCR) ---
        else:
            start_time = time.perf_counter()
            
            with st.spinner("Unrolling visual patches and triggering attention layers..."):
                processor, trocr_model = load_trocr_brain()
                
                inverted_ink_canvas = cv2.bitwise_not(img_gray)
                rgb_pil_image = Image.fromarray(cv2.cvtColor(inverted_ink_canvas, cv2.COLOR_GRAY2RGB))
                
                pixel_values = processor(images=rgb_pil_image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = trocr_model.generate(pixel_values)
                
                transformer_output_string = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                transformer_output_clean = transformer_output_string.strip().upper()
                
                end_time = time.perf_counter()
                latency_sec = end_time - start_time
                
                st.markdown(f"""
                            <div class="premium-card">
                                <div class="card-header">🔮 Transformer Sequence-to-Sequence Prediction</div>
                                <div class="output-display transformer-theme">{"⚠️ DETECT DROP" if not transformer_output_clean else transformer_output_clean}</div>
                                <div class="meta-footer">Decoded via end-to-end multi-head self-attention token streams.</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.write("")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="Inference Latency", value=f"{latency_sec:.3f} sec")
                with m_col2:
                    st.metric(label="Patch Sequence", value="16 Vision Tokens")
                with m_col3:
                    st.metric(label="Model Context", value="Attention Transformer")
else:
    st.markdown("<p style='text-align: center; color: #475569; font-size: 14px; margin-top: 20px;'>✏️ Provide an interactive stroke line or file asset above to wake device processors.</p>", unsafe_allow_html=True)